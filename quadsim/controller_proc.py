"""The G23 controller as a child process of sim_node.

Started after the ROS node is up, with the controller repo as its working directory (policies, config and the
bundled ONNX Runtime are found relative to it), its output forwarded line by line to the sim log with the
controller's prefix. If it exits, sim_node reports a `controller` event with state `exited` and stops: a legged
robot without its controller is a robot lying on the floor.

Which controller is a `ControllerSpec` chosen by `robot.controller.kind` (config/sim.yaml):

  motorcontrol   SyncAI-Device-MotorControl, `robotdog_control` (the default): no arguments, configured only by
                 ROBOTDOG_* environment variables; its state is read from its operator socket, not from stdout
  g2_deploy_rs   the original controller, `g2_deploy_rs sim ros`; its [SAFETY] / [STATE] stdout lines are scraped
"""
from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

DEFAULT_KIND = "motorcontrol"


def _arch() -> str:
    return platform.machine()


def _g2_env(repo: str, ctl_cfg: dict) -> dict[str, str]:
    env = {"G2_LMA_STATE_SOURCE": "ground_truth"}
    arch = {"x86_64": "x64", "aarch64": "aarch64"}.get(_arch(), _arch())
    lib = os.path.join(repo, f"onnxruntime-linux-{arch}-1.19.2", "lib", "libonnxruntime.so")
    if os.path.exists(lib):
        env["ORT_DYLIB_PATH"] = lib
    return env


def _g2_parse(text: str, proc: "ControllerProcess"):
    if "[SAFETY]" in text and ("lock engaged" in text or "safety triggered" in text.lower()):
        proc.safety_locked = True
    if "[STATE]" in text and "---->" in text:
        proc.state = text.split("---->")[-1].strip()


def _motorcontrol_env(repo: str, ctl_cfg: dict) -> dict[str, str]:
    mc = ctl_cfg.get("motorcontrol", {}) or {}
    env = {"ROBOTDOG_BACKEND": "sim", "ROBOTDOG_ROBOT": "g2", "ROBOTDOG_COMMAND_SOURCE": "udp",
           "ROBOTDOG_STATE_PUBLISHER": "udp", "ROBOTDOG_TELEMETRY": "off",
           "ROBOTDOG_SHUTDOWN_SETTLE_SECS": "0",                # no real motors to settle: exit on SIGINT at once
           "ROBOTDOG_SIM_TARGET": f"127.0.0.1:{int(mc.get('cmd_port', 40001))}",
           "ROBOTDOG_SIM_JOINT_PORT": str(_port(mc.get("joint_addr", "127.0.0.1:40002"))),
           "ROBOTDOG_SIM_BASE_PORT": str(_port(mc.get("base_addr", "127.0.0.1:40003"))),
           "ROBOTDOG_OPERATOR_UDP_PORT": str(_port(mc.get("operator_addr", "127.0.0.1:40020")))}
    lib = os.path.join(repo, "third_party", _arch(), "lib", "libonnxruntime.so.1.19.2")
    if os.path.exists(lib):
        env["ROBOTDOG_ONNXRUNTIME_LIB"] = lib
    return env


def _port(addr) -> int:
    return int(str(addr).rpartition(":")[2])


@dataclass(frozen=True)
class ControllerSpec:
    kind: str
    repo_env: str                                   # environment variable naming the controller repo
    repo_default: str                               # ... and its default (the container mount)
    exe_rel: str                                    # the binary, relative to the repo
    args: tuple[str, ...]
    log_prefix: str
    build_hint: str                                 # what to run when the binary is missing
    env_defaults: Callable[[str, dict], dict[str, str]]
    parse_line: Callable[[str, "ControllerProcess"], None] | None = None


SPECS: dict[str, ControllerSpec] = {
    "motorcontrol": ControllerSpec(
        kind="motorcontrol", repo_env="MOTORCONTROL_REPO", repo_default="/mc",
        exe_rel=os.path.join("target", "release", "robotdog_control"), args=(), log_prefix="[mc]",
        build_hint="docker/build_g23.sh (kind motorcontrol)", env_defaults=_motorcontrol_env),
    "g2_deploy_rs": ControllerSpec(
        kind="g2_deploy_rs", repo_env="G2_DEPLOY_RS", repo_default="/g2",
        exe_rel=os.path.join("target", "release", "g2_deploy_rs"), args=("sim", "ros"), log_prefix="[g2]",
        build_hint="docker/build_g23.sh --kind g2", env_defaults=_g2_env, parse_line=_g2_parse),
}


def controller_spec(ctl_cfg: dict) -> ControllerSpec:
    kind = str(ctl_cfg.get("kind") or DEFAULT_KIND)
    if kind not in SPECS:
        raise ValueError(f"robot.controller.kind={kind!r}: expected one of {', '.join(SPECS)}")
    return SPECS[kind]


class ControllerProcess:
    def __init__(self, ctl_cfg: dict, log, repo: str | None = None):
        self.log = log
        self.spec = controller_spec(ctl_cfg)
        self.repo = repo or os.environ.get(self.spec.repo_env, self.spec.repo_default)
        exe = ctl_cfg.get("exec") or os.path.join(self.repo, self.spec.exe_rel)
        self.exe = exe if os.path.isabs(exe) else os.path.join(self.repo, exe)
        args = ctl_cfg.get("args")
        self.args = [str(a) for a in (self.spec.args if args is None else args)]
        self.env = dict(os.environ)
        for k, v in self.spec.env_defaults(self.repo, ctl_cfg).items():
            self.env.setdefault(k, v)
        for k, v in (ctl_cfg.get("env") or {}).items():
            self.env[str(k)] = str(v)
        self.proc: subprocess.Popen | None = None
        self.lines = 0
        self.safety_locked = False                 # g2_deploy_rs: it printed its safety lock (needs RESET_SAFETY)
        self.state = None                          # g2_deploy_rs: last "[STATE] A ------------> B" target

    @property
    def available(self) -> bool:
        return os.path.isfile(self.exe) and os.access(self.exe, os.X_OK)

    def start(self):
        self.proc = subprocess.Popen([self.exe] + self.args, cwd=self.repo, env=self.env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        self.log(f"controller started: {self.exe} {' '.join(self.args)} (pid {self.proc.pid}, cwd {self.repo})")

    def _pump(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines += 1
            text = line.rstrip()
            if self.spec.parse_line is not None:
                self.spec.parse_line(text, self)
            self.log(f"{self.spec.log_prefix} {text}")

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def stop(self, timeout: float = 3.0):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            t0 = time.monotonic()
            while self.proc.poll() is None and time.monotonic() - t0 < timeout:
                time.sleep(0.05)
            if self.proc.poll() is None:
                self.proc.kill()
        except OSError:
            pass
