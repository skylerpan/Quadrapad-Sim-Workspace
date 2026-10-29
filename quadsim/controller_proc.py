"""The G23 controller (g2_deploy_rs `sim ros`) as a child process of sim_node.

Started after the ROS node is up, with the controller repo as its working directory (policies, config and
the bundled ONNX Runtime are found relative to it), its output forwarded line by line to the sim log with a
`[g2]` prefix. If it exits, sim_node reports a `controller` event with state `exited` and stops: a legged
robot without its controller is a robot lying on the floor.
"""
from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
import time


class ControllerProcess:
    def __init__(self, ctl_cfg: dict, log, repo: str | None = None):
        self.log = log
        self.repo = repo or os.environ.get("G2_DEPLOY_RS", "/g2")
        exe = ctl_cfg.get("exec") or os.path.join(self.repo, "target", "release", "g2_deploy_rs")
        self.exe = exe if os.path.isabs(exe) else os.path.join(self.repo, exe)
        self.args = [str(a) for a in (ctl_cfg.get("args") or ["sim", "ros"])]
        self.env = dict(os.environ)
        self.env.setdefault("G2_LMA_STATE_SOURCE", "ground_truth")
        for k, v in (ctl_cfg.get("env") or {}).items():
            self.env[str(k)] = str(v)
        if "ORT_DYLIB_PATH" not in self.env:
            arch = {"x86_64": "x64", "aarch64": "aarch64"}.get(platform.machine(), platform.machine())
            lib = os.path.join(self.repo, f"onnxruntime-linux-{arch}-1.19.2", "lib", "libonnxruntime.so")
            if os.path.exists(lib):
                self.env["ORT_DYLIB_PATH"] = lib
        self.proc: subprocess.Popen | None = None
        self.lines = 0

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
            self.log("[g2] " + line.rstrip())

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
