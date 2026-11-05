#!/usr/bin/env python3
"""The reference simulation end to end: their mujoco_simulation_G23.py (viewer on) + the Rust controller.

Runs inside the quadsim-g23 container (controller repo at /g2). Starts their sim, then the controller, sends the
stand sequence (MODE STAND, MODE <policy>, MODE RL) on udp:50051 and stays up until Ctrl-C / --seconds. Both
logs are relayed with [sim] / [g2] prefixes. Same UDP ports as quadsim's g23 mode, so stop that container first.

  QUADSIM_G23=1 QUADSIM_GUI=1 docker/run.sh "python3 scripts/run_theirs.py --scene stair"
  python3 scripts/run_theirs.py --send "MODE 1"                 # from the host too (host network)
  python3 scripts/run_theirs.py --send "AXES 0.4 0 0" --repeat 8  # stream it (their 300 ms watchdog)

Scenes: flat (plane), stair (their mixed terrain: three 5-step lanes 1 m ahead, risers 10 cm at y=+2.2, 15 cm at
y=0, 20 cm at y=-2.2; rough tiles behind the spawn), terrain (slope / bridge / ramps).
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time

CMD_ADDR = ("127.0.0.1", 50051)
POLICIES = {0: "PPO", 1: "HimLoco", 4: "IsaacSim"}


def _cmd_sock() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(text: str, sock: socket.socket | None = None, quiet: bool = False):
    """One text line to the controller's command port (what their ROS bridge sends)."""
    (sock or _cmd_sock()).sendto((text + "\n").encode(), CMD_ADDR)
    if not quiet:
        print(f"[theirs] sent {text}", flush=True)


def _ensure_colorama():
    """Their sim imports colorama for coloured prints; drop a no-op stub into the user site when the image lacks it."""
    try:
        import colorama  # noqa: F401
        return
    except ImportError:
        pass
    site = os.path.join(os.path.expanduser("~"), ".local", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages")
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "colorama.py"), "w") as f:
        f.write("class _P:\n    def __getattr__(self, _):\n        return ''\n\n\n"
                "def init(**_):\n    pass\n\n\nFore = _P()\nStyle = _P()\n")


def start_sim(repo: str, scene: str, headless: bool, extra_env: dict | None = None) -> subprocess.Popen:
    sim_dir = os.path.join(repo, "interface", "robot", "simulation")
    env = dict(os.environ, G2_SIM_ROS_ASCII_TELEMETRY="0", G2_SIM_SCENE=scene, PYTHONUNBUFFERED="1")
    if headless:
        env["G2_SIM_HEADLESS"] = "1"
    env.update(extra_env or {})
    _ensure_colorama()
    args = [sys.executable, "-u", "mujoco_simulation_G23.py", "--scene", scene] + (["--headless"] if headless else [])
    return subprocess.Popen(args, cwd=sim_dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def start_controller(repo: str, extra_env: dict | None = None) -> subprocess.Popen:
    env = dict(os.environ, G2_LMA_STATE_SOURCE="ground_truth")
    lib = os.path.join(repo, "onnxruntime-linux-aarch64-1.19.2", "lib", "libonnxruntime.so")
    if "ORT_DYLIB_PATH" not in env and os.path.exists(lib):
        env["ORT_DYLIB_PATH"] = lib
    env.update(extra_env or {})
    return subprocess.Popen([os.path.join(repo, "target", "release", "g2_deploy_rs"), "sim", "ros"], cwd=repo, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_all(*procs: subprocess.Popen):
    for p in procs:
        try:
            p.send_signal(signal.SIGINT)
            p.wait(timeout=3)
        except Exception:  # noqa: BLE001
            p.kill()


def _relay(proc: subprocess.Popen, tag: str, seen: list[str], lock: threading.Lock):
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(f"[{tag}] {line}", flush=True)
        with lock:
            seen.append(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.environ.get("G2_SIM_SCENE", "stair"), choices=["flat", "stair", "terrain"])
    ap.add_argument("--policy", type=int, default=0, help="RL policy slot sent before MODE RL (0 PPO, 1 HimLoco, 4 IsaacSim)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-stand", action="store_true", help="start both processes only; send MODE lines yourself")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--repo", default=os.environ.get("G2_DEPLOY_RS", "/g2"))
    ap.add_argument("--send", metavar="LINE", help="send one command line to the running controller and exit")
    ap.add_argument("--repeat", type=float, default=0.0, metavar="SECONDS",
                    help="with --send: keep sending the line at 10 Hz for this long (AXES needs it)")
    a = ap.parse_args(argv)

    if a.send:
        s = _cmd_sock()
        if a.repeat > 0:
            t0 = time.time()
            n = 0
            while time.time() - t0 < a.repeat:
                send(a.send, s, quiet=True)
                n += 1
                time.sleep(0.1)
            print(f"[theirs] sent {a.send!r} x{n} over {a.repeat:.1f} s", flush=True)
        else:
            send(a.send, s)
        return 0

    sim = start_sim(a.repo, a.scene, a.headless)
    time.sleep(2.0)
    ctl = start_controller(a.repo)
    seen: list[str] = []
    lock = threading.Lock()
    for p, tag in ((sim, "sim"), (ctl, "g2")):
        threading.Thread(target=_relay, args=(p, tag, seen, lock), daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    rc = 0
    try:
        if not a.no_stand:
            t0 = time.time()
            while time.time() - t0 < 30.0 and not stop.is_set():
                with lock:
                    idle = any("[Idle] Entered" in l for l in seen)
                if idle or ctl.poll() is not None:
                    break
                time.sleep(0.2)
            if ctl.poll() is not None:
                print("[theirs] controller exited before reaching Idle", flush=True)
                return 2
            time.sleep(1.0)
            send("MODE STAND")
            time.sleep(4.0)
            send(f"MODE {a.policy}")
            send("MODE RL")
            print(f"[theirs] {a.scene}: standing, RL with {POLICIES.get(a.policy, a.policy)}; "
                  f"drive with --send \"AXES vx vy wz\" --repeat S, switch with --send \"MODE 1\"", flush=True)
        t0 = time.time()
        while not stop.is_set():
            if sim.poll() is not None or ctl.poll() is not None:
                print(f"[theirs] exited: sim {sim.poll()} controller {ctl.poll()}", flush=True)
                rc = 1
                break
            if a.seconds and time.time() - t0 >= a.seconds:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all(ctl, sim)
    return rc


if __name__ == "__main__":
    sys.exit(main())
