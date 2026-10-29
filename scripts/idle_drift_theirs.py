#!/usr/bin/env python3
"""Reference measurement: the controller on ITS OWN simulation (mujoco_simulation_G23.py, flat scene).

Starts their sim headless and the controller inside the quadsim-g23 container (repo at /g2), sends
MODE STAND, MODE <policy>, MODE RL on udp:50051, then records the yaw from the sim's SIM_POSE ground-truth
lines (udp:30012) for --seconds and prints the yaw rate. Compare with
`walk_g23.py --idle --scene scenes/flat_test` to tell a policy habit from a quadsim emulation bug.

  QUADSIM_G23=1 docker/run.sh "python3 scripts/idle_drift_theirs.py --policy 1 --seconds 35"
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--repo", default=os.environ.get("G2_DEPLOY_RS", "/g2"))
    a = ap.parse_args(argv)
    repo = a.repo
    sim_dir = os.path.join(repo, "interface", "robot", "simulation")
    env = dict(os.environ, G2_SIM_HEADLESS="1", G2_SIM_ROS_ASCII_TELEMETRY="0", G2_SIM_SCENE="flat",
               PYTHONUNBUFFERED="1")
    try:
        import colorama  # noqa: F401  (their sim imports it)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", "colorama"], check=False)
    gt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    gt.bind(("127.0.0.1", 30012))
    gt.settimeout(1.0)
    sim = subprocess.Popen([sys.executable, "mujoco_simulation_G23.py", "--scene", "flat", "--headless"], cwd=sim_dir,
                           env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cenv = dict(os.environ, G2_LMA_STATE_SOURCE="ground_truth")
    lib = os.path.join(repo, "onnxruntime-linux-aarch64-1.19.2", "lib", "libonnxruntime.so")
    if os.path.exists(lib):
        cenv["ORT_DYLIB_PATH"] = lib
    time.sleep(2.0)
    ctl = subprocess.Popen([os.path.join(repo, "target", "release", "g2_deploy_rs"), "sim", "ros"], cwd=repo, env=cenv,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(text):
        cmd.sendto((text + "\n").encode(), ("127.0.0.1", 50051))
        print(f"[theirs] sent {text}", flush=True)

    def yaw_now():
        line = None
        try:
            for _ in range(50):
                line = gt.recv(512).decode()
        except socket.timeout:
            pass
        if not line:
            return None
        p = line.split()
        qw, qx, qy, qz = map(float, p[5:9])
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)), (float(p[2]), float(p[3]))

    try:
        time.sleep(3.0)
        send("MODE STAND")
        time.sleep(4.0)
        send(f"MODE {a.policy}")
        send("MODE RL")
        time.sleep(1.5)
        r = yaw_now()
        if r is None:
            print("[theirs] no SIM_POSE received from their sim on udp:30012")
            return 2
        yaw0, (x0, y0) = r
        yaw_acc, prev = 0.0, yaw0
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            time.sleep(1.0)
            r = yaw_now()
            if r is None:
                continue
            yaw, (x, y) = r
            yaw_acc += (yaw - prev + math.pi) % (2 * math.pi) - math.pi
            prev = yaw
        dt = time.time() - t0
        print(f"[theirs] policy {a.policy}: idle {dt:.1f} s, yaw {math.degrees(yaw_acc):.2f} deg "
              f"({math.degrees(yaw_acc) / dt:.3f} deg/s), drift {math.hypot(x - x0, y - y0):.3f} m", flush=True)
    finally:
        for p in (ctl, sim):
            try:
                p.send_signal(signal.SIGINT)
                p.wait(timeout=3)
            except Exception:  # noqa: BLE001
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
