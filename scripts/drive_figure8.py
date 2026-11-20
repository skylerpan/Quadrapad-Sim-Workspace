#!/usr/bin/env python3
"""Figure-8 driver on ground truth: pure pursuit on two tangent circles, streamed to /robot01/cmd_vel.

The figure-8 is two circles of radius `--radius` touching at the crossing point `--center` (x y, world), where
the path runs along `--heading` (rad). A lap is the left lobe (counter-clockwise) and then the right lobe
(clockwise), each starting and ending at the crossing. The robot first walks a straight lead-in from where it
stands to the crossing, then `--laps` laps at `--speed`, then stops. It also stops on a flip (|roll| or
|pitch| > 1.2 rad) or after `--seconds`. Reads /sim/ground_truth/odom like scripts/drive_gt.py and prints the
pose once a second; the last line says `reached` (all laps done), `timeout` or `flipped`, the share of the
figure-8 done and the largest cross-track error.

  docker/run.sh "python3 scripts/drive_figure8.py --center 8 0 --heading 0 --laps 3"      # scenes/slope_10
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np


def _arc(c, r, a0, sweep, ds):
    n = max(2, int(math.ceil(abs(sweep) * r / ds)) + 1)
    a = a0 + np.linspace(0.0, sweep, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def figure8_path(center, heading: float, radius: float, laps: int, ds: float = 0.02) -> np.ndarray:
    """[N, 2] points of `laps` figure-8 laps, from the crossing back to it (consecutive points <= ds apart)."""
    c = np.asarray(center, dtype=np.float64)
    left = np.array([-math.sin(heading), math.cos(heading)])
    cl, cr = c + radius * left, c - radius * left
    a_l = heading - math.pi / 2               # angle of the crossing seen from the left lobe's centre
    a_r = heading + math.pi / 2               # ... and from the right lobe's centre
    lobe_l = _arc(cl, radius, a_l, 2 * math.pi, ds)       # counter-clockwise: leaves along +heading
    lobe_r = _arc(cr, radius, a_r, -2 * math.pi, ds)      # clockwise: also leaves along +heading
    lap = np.concatenate([lobe_l, lobe_r[1:]])
    parts = [lap] + [lap[1:]] * (laps - 1)
    p = np.concatenate(parts)
    p[0] = c
    p[-1] = c
    return p


def build_path(start, center, heading: float, radius: float, laps: int, ds: float = 0.02):
    """(path [N, 2], n_lead): a straight lead-in from `start` to the crossing (n_lead points, none when the robot
    already stands within ds of it), then the figure-8."""
    start, c = np.asarray(start, dtype=np.float64), np.asarray(center, dtype=np.float64)
    fig = figure8_path(c, heading, radius, laps, ds)
    d = float(np.linalg.norm(c - start))
    if d <= ds:
        return fig, 0
    n = int(math.ceil(d / ds))
    lead = start + (c - start) * (np.arange(n) / n)[:, None]
    return np.concatenate([lead, fig]), n


class PurePursuit:
    """Pure pursuit with monotone progress: the nearest path point is searched only from the current progress up
    to `window` m ahead, so the tracker never jumps to another pass through the crossing."""

    def __init__(self, path: np.ndarray, lookahead: float = 0.5, window: float = 1.0, done_tol: float = 0.05):
        self.p = np.asarray(path, dtype=np.float64)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))])
        self.lookahead, self.window, self.done_tol = lookahead, window, done_tol
        self.i = None

    def fraction_done(self, n_lead: int = 0) -> float:
        i = self.i or 0
        s0 = self.s[n_lead]
        return float(max(0.0, (self.s[i] - s0)) / max(self.s[-1] - s0, 1e-9))

    def step(self, x: float, y: float, yaw: float):
        """(curvature 1/m (+ = turn left), done, cross-track error m) for the robot at (x, y, yaw)."""
        q = np.array([x, y])
        if self.i is None:                               # first call: nearest point overall (first on ties)
            self.i = int(np.argmin(np.linalg.norm(self.p - q, axis=1)))
        hi = int(np.searchsorted(self.s, self.s[self.i] + self.window, side="right"))
        seg = self.p[self.i:max(hi, self.i + 1)]
        self.i += int(np.argmin(np.linalg.norm(seg - q, axis=1)))
        err = float(np.linalg.norm(self.p[self.i] - q))
        if self.s[-1] - self.s[self.i] <= self.done_tol:
            return 0.0, True, err
        # target: the first point ahead that is at least `lookahead` away (the path end near the finish)
        ahead = self.p[self.i:]
        far = np.flatnonzero(np.linalg.norm(ahead - q, axis=1) >= self.lookahead)
        tgt = ahead[far[0]] if len(far) else ahead[-1]
        dx, dy = tgt - q
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy        # target in the robot frame
        d2 = lx * lx + ly * ly
        return (2.0 * ly / d2 if d2 > 1e-9 else 0.0), False, err


def _rpy(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", type=float, nargs=2, required=True, metavar=("X", "Y"), help="crossing point (world)")
    ap.add_argument("--heading", type=float, default=0.0, help="path direction at the crossing (rad, world)")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--speed", type=float, default=0.4, help="forward speed command (m/s)")
    ap.add_argument("--lookahead", type=float, default=0.5)
    ap.add_argument("--wz-max", type=float, default=0.8)
    ap.add_argument("--seconds", type=float, default=0.0, help="give up after this (0 = 2x the nominal time + 30 s)")
    ap.add_argument("--cmd-topic", default="/robot01/cmd_vel")
    ap.add_argument("--gt-topic", default="/sim/ground_truth/odom")
    a = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node

    rclpy.init()
    node = Node("drive_figure8")
    pub = node.create_publisher(Twist, a.cmd_topic, 10)
    state = {}

    def on_odom(msg: Odometry):
        p = msg.pose.pose.position
        state["x"], state["y"], state["z"] = p.x, p.y, p.z
        state["roll"], state["pitch"], state["yaw"] = _rpy(msg.pose.pose.orientation)

    node.create_subscription(Odometry, a.gt_topic, on_odom, 10)
    t0 = time.time()
    while "x" not in state and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "x" not in state:
        print("no ground truth on " + a.gt_topic, flush=True)
        return 1
    path, n_lead = build_path([state["x"], state["y"]], a.center, a.heading, a.radius, a.laps)
    pp = PurePursuit(path, lookahead=a.lookahead)
    length = float(pp.s[-1])
    seconds = a.seconds or 2.0 * length / a.speed + 30.0
    print(f"figure-8: crossing ({a.center[0]:.2f}, {a.center[1]:.2f}) heading {a.heading:.3f} radius {a.radius} "
          f"laps {a.laps}; lead-in {pp.s[n_lead]:.2f} m, path {length:.1f} m", flush=True)
    t0 = time.time()
    last = 0.0
    worst = 0.0
    result = "timeout"
    while rclpy.ok() and time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.02)
        x, y = state["x"], state["y"]
        if abs(state["roll"]) > 1.2 or abs(state["pitch"]) > 1.2:
            result = f"flipped at ({x:.2f}, {y:.2f})"
            break
        k, done, err = pp.step(x, y, state["yaw"])
        if pp.i >= n_lead:
            worst = max(worst, err)
        if done:
            result = f"reached ({x:.2f}, {y:.2f}, z {state['z']:.2f}) after {a.laps} laps"
            break
        wz = max(-a.wz_max, min(a.wz_max, a.speed * k))
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = a.speed, wz
        pub.publish(cmd)
        if time.time() - last >= 1.0:
            print(f"t={time.time() - t0:4.1f} x={x:.2f} y={y:.2f} z={state['z']:.2f} "
                  f"yaw={math.degrees(state['yaw']):6.1f} pitch={math.degrees(state['pitch']):5.1f} wz={wz:+.2f} "
                  f"done={pp.fraction_done(n_lead):.3f} err={err:.2f}", flush=True)
            last = time.time()
    pub.publish(Twist())
    time.sleep(0.5)
    pub.publish(Twist())
    print(f"{result}; figure-8 done {pp.fraction_done(n_lead):.3f} ({pp.fraction_done(n_lead) * a.laps:.2f} laps), "
          f"max cross-track {worst:.2f} m; final x={state['x']:.2f} y={state['y']:.2f} z={state['z']:.2f}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
