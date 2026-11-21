#!/usr/bin/env python3
"""Loop-closure test of the nav stack's localization: drive a route out and back on ground truth and record how
far /robot01/tracked_pose (what the stack navigates on) and /robot01/odometry/filtered (the GNSS/LiDAR EKF) are
from the truth along the way and on return to the start.

  docker/run.sh "python3 scripts/route_loop_test.py --route config/route_danuan_milksha.yaml \
      --via 34.5,46 --via 34.5,60 --vx 0.8"

The robot is steered on /sim/ground_truth/odom (pure pursuit), so the path driven does not depend on the
localization under test. Path: the robot's current position -> each --via point -> the route's waypoints -> and the
same way back to the first --via point (the loop's start and end). Stops on a fall or when stuck. Prints a line
every --print seconds and writes results/route_loop/<stamp>.json with one row per second.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix


def rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Path:
    def __init__(self, pts: np.ndarray):
        self.p = pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self.i = 0                                     # the segment the robot was last on: never search backwards

    def at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        return np.array([np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])])

    def project(self, xy: np.ndarray) -> float:
        """Arc length of the closest point, looking only a little ahead of the last one (the path overlaps
        itself on the way back)."""
        best_s, best_d = self.s[self.i], math.inf
        for k in range(self.i, min(self.i + 15, len(self.p) - 1)):
            a, b = self.p[k], self.p[k + 1]
            ab = b - a
            t = 0.0 if not ab.any() else float(np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            d = float(np.linalg.norm(a + t * ab - xy))
            if d < best_d:
                best_d, best_s, best_k = d, self.s[k] + t * float(np.linalg.norm(ab)), k
        self.i = best_k
        return float(best_s)


class LoopTest(Node):
    def __init__(self, a):
        super().__init__("route_loop_test")
        self.a = a
        self.gt = self.nav = self.ekf = self.fix = self.corr = None
        self.lidar_updates: list[float] = []
        self.seeds: list[dict] = []
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, "/sim/ground_truth/odom", lambda m: setattr(self, "gt", m), 10)
        self.create_subscription(PoseStamped, "/robot01/tracked_pose", lambda m: setattr(self, "nav", m), 10)
        self.create_subscription(Odometry, "/robot01/odometry/filtered", lambda m: setattr(self, "ekf", m), q)
        self.create_subscription(NavSatFix, "/robot01/gnss/fix", lambda m: setattr(self, "fix", m), be)
        self.create_subscription(Vector3Stamped, "/robot01/gnss_fusion/correction", lambda m: setattr(self, "corr", m), 10)
        self.create_subscription(PoseWithCovarianceStamped, "/robot01/gnss_fusion/lidar_pose", self.on_lidar, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/robot01/initialpose", self.on_seed, 10)
        self.pub = self.create_publisher(Twist, "/robot01/cmd_vel", 10)

    def t(self) -> float:
        s = self.gt.header.stamp
        return s.sec + s.nanosec * 1e-9

    def on_lidar(self, m):
        if self.gt is not None:
            self.lidar_updates.append(self.t())

    def on_seed(self, m):
        if self.gt is not None:
            self.seeds.append({"t": self.t(), "x": m.pose.pose.position.x, "y": m.pose.pose.position.y})

    def row(self, t0: float, s: float, phase: str) -> dict:
        p = self.gt.pose.pose.position
        _, _, yaw = rpy(self.gt.pose.pose.orientation)
        now = self.t()
        r = {"t": round(now - t0, 1), "s": round(s, 1), "phase": phase, "gt": [round(p.x, 2), round(p.y, 2), round(yaw, 3)],
             "lidar_updates_10s": sum(1 for u in self.lidar_updates if u > now - 10.0)}
        if self.nav is not None:
            n = self.nav.pose
            r["nav"] = [round(n.position.x, 2), round(n.position.y, 2)]
            r["nav_dxy"] = round(math.hypot(n.position.x - p.x, n.position.y - p.y), 3)
            r["nav_dyaw"] = round(math.degrees(wrap(rpy(n.orientation)[2] - yaw)), 2)
        if self.ekf is not None:
            e = self.ekf.pose.pose
            r["ekf_dxy"] = round(math.hypot(e.position.x - p.x, e.position.y - p.y), 3)
            r["ekf_dyaw"] = round(math.degrees(wrap(rpy(e.orientation)[2] - yaw)), 2)
        if self.corr is not None:
            r["ekf_minus_lidar"] = round(math.hypot(self.corr.vector.x, self.corr.vector.y), 3)
        if self.fix is not None:
            r["fix"] = None if math.isnan(self.fix.latitude) else round(math.sqrt(self.fix.position_covariance[0]), 2)
        return r


def build_path(start_xy, vias, route):
    """current position -> vias -> route -> route reversed -> vias reversed (ends at the first via)."""
    out = [np.asarray(start_xy, float)] + [np.asarray(v, float) for v in vias] + [np.asarray(w, float) for w in route]
    n_out = len(out)
    back = [np.asarray(w, float) for w in route[::-1][1:]] + [np.asarray(v, float) for v in vias[::-1]]
    pts = np.array(out + back)
    # densify long straight legs so the projection window (15 segments) always holds the robot
    dense = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / 2.0))
        dense += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    pts = np.array(dense)
    turn = int(np.argmin(np.linalg.norm(pts - np.asarray(route[-1], float), axis=1)))
    return pts, turn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True, help="YAML with `waypoints: [[x, y], ...]` (scripts/osm_route.py)")
    ap.add_argument("--via", action="append", default=[], metavar="X,Y", help="map points before the route; the first is the loop's start and end")
    ap.add_argument("--vx", type=float, default=0.8)
    ap.add_argument("--lookahead", type=float, default=2.5)
    ap.add_argument("--k-yaw", type=float, default=1.5)
    ap.add_argument("--wz-max", type=float, default=0.8)
    ap.add_argument("--settle", type=float, default=30.0, help="seconds to stand at the end, still recording")
    ap.add_argument("--print", dest="print_every", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=4000.0)
    a = ap.parse_args()
    with open(a.route) as f:
        route = yaml.safe_load(f)["waypoints"]
    vias = [[float(v) for v in s.split(",")] for s in a.via]

    # Ctrl-C / docker stop must still leave a result file: rclpy would tear its context down on the signal, so
    # take the signals ourselves and leave the loop normally.
    stop = {"flag": False}
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    node = LoopTest(a)
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    wall0 = time.time()
    while rclpy.ok() and node.gt is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - wall0 > 30:
            print("no ground truth", file=sys.stderr)
            return 1
    p = node.gt.pose.pose.position
    pts, turn = build_path([p.x, p.y], vias, route)
    path = Path(pts)
    s_turn = float(path.s[turn])
    s_home = float(path.s[1 + 0]) if not vias else float(np.linalg.norm(np.asarray(vias[0]) - np.array([p.x, p.y])))
    print(f"path {path.length:.0f} m: turnaround at {s_turn:.0f} m, loop start/end {vias[0] if vias else [p.x, p.y]}", flush=True)

    out = os.path.join("results", "route_loop", time.strftime("%Y%m%d-%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def save(result):
        with open(out + ".tmp", "w") as f:
            json.dump({"result": result, "route": a.route, "vias": vias, "vx": a.vx, "path_length": path.length,
                       "s_turn": s_turn, "t0": t0, "seeds": [{**sd, "t": round(sd["t"] - t0, 1)} for sd in node.seeds],
                       "path": pts.round(2).tolist(), "rows": rows}, f)
        os.replace(out + ".tmp", out)

    t0 = node.t()
    rows, result = [], "done"
    last_save = t0
    last_row = last_print = -1e9
    progress_s, progress_t = 0.0, t0
    end_t = None
    try:
        while rclpy.ok():
            if stop["flag"]:
                result = "interrupted"
                break
            rclpy.spin_once(node, timeout_sec=0.02)
            now = node.t()
            if now - last_save > 60.0:
                last_save = now
                save("running")
            g = node.gt.pose.pose
            roll, pitch, yaw = rpy(g.orientation)
            xy = np.array([g.position.x, g.position.y])
            s = path.project(xy)
            cmd = Twist()
            if end_t is None:
                if abs(roll) > 1.0 or abs(pitch) > 1.0:
                    result = "fell"
                    break
                if s > progress_s + 0.5:
                    progress_s, progress_t = s, now
                elif now - progress_t > 25.0:
                    result = "stuck"
                    break
                if now - t0 > a.max_seconds:
                    result = "timeout"
                    break
                if path.length - s < 0.4:
                    end_t = now
                else:
                    tgt = path.at(s + a.lookahead)
                    err = wrap(math.atan2(tgt[1] - xy[1], tgt[0] - xy[0]) - yaw)
                    cmd.angular.z = max(-a.wz_max, min(a.wz_max, a.k_yaw * err))
                    cmd.linear.x = a.vx * max(0.0, math.cos(err)) if abs(err) < 1.2 else 0.0
            elif now - end_t > a.settle:
                break
            node.pub.publish(cmd)
            if now - last_row >= 1.0:
                last_row = now
                phase = "settle" if end_t is not None else ("out" if s < s_turn else "back")
                rows.append(node.row(t0, s, phase))
                if now - last_print >= a.print_every:
                    last_print = now
                    r = rows[-1]
                    print(f"t={r['t']:7.1f} s={r['s']:7.1f}/{path.length:.0f} {phase:6s} gt ({r['gt'][0]:8.2f},{r['gt'][1]:8.2f}) "
                          f"nav {r.get('nav_dxy', float('nan')):6.2f} m {r.get('nav_dyaw', float('nan')):6.1f} deg | "
                          f"ekf {r.get('ekf_dxy', float('nan')):6.2f} m | lidar upd/10s {r['lidar_updates_10s']} | "
                          f"fix sigma {r.get('fix')} | seeds {len(node.seeds)}", flush=True)
    finally:
        save(result)
        try:
            for _ in range(5):
                node.pub.publish(Twist())
                time.sleep(0.05)
        except Exception:                              # the sim zeroes cmd_vel on its own timeout anyway
            pass
    print(f"result: {result}; {len(rows)} rows -> {out}", flush=True)
    rclpy.try_shutdown()
    return 0 if result == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
