#!/usr/bin/env python3
"""Drive the robot and score the nav stack's GNSS fusion (syncai_gnss_localization: navsat_transform_node + a
map-frame EKF, started by location_3d_gnss.launch.py) against the sim's ground truth. Runs in the
container (ROS 2), with the sim and the nav stack up.

  docker/run.sh "python3 scripts/gnss_ekf_check.py"             # forward 10 s, turn 4 s, forward 10 s, stop 3 s
  docker/run.sh "python3 scripts/gnss_ekf_check.py --no-drive --seconds 30"   # observe only
  ... --expect ekf_nav_dxy=0.05 --expect datum_err=0.05         # fail (exit 1) when an RMS is above its limit

Errors against ground truth: ekf_* = /robot01/odometry/filtered (the EKF), nav_* = /robot01/tracked_pose (what
the stack navigates on: the LiDAR pose, or the EKF's with drive_tf:=true), fix_dxy = the raw fix through
the scene's datum. Two more that need no ground truth: ekf_nav_dxy = EKF vs tracked_pose, and datum_err =
navsat_transform_node's /robot01/odometry/gps vs this repo's exact WGS84 conversion of the very same fix, which
isolates a datum or projection mismatch from receiver noise.

Prints a line per second, then RMS / max of each over the run, and writes results/gnss_ekf/<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quadsim.config import load_config, load_scene_meta  # noqa: E402
from quadsim.gnss import enu_to_map, lla_to_enu, resolve_datum  # noqa: E402

PATTERN = [(10.0, 0.3, 0.0), (4.0, 0.0, 0.5), (10.0, 0.3, 0.0), (3.0, 0.0, 0.0)]      # (seconds, vx, wz)


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Check(Node):
    def __init__(self, cfg: dict, drive: bool, seconds: float):
        super().__init__("gnss_ekf_check")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        tp = cfg["ros"]["topics"]
        self.datum = resolve_datum(load_scene_meta(cfg["scene"]), cfg["gnss"].get("datum"))
        if self.datum is None:
            raise SystemExit(f"no datum: scene {cfg['scene']} has no geo and config gnss.datum is null")
        self.mount = np.asarray(cfg["gnss"]["mount_xyz"], float)
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.gt = self.ekf = self.nav = self.fix = None
        self.fixes: dict[tuple[int, int], NavSatFix] = {}       # by stamp, to pair with odometry/gps
        self.datum_errs: list[float] = []
        self.create_subscription(Odometry, tp["gt_odom"], self._gt, 10)
        self.create_subscription(Odometry, "/robot01/odometry/filtered", self._ekf, q)
        self.create_subscription(Odometry, "/robot01/odometry/gps", self._gps, q)
        self.create_subscription(PoseStamped, "/robot01/tracked_pose", self._nav, 10)
        self.create_subscription(NavSatFix, tp["gnss_fix"], self._fix, q)
        self.pub_cmd = self.create_publisher(Twist, tp["cmd_vel"], 10)
        self.drive = drive
        self.plan = [(s, vx, wz) for s, vx, wz in PATTERN] if drive else [(seconds, 0.0, 0.0)]
        self.t0 = None
        self.rows: list[dict] = []
        self.last_print = -1.0
        self.done = False
        self.create_timer(0.1, self._tick)

    def _gt(self, m):
        self.gt = m

    def _ekf(self, m):
        self.ekf = m

    def _nav(self, m):
        self.nav = m

    def _fix(self, m):
        self.fix = m
        if not math.isnan(m.latitude):
            self.fixes[(m.header.stamp.sec, m.header.stamp.nanosec)] = m
            while len(self.fixes) > 50:
                self.fixes.pop(next(iter(self.fixes)))

    def _gps(self, m):
        """navsat_transform_node's output carries its fix's stamp: compare it with our conversion of that fix."""
        fix = self.fixes.get((m.header.stamp.sec, m.header.stamp.nanosec))
        if fix is None or self.gt is None:
            return
        want = self.fix_in_map(yaw_of(self.gt.pose.pose.orientation), fix)
        self.datum_errs.append(float(np.hypot(m.pose.pose.position.x - want[0], m.pose.pose.position.y - want[1])))

    def fix_in_map(self, gt_yaw: float, fix: NavSatFix | None = None) -> np.ndarray | None:
        """A fix (default: the latest) as a base position in the map frame: lat/lon -> ENU -> map, minus the
        antenna offset."""
        fix = fix or self.fix
        if fix is None or math.isnan(fix.latitude):
            return None
        ant = enu_to_map(lla_to_enu(fix.latitude, fix.longitude, self.datum.alt, self.datum), self.datum)[:2]
        cy, sy = math.cos(gt_yaw), math.sin(gt_yaw)
        return ant - np.array([cy * self.mount[0] - sy * self.mount[1], sy * self.mount[0] + cy * self.mount[1]])

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.gt is None or self.ekf is None:
            if self.t0 is None and now > 0 and int(now * 10) % 20 == 0:
                self.get_logger().info(f"waiting: gt {self.gt is not None}, ekf {self.ekf is not None}")
            return
        if self.t0 is None:
            self.t0 = now
            self.get_logger().info("ground truth and /robot01/odometry/filtered both seen: starting")
        t = now - self.t0
        # the command for this moment of the pattern
        elapsed, cmd = 0.0, None
        for secs, vx, wz in self.plan:
            if t < elapsed + secs:
                cmd = (vx, wz)
                break
            elapsed += secs
        if cmd is None:
            self.pub_cmd.publish(Twist())
            self.done = True
            return
        tw = Twist()
        tw.linear.x, tw.angular.z = cmd
        if self.drive:
            self.pub_cmd.publish(tw)
        p, g = self.gt.pose.pose.position, self.gt.pose.pose.orientation
        gt_xy, gt_yaw = np.array([p.x, p.y]), yaw_of(g)
        e = self.ekf.pose.pose
        ekf_xy, ekf_yaw = np.array([e.position.x, e.position.y]), yaw_of(e.orientation)
        row = {"t": round(t, 2), "gt": [round(p.x, 3), round(p.y, 3), round(gt_yaw, 3)],
               "ekf_dxy": float(np.linalg.norm(ekf_xy - gt_xy)), "ekf_dyaw": wrap(ekf_yaw - gt_yaw)}
        if self.nav is not None:
            n = self.nav.pose
            row["nav_dxy"] = float(np.hypot(n.position.x - p.x, n.position.y - p.y))
            row["nav_dyaw"] = wrap(yaw_of(n.orientation) - gt_yaw)
            row["ekf_nav_dxy"] = float(np.hypot(e.position.x - n.position.x, e.position.y - n.position.y))
        raw = self.fix_in_map(gt_yaw)
        if raw is not None:
            row["fix_dxy"] = float(np.linalg.norm(raw - gt_xy))
        self.rows.append(row)
        if t - self.last_print >= 1.0:
            self.last_print = t
            nav = f"nav {row['nav_dxy']:.2f} m" if "nav_dxy" in row else "nav -"
            raw_s = f"fix {row['fix_dxy']:.2f} m" if "fix_dxy" in row else "fix -"
            self.get_logger().info(f"t={t:5.1f} gt ({p.x:6.2f},{p.y:6.2f},{math.degrees(gt_yaw):6.1f}deg) "
                                   f"cmd vx {cmd[0]:.1f} wz {cmd[1]:.1f} | ekf {row['ekf_dxy']:.2f} m "
                                   f"{math.degrees(row['ekf_dyaw']):5.1f}deg | {nav} | {raw_s}")


def stats(values, scale: float = 1.0) -> dict:
    v = np.asarray(values, float)
    return {"rms": round(float(np.sqrt(np.mean(v ** 2)) * scale), 3),
            "max": round(float(np.abs(v).max() * scale), 3), "n": int(len(v))}


def summarize(rows: list[dict], datum_errs: list[float] | None = None) -> dict:
    """RMS / max / count per quantity: metres, or degrees for the *_dyaw ones."""
    out = {}
    for key in ("ekf_dxy", "ekf_dyaw", "nav_dxy", "nav_dyaw", "ekf_nav_dxy", "fix_dxy"):
        v = [r[key] for r in rows if key in r]
        if v:
            out[key] = stats(v, 180 / math.pi if key.endswith("dyaw") else 1.0)
    if datum_errs:
        out["datum_err"] = stats(datum_errs)
    return out


def check_expectations(summary: dict, expect: list[str]) -> list[str]:
    """--expect key=limit: the failures, as text. A quantity that was never measured is a failure too."""
    failures = []
    for item in expect:
        key, sep, raw = item.partition("=")
        try:
            limit = float(raw)
        except ValueError:
            limit = None
        if not sep or limit is None:
            failures.append(f"--expect {item!r}: want key=number")
        elif key not in summary:
            failures.append(f"{key}: not measured (expected RMS <= {limit})")
        elif summary[key]["rms"] > limit:
            failures.append(f"{key}: RMS {summary[key]['rms']} > {limit}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None, help="the sim's scene (its meta.json geo is the datum); default config scene")
    ap.add_argument("--no-drive", action="store_true", help="observe only, send no cmd_vel")
    ap.add_argument("--seconds", type=float, default=30.0, help="--no-drive: how long to observe")
    ap.add_argument("--wait", type=float, default=60.0, help="give up after this many wall seconds without data")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=LIMIT",
                    help="fail when a summary RMS is above the limit (m, or deg for *_dyaw), e.g. ekf_nav_dxy=0.05; repeatable")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    if a.scene:
        cfg["scene"] = a.scene
    rclpy.init()
    node = Check(cfg, not a.no_drive, a.seconds)
    wall0 = time.time()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.t0 is None and time.time() - wall0 > a.wait:
                print(f"no data after {a.wait:.0f} s: gt {node.gt is not None}, ekf {node.ekf is not None}", file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        rclpy.shutdown()
    summary = summarize(node.rows, node.datum_errs)
    failures = check_expectations(summary, a.expect)
    print(json.dumps(summary, indent=1))
    out = os.path.join("results", "gnss_ekf", time.strftime("%Y%m%d-%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"drive": not a.no_drive, "pattern": PATTERN if not a.no_drive else None, "summary": summary,
                   "expect": a.expect, "failures": failures, "rows": node.rows}, f, indent=1)
    print(f"result: {out}")
    for line in failures:
        print(f"FAIL {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
