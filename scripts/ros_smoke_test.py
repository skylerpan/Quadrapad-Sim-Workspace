"""End-to-end ROS 2 check of a running quadsim node (no Nav2 needed).

Usage (container, with quadsim.sim_node running):
  python3 scripts/ros_smoke_test.py [--drive-seconds 6]

Phases: topic rates in sim time -> TF chain -> drive forward -> drive until a
collision/stuck event -> /sim/set_pose back to start. Exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    from livox_ros_driver2.msg import CustomMsg
except ImportError:
    CustomMsg = None


class Smoke(Node):
    def __init__(self):
        super().__init__("quadsim_smoke", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.counts = {k: 0 for k in ("clock", "imu", "points", "custom", "odom")}
        self.sim_t = 0.0
        self.odom = None
        self.events = []
        self.custom_pts = []
        self.create_subscription(Clock, "/clock", self._on_clock, 50)
        self.create_subscription(Imu, "/livox/imu", lambda m: self._count("imu"), 50)
        self.create_subscription(PointCloud2, "/livox/points", lambda m: self._count("points"), 10)
        if CustomMsg is not None:
            self.create_subscription(CustomMsg, "/livox/lidar", self._custom, 10)
        self.create_subscription(Odometry, "/sim/ground_truth/odom", self._odom, 10)
        self.create_subscription(String, "/sim/events", lambda m: self.events.append(json.loads(m.data)), 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.set_pose = self.create_publisher(PoseStamped, "/sim/set_pose", 10)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

    def _count(self, k):
        self.counts[k] += 1

    def _on_clock(self, m):
        self.counts["clock"] += 1
        self.sim_t = m.clock.sec + 1e-9 * m.clock.nanosec

    def _custom(self, m):
        self.counts["custom"] += 1
        self.custom_pts.append(m.point_num)
        if m.point_num > 1:
            offs = [p.offset_time for p in m.points[:: max(1, m.point_num // 200)]]
            assert all(b >= a for a, b in zip(offs, offs[1:])), "offset_time not monotonic"

    def _odom(self, m):
        self.counts["odom"] += 1
        self.odom = m

    def spin_sim(self, seconds: float, cmd: Twist | None = None, until=None, wall_timeout: float = 120.0):
        t_end = self.sim_t + seconds
        w0 = time.time()
        while rclpy.ok() and self.sim_t < t_end and time.time() - w0 < wall_timeout:
            if cmd is not None:
                self.cmd.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.02)
            if until is not None and until():
                return True
        return False

    def xy(self):
        p = self.odom.pose.pose.position
        return p.x, p.y, p.z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-seconds", type=float, default=6.0)
    a = ap.parse_args()
    rclpy.init()
    n = Smoke()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg, flush=True)
        ok &= bool(cond)

    w0 = time.time()
    while rclpy.ok() and (n.odom is None or n.counts["clock"] < 5) and time.time() - w0 < 60:
        rclpy.spin_once(n, timeout_sec=0.1)
    check(n.odom is not None, "sim is publishing (/clock, ground truth)")
    if n.odom is None:
        sys.exit(1)

    # 1) rates in sim time
    for k in n.counts:
        n.counts[k] = 0
    t0 = n.sim_t
    n.spin_sim(3.0)
    dt = max(n.sim_t - t0, 1e-6)
    rates = {k: v / dt for k, v in n.counts.items()}
    print("rates (msgs per sim-second): " + ", ".join(f"{k} {v:.1f}" for k, v in rates.items()))
    check(abs(rates["imu"] - 200) < 20, "IMU ~200 Hz")
    check(abs(rates["points"] - 10) < 2, "PointCloud2 ~10 Hz")
    if CustomMsg is not None:
        check(abs(rates["custom"] - 10) < 2, "CustomMsg ~10 Hz")
        check(min(n.custom_pts) > 10000, f"CustomMsg points/frame >= 10k (min {min(n.custom_pts)})")
    check(rates["clock"] > 100, "/clock > 100 Hz")

    # 2) TF chain
    try:
        tf = n.tf_buf.lookup_transform("map", "livox_frame", rclpy.time.Time())
        check(True, f"TF map->livox_frame available (z={tf.transform.translation.z:.2f})")
    except Exception as e:  # noqa: BLE001
        check(False, f"TF map->livox_frame: {e}")

    # 3) drive forward
    start = n.xy()
    fwd = Twist()
    fwd.linear.x = 0.5
    n.spin_sim(a.drive_seconds, fwd)
    n.spin_sim(1.0, Twist())
    end = n.xy()
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    blocked = [e for e in n.events if e.get("type") in ("collision", "stuck")]
    print(f"drove {dist:.2f} m in {a.drive_seconds}s at 0.5 m/s (events so far: {len(blocked)})")
    check(dist > 0.5 or blocked, "base moves on cmd_vel (or reports why it cannot)")

    # 4) drive until something stops us
    n.events.clear()
    fast = Twist()
    fast.linear.x = 1.0
    hit = n.spin_sim(60.0, fast, until=lambda: any(e.get("type") in ("collision", "stuck") for e in n.events))
    n.spin_sim(0.5, Twist())
    ev = [e for e in n.events if e.get("type") in ("collision", "stuck")]
    check(hit, f"obstacle reported: {ev[0] if ev else 'none within 60 s'}")

    # 5) teleport back
    ps = PoseStamped()
    ps.header.frame_id = "map"
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = start[0], start[1], start[2] - 0.3
    ps.pose.orientation.w = 1.0
    n.set_pose.publish(ps)
    n.spin_sim(1.0)
    back = n.xy()
    check(math.hypot(back[0] - start[0], back[1] - start[1]) < 0.1, "set_pose teleports the base")

    n.cmd.publish(Twist())
    n.destroy_node()
    rclpy.shutdown()
    print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
