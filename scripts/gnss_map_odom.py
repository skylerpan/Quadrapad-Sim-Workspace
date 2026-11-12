#!/usr/bin/env python3
"""NavSatFix -> nav_msgs/Odometry in the map frame through a fixed datum, for robot_localization's map EKF.

robot_localization's own navsat_transform_node anchors its datum to the robot's pose at start-up (a manual
`datum` means "the robot is at this lat/lon with this heading now"), so a noisy first fix or a robot that is
not at the map origin becomes a permanent offset. The datum is a property of the map, not of the start pose:
this node takes it from config/sim.yaml `gnss.datum` (on the real robot the same numbers would come from the
map's own metadata) and converts every fix with quadsim.gnss, the same WGS84 math the sim used to make it.

  docker/run.sh "python3 scripts/gnss_map_odom.py"      # /robot01/gnss/fix -> /odometry/gps (frame map)

The antenna offset (gnss.mount_xyz) is taken off with the robot's yaw from the odometry topic, so the
published position is base_link's. No fix (NaN) publishes nothing. The pose covariance carries the fix's own
sigma (x, y from position_covariance), the rest is left huge so the EKF only takes x and y.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quadsim.config import load_config, load_scene_meta  # noqa: E402
from quadsim.gnss import Datum, enu_to_map, lla_to_enu, resolve_datum  # noqa: E402

BIG = 1e6


def fix_to_map(lat: float, lon: float, datum: Datum, mount_xy, yaw: float) -> tuple[float, float]:
    """The base position in the map frame for an antenna fix: lat/lon -> ENU -> map (minus the datum heading),
    then the antenna offset rotated by the robot's yaw is taken off."""
    ax, ay, _ = enu_to_map(lla_to_enu(lat, lon, datum.alt, datum), datum)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return ax - (cy * mount_xy[0] - sy * mount_xy[1]), ay - (sy * mount_xy[0] + cy * mount_xy[1])


class GnssMapOdom(Node):
    def __init__(self, cfg: dict, odom_topic: str, out_topic: str):
        super().__init__("gnss_map_odom")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.datum = resolve_datum(load_scene_meta(cfg["scene"]), cfg["gnss"].get("datum"))
        if self.datum is None:
            raise SystemExit(f"no datum: scene {cfg['scene']} has no geo and config gnss.datum is null")
        self.mount = [float(v) for v in cfg["gnss"]["mount_xyz"][:2]]
        self.frames = cfg["ros"]["frames"]
        self.yaw = 0.0
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, odom_topic, self._odom, 10)
        self.create_subscription(NavSatFix, cfg["ros"]["topics"]["gnss_fix"], self._fix, q)
        self.pub = self.create_publisher(Odometry, out_topic, q)
        self.n = 0
        self.get_logger().info(f"datum {self.datum}, antenna {self.mount}, fix -> {out_topic} in {self.frames['map']}")

    def _odom(self, m):
        o = m.pose.pose.orientation
        self.yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))

    def _fix(self, m: NavSatFix):
        if m.status.status == NavSatStatus.STATUS_NO_FIX or math.isnan(m.latitude):
            return
        x, y = fix_to_map(m.latitude, m.longitude, self.datum, self.mount, self.yaw)
        od = Odometry()
        od.header.stamp = m.header.stamp
        od.header.frame_id = self.frames["map"]
        od.child_frame_id = self.frames["base"]
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        od.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        for i in range(6):
            cov[i * 7] = BIG
        cov[0], cov[7] = float(m.position_covariance[0]), float(m.position_covariance[4])
        od.pose.covariance = cov
        self.pub.publish(od)
        self.n += 1
        if self.n % 100 == 1:
            self.get_logger().info(f"fix {self.n}: map ({x:.2f}, {y:.2f}) sigma {math.sqrt(cov[0]):.2f} m, status {m.status.status}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None, help="the sim's scene (its meta.json geo is the datum); default config scene")
    ap.add_argument("--odom", default="/robot01/odom", help="odometry whose yaw turns the antenna offset")
    ap.add_argument("--out", default="/odometry/gps")
    a = ap.parse_args()
    rclpy.init()
    cfg = load_config(a.config, [])
    if a.scene:
        cfg["scene"] = a.scene
    node = GnssMapOdom(cfg, a.odom, a.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
