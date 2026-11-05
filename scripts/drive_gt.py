#!/usr/bin/env python3
"""Heading-hold driver on ground truth: stream /robot01/cmd_vel so the robot walks along a lane at a fixed heading.

Reads /sim/ground_truth/odom, holds heading `--yaw` (rad) and pulls back to the lane line (`--lane` = the cross-track
coordinate: x when walking along y, y when walking along x), stops when the along-track coordinate passes `--stop`,
on a flip (|roll| or |pitch| > 1.2 rad) or after `--seconds`. Prints the pose once a second. For stairs / straight
runs with the live viewer (same loop as the reference-sim driver used on their mixed terrain):

  docker/run.sh "python3 scripts/drive_gt.py --vx 0.4 --yaw 1.5708 --lane 6.35 --stop -7.0 --seconds 60"   # E1 up
"""
from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _rpy(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--yaw", type=float, default=0.0, help="heading to hold (rad, world)")
    ap.add_argument("--lane", type=float, default=0.0, help="cross-track line to hold (x for yaw ±pi/2, else y)")
    ap.add_argument("--stop", type=float, required=True, help="stop once the along-track coordinate passes this")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--k-yaw", type=float, default=1.5)
    ap.add_argument("--k-lane", type=float, default=0.8)
    ap.add_argument("--wz-max", type=float, default=0.6)
    ap.add_argument("--cmd-topic", default="/robot01/cmd_vel")
    ap.add_argument("--gt-topic", default="/sim/ground_truth/odom")
    a = ap.parse_args()

    rclpy.init()
    node = Node("drive_gt")
    pub = node.create_publisher(Twist, a.cmd_topic, 10)
    state = {}

    def on_odom(msg: Odometry):
        p = msg.pose.pose.position
        state["x"], state["y"], state["z"] = p.x, p.y, p.z
        state["roll"], state["pitch"], state["yaw"] = _rpy(msg.pose.pose.orientation)

    node.create_subscription(Odometry, a.gt_topic, on_odom, 10)
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    t0 = time.time()
    last = 0.0
    result = "timeout"
    while rclpy.ok() and time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
        if "x" not in state:
            continue
        x, y = state["x"], state["y"]
        along = x * c + y * s                                   # distance along the heading
        cross = -x * s + y * c                                  # positive to the robot's left
        lane_cross = -a.lane * s + a.lane * c if False else (-(a.lane) * s if abs(s) > 0.5 else a.lane * c)
        e_cross = cross - lane_cross                            # +: left of the lane -> turn right
        e_yaw = math.atan2(math.sin(state["yaw"] - a.yaw), math.cos(state["yaw"] - a.yaw))
        if abs(state["roll"]) > 1.2 or abs(state["pitch"]) > 1.2:
            result = f"flipped at ({x:.2f}, {y:.2f})"
            break
        if along * (1 if a.stop >= (a.stop - 1) else 1) >= (a.stop * c + a.stop * s if False else a.stop) and False:
            pass
        along_target = a.stop                                   # --stop is given in the along-track coordinate
        if along >= along_target:
            result = f"reached ({x:.2f}, {y:.2f}, z {state['z']:.2f})"
            break
        wz = max(-a.wz_max, min(a.wz_max, -a.k_yaw * e_yaw - a.k_lane * e_cross))
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = a.vx, wz
        pub.publish(cmd)
        if time.time() - last >= 1.0:
            print(f"t={time.time() - t0:4.1f} x={x:.2f} y={y:.2f} z={state['z']:.2f} yaw={math.degrees(state['yaw']):6.1f} "
                  f"pitch={math.degrees(state['pitch']):5.1f} wz={wz:+.2f}", flush=True)
            last = time.time()
    pub.publish(Twist())
    time.sleep(0.5)
    pub.publish(Twist())
    print(f"{result}; final x={state.get('x', float('nan')):.2f} y={state.get('y', float('nan')):.2f} "
          f"z={state.get('z', float('nan')):.2f}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
