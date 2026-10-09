"""ROS 2 front-end for the quadruped navigation sim.

Usage (inside the container):
  python3 -m quadsim.sim_node [--config config/sim.yaml] [--scene scenes/dp2f] [--rtf 1.0]
                              [--no-gt-tf] [--viewer] [key=value ...]

The sim owns /clock: run every other node with use_sim_time:=true.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from pcd2sim.pcd_io import read_pcd

from .config import load_config
from .ros_io import lidar_pointcloud2, serialize_custom_msg, stamp_parts, xyzi_pointcloud2
from .sim import Simulator

try:
    from livox_ros_driver2.msg import CustomMsg
except ImportError:  # message package not built: PointCloud2 only
    CustomMsg = None


def _yaw_from_quat(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class SimNode(Node):
    def __init__(self, cfg: dict, sim: Simulator):
        super().__init__("quadsim")
        self.cfg, self.sim = cfg, sim
        ros = cfg["ros"]
        self.fr = ros["frames"]
        tp = ros["topics"]
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_clock = self.create_publisher(Clock, tp["clock"], 10)
        self.pub_imu = self.create_publisher(Imu, tp["imu"], sensor_qos)
        self.pub_odom = self.create_publisher(Odometry, tp["gt_odom"], 10)
        self.pub_events = self.create_publisher(String, tp["events"], 10)
        self.pub_points = (self.create_publisher(PointCloud2, tp["lidar_points"], sensor_qos)
                           if ros["publish_pointcloud2"] else None)
        self.pub_custom = None
        if ros["publish_custom_msg"]:
            if CustomMsg is None:
                self.get_logger().warn("livox_ros_driver2 msgs not found: CustomMsg output disabled")
            else:
                self.pub_custom = self.create_publisher(CustomMsg, tp["lidar_custom"], sensor_qos)
        self.tf = TransformBroadcaster(self) if ros["publish_gt_tf"] else None
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()
        if ros["publish_map_cloud"]:
            self.pub_map = self.create_publisher(PointCloud2, tp["map_cloud"], latched)
            xyz, inten = read_pcd(os.path.join(cfg["scene"], "map_ds.pcd"))
            self.pub_map.publish(xyzi_pointcloud2(0.0, self.fr["map"], xyz.astype(np.float32),
                                                  inten if inten is not None else np.zeros(len(xyz))))

        self.lock = threading.Lock()
        self.pending_cmd = None
        self.pending_pose = None
        self.create_subscription(Twist, tp["cmd_vel"], self._on_cmd, 10)
        self.create_subscription(PoseStamped, tp["set_pose"], self._on_set_pose, 10)
        self.odom_every = max(1, int(round(cfg["physics"]["control_hz"] / ros["odom_hz"])))
        self.tick = 0

    # -- callbacks (executor thread) ------------------------------------------
    def _on_cmd(self, msg: Twist):
        with self.lock:
            self.pending_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_set_pose(self, msg: PoseStamped):
        p = msg.pose.position
        with self.lock:
            self.pending_pose = (p.x, p.y, _yaw_from_quat(msg.pose.orientation), p.z)

    # -- main-thread stepping ---------------------------------------------------
    def step(self):
        with self.lock:
            cmd, pose = self.pending_cmd, self.pending_pose
            self.pending_cmd = self.pending_pose = None
        if pose is not None:
            ok = self.sim.set_pose(*pose)
            self._event({"type": "set_pose", "ok": ok, "pose": list(pose), "t": self.sim.d.time})
        if cmd is not None:
            self.sim.set_cmd(*cmd)
        out = self.sim.control_tick()
        for t in out.clock:
            msg = Clock()
            msg.clock.sec, msg.clock.nanosec = stamp_parts(t)
            self.pub_clock.publish(msg)
        for s in out.imu:
            self._publish_imu(s)
        for f in out.frames:
            if self.pub_custom is not None:
                self.pub_custom.publish(serialize_custom_msg(f.stamp, self.fr["lidar"], f))
            if self.pub_points is not None:
                self.pub_points.publish(lidar_pointcloud2(f.stamp, self.fr["lidar"], f,
                                                          self.cfg["ros"]["pointcloud2_format"]))
        for e in out.events:
            self._event(e)
        self.tick += 1
        if self.tick % self.odom_every == 0:
            self._publish_ground_truth(out.t)
        return out

    def _event(self, e: dict):
        self.pub_events.publish(String(data=json.dumps(e)))
        self.get_logger().info(f"event: {e}")

    def _publish_imu(self, s):
        m = Imu()
        m.header.stamp.sec, m.header.stamp.nanosec = stamp_parts(s.t)
        m.header.frame_id = self.fr["lidar"]
        m.orientation_covariance[0] = -1.0          # no orientation (like the Mid-360 IMU)
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, s.acc)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, s.gyro)
        self.pub_imu.publish(m)

    def _publish_ground_truth(self, t: float):
        pos, quat, v_body, w_body = self.sim.ground_truth()
        stamp = stamp_parts(t)
        od = Odometry()
        od.header.stamp.sec, od.header.stamp.nanosec = stamp
        od.header.frame_id = self.fr["map"]
        od.child_frame_id = self.fr["base"]
        p, o = od.pose.pose.position, od.pose.pose.orientation
        p.x, p.y, p.z = map(float, pos)
        o.w, o.x, o.y, o.z = map(float, quat)
        tw = od.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = map(float, v_body)
        tw.angular.x, tw.angular.y, tw.angular.z = map(float, w_body)
        self.pub_odom.publish(od)
        if self.tf is not None:
            ident = TransformStamped()
            ident.header.stamp = od.header.stamp
            ident.header.frame_id, ident.child_frame_id = self.fr["map"], self.fr["odom"]
            ident.transform.rotation.w = 1.0
            base = TransformStamped()
            base.header.stamp = od.header.stamp
            base.header.frame_id, base.child_frame_id = self.fr["odom"], self.fr["base"]
            base.transform.translation.x, base.transform.translation.y, base.transform.translation.z = map(float, pos)
            base.transform.rotation = o
            self.tf.sendTransform([ident, base])

    def _publish_static_tf(self):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = self.fr["base"], self.fr["lidar"]
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = \
            map(float, self.cfg["lidar"]["mount_xyz"])
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)


def _spin(node, stop: threading.Event, failed: threading.Event):
    """Executor thread. A callback exception would otherwise end it silently and leave the sim
    running deaf to /cmd_vel: log it and stop the whole node instead."""
    try:
        rclpy.spin(node)
    except Exception:  # noqa: BLE001
        if rclpy.ok():                     # not a normal shutdown
            traceback.print_exc()
            os.write(2, b"quadsim: ROS executor died, stopping the sim\n")
            failed.set()
            stop.set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--rtf", type=float, default=None, help="real-time factor (0 = as fast as possible)")
    ap.add_argument("--no-gt-tf", action="store_true", help="do not publish map->odom->base_link")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this much sim time (0 = forever)")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo passive viewer (needs a display)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. lidar.range_noise=0.0")
    a, ros_args = ap.parse_known_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.scene:
        cfg["scene"] = a.scene
    if a.rtf is not None:
        cfg["physics"]["rtf"] = a.rtf
    if a.no_gt_tf:
        cfg["ros"]["publish_gt_tf"] = False

    # Own SIGINT/SIGTERM handling: rclpy's handler races with the background spin thread and
    # destroy_node, which could leave the process hanging after Ctrl+C. 2nd Ctrl+C exits at once.
    stop = threading.Event()
    failed = threading.Event()               # set by the executor thread on a callback error

    def _on_signal(signum, _frame):
        if stop.is_set():
            os._exit(130)
        stop.set()
        # os.write, not print: print from a signal handler can hit a reentrant stdout write
        os.write(2, b"\nquadsim: shutting down (Ctrl+C again to force)\n")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sim = Simulator(cfg, visual_map=a.viewer)
    if stop.is_set():
        os._exit(130)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = SimNode(cfg, sim)
    node.get_logger().info(
        f"scene {cfg['scene']} ({sim.meta['boxes']} boxes), spawn {np.round(sim.d.qpos[:3], 2).tolist()}, "
        f"rtf {cfg['physics']['rtf']}, gt_tf {cfg['ros']['publish_gt_tf']}, "
        f"custom_msg {node.pub_custom is not None}")
    spin = threading.Thread(target=_spin, args=(node, stop, failed), daemon=True)
    spin.start()

    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)
        with viewer.lock():                      # follow the base; the map mesh is far larger than the robot
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.base_body
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 6.0, -35.0, 180.0

    rtf = cfg["physics"]["rtf"]
    wall0, sim0 = time.perf_counter(), sim.d.time
    last_report = wall0
    code = 0
    try:
        while not stop.is_set() and rclpy.ok() and (a.duration <= 0 or sim.d.time < a.duration):
            node.step()
            if viewer is not None:
                if not viewer.is_running():          # viewer window closed: stop the sim too
                    break
                viewer.sync()
            now = time.perf_counter()
            if rtf > 0:
                ahead = (sim.d.time - sim0) / rtf - (now - wall0)
                if ahead > 0:
                    stop.wait(ahead)
                elif ahead < -1.0:                       # fell behind: don't try to catch up
                    wall0, sim0 = now, sim.d.time
            if now - last_report > 10.0:
                node.get_logger().info(f"sim t={sim.d.time:.1f}s pose={np.round(sim.d.qpos[:3], 2).tolist()}")
                last_report = now
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        rclpy.try_shutdown()                     # wakes and ends the spin thread
        spin.join(timeout=2.0)
        if failed.is_set():
            code = 1
        if viewer is not None:
            viewer.close()
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip interpreter teardown (DDS/open3d/GLFW threads can block it); everything is already closed.
        os._exit(code)


if __name__ == "__main__":
    main()
