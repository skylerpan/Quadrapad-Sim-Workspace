"""ROS 2 front-end for the quadruped navigation sim.

Usage (inside the container):
  python3 -m quadsim.sim_node [--config config/sim.yaml] [--scene scenes/dp_1012] [--rtf 1.0]
                              [--viewer] [--no-web] [key=value ...]

The sim owns /clock: run every other node with use_sim_time:=true.
A browser viewer (viser) is served on http://<host>:8755 by default (config web:, --no-web to skip it).
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
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Int32, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from pcd2sim.pcd_io import read_pcd
from quadsim.base import lidar_mount

from .config import load_config
from .controller_proc import ControllerProcess
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
        # transient_local: a late subscriber asking for it still gets the start-up `spawn` event
        self.pub_events = self.create_publisher(String, tp["events"], QoSProfile(
            depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
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
        self.pending_policy = None
        self.pending_motion = None                 # web viewer stand / walk / lie / damp
        self.pending_path = None
        self.create_subscription(Twist, tp["cmd_vel"], self._on_cmd, 10)
        web = cfg.get("web") or {}
        if web.get("enabled", True) and web.get("path_topic"):     # the planner's path, drawn by the web viewer
            self.create_subscription(Path, web["path_topic"], self._on_path, 10)
        self.create_subscription(PoseStamped, tp["set_pose"], self._on_set_pose, 10)
        if sim.legged and tp.get("policy_mode"):
            self.create_subscription(Int32, tp["policy_mode"], self._on_policy_mode, 10)
        self.odom_every = max(1, int(round(cfg["physics"]["control_hz"] / ros["odom_hz"])))
        self.tick = 0
        self._report_spawn()

    def _pose_event(self, kind: str, requested, ok: bool, moved: bool) -> dict:
        """Event describing where the base actually is after a spawn / set_pose request."""
        b = self.sim.base
        return {"type": kind, "requested": [float(v) for v in requested], "ok": ok, "moved": moved,
                "pose": [b.x, b.y, b.yaw], "z": float(self.sim.d.qpos[2]), "floor_z": float(b.ground.z0),
                "t": self.sim.d.time}

    def _report_spawn(self):
        """Log + publish the real spawn pose. The requested one is not always where the base ends up:
        no ground near z_floor, or the body overlapped the map and was moved to the nearest free pose
        (the UserWarning for that is printed before ROS logging exists and is easy to miss)."""
        sim = self.sim
        e = self._pose_event("spawn", sim.spawn_requested, sim.spawn_ok, sim.spawn_moved)
        self._event(e)
        req = e["requested"]
        if sim.spawn_moved:
            self.get_logger().warn(
                f"spawn ({req[0]:.2f}, {req[1]:.2f}, yaw {req[2]:.2f}) overlaps the map: base moved to "
                f"({e['pose'][0]:.2f}, {e['pose'][1]:.2f}, yaw {e['pose'][2]:.2f}); TF/odom follow the moved "
                f"pose. Set `spawn: [x, y, yaw, z_floor]` in the config to a free spot")
        if not sim.spawn_ok:
            self.get_logger().warn(
                f"spawn: no ground within -1.0..+0.5 m of z_floor={req[3]:.2f} at "
                f"({req[0]:.2f}, {req[1]:.2f}); using z_floor as the floor. z_floor should be the floor "
                f"height of that storey (see scenes/<name>/meta.json spawn or map_ds.pcd)")

    # -- callbacks (executor thread) ------------------------------------------
    def _on_cmd(self, msg: Twist):
        with self.lock:
            self.pending_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_set_pose(self, msg: PoseStamped):
        p = msg.pose.position
        with self.lock:
            self.pending_pose = (p.x, p.y, _yaw_from_quat(msg.pose.orientation), p.z)

    def _on_policy_mode(self, msg: Int32):
        with self.lock:
            self.pending_policy = int(msg.data)

    def _on_path(self, msg: Path):
        pts = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                       dtype=np.float32).reshape(-1, 3)
        with self.lock:
            self.pending_path = pts

    def take_path(self):
        """The latest planned path since the last call (main thread), or None."""
        with self.lock:
            pts, self.pending_path = self.pending_path, None
        return pts

    # -- main-thread stepping ---------------------------------------------------
    def step(self):
        with self.lock:
            cmd, pose, policy, motion = self.pending_cmd, self.pending_pose, self.pending_policy, self.pending_motion
            self.pending_cmd = self.pending_pose = self.pending_policy = self.pending_motion = None
        if pose is not None:
            ok, moved = self.sim.set_pose(*pose)
            self._event(self._pose_event("set_pose", pose, ok, moved))
        if policy is not None:
            e = self.sim.set_policy_mode(policy)
            if e is not None:
                self._event(e)
        if motion is not None:
            e = self.sim.set_motion(motion)
            if e is not None:
                self._event(e)
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
        pos, (qw, qx, qy, qz) = lidar_mount(self.cfg["lidar"])
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = qw, qx, qy, qz
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
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this much sim time (0 = forever)")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo passive viewer (needs a display)")
    ap.add_argument("--no-web", action="store_true", help="do not serve the browser viewer (config web.enabled)")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. lidar.range_noise=0.0")
    a, ros_args = ap.parse_known_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.scene:
        cfg["scene"] = a.scene
    if a.rtf is not None:
        cfg["physics"]["rtf"] = a.rtf
    if a.no_web:
        cfg.setdefault("web", {})["enabled"] = False

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
        f"scene {cfg['scene']} ({sim.meta['boxes']} boxes), spawn requested [x, y, yaw, z_floor] "
        f"{np.round(sim.spawn_requested, 2).tolist()} -> base at {np.round(sim.d.qpos[:3], 2).tolist()} "
        f"yaw {sim.base.yaw:.2f} (floor z {sim.spawn_floor_z:.2f}, moved {sim.spawn_moved}), "
        f"rtf {cfg['physics']['rtf']}, gt_tf {cfg['ros']['publish_gt_tf']}, "
        f"custom_msg {node.pub_custom is not None}, robot {sim.robot_type}"
        + (f" (gait {cfg['robot'].get('gait', 'rl')}, policy {cfg['robot'].get('policy_mode')}, gravity "
           f"{np.round(sim.m.opt.gravity, 3).tolist()}, torque cmds on udp:{sim.g23.cmd_port}, telemetry -> "
           f"{sim.g23.telemetry_to}, commands -> {sim.g23.command_to}; reference sim options: {sim.g23.describe()})"
           if sim.legged else ""))
    spin = threading.Thread(target=_spin, args=(node, stop, failed), daemon=True)
    spin.start()

    # legged mode: the controller runs as our child (robot.controller.spawn, default true) when its binary
    # exists in the mounted controller repo (docker/run.sh QUADSIM_G23=1, docker/build_g23.sh)
    ctl = None
    ctl_cfg = cfg["robot"].get("controller", {}) or {}
    if sim.legged and ctl_cfg.get("spawn", True):
        ctl = ControllerProcess(ctl_cfg, lambda msg: node.get_logger().info(msg))
        if ctl.available:
            ctl.start()
        else:
            node.get_logger().warn(
                f"controller binary not found at {ctl.exe}: start it yourself (g2_deploy_rs sim ros) or build "
                f"it with docker/build_g23.sh; the robot lies still until torque commands arrive on "
                f"udp:{sim.g23.cmd_port}")
            ctl = None

    viewer = None
    viewer_cfg = cfg.get("viewer", {})
    sync_period = 1.0 / max(1.0, float(viewer_cfg.get("sync_hz", 30)))
    next_sync = 0.0
    sync_wall = 0.0                              # wall time spent inside viewer.sync() since the last report
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)
        with viewer.lock():                      # follow the base; the map mesh is far larger than the robot
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.base_body
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 6.0, -35.0, 180.0
        node.get_logger().info(
            f"viewer: gl={os.environ.get('QUADSIM_GL', 'unknown')} (docker/run.sh QUADSIM_GL), map mesh "
            f"{sim.m.nmeshface} faces, shadows {bool(viewer_cfg.get('shadows', False))}, "
            f"sync <= {1.0 / sync_period:.0f} Hz")

    # browser viewer: on by default, never fatal (a missing viser or a bind error just means no web view)
    web = None
    web_wall = 0.0
    if (cfg.get("web") or {}).get("enabled", True):
        try:
            from .web_viewer import WebViewer
            web = WebViewer(cfg, sim, log=lambda msg: node.get_logger().info(msg))
        except ImportError as e:
            node.get_logger().warn(f"web viewer disabled: {e} (pip3 install viser; docker/Dockerfile installs it)")
        except Exception as e:  # noqa: BLE001
            node.get_logger().warn(f"web viewer failed to start: {e!r}; running without it")
        else:
            node.get_logger().info(f"web viewer: {web.url} ({web.describe()})")
            if web.port != int((cfg.get("web") or {}).get("port", web.port)):
                node.get_logger().warn(f"web viewer: port {(cfg.get('web') or {}).get('port')} was busy, serving on {web.port}")

    rtf = cfg["physics"]["rtf"]
    wall0, sim0 = time.perf_counter(), sim.d.time
    last_report = wall0
    report_sim = sim.d.time
    code = 0
    try:
        while not stop.is_set() and rclpy.ok() and (a.duration <= 0 or sim.d.time < a.duration):
            if web is not None:                      # browser controls go through the same path as the topics
                tele, reset = web.teleop_cmd(), web.take_reset(sim)
                motion, policy = web.take_motion(), web.take_policy()
                if tele is not None or reset is not None or motion is not None or policy is not None:
                    with node.lock:
                        if tele is not None:
                            node.pending_cmd = tele
                        if reset is not None:
                            node.pending_pose = reset
                        if motion is not None:
                            node.pending_motion = motion
                        if policy is not None:
                            node.pending_policy = policy
            out = node.step()
            now = time.perf_counter()
            if ctl is not None and ctl.poll() is not None:
                node._event({"type": "controller", "state": "exited", "code": ctl.poll(), "t": sim.d.time})
                node.get_logger().error(f"controller exited with code {ctl.poll()}: stopping the sim")
                code = 1
                break
            if viewer is not None:
                if not viewer.is_running():          # viewer window closed: stop the sim too
                    break
                if now >= next_sync:                 # sync() takes the render lock: no more often than the display needs
                    viewer.sync()
                    t_after = time.perf_counter()
                    sync_wall += t_after - now
                    now = t_after
                    next_sync = now + sync_period
            if web is not None:
                path = node.take_path()
                if path is not None:
                    web.set_path(path)
                web_wall += web.update(sim, out.frames)
            if rtf > 0:
                ahead = (sim.d.time - sim0) / rtf - (now - wall0)
                if ahead > 0:
                    stop.wait(ahead)
                elif ahead < -1.0:                       # fell behind: don't try to catch up
                    wall0, sim0 = now, sim.d.time
            if now - last_report > 10.0:
                achieved = (sim.d.time - report_sim) / (now - last_report)
                msg = f"sim t={sim.d.time:.1f}s pose={np.round(sim.d.qpos[:3], 2).tolist()}, rtf {achieved:.2f}"
                if sim.legged:
                    msg += f", lidar worker {100.0 * sim.lidar.busy_s / (now - last_report):.0f}% of wall"
                    sim.lidar.busy_s = 0.0
                if viewer is not None:
                    msg += f", viewer.sync {100.0 * sync_wall / (now - last_report):.0f}% of wall"
                    sync_wall = 0.0
                if web is not None:
                    msg += f", web {100.0 * web_wall / (now - last_report):.0f}% of wall ({web.n_clients} clients)"
                    web_wall = 0.0
                node.get_logger().info(msg)
                if sim.legged and rtf > 0 and achieved < 0.98 * rtf:
                    # the controller is a wall-clock 1 kHz loop: a slow sim starves it of telemetry
                    node.get_logger().warn(f"real-time factor {achieved:.2f} < {rtf} in legged mode: the "
                                           f"controller sees late telemetry (reduce lidar.points_per_sec, "
                                           f"robot.collision_pool, or close the viewer)")
                last_report = now
                report_sim = sim.d.time
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
        if web is not None:
            web.close()
        if ctl is not None:
            ctl.stop()
        sim.close()
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip interpreter teardown (DDS/open3d/GLFW threads can block it); everything is already closed.
        os._exit(code)


if __name__ == "__main__":
    main()
