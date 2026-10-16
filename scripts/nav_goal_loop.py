#!/usr/bin/env python3
"""Navigation goal-loop test: send map-frame waypoints to the running navigation stack one after
another and verify arrival against the simulator's ground truth.

Run inside the sim container (host has no ROS):
  scripts/nav_goal_loop.sh --dry-run                       # plan only: which legs the planner can solve
  scripts/nav_goal_loop.sh --loops 2 --order shuffle       # drive the loop and verify every leg
  scripts/nav_goal_loop.sh --select wp03,wp04 --loops 1    # a few named points

Per leg the script mirrors syncai_waypoint_sequencer's protocol:
  1. /initialpose  <- current /robot01/tracked_pose (the planner does not read the robot pose itself)
  2. /goal_pose    <- the waypoint (0.15 s later)
  3. wait for a /robot01/planned_path received after the goal, non-empty, ending near the waypoint
  4. send it to the /robot01/follow_path action and wait for the result
  5. settle, then compare /sim/ground_truth/odom with the path endpoint, check localization error,
     and look for collision / stuck events on /sim/events during the leg
The planner's `auto_follow_path` is switched off for the run (else it would send FollowPath too)
and restored at exit. Reports go to results/nav_goal_loop/<timestamp>/.
Exit code: 0 all legs PASS, 1 any FAIL, 2 preflight failed, 130 interrupted.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from typing import Callable, Optional

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

try:
    from rclpy.signals import SignalHandlerOptions
except ImportError:  # older rclpy
    SignalHandlerOptions = None

STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN", GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING", GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


class Interrupted(Exception):
    pass


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def path_length(path: Path) -> float:
    pts = [p.pose.position for p in path.poses]
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:]))


def path_max_dz(path: Path) -> float:
    zs = [p.pose.position.z for p in path.poses]
    return max(zs) - min(zs) if zs else 0.0


def load_waypoints(path: str) -> tuple[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc = doc.get("waypoint_sequencer", doc)  # accept the sequencer's file layout too
    frame = str(doc.get("frame_id", "map"))
    wps = []
    for i, w in enumerate(doc["waypoints"]):
        wps.append({"name": str(w.get("name", f"wp{i + 1:02d}")), "x": float(w["x"]), "y": float(w["y"]),
                    "z": float(w.get("z", 0.0)), "yaw": float(w.get("yaw", 0.0))})
    if not wps:
        raise ValueError(f"{path}: no waypoints")
    return frame, wps


class NavGoalLoop(Node):
    def __init__(self, args: argparse.Namespace, frame: str):
        super().__init__("nav_goal_loop", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.a = args
        self.frame = frame
        self.sim_t: Optional[float] = None
        self.gt: Optional[Odometry] = None
        self.tracked: Optional[PoseStamped] = None
        self.tracked_wall = 0.0
        self.paths: list[tuple[float, Path]] = []        # (wall time received, path)
        self.events: list[dict] = []
        self.feedback: Optional[FollowPath.Feedback] = None
        self.stop_requested = False

        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Clock, "/clock", self._on_clock, 50)
        self.create_subscription(Odometry, args.gt_topic, self._on_gt, 20)
        self.create_subscription(PoseStamped, args.tracked_pose_topic, self._on_tracked, 20)
        # volatile on purpose: we must not receive the last latched path from before this run
        self.create_subscription(Path, args.planned_path_topic, self._on_path, 10)
        self.create_subscription(String, args.events_topic, self._on_event, latched)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, args.initial_pose_topic, 10)
        self.pub_goal = self.create_publisher(PoseStamped, args.goal_pose_topic, 10)
        self.follow = ActionClient(self, FollowPath, args.follow_path_action)
        planner = args.planner_node.rstrip("/")
        self.param_get = self.create_client(GetParameters, planner + "/get_parameters")
        self.param_set = self.create_client(SetParameters, planner + "/set_parameters")
        self.seq_status: Optional[str] = None
        self.create_subscription(String, "/waypoint_sequencer/status", self._on_seq_status,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))

    # ---- callbacks -------------------------------------------------------------------------
    def _on_clock(self, m: Clock):
        self.sim_t = m.clock.sec + 1e-9 * m.clock.nanosec

    def _on_gt(self, m: Odometry):
        self.gt = m

    def _on_tracked(self, m: PoseStamped):
        self.tracked = m
        self.tracked_wall = time.monotonic()

    def _on_path(self, m: Path):
        self.paths.append((time.monotonic(), m))

    def _on_event(self, m: String):
        try:
            self.events.append(json.loads(m.data))
        except json.JSONDecodeError:
            pass

    def _on_seq_status(self, m: String):
        self.seq_status = m.data

    # ---- helpers ---------------------------------------------------------------------------
    def now_sim(self) -> float:
        return self.sim_t if self.sim_t is not None else 0.0

    def spin(self, sim_seconds: float, until: Optional[Callable[[], bool]] = None, wall_timeout: float = 0.0) -> bool:
        """Spin for `sim_seconds` of sim time (or until `until()`); True when `until` fired.
        A wall-clock guard (3x + 30 s by default) keeps us from hanging if the sim stalls."""
        t_end = self.now_sim() + sim_seconds
        w_end = time.monotonic() + (wall_timeout or (3.0 * sim_seconds + 30.0))
        while rclpy.ok():
            if self.stop_requested:
                raise Interrupted()
            rclpy.spin_once(self, timeout_sec=0.02)
            if until is not None and until():
                return True
            if self.now_sim() >= t_end or time.monotonic() >= w_end:
                return False
        raise Interrupted()

    def wait_future(self, fut, sim_timeout: float):
        self.spin(sim_timeout, until=fut.done)
        return fut.result() if fut.done() else None

    def gt_xy(self) -> tuple[float, float, float]:
        p = self.gt.pose.pose.position
        return p.x, p.y, p.z

    def gt_speed(self) -> float:
        v = self.gt.twist.twist.linear
        return math.hypot(v.x, v.y)

    def tracked_fresh(self) -> bool:
        return self.tracked is not None and time.monotonic() - self.tracked_wall < self.a.pose_timeout

    def loc_error(self) -> float:
        if self.tracked is None or self.gt is None:
            return float("nan")
        t, g = self.tracked.pose.position, self.gt.pose.pose.position
        return math.hypot(t.x - g.x, t.y - g.y)

    def pose_msg(self, x: float, y: float, z: float, yaw: float) -> PoseStamped:
        m = PoseStamped()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m

    def publish_start(self, pose: PoseStamped):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose = pose.pose
        self.pub_init.publish(m)

    # ---- planner parameter ---------------------------------------------------------------------
    def get_auto_follow(self) -> Optional[bool]:
        req = GetParameters.Request(names=["auto_follow_path"])
        res = self.wait_future(self.param_get.call_async(req), 5.0)
        if res is None or not res.values:
            return None
        return bool(res.values[0].bool_value)

    def set_auto_follow(self, value: bool) -> bool:
        req = SetParameters.Request()
        req.parameters = [Parameter("auto_follow_path", Parameter.Type.BOOL, value).to_parameter_msg()]
        res = self.wait_future(self.param_set.call_async(req), 5.0)
        return bool(res and res.results and res.results[0].successful)


class Runner:
    def __init__(self, node: NavGoalLoop, args: argparse.Namespace, wps: list[dict], out_dir: str):
        self.n = node
        self.a = args
        self.wps = wps
        self.out_dir = out_dir
        self.legs: list[dict] = []
        self.log_f = open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8")
        self.auto_follow_before: Optional[bool] = None
        self.restored = False
        self.active_goal = None
        self.run_start_sim = 0.0
        self.run_start_wall = dt.datetime.now().isoformat(timespec="seconds")

    def log(self, msg: str):
        line = f"[{self.n.now_sim():9.2f}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    # ---- preflight -----------------------------------------------------------------------------
    def preflight(self) -> bool:
        n, a = self.n, self.a
        self.log("waiting for /clock, ground truth, tracked pose, FollowPath server, planner parameters ...")
        ready = lambda: (n.sim_t is not None and n.gt is not None and n.tracked_fresh()  # noqa: E731
                         and n.follow.server_is_ready() and n.param_get.service_is_ready()
                         and n.param_set.service_is_ready())
        if not n.spin(1e9, until=ready, wall_timeout=a.preflight_timeout):
            missing = [name for name, ok in (("/clock", n.sim_t is not None), ("ground truth", n.gt is not None),
                                              ("tracked pose", n.tracked_fresh()),
                                              ("FollowPath action", n.follow.server_is_ready()),
                                              ("planner param service", n.param_get.service_is_ready()))
                       if not ok]
            self.log(f"PREFLIGHT FAIL: not available: {', '.join(missing)}")
            return False
        self.run_start_sim = n.now_sim()
        err = n.loc_error()
        gx, gy, _ = n.gt_xy()
        self.log(f"robot at ({gx:.2f}, {gy:.2f}) ground truth; localization error {err:.3f} m")
        if err > a.loc_preflight:
            self.log(f"PREFLIGHT FAIL: localization error {err:.2f} m > {a.loc_preflight} m (relocalize first)")
            return False
        if n.tracked.header.frame_id != n.frame:
            self.log(f"PREFLIGHT FAIL: tracked pose frame '{n.tracked.header.frame_id}' != '{n.frame}'")
            return False
        if n.seq_status:
            try:
                st = json.loads(n.seq_status).get("state")
            except json.JSONDecodeError:
                st = n.seq_status
            if st not in (None, "IDLE", "COMPLETED", "CANCELED", "FAILED", "PAUSED"):
                self.log(f"WARNING: /waypoint_sequencer is {st}; it will fight over /goal_pose")
        self.auto_follow_before = n.get_auto_follow()
        if self.auto_follow_before is None:
            self.log("PREFLIGHT FAIL: could not read auto_follow_path from the planner")
            return False
        if not n.set_auto_follow(False):
            self.log("PREFLIGHT FAIL: could not set auto_follow_path=false on the planner")
            return False
        self.log(f"planner auto_follow_path was {self.auto_follow_before}; now false for this run")
        return True

    def restore(self):
        if self.restored or self.auto_follow_before is None:
            return
        self.restored = True
        ok = False
        try:
            ok = self.n.set_auto_follow(self.auto_follow_before)
        except Exception:  # noqa: BLE001
            pass
        self.log(f"planner auto_follow_path restored to {self.auto_follow_before}: {'ok' if ok else 'FAILED'}")
        if not ok:
            self.log(f"  -> run manually: ros2 param set {self.a.planner_node} auto_follow_path "
                     f"{str(self.auto_follow_before).lower()}")

    def cancel_active(self):
        if self.active_goal is not None:
            try:
                fut = self.active_goal.cancel_goal_async()
                self.n.spin(3.0, until=fut.done, wall_timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            self.active_goal = None

    # ---- one leg -------------------------------------------------------------------------------
    def plan_leg(self, start: Optional[dict], wp: dict, leg: dict) -> Optional[Path]:
        """initialpose -> goal_pose -> matching planned_path. None (with leg['reason']) on failure."""
        n, a = self.n, self.a
        if start is None:
            if not n.tracked_fresh():
                n.spin(a.pose_timeout, until=n.tracked_fresh)
            if not n.tracked_fresh():
                leg["reason"] = "NO_POSE"
                return None
            start_pose = n.tracked
        else:
            start_pose = n.pose_msg(start["x"], start["y"], start["z"], start["yaw"])
        sp = start_pose.pose.position
        leg["start"] = [round(sp.x, 3), round(sp.y, 3), round(sp.z, 3)]
        n.publish_start(start_pose)
        n.spin(a.start_goal_delay)
        n.paths.clear()
        t_goal = time.monotonic()
        n.pub_goal.publish(n.pose_msg(wp["x"], wp["y"], wp["z"], wp["yaw"]))
        t0 = n.now_sim()
        deadline = t0 + a.plan_timeout
        empty_deadline = None
        seen = 0
        rejected = []
        while True:
            n.spin(0.05)
            while seen < len(n.paths):
                recv, p = n.paths[seen]
                seen += 1
                if recv < t_goal:
                    continue
                if not p.poses:
                    # may be the planner re-planning the OLD goal after our initialpose: wait a bit more
                    if empty_deadline is None:
                        empty_deadline = n.now_sim() + a.empty_grace
                    continue
                e = p.poses[-1].pose.position
                dxy, dz = math.hypot(e.x - wp["x"], e.y - wp["y"]), abs(e.z - wp["z"])
                if dxy > a.endpoint_xy_tol or dz > a.endpoint_z_tol:
                    rejected.append((round(dxy, 2), round(dz, 2)))
                    continue
                leg.update(path_len_m=round(path_length(p), 2), path_poses=len(p.poses),
                           path_max_dz=round(path_max_dz(p), 2), plan_time_s=round(n.now_sim() - t0, 2),
                           path_endpoint=[round(e.x, 3), round(e.y, 3), round(e.z, 3)],
                           dist_endpoint_to_waypoint_m=round(dxy, 3))
                if dxy > a.snap_warn:
                    leg["warnings"].append("SNAPPED_FAR")
                return p
            now = n.now_sim()
            if now >= deadline or (empty_deadline is not None and now >= empty_deadline):
                leg["reason"] = "NO_PATH"
                leg["plan_time_s"] = round(now - t0, 2)
                if rejected:
                    leg["detail"] = f"paths ignored, endpoint off by (xy, z) {rejected}"
                elif empty_deadline is None:
                    leg["detail"] = "no path message at all"
                return None

    def execute_leg(self, path: Path, wp: dict, leg: dict) -> bool:
        n, a = self.n, self.a
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = a.controller_id
        goal.goal_checker_id = a.goal_checker_id
        n.feedback = None
        send = n.follow.send_goal_async(goal, feedback_callback=lambda f: setattr(n, "feedback", f.feedback))
        handle = n.wait_future(send, a.action_server_timeout)
        if handle is None or not handle.accepted:
            leg["reason"] = "ACTION_REJECTED"
            return False
        self.active_goal = handle
        result_fut = handle.get_result_async()
        leg_timeout = max(a.leg_timeout_min, leg["path_len_m"] / a.expected_speed + a.leg_timeout_margin)
        leg["leg_timeout_s"] = round(leg_timeout, 1)
        t_exec = n.now_sim()
        leg_start = leg["t_start_sim"]
        max_loc, max_speed = 0.0, 0.0
        status_name = "TIMEOUT"
        while True:
            n.spin(0.1, until=result_fut.done)
            max_loc = max(max_loc, n.loc_error())
            max_speed = max(max_speed, n.gt_speed())
            ev = [e for e in n.events if e.get("t", 0.0) >= leg_start and e.get("type") in ("collision", "stuck")]
            if result_fut.done():
                status = result_fut.result().status
                status_name = STATUS_NAMES.get(status, str(status))
                break
            if any(e["type"] == "collision" for e in ev):
                status_name = "COLLISION"
                break
            if n.now_sim() - t_exec > leg_timeout:
                break
        fb = n.feedback
        leg.update(action_status=status_name, exec_time_s=round(n.now_sim() - t_exec, 2),
                   loc_err_max_m=round(max_loc, 3), max_speed_mps=round(max_speed, 2),
                   last_distance_to_goal=round(fb.distance_to_goal, 2) if fb else None,
                   events=[{k: e.get(k) for k in ("type", "reason", "t")} for e in ev])
        if ev:
            leg["detail"] = "sim events: " + ", ".join(sorted({f"{e['type']}/{e.get('reason')}" for e in ev}))
        if status_name in ("COLLISION", "TIMEOUT"):
            self.cancel_active()
            leg["reason"] = status_name
            return False
        self.active_goal = None
        if status_name != "SUCCEEDED":
            leg["reason"] = status_name
            return False
        return True

    def verify_leg(self, wp: dict, leg: dict) -> bool:
        n, a = self.n, self.a
        # settle: the base must stay still
        for attempt in range(2):
            x0, y0, _ = n.gt_xy()
            moved = 0.0
            t_end = n.now_sim() + a.settle
            while n.now_sim() < t_end:
                n.spin(0.05)
                x, y, _ = n.gt_xy()
                moved = max(moved, math.hypot(x - x0, y - y0))
            if moved < a.speed_eps * a.settle:
                break
        else:
            leg["reason"] = "STILL_MOVING"
            return False
        gx, gy, gz = n.gt_xy()
        ex, ey, _ = leg["path_endpoint"]
        d_end = math.hypot(gx - ex, gy - ey)
        d_wp = math.hypot(gx - wp["x"], gy - wp["y"])
        loc = n.loc_error()
        leg.update(arrival_gt=[round(gx, 3), round(gy, 3), round(gz, 3)], dist_to_endpoint_m=round(d_end, 3),
                   dist_to_waypoint_m=round(d_wp, 3), loc_err_arrival_m=round(loc, 3),
                   arrival_yaw_err=round(abs(math.remainder(yaw_of(n.gt.pose.pose.orientation) - wp["yaw"],
                                                            2 * math.pi)), 2))
        ev = [e for e in n.events if e.get("t", 0.0) >= leg["t_start_sim"] and e.get("type") in ("collision", "stuck")]
        leg["events"] = [{k: e.get(k) for k in ("type", "reason", "t")} for e in ev]
        if any(e["type"] == "collision" for e in ev):
            leg["reason"] = "COLLISION"
            return False
        if any(e["type"] == "stuck" for e in ev):
            if a.stuck_is_fail:
                leg["reason"] = "STUCK"
                return False
            leg["warnings"].append("STUCK")
        if d_end > a.xy_tol:
            leg["reason"] = "GT_NOT_AT_GOAL"
            return False
        if leg["loc_err_max_m"] > a.loc_fail or loc > a.loc_fail:
            leg["reason"] = "LOC_ERROR"
            return False
        if loc > a.loc_warn:
            leg["warnings"].append("LOC_DRIFT")
        return True

    def run_leg(self, idx: int, loop: int, start: Optional[dict], wp: dict) -> dict:
        n, a = self.n, self.a
        leg = {"idx": idx, "loop": loop, "from": start["name"] if start else "robot", "to": wp["name"],
               "goal": [wp["x"], wp["y"], wp["z"]], "t_start_sim": round(n.now_sim(), 2),
               "result": "FAIL", "reason": None, "warnings": []}
        try:
            path = self.plan_leg(start, wp, leg)
            if path is not None:
                if a.dry_run:
                    leg["result"] = "PASS"
                elif self.execute_leg(path, wp, leg) and self.verify_leg(wp, leg):
                    leg["result"] = "PASS"
        except Interrupted:
            leg["reason"] = "INTERRUPTED"
            leg["t_end_sim"] = round(n.now_sim(), 2)
            self.legs.append(leg)
            self.log(self.format_leg(leg))
            raise
        if leg["result"] == "FAIL" and leg["reason"] is None:
            leg["reason"] = "UNKNOWN"
        leg["t_end_sim"] = round(n.now_sim(), 2)
        self.legs.append(leg)
        self.log(self.format_leg(leg))
        return leg

    @staticmethod
    def format_leg(leg: dict) -> str:
        parts = [f"LEG {leg['idx']} loop{leg['loop']} {leg['from']}->{leg['to']}"]
        if "path_len_m" in leg:
            parts.append(f"path {leg['path_len_m']:.1f} m plan {leg.get('plan_time_s', 0):.2f} s")
        if "action_status" in leg:
            parts.append(f"exec {leg['exec_time_s']:.1f} s {leg['action_status']}")
        if "dist_to_endpoint_m" in leg:
            parts.append(f"gt-dist {leg['dist_to_endpoint_m']:.2f} (wp {leg['dist_to_waypoint_m']:.2f})"
                         f" loc {leg['loc_err_arrival_m']:.2f}/{leg['loc_err_max_m']:.2f}")
        if leg.get("events"):
            parts.append(f"events {len(leg['events'])}")
        parts.append(leg["result"] + (f"({leg['reason']})" if leg["reason"] else ""))
        if leg.get("warnings"):
            parts.append("warn:" + ",".join(leg["warnings"]))
        if leg.get("detail"):
            parts.append(leg["detail"])
        return "  ".join(parts)

    # ---- ordering ------------------------------------------------------------------------------
    def nearest_index(self) -> int:
        gx, gy, _ = self.n.gt_xy()
        return min(range(len(self.wps)), key=lambda i: math.hypot(self.wps[i]["x"] - gx, self.wps[i]["y"] - gy))

    def order_for_loop(self, loop: int, rng: random.Random, last: Optional[dict]) -> list[dict]:
        wps = list(self.wps)
        if self.a.order == "shuffle":
            rng.shuffle(wps)
            if last is not None and len(wps) > 1 and wps[0] is last:
                wps.append(wps.pop(0))
            return wps
        if self.a.order == "random":
            out = []
            prev = last
            for _ in range(len(wps)):
                choices = [w for w in wps if w is not prev] or wps
                prev = rng.choice(choices)
                out.append(prev)
            return out
        start = 0
        if self.a.start_at:
            names = [w["name"] for w in wps]
            if self.a.start_at not in names:
                raise SystemExit(f"--start-at {self.a.start_at}: no such waypoint")
            start = names.index(self.a.start_at)
        elif self.a.start_nearest:
            start = self.nearest_index()
        return wps[start:] + wps[:start]

    # ---- main loops ----------------------------------------------------------------------------
    def run(self) -> int:
        a = self.a
        rng = random.Random(a.seed)
        idx = 0
        consecutive_fail = 0
        aborted = None
        try:
            if a.dry_run and a.dry_run_from == "matrix":
                pairs = [(s, g) for s in self.wps for g in self.wps if s is not g]
                self.log(f"dry-run matrix: {len(pairs)} ordered pairs")
                for s, g in pairs:
                    idx += 1
                    self.run_leg(idx, 1, s, g)
            else:
                last = None
                loop = 0
                while True:
                    loop += 1
                    if a.loops and loop > a.loops:
                        break
                    for wp in self.order_for_loop(loop, rng, last):
                        if a.duration and self.n.now_sim() - self.run_start_sim >= a.duration:
                            aborted = "duration reached"
                            break
                        idx += 1
                        start = None
                        if a.dry_run and a.dry_run_from == "chain" and last is not None:
                            start = last
                        leg = self.run_leg(idx, loop, start, wp)
                        if a.dry_run:
                            # chain on from the last point the planner could reach
                            last = wp if leg["result"] == "PASS" else last
                            self.n.spin(a.dry_dwell)
                        else:
                            last = wp
                            self.n.spin(a.dwell)
                        if leg["result"] == "PASS":
                            consecutive_fail = 0
                        else:
                            consecutive_fail += 1
                            if a.stop_on_fail:
                                aborted = "stop-on-fail"
                                break
                            if consecutive_fail >= a.max_consecutive_fail:
                                aborted = f"{consecutive_fail} consecutive failures"
                                break
                    if aborted:
                        break
                    if a.duration and self.n.now_sim() - self.run_start_sim >= a.duration:
                        break
        except Interrupted:
            aborted = "interrupted"
            self.log("interrupted: cancelling the active goal (press Ctrl-C again to skip the cleanup)")
        finally:
            self.n.stop_requested = False  # cleanup still needs to spin; a second Ctrl-C aborts it
            try:
                self.cancel_active()
            finally:
                try:
                    self.restore()
                finally:
                    self.write_report(aborted)
        if aborted == "interrupted":
            return 130
        if aborted and aborted != "duration reached":
            return 1
        return 0 if all(l["result"] == "PASS" for l in self.legs) else 1

    # ---- report --------------------------------------------------------------------------------
    def write_report(self, aborted: Optional[str]):
        legs = self.legs
        fails = [l for l in legs if l["result"] != "PASS"]
        by_reason: dict[str, int] = {}
        for l in fails:
            by_reason[l["reason"]] = by_reason.get(l["reason"], 0) + 1
        per_wp: dict[str, dict] = {}
        for l in legs:
            d = per_wp.setdefault(l["to"], {"pass": 0, "fail": 0})
            d["pass" if l["result"] == "PASS" else "fail"] += 1
        # mean speed over the legs that were actually driven to the end
        exec_legs = [l for l in legs if l.get("action_status") == "SUCCEEDED"]
        total_m = sum(l.get("path_len_m", 0.0) for l in (legs if self.a.dry_run else exec_legs))
        total_s = sum(l["exec_time_s"] for l in exec_legs)
        summary = {
            "legs": len(legs), "pass": len(legs) - len(fails), "fail": len(fails), "fail_by_reason": by_reason,
            "total_path_m": round(total_m, 1), "total_exec_s": round(total_s, 1),
            "mean_speed_mps": round(total_m / total_s, 2) if total_s > 0 else None,
            "loc_err_max_m": max((l.get("loc_err_max_m", 0.0) for l in legs), default=0.0),
            "per_waypoint": per_wp, "aborted": aborted,
        }
        report = {
            "run": {"started_wall": self.run_start_wall, "ended_wall": dt.datetime.now().isoformat(timespec="seconds"),
                    "started_sim_t": self.run_start_sim, "ended_sim_t": self.n.now_sim(),
                    "mode": "dry-run" if self.a.dry_run else "live", "args": vars(self.a), "git_sha": git_sha(),
                    "planner_auto_follow_path_before": self.auto_follow_before,
                    "planner_auto_follow_path_restored": self.restored},
            "waypoints": self.wps, "legs": legs, "summary": summary,
        }
        if self.a.dry_run and self.a.dry_run_from == "matrix":
            matrix: dict[str, dict] = {}
            for l in legs:
                matrix.setdefault(l["from"], {})[l["to"]] = l.get("path_len_m")
            report["matrix"] = matrix
        with open(os.path.join(self.out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        cols = ["idx", "loop", "from", "to", "result", "reason", "path_len_m", "plan_time_s", "action_status",
                "exec_time_s", "dist_to_endpoint_m", "dist_to_waypoint_m", "loc_err_arrival_m", "loc_err_max_m",
                "max_speed_mps", "path_max_dz", "last_distance_to_goal", "t_start_sim", "t_end_sim"]
        with open(os.path.join(self.out_dir, "legs.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols + ["events", "warnings"])
            for l in legs:
                w.writerow([l.get(c, "") for c in cols] +
                           [";".join(f"{e['type']}:{e.get('reason')}" for e in l.get("events", [])),
                            ";".join(l.get("warnings", []))])
        self.log(f"SUMMARY legs {summary['legs']} pass {summary['pass']} fail {summary['fail']} "
                 f"{by_reason if by_reason else ''} path {summary['total_path_m']} m"
                 + (f" mean speed {summary['mean_speed_mps']} m/s" if summary["mean_speed_mps"] else "")
                 + (f"  [{aborted}]" if aborted else ""))
        self.log(f"report: {self.out_dir}/report.json, legs.csv, run.log")


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    except Exception:  # noqa: BLE001
        return None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waypoints", default="config/nav_waypoints_dp_1008.yaml")
    ap.add_argument("--select", help="comma-separated waypoint names to use (in file order)")
    ap.add_argument("--skip", help="comma-separated waypoint names to leave out")
    ap.add_argument("--order", choices=["sequential", "shuffle", "random"], default="sequential",
                    help="shuffle: new permutation per loop; random: independent draw per leg")
    ap.add_argument("--start-at", help="sequential: rotate the list to begin at this waypoint")
    ap.add_argument("--start-nearest", action="store_true", default=True,
                    help="sequential: begin at the waypoint nearest the robot (default)")
    ap.add_argument("--no-start-nearest", dest="start_nearest", action="store_false")
    ap.add_argument("--loops", type=int, default=1, help="number of passes over the list (0 = until --duration)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many sim seconds (after the leg)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="plan only; never send FollowPath")
    ap.add_argument("--dry-run-from", choices=["robot", "chain", "matrix"], default="chain",
                    help="robot: every goal from the robot pose; chain: from the previous waypoint; "
                         "matrix: every ordered pair (reachability table)")
    g = ap.add_argument_group("tolerances and timeouts (metres, m/s, sim seconds)")
    g.add_argument("--xy-tol", type=float, default=0.35, help="ground-truth distance to the path endpoint")
    g.add_argument("--snap-warn", type=float, default=1.0, help="warn when the planner moved the goal this far")
    g.add_argument("--loc-warn", type=float, default=0.3)
    g.add_argument("--loc-fail", type=float, default=1.0)
    g.add_argument("--loc-preflight", type=float, default=0.5)
    g.add_argument("--settle", type=float, default=1.0)
    g.add_argument("--speed-eps", type=float, default=0.05)
    g.add_argument("--plan-timeout", type=float, default=10.0)
    g.add_argument("--empty-grace", type=float, default=2.0, help="extra wait after an empty path before NO_PATH")
    g.add_argument("--endpoint-xy-tol", type=float, default=1.5,
                   help="accept a path ending this close to the goal (the planner snaps goals up to 2.4 m)")
    g.add_argument("--endpoint-z-tol", type=float, default=1.0)
    g.add_argument("--expected-speed", type=float, default=0.5)
    g.add_argument("--leg-timeout-min", type=float, default=30.0)
    g.add_argument("--leg-timeout-margin", type=float, default=15.0)
    g.add_argument("--action-server-timeout", type=float, default=5.0)
    g.add_argument("--start-goal-delay", type=float, default=0.15)
    g.add_argument("--pose-timeout", type=float, default=1.0, help="tracked pose must be younger than this (wall s)")
    g.add_argument("--preflight-timeout", type=float, default=60.0, help="wall seconds")
    g.add_argument("--dwell", type=float, default=1.0)
    g.add_argument("--dry-dwell", type=float, default=0.2)
    g.add_argument("--max-consecutive-fail", type=int, default=3)
    g.add_argument("--stop-on-fail", action="store_true")
    g.add_argument("--stuck-is-fail", type=lambda s: s.lower() not in ("0", "false", "no"), default=True)
    t = ap.add_argument_group("ROS names")
    t.add_argument("--tracked-pose-topic", default="/robot01/tracked_pose")
    t.add_argument("--gt-topic", default="/sim/ground_truth/odom")
    t.add_argument("--events-topic", default="/sim/events")
    t.add_argument("--initial-pose-topic", default="/initialpose")
    t.add_argument("--goal-pose-topic", default="/goal_pose")
    t.add_argument("--planned-path-topic", default="/robot01/planned_path")
    t.add_argument("--follow-path-action", default="/robot01/follow_path")
    t.add_argument("--planner-node", default="/octo_planner_rviz_node")
    t.add_argument("--controller-id", default="FollowPath")
    t.add_argument("--goal-checker-id", default="general_goal_checker")
    ap.add_argument("--results-dir", default="results/nav_goal_loop")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    frame, wps = load_waypoints(a.waypoints)
    if a.select:
        keep = [s.strip() for s in a.select.split(",") if s.strip()]
        missing = [k for k in keep if k not in {w["name"] for w in wps}]
        if missing:
            print(f"--select: unknown waypoints {missing}", file=sys.stderr)
            return 2
        wps = [w for w in wps if w["name"] in keep]
    if a.skip:
        drop = {s.strip() for s in a.skip.split(",")}
        wps = [w for w in wps if w["name"] not in drop]
    if not wps:
        print("no waypoints left after --select/--skip", file=sys.stderr)
        return 2
    out_dir = os.path.join(a.results_dir, dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ("-dry" if a.dry_run else ""))
    os.makedirs(out_dir, exist_ok=True)

    if SignalHandlerOptions is not None:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    else:
        rclpy.init()
    node = NavGoalLoop(a, frame)
    runner = Runner(node, a, wps, out_dir)

    def on_sigint(_sig, _frm):
        node.stop_requested = True
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    runner.log(f"nav_goal_loop {'DRY-RUN (' + a.dry_run_from + ')' if a.dry_run else 'LIVE'}: "
               f"{len(wps)} waypoints from {a.waypoints}, order {a.order}, loops {a.loops or 'until duration'}"
               + (f", duration {a.duration:g} s" if a.duration else ""))
    rc = 2
    try:
        if runner.preflight():
            rc = runner.run()
        else:
            runner.restore()
    except Interrupted:  # interrupted during preflight or cleanup
        node.stop_requested = False
        runner.restore()
        rc = 130
    finally:
        runner.log_f.close()
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
