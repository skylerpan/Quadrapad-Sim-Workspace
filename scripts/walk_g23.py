#!/usr/bin/env python3
"""Robot-layer evaluation: the G23 walks a trajectory on a scene with its real controller, no ROS, no nav stack.

Runs the Simulator in legged mode at real time (the controller is a wall-clock 1 kHz loop), starts the
controller as a child process (robot/controller settings of config/sim.yaml, docker/build_g23.sh), waits for
the stand -> gait sequence, then drives the torso along a polyline with pure pursuit on the sim's ground
truth: Maps/<map>/poses.txt (SLAM keyframes, every --sample metres), a waypoint yaml, or a straight line.
Logs falls, controller events, foot contact force peaks, torso height / attitude, distance covered, and
writes results/walk_g23/<timestamp>.json. This is the "robot layer" of the Phase 1 metrics: where the real
robot falls or stalls on the map, which the geometry numbers alone cannot say.

  docker/build_g23.sh                      # once: image + controller
  QUADSIM_G23=1 docker/run.sh "python3 scripts/walk_g23.py --poses Maps/dp_1008/poses.txt --seconds 120"
  QUADSIM_G23=1 docker/run.sh "python3 scripts/walk_g23.py --line 5 --vx 0.4"         # 5 m straight ahead
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pcd2sim.geometry import resample  # noqa: E402
from quadsim.config import load_config  # noqa: E402
from quadsim.controller_proc import ControllerProcess  # noqa: E402
from quadsim.sim import Simulator  # noqa: E402


def load_poses(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 4:
                try:
                    rows.append([float(t[1]), float(t[2]), float(t[3])])
                except ValueError:
                    pass
    return np.array(rows)


def foot_force_peak(sim: Simulator) -> float:
    """Largest normal force on a foot this step (N)."""
    m, d = sim.m, sim.d
    f = np.zeros(6)
    peak = 0.0
    for i in range(d.ncon):
        con = d.contact[i]
        if con.geom1 in sim.g23.feet or con.geom2 in sim.g23.feet:
            mujoco.mj_contactForce(m, d, i, f)
            peak = max(peak, abs(float(f[0])))
    return peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--poses", help="poses.txt to follow (x, y from the keyframes)")
    ap.add_argument("--waypoints", help="waypoint yaml to follow as straight legs")
    ap.add_argument("--line", type=float, default=None, help="walk this many metres straight ahead instead")
    ap.add_argument("--idle", action="store_true", help="send no velocity at all: measure the drift of the standing robot")
    ap.add_argument("--sample", type=float, default=0.3, help="polyline resampling (m)")
    ap.add_argument("--start-at", type=float, default=0.0, help="start this many metres along the trajectory")
    ap.add_argument("--vx", type=float, default=0.4, help="forward speed (m/s); the RL policy wants >= 0.3")
    ap.add_argument("--wz", type=float, default=0.6, help="turn rate when turning (rad/s); the policy wants 0.5-1.0")
    ap.add_argument("--lookahead", type=float, default=0.6)
    ap.add_argument("--switch", action="append", default=[], metavar="DIST:MODE",
                    help="switch the RL policy (0 PPO, 1 HIMLoco) once the progress passes DIST metres, e.g. 12:1 "
                         "(what a nav stack would do before stairs); repeatable")
    ap.add_argument("--seconds", type=float, default=120.0, help="stop after this much sim time (0 = to the end)")
    ap.add_argument("--rtf", type=float, default=1.0, help="real-time factor; the controller needs 1.0")
    ap.add_argument("--no-controller", action="store_true", help="do not spawn the controller (it runs elsewhere)")
    ap.add_argument("--out", default=None, help="result json (default results/walk_g23/<timestamp>.json)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, ["robot.type=g23"] + list(a.overrides))
    if a.scene:
        cfg["scene"] = a.scene
    sim = Simulator(cfg)
    log = lambda m: print(f"[walk t={sim.d.time:7.2f}] {m}", flush=True)
    ctl = None
    if not a.no_controller:
        ctl = ControllerProcess(cfg["robot"].get("controller", {}) or {}, print)
        if ctl.available:
            ctl.start()
        else:
            log(f"controller binary not found at {ctl.exe}; waiting for an external controller on udp:{sim.g23.cmd_port}")
            ctl = None

    # the path
    x0, y0 = sim.base.x, sim.base.y
    if a.poses:
        pts = load_poses(a.poses)[:, :2]
    elif a.waypoints:
        with open(a.waypoints) as f:
            pts = np.array([[w["x"], w["y"]] for w in yaml.safe_load(f)["waypoints"]], dtype=float)
    else:
        L = a.line if a.line is not None else 5.0
        yaw = sim.base.yaw
        pts = np.array([[x0, y0], [x0 + L * math.cos(yaw), y0 + L * math.sin(yaw)]])
    path = resample(pts, a.sample)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    i0 = int(np.searchsorted(seg, a.start_at))
    path = path[i0:]
    if a.poses or a.waypoints:
        # start where the path starts: teleport the robot there (the controller re-stands)
        d = np.diff(path[:2], axis=0)[0]
        sim.set_pose(float(path[0, 0]), float(path[0, 1]), float(math.atan2(d[1], d[0])), None)
        log(f"placed at the start of the path ({path[0, 0]:.2f}, {path[0, 1]:.2f}), floor z {sim.base.ground.z0:.2f}")
    total = float(seg[-1] - seg[i0])
    log(f"path: {len(path)} points, {total:.1f} m; scene {cfg['scene']}; controller {'child' if ctl else 'external'}")

    rec = {"scene": cfg["scene"], "path_m": round(total, 2), "gait": cfg["robot"].get("gait"), "events": [],
           "force_peaks": [], "height": [], "pitch": [], "t_gait": None, "t_fall": None, "t_safety": None,
           "progress_m": 0.0, "idle": bool(a.idle), "t_idle": None}
    yaw_unwrapped, yaw_prev = 0.0, None
    wall0, sim0 = time.perf_counter(), sim.d.time
    target = 0
    gait_on = False
    turning = False
    done = False
    switches = sorted((float(v.split(":")[0]), int(v.split(":")[1])) for v in a.switch)
    last_log = 0.0
    try:
        while not done and (a.seconds <= 0 or sim.d.time - sim0 < a.seconds):
            if ctl is not None and ctl.poll() is not None:
                log(f"controller exited with code {ctl.poll()}")
                rec["events"].append({"type": "controller", "state": "exited", "code": ctl.poll(), "t": sim.d.time})
                break
            locked = sim.g23.safety_locked or (ctl is not None and ctl.safety_locked)   # the link's status or g2's stdout
            if locked and rec.get("t_safety") is None:
                rec["t_safety"] = sim.d.time
                rec["events"].append({"type": "controller", "state": "safety", "t": sim.d.time,
                                      "pose": [sim.base.x, sim.base.y, sim.base.yaw]})
                log(f"controller safety lock at ({sim.base.x:.2f}, {sim.base.y:.2f}): stopping")
                done = True
            x, y, yaw = sim.base.x, sim.base.y, sim.base.yaw
            if yaw_prev is not None:
                yaw_unwrapped += (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_prev = yaw
            if gait_on and a.idle:
                if rec["t_idle"] is None:
                    rec["t_idle"], rec["idle_start"] = sim.d.time, [x, y, yaw_unwrapped]
            elif gait_on:
                # pure pursuit: advance the target to the first point at least `lookahead` ahead
                while target < len(path) - 1 and np.hypot(*(path[target] - [x, y])) < a.lookahead:
                    target += 1
                dx, dy = path[target] - [x, y]
                heading = math.atan2(dy, dx) - yaw
                heading = (heading + math.pi) % (2 * math.pi) - math.pi
                # the RL policy is unstable at very small non-zero commands (G2 API spec section 7): steer
                # bang-bang with hysteresis, walk at full --vx or not at all
                if abs(heading) > 0.35:
                    turning = True
                elif abs(heading) < 0.12:
                    turning = False
                wz = math.copysign(a.wz, heading) if turning else 0.0
                vx = 0.0 if abs(heading) > 1.0 else a.vx
                sim.set_cmd(vx, 0.0, wz)
                if target == len(path) - 1 and np.hypot(dx, dy) < 0.3:
                    log("reached the end of the path")
                    done = True
            out = sim.control_tick()
            for e in out.events:
                rec["events"].append(e)
                log(f"event {e}")
                if e["type"] == "controller" and e.get("state") in ("rl", "lma"):
                    rec["t_gait"] = sim.d.time
                if e["type"] == "fall":
                    rec["t_fall"] = sim.d.time
                    done = True
            if rec["t_gait"] is not None and not gait_on and sim.d.time - rec["t_gait"] > 1.5:
                gait_on = True                                   # the gait needs a moment to settle after MODE
                log("gait on: following the path")
            z_floor = sim.base.ground.z0
            R = sim.d.xmat[sim.base_body].reshape(3, 3)
            rec["force_peaks"].append(foot_force_peak(sim))
            if a.idle and rec.get("t_idle") is not None:
                rec.setdefault("idle_dq", []).append(float(np.abs(sim.d.qvel[sim.g23.jv]).mean()))
                rec.setdefault("idle_contacts", []).append(float(sim.g23.foot_contacts().sum()))
            rec["height"].append(float(sim.d.qpos[2] - z_floor))
            rec["pitch"].append(float(math.degrees(math.asin(-R[2, 0]))))
            rec["progress_m"] = round(float(seg[i0 + min(target, len(path) - 1)] - seg[i0]), 2)
            while switches and gait_on and rec["progress_m"] >= switches[0][0]:
                dist, mode = switches.pop(0)
                e = sim.set_policy_mode(mode)
                rec["events"].append(e)
                log(f"policy -> {mode} at {rec['progress_m']} m (switch {dist}:{mode})")
            if sim.d.time - last_log >= 5.0:
                last_log = sim.d.time
                log(f"pose ({x:.2f}, {y:.2f}) yaw {yaw:.2f}, target {target}/{len(path) - 1}, "
                    f"progress {rec['progress_m']} m, height {rec['height'][-1]:.2f}, "
                    f"force peak {max(rec['force_peaks'][-250:]):.0f} N, stage {sim.g23.stage}")
            if a.rtf > 0:
                ahead = (sim.d.time - sim0) / a.rtf - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        if ctl is not None:
            ctl.stop()
        sim.close()
    fp = np.array(rec["force_peaks"])
    h = np.array(rec["height"])
    if a.idle and rec.get("t_idle") is not None:
        x0_, y0_, yaw0_ = rec["idle_start"]
        dt_ = sim.d.time - rec["t_idle"]
        rec["idle_seconds"] = round(dt_, 1)
        rec["idle_yaw_deg"] = round(math.degrees(yaw_unwrapped - yaw0_), 2)
        rec["idle_yaw_rate_deg_s"] = round(math.degrees(yaw_unwrapped - yaw0_) / max(dt_, 1e-6), 3)
        rec["idle_drift_m"] = round(float(np.hypot(sim.base.x - x0_, sim.base.y - y0_)), 3)
        if rec.get("idle_dq"):
            rec["idle_joint_speed_rad_s"] = round(float(np.mean(rec["idle_dq"])), 3)
            rec["idle_feet_down_mean"] = round(float(np.mean(rec["idle_contacts"])), 2)
        rec.pop("idle_dq", None); rec.pop("idle_contacts", None)
    rec.update(sim_seconds=round(sim.d.time - sim0, 2), wall_seconds=round(time.perf_counter() - wall0, 2),
               fell=rec["t_fall"] is not None, safety=rec["t_safety"] is not None,
               reached_end=done and rec["t_fall"] is None and rec["t_safety"] is None,
               force_peak_max=round(float(fp.max()), 1) if len(fp) else None,
               force_peak_p99=round(float(np.percentile(fp, 99)), 1) if len(fp) else None,
               height_mean=round(float(h.mean()), 3) if len(h) else None,
               height_min=round(float(h.min()), 3) if len(h) else None,
               pitch_abs_p95=round(float(np.percentile(np.abs(rec["pitch"]), 95)), 1) if rec["pitch"] else None)
    for k in ("force_peaks", "height", "pitch"):
        del rec[k]
    out = a.out or os.path.join("results", "walk_g23", time.strftime("%Y%m%d-%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in rec.items() if k != "events"}, indent=1))
    print(f"result: {out}")
    return 0 if rec["reached_end"] else 1


if __name__ == "__main__":
    sys.exit(main())
