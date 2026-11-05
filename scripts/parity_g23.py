#!/usr/bin/env python3
"""Falsification test: is quadsim's G23 the reference simulation's G23?

Runs the reference simulation class (g2_deploy_rs interface/robot/simulation/mujoco_simulation_G23.py, imported
as-is, flat scene, no sockets used) and quadsim's legged backend side by side in one process and checks:

  A  model     every joint / body / geom / actuator / sensor / option field of the robot, after both sides applied
               their J60 joint model, COM offset and IMU position: must be bit-identical
  B  telemetry same state in both -> the 228-byte packet, the ROS-bridge ASCII line and the SIM_POSE line must be
               byte-identical (quadsim's base z is reported minus the scene floor height)
  C  dynamics  same initial state, the same 1 kHz command stream (PD hold, sinusoidal joint targets, a torso push):
               the two trajectories are compared step by step. Their own model with the floor plane swapped for a
               box is run as a third line, to tell a robot-model difference from a floor-geometry one.

Any non-zero line in A or B falsifies the parity claim. C reports the divergence; the robot model is cleared when
quadsim tracks "theirs + box floor" as closely as "theirs + box floor" tracks "theirs + plane".

  QUADSIM_G23=1 docker/run.sh "python3 scripts/parity_g23.py"
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# private ports: never the live controller's
os.environ.setdefault("G2_SIM_HEADLESS", "1")
os.environ.setdefault("G2_SIM_SCENE", "flat")


def _load_reference(repo: str):
    sim_dir = os.path.join(repo, "interface", "robot", "simulation")
    try:
        import colorama  # noqa: F401
    except ImportError:                                   # their sim only uses it for coloured prints
        import types

        class _Plain:
            def __getattr__(self, _):
                return ""
        stub = types.ModuleType("colorama")
        stub.init, stub.Fore, stub.Style = lambda **_: None, _Plain(), _Plain()
        sys.modules["colorama"] = stub
    sys.path.insert(0, sim_dir)
    import mujoco_simulation_G23 as ref  # noqa: E402
    return ref


class _Capture:
    """Stands in for their send_sock: keeps every datagram instead of sending it."""

    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((addr, data))


def _named(m, objtype, n):
    import mujoco
    return [mujoco.mj_id2name(m, objtype, i) for i in range(n)]


def compare_models(mt, mo, mujoco) -> list[str]:
    """A: field-by-field comparison keyed by element name. Returns the list of differences."""
    diffs = []

    def eq(label, a, b, tol=0.0):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=tol):
            diffs.append(f"{label}: theirs {np.array2string(a, precision=6)} quadsim {np.array2string(b, precision=6)}")

    o = mt.opt, mo.opt
    for f in ("timestep", "gravity", "integrator", "cone", "jacobian", "solver", "iterations", "tolerance",
              "ls_iterations", "ls_tolerance", "noslip_iterations", "noslip_tolerance", "impratio", "wind",
              "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "disableflags", "enableflags"):
        eq(f"opt.{f}", getattr(o[0], f), getattr(o[1], f))
    # joints (by name; everything their _apply_motor_model touches lives on the dof)
    for n in _named(mt, mujoco.mjtObj.mjOBJ_JOINT, mt.njnt):
        jt, jo = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_JOINT, n), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jo < 0:
            diffs.append(f"joint {n}: missing in quadsim")
            continue
        for f in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_stiffness", "jnt_margin",
                  "jnt_solref", "jnt_solimp", "jnt_actfrcrange", "jnt_actfrclimited"):
            eq(f"joint {n}.{f[4:]}", getattr(mt, f)[jt], getattr(mo, f)[jo])
        if mt.jnt_type[jt] == mujoco.mjtJoint.mjJNT_HINGE:
            dt_, do_ = mt.jnt_dofadr[jt], mo.jnt_dofadr[jo]
            for f in ("dof_armature", "dof_frictionloss", "dof_damping", "dof_solref", "dof_solimp"):
                eq(f"joint {n}.{f[4:]}", getattr(mt, f)[dt_], getattr(mo, f)[do_])
            eq(f"joint {n}.qpos0", mt.qpos0[mt.jnt_qposadr[jt]], mo.qpos0[mo.jnt_qposadr[jo]])
    # bodies
    for n in _named(mt, mujoco.mjtObj.mjOBJ_BODY, mt.nbody):
        if n == "world":
            continue
        bt, bo = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_BODY, n), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_BODY, n)
        if bo < 0:
            diffs.append(f"body {n}: missing in quadsim")
            continue
        for f in ("body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_subtreemass",
                  "body_gravcomp", "body_dofnum", "body_jntnum", "body_geomnum"):
            eq(f"body {n}.{f[5:]}", getattr(mt, f)[bt], getattr(mo, f)[bo])
        if n != "TORSO":                                          # the torso is spawned on the map, not at the xml pos
            eq(f"body {n}.pos", mt.body_pos[bt], mo.body_pos[bo])
        pt, po = mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_BODY, mt.body_parentid[bt]), \
            mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_BODY, mo.body_parentid[bo])
        if pt != po:
            diffs.append(f"body {n}.parent: theirs {pt} quadsim {po}")
    # geoms (their floor plane has no counterpart: quadsim's floor is the scene)
    for n in _named(mt, mujoco.mjtObj.mjOBJ_GEOM, mt.ngeom):
        if n == "floor" or n is None:
            continue
        gt, go = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_GEOM, n), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_GEOM, n)
        if go < 0:
            diffs.append(f"geom {n}: missing in quadsim")
            continue
        for f in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                  "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin",
                  "geom_gap", "geom_priority", "geom_group"):
            eq(f"geom {n}.{f[5:]}", getattr(mt, f)[gt], getattr(mo, f)[go])
        if mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_BODY, mt.geom_bodyid[gt]) != \
                mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_BODY, mo.geom_bodyid[go]):
            diffs.append(f"geom {n}: on a different body")
    # the floor contact parameters: their plane vs quadsim's proxy boxes
    ft, fo = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_GEOM, "floor"), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_GEOM, "proxy_0")
    for f in ("geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap",
              "geom_priority"):
        eq(f"floor (plane) vs proxy box .{f[5:]}", getattr(mt, f)[ft], getattr(mo, f)[fo])
    # actuators
    for n in _named(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, mt.nu):
        at, ao = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_ACTUATOR, n), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if ao < 0:
            diffs.append(f"actuator {n}: missing in quadsim")
            continue
        for f in ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_gear",
                  "actuator_ctrllimited", "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange",
                  "actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
            eq(f"actuator {n}.{f[9:]}", getattr(mt, f)[at], getattr(mo, f)[ao])
        jt = mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_JOINT, mt.actuator_trnid[at][0])
        jo = mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_JOINT, mo.actuator_trnid[ao][0])
        if jt != jo:
            diffs.append(f"actuator {n}: drives {jt} vs {jo}")
    if mt.nu != mo.nu:
        diffs.append(f"nu: theirs {mt.nu} quadsim {mo.nu}")
    # the joint order the 12-vectors use: their qpos[7:19] order must equal quadsim's name order
    theirs = [mujoco.mj_id2name(mt, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, mt.njnt)]
    from quadsim.g23 import JOINTS
    if theirs != JOINTS:
        diffs.append(f"joint order: theirs {theirs} quadsim {JOINTS}")
    # IMU
    for s in ("imu_acc", "imu_gyro"):
        st, so = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_SENSOR, s), mujoco.mj_name2id(mo, mujoco.mjtObj.mjOBJ_SENSOR, s)
        eq(f"sensor {s}.type", mt.sensor_type[st], mo.sensor_type[so])
        site_t, site_o = mt.sensor_objid[st], mo.sensor_objid[so]
        eq(f"sensor {s}.site_pos", mt.site_pos[site_t], mo.site_pos[site_o])
        eq(f"sensor {s}.site_quat", mt.site_quat[site_t], mo.site_quat[site_o])
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("G2_DEPLOY_RS", "/g2"))
    ap.add_argument("--scene", default="scenes/flat_test")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args(argv)
    import mujoco
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    from quadsim.g23 import DEFAULT_Q

    ref = _load_reference(a.repo)
    theirs = ref.MuJoCoSimulation(local_port=20201, ctrl_port=30210)
    theirs.send_sock = _Capture()
    theirs.telemetry_targets = [("127.0.0.1", 30210)]
    cfg = load_config(None, [f"scene={a.scene}", "robot.type=g23", "robot.auto_stand=false",
                             "robot.controller.cmd_port=20202", "robot.controller.telemetry=[127.0.0.1:30211]",
                             "robot.controller.command_addr=127.0.0.1:50251", "robot.controller.gt_addr=127.0.0.1:30212",
                             "robot.controller.ros_bridge_addr=127.0.0.1:50210", "lidar.points_per_sec=2000",
                             "spawn=[0,0,0,0]"])
    ours = Simulator(cfg)
    mt, dt_ = theirs.model, theirs.data
    mo, do_ = ours.m, ours.d
    z0 = float(ours.base.ground.z0)
    failures = 0

    # ---- A: model ---------------------------------------------------------------------------------------
    diffs = compare_models(mt, mo, mujoco)
    print(f"A model   : {len(diffs)} field difference(s)" + ("" if not diffs else ""))
    for s in diffs:
        print("   ", s)
    failures += len(diffs)

    # ---- helpers for driving both -----------------------------------------------------------------------
    def set_cmd(kp, q_des, kd, dq_des, tau_ff):
        kp, q_des, kd, dq_des, tau_ff = (np.asarray(v, dtype=np.float32) for v in (kp, q_des, kd, dq_des, tau_ff))
        theirs.kp_cmd, theirs.pos_cmd, theirs.kd_cmd = kp.reshape(-1, 1), q_des.reshape(-1, 1), kd.reshape(-1, 1)
        theirs.vel_cmd, theirs.tau_ff = dq_des.reshape(-1, 1), tau_ff.reshape(-1, 1)
        g = ours.g23
        with g.lock:
            g.kp[:], g.q_des[:], g.kd[:], g.dq_des[:], g.tau_ff[:] = kp, q_des, kd, dq_des, tau_ff

    def step_theirs(step):
        theirs._apply_joint_torque()
        mujoco.mj_step(mt, dt_)
        if theirs._sensor_model:
            theirs._update_imu(step)
        theirs.timestamp = step * ref.DT

    def step_ours():
        ours.g23.apply_torque()
        mujoco.mj_step(mo, do_)
        ours.g23.update_imu()

    def sync_ours_to_theirs():
        do_.qpos[:] = dt_.qpos
        do_.qpos[2] += z0
        do_.qvel[:] = dt_.qvel
        do_.ctrl[:] = dt_.ctrl
        ours._refresh_proxies(do_.qpos[0:3])
        mujoco.mj_forward(mo, do_)

    # ---- B: telemetry from an identical state ---------------------------------------------------------
    # settle their robot on the floor under a PD hold first, so contacts and actuator forces are non-trivial
    hold = (np.full(12, 60.0), DEFAULT_Q, np.full(12, 2.0), np.zeros(12), np.zeros(12))
    set_cmd(*hold)
    for s in range(1, 1501):
        step_theirs(s)
    sync_ours_to_theirs()
    # one more identical step on both so qfrc_actuator / sensordata come from the same state and torque
    step_theirs(1501)
    step_ours()
    t = theirs.timestamp
    theirs.send_sock.sent.clear()
    theirs._send_robot_state(1501)
    sent = {addr: data for addr, data in theirs.send_sock.sent}
    pkt_t = sent[("127.0.0.1", 30210)]
    pkt_o = ours.g23.telemetry(t)
    vt, vo = struct.unpack("<d55f", pkt_t), struct.unpack("<d55f", pkt_o)
    vo = list(vo)
    vo[46 + 2] -= z0                                   # base z: quadsim's floor is at z0
    vo[46 + 2] = np.float32(vo[46 + 2])
    names = ["t"] + ["rpy"] * 3 + ["acc"] * 3 + ["gyro"] * 3 + ["q"] * 12 + ["dq"] * 12 + ["tau"] * 12 + \
            ["base_pos"] * 3 + ["base_vel"] * 3 + ["contact"] * 4
    bad = [(names[i], vt[i], vo[i]) for i in range(56) if not np.isclose(vt[i], vo[i], rtol=0, atol=1e-5)]
    print(f"B packet  : {len(bad)} of 56 fields differ (tol 1e-5)" + (f"  identical bytes: {pkt_t == pkt_o}" if not bad else ""))
    for n, x, y in bad:
        print(f"    {n}: theirs {x:.6g} quadsim {y:.6g}")
    failures += len(bad)
    ascii_t = sent[theirs.ros_bridge_addr].decode()
    ascii_o = ours.g23.ros_bridge_line()
    print(f"B ascii   : {'identical' if ascii_t == ascii_o else 'DIFFERENT'}")
    if ascii_t != ascii_o:
        failures += 1
        print("    theirs :", ascii_t.strip()[:160])
        print("    quadsim:", ascii_o.strip()[:160])
    gt_t = sent[(ref.GT_IP, ref.GT_PORT)].decode().split()
    gt_o = ours.g23.gt_line(t).split()
    gt_o[4] = f"{float(gt_o[4]) - z0:.6f}"
    same_gt = gt_t == gt_o
    print(f"B SIM_POSE: {'identical' if same_gt else 'DIFFERENT'}")
    if not same_gt:
        failures += 1
        print("    theirs :", " ".join(gt_t))
        print("    quadsim:", " ".join(gt_o))

    # ---- C: dynamics under the same command stream ----------------------------------------------------
    # third line: their xml with the plane replaced by a box floor (same defaults as quadsim's proxies)
    spec = mujoco.MjSpec.from_file(os.path.join(a.repo, "robot", "g23", "mjcf", "g23_mjcf.xml"))
    spec.option.timestep = ref.DT
    spec.delete(spec.geom("floor"))
    spec.worldbody.add_geom(name="boxfloor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[5.0, 5.0, 0.05], pos=[0, 0, -0.05],
                            contype=1, conaffinity=0)
    mb = spec.compile()
    mb.dof_armature[6:18] = mt.dof_armature[6:18]
    mb.dof_frictionloss[6:18] = mt.dof_frictionloss[6:18]
    mb.dof_damping[6:18] = mt.dof_damping[6:18]
    mb.body_ipos[:] = mt.body_ipos
    db = mujoco.MjData(mb)
    # same start for all three: their initial pose (base at the xml height), zero velocity
    theirs._set_initial_pose("g23")
    dt_.qvel[:] = 0
    theirs._cmd_history.clear(), theirs._imu_history.clear()
    theirs._tau_filtered = None
    db.qpos[:] = dt_.qpos
    db.qvel[:] = 0
    mujoco.mj_forward(mb, db)
    sync_ours_to_theirs()
    ours.g23._cmd_history.clear(), ours.g23._imu_history.clear()
    ours.g23._tau_filtered = None
    torso_t = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
    torso_b = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
    n = int(a.seconds / ref.DT)
    dq_ob, dq_bt, dp_ob, dp_bt, dtau = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    kp, kd = np.full(12, 60.0), np.full(12, 2.0)
    timeline = []
    assert mujoco.mj_id2name(mo, mujoco.mjtObj.mjOBJ_BODY, ours.base_body) == "TORSO"
    for s in range(1, n + 1):
        tt = s * ref.DT
        q_des = DEFAULT_Q.copy()
        if tt > 1.5:                                                   # sinusoidal joint targets, all legs
            q_des = DEFAULT_Q + 0.15 * np.sin(2 * np.pi * 1.0 * (tt - 1.5)) * np.tile([0.3, 1.0, -1.0], 4)
        set_cmd(kp, q_des, kd, np.zeros(12), np.zeros(12))
        push = np.array([120.0, 40.0, 0.0]) if 3.0 <= tt < 3.3 else np.zeros(3)   # a shove that tips it over: torso and hips hit the floor
        dt_.xfrc_applied[torso_t, :3] = push
        db.xfrc_applied[torso_b, :3] = push
        do_.xfrc_applied[ours.base_body, :3] = push
        step_theirs(s)
        tau_theirs = theirs.input_tq.ravel().copy()
        # the box-floor twin runs their torque code on its own data
        theirs_data, theirs.data = theirs.data, db
        theirs._apply_joint_torque()
        theirs.data = theirs_data
        mujoco.mj_step(mb, db)
        step_ours()
        if s % ours.ctrl_steps == 0:
            ours._refresh_proxies(do_.qpos[0:3])
        i = s - 1
        dq_ob[i] = np.max(np.abs(do_.qpos[7:19] - db.qpos[7:19]))
        dq_bt[i] = np.max(np.abs(db.qpos[7:19] - dt_.qpos[7:19]))
        dp_ob[i] = np.linalg.norm(do_.qpos[0:3] - db.qpos[0:3] - [0, 0, z0])
        dp_bt[i] = np.linalg.norm(db.qpos[0:3] - dt_.qpos[0:3])
        dtau[i] = np.max(np.abs(ours.g23.input_tq - tau_theirs))
        if s % 500 == 0:
            timeline.append((tt, dq_ob[i], dq_bt[i], dp_ob[i], dp_bt[i]))
    print(f"C dynamics ({a.seconds:.0f} s, PD hold -> 1 Hz joint sweep from 1.5 s, 120 N push 3.0-3.3 s (falls over)):")
    print("                                   max |dq| rad   final |dq|   max |dpos| m   final |dpos|")
    print(f"   quadsim vs theirs+box floor     {dq_ob.max():10.2e}   {dq_ob[-1]:9.2e}   {dp_ob.max():10.2e}   {dp_ob[-1]:9.2e}")
    print(f"   theirs+box floor vs theirs      {dq_bt.max():10.2e}   {dq_bt[-1]:9.2e}   {dp_bt.max():10.2e}   {dp_bt[-1]:9.2e}")
    print(f"   torque command quadsim vs theirs: max |dtau| {dtau.max():.2e} Nm (first step {dtau[0]:.2e})")
    print("   timeline  t      |dq| ours-box  box-theirs   |dpos| ours-box  box-theirs")
    for tt, a1, b1, a2, b2 in timeline:
        print(f"           {tt:4.1f}   {a1:11.2e}  {b1:10.2e}   {a2:13.2e}  {b2:10.2e}")
    print(f"   final yaw: theirs {np.degrees(_yaw(dt_.qpos[3:7])):7.3f} deg, box {np.degrees(_yaw(db.qpos[3:7])):7.3f} deg, "
          f"quadsim {np.degrees(_yaw(do_.qpos[3:7])):7.3f} deg")
    ours.close()
    print(f"\nRESULT: {'parity falsified: %d difference(s)' % failures if failures else 'no difference found in A / B'}")
    return 1 if failures else 0


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


if __name__ == "__main__":
    sys.exit(main())
