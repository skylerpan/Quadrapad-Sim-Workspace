"""Ground truth the controller cannot see, once per physics step, for diagnosing its state estimator.

For every foot: the true world-frame velocity and angular velocity of the foot sphere, the normal force
the world puts on it and the direction of that force. For everything else on the robot that touches the
world: the normal force per shank, and the rest lumped together (thighs, hips, torso).

The time stamp is the one of the telemetry packet sent in the same step, so the log lines up sample for
sample with what the controller received (its G2_KF_INPUT_LOG recording).

Off unless robot.controller.truth_log names a file; written as one .npz when the simulator closes.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

LEGS = ["FL", "FR", "HL", "HR"]


class TruthLog:
    def __init__(self, m, d, foot_geoms: list[int], root: int, path: str):
        self.m, self.d, self.root, self.path = m, d, root, path
        self.foot_geoms = list(foot_geoms)
        self.foot_of = {g: i for i, g in enumerate(self.foot_geoms)}
        self.shank_of = {}
        for i, leg in enumerate(LEGS):
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_SHANK")
            if body >= 0:
                self.shank_of[body] = i
        self.rows: dict[str, list] = {k: [] for k in ("t", "base_p", "base_v", "foot_v", "foot_w", "foot_force",
                                                      "foot_normal", "shank_force", "other_force")}
        self._vel = np.zeros(6)
        self._force = np.zeros(6)

    def record(self, t: float):
        m, d = self.m, self.d
        foot_force, foot_normal = np.zeros(4), np.zeros((4, 3))
        shank_force, other = np.zeros(4), 0.0
        for k in range(d.ncon):
            con = d.contact[k]
            in1 = m.body_rootid[m.geom_bodyid[con.geom1]] == self.root
            in2 = m.body_rootid[m.geom_bodyid[con.geom2]] == self.root
            if in1 == in2:                                   # robot against itself, or world against world
                continue
            geom = con.geom1 if in1 else con.geom2
            mujoco.mj_contactForce(m, d, k, self._force)
            fn = float(self._force[0])
            foot = self.foot_of.get(int(geom))
            if foot is not None:
                foot_force[foot] += fn
                to_centre = d.geom_xpos[geom] - con.pos      # from the ground contact into the sphere
                foot_normal[foot] += fn * to_centre / (np.linalg.norm(to_centre) + 1e-12)
                continue
            shank = self.shank_of.get(int(m.geom_bodyid[geom]))
            if shank is not None:
                shank_force[shank] += fn
            else:
                other += fn
        norms = np.linalg.norm(foot_normal, axis=1, keepdims=True)
        foot_normal = np.where(norms > 0, foot_normal / np.where(norms > 0, norms, 1.0), 0.0)
        foot_v, foot_w = np.zeros((4, 3)), np.zeros((4, 3))
        for i, geom in enumerate(self.foot_geoms):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, geom, self._vel, 0)   # world axes: [angular, linear]
            foot_w[i], foot_v[i] = self._vel[:3], self._vel[3:]
        r = self.rows
        r["t"].append(t)
        r["base_p"].append(d.qpos[0:3].copy())
        r["base_v"].append(d.qvel[0:3].copy())
        r["foot_v"].append(foot_v)
        r["foot_w"].append(foot_w)
        r["foot_force"].append(foot_force)
        r["foot_normal"].append(foot_normal)
        r["shank_force"].append(shank_force)
        r["other_force"].append(other)

    def close(self):
        if not self.rows["t"]:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(self.path, feet=np.array(LEGS), **{k: np.asarray(v) for k, v in self.rows.items()})
        self.rows = {k: [] for k in self.rows}
