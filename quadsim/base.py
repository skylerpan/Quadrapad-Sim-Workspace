"""Ideal quadruped base: cmd_vel -> target pose that follows the terrain.

No legs are simulated. A gait-controller-like response (lag, velocity and
acceleration limits) turns cmd_vel into planar motion. Downward rays under the
four feet give the ground: per-foot medians drive the step checks, and a
least-squares plane through all foot rays gives height, roll and pitch (robust
to the few-cm noise of SLAM floors). Moves into obstacles, over too-high steps
or too-steep slopes are rejected and reported.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class Ground:
    """Terrain sample under the four feet at one candidate pose."""

    def __init__(self, feet_h: np.ndarray, z0: float, roll: float, pitch: float):
        self.feet_h, self.z0, self.roll, self.pitch = feet_h, z0, roll, pitch

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.feet_h).all())


class IdealBase:
    def __init__(self, cfg: dict, ray_scene, voxel_lookup=None,
                 blocked: Callable[[float, float, float, float], bool] | None = None):
        self.c = cfg
        self.scene = ray_scene
        self.lookup = voxel_lookup
        self.blocked = blocked
        self.feet = np.asarray(cfg["foot_offsets"], dtype=np.float64)           # [4,2]
        s = cfg["foot_patch"]
        g = np.array([-s, 0.0, s])
        self.patch = np.array([(a, b) for a in g for b in g])                    # [9,2]
        self.local = (self.feet[:, None, :] + self.patch[None, :, :]).reshape(-1, 2)  # [36,2]
        self.vel_lim = np.asarray(cfg["vel_limits"], dtype=np.float64)
        self.acc_lim = np.asarray(cfg["acc_limits"], dtype=np.float64)
        self.x = self.y = self.yaw = 0.0
        self.ground = Ground(np.zeros(4), 0.0, 0.0, 0.0)
        self.v = np.zeros(3)
        self.smooth = np.zeros(3)            # low-passed (z0, roll, pitch) for the target pose
        self.smooth_tau = float(cfg.get("attitude_tau", 0.15))

    # -- terrain ---------------------------------------------------------------
    def _sample(self, x, y, yaw, z_from: np.ndarray, length: float) -> Ground:
        c, s = np.cos(yaw), np.sin(yaw)
        lx, ly = self.local[:, 0], self.local[:, 1]
        oz = np.repeat(z_from, len(self.patch))
        origins = np.stack([x + c * lx - s * ly, y + s * lx + c * ly, oz], axis=1)
        dirs = np.tile(np.array([0.0, 0.0, -1.0]), (len(origins), 1))
        t, _ = self.scene.cast(origins, dirs)
        hit = t <= length
        z = np.where(hit, oz - t, np.nan)
        if self.lookup is not None and hit.any():
            # Box tops are voxel-quantised; use the mean z of the PCD points in the hit voxel.
            inside = origins[hit].copy()
            inside[:, 2] = z[hit] - 0.25 * self.lookup.voxel
            z[hit] = self.lookup.surface_z(inside, z[hit])
        per_foot = z.reshape(4, -1)
        n_ok = np.isfinite(per_foot).sum(axis=1)
        feet_h = np.where(n_ok >= 3, np.nanmedian(np.where(n_ok[:, None] >= 3, per_foot, 0.0), axis=1), np.nan)
        if not np.isfinite(feet_h).all():
            return Ground(feet_h, np.nan, 0.0, 0.0)
        # robust least-squares plane z = a*lx + b*ly + z0 in the body frame:
        # refit after dropping samples > 3 cm off the plane (clutter, map noise)
        ok = np.isfinite(z)
        A = np.stack([lx, ly, np.ones(len(lx))], axis=1)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[ok], z[ok], rcond=None)
            inl = ok & (np.abs(A @ coef - np.nan_to_num(z)) < 0.03)
            if inl.sum() < 6 or (inl == ok).all():
                break
            ok = inl
        a, b, z0 = coef
        return Ground(feet_h, float(z0), float(np.arctan(b)), float(-np.arctan(a)))

    # -- API -------------------------------------------------------------------
    def reset(self, x: float, y: float, yaw: float, z_floor_hint: float) -> bool:
        """Place the base; searches ground from 0.5 m above the hint down 1.5 m."""
        g = self._sample(x, y, yaw, np.full(4, z_floor_hint + 0.5), 1.5)
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.v[:] = 0.0
        ok = g.valid
        if not ok:
            g = Ground(np.where(np.isfinite(g.feet_h), g.feet_h, z_floor_hint), z_floor_hint, 0.0, 0.0)
        self.ground = g
        self.smooth = np.array([g.z0, g.roll, g.pitch])
        return ok

    def update(self, dt: float, cmd: np.ndarray, cmd_age: float) -> str | None:
        """Advance one control tick. Returns a rejection reason or None."""
        c = self.c
        cmd = np.zeros(3) if cmd_age > c["cmd_timeout"] else np.clip(cmd, -self.vel_lim, self.vel_lim)
        alpha = 1.0 - np.exp(-dt / max(c["cmd_tau"], 1e-6))
        dv = np.clip((cmd - self.v) * alpha, -self.acc_lim * dt, self.acc_lim * dt)
        v = self.v + dv
        reason = None
        if not np.allclose(v, 0.0, atol=1e-6):
            yaw = self.yaw + v[2] * dt
            cy, sy = np.cos(yaw), np.sin(yaw)
            x = self.x + (cy * v[0] - sy * v[1]) * dt
            y = self.y + (sy * v[0] + cy * v[1]) * dt
            reason = self._check(x, y, yaw)
            if reason is None:
                self.x, self.y, self.yaw = x, y, yaw
            else:
                v = np.zeros(3)
        self.v = v
        a = 1.0 - np.exp(-dt / max(self.smooth_tau, 1e-6))
        g = self.ground
        self.smooth += a * (np.array([g.z0, g.roll, g.pitch]) - self.smooth)
        return reason

    def _check(self, x, y, yaw) -> str | None:
        c = self.c
        g = self._sample(x, y, yaw, self.ground.feet_h + c["max_step_up"] + 0.05,
                         c["max_step_up"] + c["max_step_down"] + 0.05)
        if not g.valid:
            return "no_ground"
        dh = g.feet_h - self.ground.feet_h
        if (dh > c["max_step_up"]).any():
            return "step_up"
        if (dh < -c["max_step_down"]).any():
            return "step_down"
        if np.degrees(np.arccos(np.cos(g.roll) * np.cos(g.pitch))) > c["max_slope_deg"]:
            return "slope"
        if self.blocked is not None and self.blocked(x, y, yaw, g.z0 + c["stand_height"]):
            return "collision"
        self.ground = g
        return None

    def target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Body-centre position and wxyz quaternion for the mocap target."""
        z0, roll, pitch = self.smooth
        R = rot_zyx(self.yaw, pitch, roll)
        return np.array([self.x, self.y, z0 + self.c["stand_height"]]), quat_wxyz(R)
