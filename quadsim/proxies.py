"""Collision proxies: a small pool of mocap boxes moved onto the map boxes near the robot.

The MuJoCo model stays tiny (robot + pool) regardless of map size; the full
map lives in Embree for LiDAR/terrain rays. Proxy geoms collide only with the
robot (contype/conaffinity), never with each other.
"""
from __future__ import annotations

import numpy as np

PARK_Z = -1000.0


class BoxIndex:
    """2D uniform-grid index over box AABBs for fast window queries."""

    def __init__(self, boxes: np.ndarray, cell: float = 2.0):
        self.boxes = boxes.astype(np.float64)
        self.lo = self.boxes[:, :3] - self.boxes[:, 3:]
        self.hi = self.boxes[:, :3] + self.boxes[:, 3:]
        self.cell = cell
        self.origin = self.lo[:, :2].min(axis=0)
        c0 = np.floor((self.lo[:, :2] - self.origin) / cell).astype(np.int64)
        c1 = np.floor((self.hi[:, :2] - self.origin) / cell).astype(np.int64)
        self.ny = int(c1[:, 1].max()) + 1
        span = (c1 - c0 + 1)
        reps = span[:, 0] * span[:, 1]
        box_id = np.repeat(np.arange(len(boxes)), reps)
        # enumerate every covered cell of every box (boxes are small; a few span several cells)
        k = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
        sy = np.repeat(span[:, 1], reps)
        cx = np.repeat(c0[:, 0], reps) + k // sy
        cy = np.repeat(c0[:, 1], reps) + k % sy
        code = cx * self.ny + cy
        order = np.argsort(code, kind="stable")
        self.codes = code[order]
        self.ids = box_id[order]

    def query(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Indices of boxes whose AABB overlaps [lo, hi] (3D)."""
        c0 = np.floor((lo[:2] - self.origin) / self.cell).astype(np.int64)
        c1 = np.floor((hi[:2] - self.origin) / self.cell).astype(np.int64)
        cand = []
        for cx in range(c0[0], c1[0] + 1):
            a = cx * self.ny + max(c0[1], 0)
            b = cx * self.ny + min(c1[1], self.ny - 1)
            if cx < 0 or b < a:
                continue
            i0, i1 = np.searchsorted(self.codes, a, "left"), np.searchsorted(self.codes, b, "right")
            cand.append(self.ids[i0:i1])
        if not cand:
            return np.zeros(0, dtype=np.int64)
        ids = np.unique(np.concatenate(cand))
        ok = np.all((self.lo[ids] <= hi) & (self.hi[ids] >= lo), axis=1)
        return ids[ok]


class ProxyPool:
    def __init__(self, model, data, index: BoxIndex, n: int):
        import mujoco

        self.m, self.d, self.index, self.n = model, data, index, n
        self.geom = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"proxy_{i}") for i in range(n)])
        body = np.array([model.geom_bodyid[g] for g in self.geom])
        self.mocap = np.array([model.body_mocapid[b] for b in body])
        self.assigned = np.full(n, -1, dtype=np.int64)
        self.overflow = 0
        self.park_all()

    def park_all(self):
        for i in range(self.n):
            self.d.mocap_pos[self.mocap[i]] = (0.0, 0.0, PARK_Z - 10.0 * i)
        self.assigned[:] = -1

    def update(self, center: np.ndarray, half_xy: float, z_lo: float, z_hi: float, margin: float):
        """Assign proxies to the map boxes overlapping the window around the robot body."""
        lo = np.array([center[0] - half_xy - margin, center[1] - half_xy - margin, z_lo])
        hi = np.array([center[0] + half_xy + margin, center[1] + half_xy + margin, z_hi])
        ids = self.index.query(lo, hi)
        if len(ids) > self.n:
            b = self.index.boxes[ids]
            dist = np.linalg.norm(np.maximum(np.abs(b[:, :2] - center[:2]) - b[:, 3:5], 0.0), axis=1)
            ids = ids[np.argsort(dist)[: self.n]]
            self.overflow += 1
        # Slot-stable assignment: a box already held by a proxy keeps its slot, so a proxy that is
        # in contact with the robot is never moved/resized under it; only new boxes take free slots.
        m, d = self.m, self.d
        keep = (self.assigned >= 0) & np.isin(self.assigned, ids)
        for i in np.flatnonzero((self.assigned >= 0) & ~keep):          # box left the window
            d.mocap_pos[self.mocap[i]] = (0.0, 0.0, PARK_Z - 10.0 * i)
            self.assigned[i] = -1
        new = ids[~np.isin(ids, self.assigned[keep])]
        for i, bid in zip(np.flatnonzero(self.assigned < 0), new):
            g = self.geom[i]
            box = self.index.boxes[bid]
            m.geom_size[g] = box[3:]
            m.geom_rbound[g] = float(np.linalg.norm(box[3:]))
            m.geom_aabb[g] = (0.0, 0.0, 0.0, box[3], box[4], box[5])
            d.mocap_pos[self.mocap[i]] = box[:3]
            self.assigned[i] = bid
        return len(ids)
