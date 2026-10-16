#!/usr/bin/env python3
"""Evaluate a built scene: floor statistics, staircase recovery, walkability of every reachable
floor cell through the ideal base, and the spawn. Runs inside the sim container (needs the
Embree ray scene and MuJoCo):

  docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --auto \\
                 --stairs-box=-46,-30,-3,11 --stairs-box=-5,-2,-18,-14"
  docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1012 --pcd Maps/dp_1012_fixed.pcd --find-stairs"
  docker/run.sh "python3 scripts/eval_scene.py --scene scenes/dp_1008 --poses Maps/dp_1008/poses.txt"

Sections of the report (also written to <scene>/eval.json, ASCII stair profiles to <scene>/eval_profile.txt):
  ground   meta.json ground stats (planar_fraction, plane_rms_m, cells_filled, ...)
  stairs   per --stairs-box (xmin,xmax,ymin,ymax[,zmin,zmax]): distinct slab levels, rises, coverage of
           the raw standable columns by a slab, |slab - raw top| residual, cells per level
  walk     --auto: BFS over the flattened floor cells from the spawn (8-connected, |dz| <= 0.2 so stairs
           are included); on a 0.3 m lattice every reachable cell is entered from its BFS parent through
           IdealBase.reset + _check. Samples where the body does not fit even at rest (walls, furniture) are
           dropped; samples with a foot over a column without flattened floor are counted as edge_fraction
           (floor edges and holes); the rest are "valid" and their no_ground / step_up / step_down / slope /
           collision counts are the floor defects, plus the share with a foot patch over nothing (rays).
           Stair boxes are replayed again with --climb overrides (default robot.max_step_up=0.22
           robot.max_slope_deg=45) and no collision test, to separate build geometry from robot limits.
           --poses poses.txt / --waypoints yaml replay a polyline instead (poses: SLAM keyframes).
  spawn    config spawn: ground found, moved to a free pose, floor z
key=value arguments override config/sim.yaml (e.g. robot.max_step_up=0.2).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pcd2sim.pcd_io import read_pcd  # noqa: E402
from pcd2sim.voxel import crop_percentile, remove_isolated, standable_mask, voxelize  # noqa: E402
from quadsim.base import IdealBase  # noqa: E402
from quadsim.config import load_config  # noqa: E402
from quadsim.sim import Simulator  # noqa: E402

REASONS = ("no_ground", "step_up", "step_down", "slope", "collision")
_XY8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


# -- scene geometry ----------------------------------------------------------------------------------
class SlabCells:
    """Flattened floor slabs rasterised to voxel-sized cells: (ix, iy, z) sorted by column, several
    entries per column on multi-storey maps."""

    def __init__(self, slabs: np.ndarray, origin: np.ndarray, voxel: float):
        self.origin, self.v = origin[:2].astype(np.float64), float(voxel)
        if slabs is None or len(slabs) == 0:
            self.ix = self.iy = np.zeros(0, np.int64)
            self.z = np.zeros(0)
            self.col = np.zeros(0, np.int64)
            self.tilt = np.zeros(0)
            self.ny = 1
            self.iy_min = 0
            return
        x0, y0 = slabs[:, 0, 0].astype(np.float64), slabs[:, 0, 1].astype(np.float64)
        x1, y1 = slabs[:, 7, 0].astype(np.float64), slabs[:, 7, 1].astype(np.float64)
        z1, z3, z5 = slabs[:, 1, 2].astype(np.float64), slabs[:, 3, 2].astype(np.float64), slabs[:, 5, 2].astype(np.float64)
        i0 = np.round((x0 - self.origin[0]) / self.v).astype(np.int64)
        i1 = np.round((x1 - self.origin[0]) / self.v).astype(np.int64)
        j0 = np.round((y0 - self.origin[1]) / self.v).astype(np.int64)
        j1 = np.round((y1 - self.origin[1]) / self.v).astype(np.int64)
        n = (i1 - i0) * (j1 - j0)
        s = np.repeat(np.arange(len(slabs)), n)
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        w = np.repeat(j1 - j0, n)
        ix = np.repeat(i0, n) + off // w
        iy = np.repeat(j0, n) + off % w
        cx = self.origin[0] + (ix + 0.5) * self.v
        cy = self.origin[1] + (iy + 0.5) * self.v
        ax = np.where(x1 > x0, (z5 - z1) / np.maximum(x1 - x0, 1e-9), 0.0)
        ay = np.where(y1 > y0, (z3 - z1) / np.maximum(y1 - y0, 1e-9), 0.0)
        z = z1[s] + ax[s] * (cx - x0[s]) + ay[s] * (cy - y0[s])
        tilt = np.degrees(np.arctan(np.hypot(ax, ay)))[s]
        self.ny = int(iy.max() - iy.min() + 2)
        self.iy_min = int(iy.min())
        col = ix * self.ny + (iy - self.iy_min)
        order = np.lexsort((z, col))
        self.ix, self.iy, self.z, self.col = ix[order], iy[order], z[order], col[order]
        self.tilt = tilt[order]

    def __len__(self):
        return len(self.z)

    def xy(self, idx=slice(None)):
        return np.column_stack([self.origin[0] + (self.ix[idx] + 0.5) * self.v,
                                self.origin[1] + (self.iy[idx] + 0.5) * self.v])

    def column_range(self, col: np.ndarray):
        lo = np.searchsorted(self.col, col, side="left")
        hi = np.searchsorted(self.col, col, side="right")
        return lo, hi

    def edges(self, dz_max: float):
        """(rows, cols) of 8-neighbour cell pairs whose heights differ by <= dz_max."""
        rows, cols = [], []
        depth = int(np.max(np.bincount(self.col - self.col.min()))) if len(self) else 0
        for dx, dy in _XY8:
            nc = (self.ix + dx) * self.ny + (self.iy + dy - self.iy_min)
            lo, hi = self.column_range(nc)
            for k in range(depth):
                j = lo + k
                f = j < hi
                i = np.flatnonzero(f)
                j = j[f]
                close = np.abs(self.z[i] - self.z[j]) <= dz_max
                rows.append(i[close])
                cols.append(j[close])
        return np.concatenate(rows), np.concatenate(cols)

    def nearest(self, x: float, y: float, z: float, r: float = 0.5, dz: float = 0.6) -> int | None:
        d = np.hypot(self.ix * self.v + self.origin[0] + 0.5 * self.v - x,
                     self.iy * self.v + self.origin[1] + 0.5 * self.v - y)
        ok = (d <= r) & (np.abs(self.z - z) <= dz)
        if not ok.any():
            return None
        cand = np.flatnonzero(ok)
        return int(cand[np.argmin(d[cand] + np.abs(self.z[cand] - z))])


def parse_box(s: str) -> dict:
    v = [float(t) for t in s.split(",")]
    if len(v) not in (4, 6):
        raise argparse.ArgumentTypeError("--stairs-box xmin,xmax,ymin,ymax[,zmin,zmax]")
    return {"xmin": v[0], "xmax": v[1], "ymin": v[2], "ymax": v[3],
            "zmin": v[4] if len(v) == 6 else -1e9, "zmax": v[5] if len(v) == 6 else 1e9}


def in_box(box: dict, xy: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    m = (xy[:, 0] >= box["xmin"]) & (xy[:, 0] <= box["xmax"]) & (xy[:, 1] >= box["ymin"]) & (xy[:, 1] <= box["ymax"])
    if z is not None:
        m &= (z >= box["zmin"]) & (z <= box["zmax"])
    return m


# -- raw PCD side (stair detection, coverage) -------------------------------------------------------
def raw_surface(pcd_path: str, voxel: float, boxes: list[dict] | None = None):
    """Standable raw columns (x, y, z of the top surface voxel) from the PCD, optionally only inside boxes
    (grown by 1 m; the whole column height is kept so the roof test works)."""
    xyz, _ = read_pcd(pcd_path)
    xyz = xyz[crop_percentile(xyz, 0.05)]
    if boxes:
        keep = np.zeros(len(xyz), dtype=bool)
        for b in boxes:
            keep |= (xyz[:, 0] >= b["xmin"] - 1) & (xyz[:, 0] <= b["xmax"] + 1) & \
                    (xyz[:, 1] >= b["ymin"] - 1) & (xyz[:, 1] <= b["ymax"] + 1)
        xyz = xyz[keep]
    grid = voxelize(xyz, None, voxel)
    grid = remove_isolated(grid, 2)
    stand = standable_mask(grid)
    c = grid.centers()[stand]
    z = grid.zmean[stand].astype(np.float64)
    return c[:, 0], c[:, 1], z, grid.counts[stand]


def find_stairs(x, y, z, cell: float = 1.0, min_levels: int = 4, rise=(0.10, 0.26), min_vox: int = 3):
    """Candidate staircases: 1 m cells whose standable surface forms >= min_levels distinct heights with
    consecutive gaps inside `rise`, merged over 8-neighbour cells. Returns a list of candidate dicts."""
    ij = np.floor(np.column_stack([x, y]) / cell).astype(np.int64)
    ij0 = ij.min(axis=0)
    ij -= ij0                                            # non-negative, so key // and % recover (i, j)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    stair_cells = {}
    for a, b in zip(starts, ends):
        zz = np.sort(zs[a:b])
        if len(zz) < min_levels * min_vox:
            continue
        cut = np.flatnonzero(np.diff(zz) > 0.06)
        groups = np.split(zz, cut + 1)
        levels = np.array([g.mean() for g in groups if len(g) >= min_vox])
        if len(levels) < min_levels:
            continue
        gaps = np.diff(levels)
        ok = (gaps >= rise[0]) & (gaps <= rise[1])
        best = 0
        run = 0
        for o in ok:                                  # longest run of stair-like gaps
            run = run + 1 if o else 0
            best = max(best, run)
        if best + 1 >= min_levels:
            stair_cells[int(ks[a])] = (int(ks[a]) // 1_000_003, int(ks[a]) % 1_000_003, levels)
    if not stair_cells:
        return []
    keys = np.array(list(stair_cells.keys()))
    ij = np.array([(v[0], v[1]) for v in stair_cells.values()])
    idx = {k: i for i, k in enumerate(keys.tolist())}
    rows, cols = [], []
    for i, (a, b) in enumerate(ij):
        for dx, dy in _XY8:
            k = (a + dx) * 1_000_003 + (b + dy)
            if k in idx:
                rows.append(i)
                cols.append(idx[k])
    n, lab = connected_components(coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(keys),) * 2),
                                  directed=False)
    out = []
    for c in range(n):
        m = lab == c
        cells = ij[m]
        levels = np.concatenate([stair_cells[int(k)][2] for k in keys[m]])
        cells = cells + ij0
        xmin, xmax = cells[:, 0].min() * cell, (cells[:, 0].max() + 1) * cell
        ymin, ymax = cells[:, 1].min() * cell, (cells[:, 1].max() + 1) * cell
        sel = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
        A = np.column_stack([x[sel], y[sel], np.ones(sel.sum())])
        coef, *_ = np.linalg.lstsq(A, z[sel], rcond=None)
        ang = math.degrees(math.atan2(coef[1], coef[0]))
        out.append({"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax),
                    "zmin": float(levels.min()), "zmax": float(levels.max()), "cells_1m": int(m.sum()),
                    "n_levels": int(len(np.unique(np.round(levels, 2)))), "axis_deg": round(ang, 1)})
    out.sort(key=lambda d: -d["cells_1m"])
    return out


# -- stair metrics -----------------------------------------------------------------------------------
def cluster_levels(z: np.ndarray, gap: float = 0.06, min_n: int = 4):
    zz = np.sort(z)
    cut = np.flatnonzero(np.diff(zz) > gap)
    groups = [g for g in np.split(zz, cut + 1) if len(g) >= min_n]
    return [(float(np.median(g)), int(len(g)), float(g.max() - g.min())) for g in groups]


def stair_metrics(cells: SlabCells, box: dict, raw=None) -> dict:
    xy = cells.xy()
    m = in_box(box, xy, cells.z)
    out = {"box": box, "slab_cells": int(m.sum())}
    if m.sum() == 0:
        out["n_levels"] = 0
        return out
    z = cells.z[m]
    levels = cluster_levels(z)
    out["levels"] = [{"z": round(l[0], 3), "cells": l[1], "spread": round(l[2], 3)} for l in levels]
    out["n_levels"] = len(levels)
    out["ramp_fraction"] = round(float((cells.tilt[m] > 12.0).mean()), 3)   # treads merged into a slope
    if len(levels) >= 2:
        gaps = np.diff([l[0] for l in levels])
        rises = gaps[gaps <= 0.30]                       # tread-to-tread; larger gaps are landings/storeys
        out["n_rises"] = int(len(rises))
        if len(rises):
            out["rise_mean"] = round(float(rises.mean()), 3)
            out["rise_std"] = round(float(rises.std()), 3)
            out["rise_min"], out["rise_max"] = round(float(rises.min()), 3), round(float(rises.max()), 3)
    # cells not inside any level (a ramp-like continuum or seams)
    lv = np.array([l[0] for l in levels]) if levels else np.zeros(0)
    off = np.ones(len(z), dtype=bool)
    if len(lv):
        off = np.min(np.abs(z[:, None] - lv[None]), axis=1) > 0.05
    out["off_level_fraction"] = round(float(off.mean()), 3)
    if raw is not None:
        rx, ry, rz, _ = raw
        rm = in_box(box, np.column_stack([rx, ry]), rz)
        rx, ry, rz = rx[rm], ry[rm], rz[rm]
        out["raw_columns"] = int(len(rz))
        if len(rz):
            # nearest slab cell in the same column (same voxel grid), any storey: closest in z
            key_s = np.round((xy[m] - cells.origin) / cells.v - 0.5).astype(np.int64)
            key_r = np.round((np.column_stack([rx, ry]) - cells.origin) / cells.v - 0.5).astype(np.int64)
            cs = key_s[:, 0] * cells.ny + key_s[:, 1]
            cr = key_r[:, 0] * cells.ny + key_r[:, 1]
            order = np.argsort(cs, kind="stable")
            cs_s, z_s = cs[order], z[order]
            lo = np.searchsorted(cs_s, cr, side="left")
            hi = np.searchsorted(cs_s, cr, side="right")
            res = np.full(len(rz), np.nan)
            for k in range(int((hi - lo).max()) if len(rz) else 0):
                j = lo + k
                f = j < hi
                d = z_s[j[f]] - rz[f]
                cur = res[f]
                res[f] = np.where(np.isnan(cur) | (np.abs(d) < np.abs(cur)), d, cur)
            has = np.isfinite(res) & (np.abs(res) <= 0.30)
            out["coverage"] = round(float(has.mean()), 3)
            if has.any():
                r = np.abs(res[has])
                out["residual_p95"] = round(float(np.percentile(r, 95)), 3)
                out["residual_max"] = round(float(r.max()), 3)
                out["buried_fraction"] = round(float((res[has] < -0.08).mean()), 3)   # slab well below raw top
    return out


def ascii_profile(cells: SlabCells, box: dict, axis_deg: float | None, width: int = 100) -> str:
    xy = cells.xy()
    m = in_box(box, xy, cells.z)
    if m.sum() == 0:
        return "(no slab cells in box)\n"
    p, z = xy[m], cells.z[m]
    if axis_deg is None:
        A = np.column_stack([p, np.ones(len(p))])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        axis_deg = math.degrees(math.atan2(coef[1], coef[0]))
    u = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg))])
    s = p @ u
    s -= s.min()
    ds = max(0.1, s.max() / width)
    dz = 0.05
    si = np.floor(s / ds).astype(int)
    zi = np.floor((z - z.min()) / dz).astype(int)
    grid = np.full((zi.max() + 1, si.max() + 1), " ")
    cnt = {}
    for a, b in zip(zi, si):
        cnt[(a, b)] = cnt.get((a, b), 0) + 1
    for (a, b), n in cnt.items():
        grid[a, b] = "#" if n >= 3 else "."
    lines = [f"stair profile along {axis_deg:.0f} deg, column = {ds:.2f} m, row = {dz:.2f} m, '#' >= 3 cells"]
    for r in range(grid.shape[0] - 1, -1, -1):
        lines.append(f"{z.min() + r * dz:7.2f} |" + "".join(grid[r]))
    return "\n".join(lines) + "\n"


# -- walkability --------------------------------------------------------------------------------------
def make_base(sim: Simulator, overrides: list[str] | None = None, collisions: bool = True) -> IdealBase:
    r = copy.deepcopy(sim.cfg["robot"])
    for item in overrides or []:
        key, _, raw = item.partition("=")
        key = key.strip()
        if key.startswith("robot."):
            key = key[6:]
        r[key] = yaml.safe_load(raw)
    return IdealBase(r, sim.rays, sim.voxels, blocked=sim._blocked if collisions else None)


def strict_base(sim: Simulator, base: IdealBase) -> IdealBase:
    r = copy.deepcopy(base.c)
    r["max_feet_over_gap"] = 0
    return IdealBase(r, sim.rays, sim.voxels, blocked=None)


def feet_on_floor(cells: SlabCells, base: IdealBase, x, y, yaw, z, dz: float = 0.25) -> bool:
    """All four foot centres over a flattened floor cell within dz of z."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    fx = x + cy * base.feet[:, 0] - sy * base.feet[:, 1]
    fy = y + sy * base.feet[:, 0] + cy * base.feet[:, 1]
    ix = np.floor((fx - cells.origin[0]) / cells.v).astype(np.int64)
    iy = np.floor((fy - cells.origin[1]) / cells.v).astype(np.int64)
    lo, hi = cells.column_range(ix * cells.ny + (iy - cells.iy_min))
    for a, b in zip(lo, hi):
        if b <= a or np.abs(cells.z[a:b] - z).min() > dz:
            return False
    return True


def step_check(base: IdealBase, strict: IdealBase, x0, y0, z0, x1, y1, blocked=None, cells=None, z1=None):
    """Enter (x1, y1) from (x0, y0). Returns (reason or None, foot-over-nothing, body fits, feet on floor)."""
    yaw = math.atan2(y1 - y0, x1 - x0) if (x1 != x0 or y1 != y0) else 0.0
    fits = True if blocked is None else not blocked(x1, y1, yaw, (z1 if z1 is not None else z0) + base.c["stand_height"])
    on_floor = True if cells is None else feet_on_floor(cells, base, x1, y1, yaw, z1 if z1 is not None else z0)
    if not base.reset(x0, y0, yaw, z0):
        return "no_ground_at_start", False, fits, on_floor
    reason = base._check(x1, y1, yaw)
    c = base.c
    g = strict._sample(x1, y1, yaw, base.ground.feet_h + c["max_step_up"] + 0.05,
                       c["max_step_up"] + c["max_step_down"] + 0.05)
    return reason, not g.valid, fits, on_floor


def cluster_events(events: list, size: float = 2.0, top: int = 5):
    buckets = {}
    for reason, x, y in events:
        k = (reason, math.floor(x / size), math.floor(y / size))
        buckets.setdefault(k, []).append((x, y))
    rows = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:top]
    return [{"reason": k[0], "x": round(float(np.mean([p[0] for p in v])), 1),
             "y": round(float(np.mean([p[1] for p in v])), 1), "count": len(v)} for k, v in rows]


def walk_auto(sim: Simulator, cells: SlabCells, base: IdealBase, lattice: int = 3, mask=None,
              label: str = "walk") -> dict:
    """BFS over slab cells from the spawn, then enter every lattice cell from its BFS parent."""
    t0 = time.time()
    sx, sy = sim.base.x, sim.base.y
    start = cells.nearest(sx, sy, sim.base.ground.z0)
    if start is None:
        return {"error": "spawn is not on a flattened floor cell"}
    rows, cols = cells.edges(0.2)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(cells),) * 2).tocsr()
    order, pred = breadth_first_order(graph, start, directed=False, return_predecessors=True)
    reach = np.zeros(len(cells), dtype=bool)
    reach[order] = True
    out = {"reachable_cells": int(reach.sum()), "reachable_m2": round(float(reach.sum()) * cells.v ** 2, 1),
           "total_cells": int(len(cells))}
    sel = reach & (cells.ix % lattice == 0) & (cells.iy % lattice == 0) & (pred >= 0)
    if mask is not None:
        sel &= mask
    idx = np.flatnonzero(sel)
    strict = strict_base(sim, base)
    xy = cells.xy()
    counts = {r: 0 for r in REASONS}
    counts["no_ground_at_start"] = 0
    events = []
    unsupported = n_blocked = n_edge = n_valid = 0
    for i in idx:
        p = pred[i]
        reason, foot, fits, on_floor = step_check(base, strict, xy[p, 0], xy[p, 1], cells.z[p], xy[i, 0], xy[i, 1],
                                                  blocked=sim._blocked, cells=cells, z1=cells.z[i])
        if not fits:                 # the body does not fit here even at rest (walls, furniture): not a floor defect
            n_blocked += 1
            continue
        if not on_floor:             # a foot over a column without flattened floor: edge of a floor, or a hole
            n_edge += 1
            events.append(("foot_off_floor", xy[i, 0], xy[i, 1]))
            continue
        n_valid += 1
        unsupported += foot
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            events.append((reason, xy[i, 0], xy[i, 1]))
    n = max(1, n_valid)
    out.update(samples=int(len(idx)), blocked_samples=n_blocked, edge_samples=n_edge, valid_samples=n_valid,
               edge_fraction=round(n_edge / max(1, len(idx) - n_blocked), 4), events=counts,
               fail_fraction=round(sum(counts.values()) / n, 4),
               unsupported_foot_fraction=round(unsupported / n, 4), clusters=cluster_events(events),
               seconds=round(time.time() - t0, 1))
    return out


def walk_polyline(sim: Simulator, base: IdealBase, pts: np.ndarray, z_hint: float | None, step: float = 0.1) -> dict:
    """Replay a polyline [N,2|3]; re-anchor at the failing sample after every rejection."""
    seg = np.diff(pts[:, :2], axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s = np.r_[0, np.cumsum(L)]
    n = max(2, int(s[-1] / step))
    t = np.linspace(0, s[-1], n)
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    z = np.interp(t, s, pts[:, 2]) if pts.shape[1] > 2 and z_hint is None else np.full(n, z_hint if z_hint is not None else 0.0)
    strict = strict_base(sim, base)
    counts = {r: 0 for r in REASONS}
    counts["no_ground_at_start"] = 0
    events = []
    unsupported = 0
    yaw = math.atan2(y[1] - y[0], x[1] - x[0])
    ok = base.reset(x[0], y[0], yaw, z[0])
    for i in range(1, n):
        yaw = math.atan2(y[i] - y[i - 1], x[i] - x[i - 1]) if (x[i] != x[i - 1] or y[i] != y[i - 1]) else yaw
        reason = base._check(x[i], y[i], yaw) if ok else "no_ground_at_start"
        c = base.c
        g = strict._sample(x[i], y[i], yaw, base.ground.feet_h + c["max_step_up"] + 0.05,
                           c["max_step_up"] + c["max_step_down"] + 0.05)
        unsupported += not g.valid
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            events.append((reason, x[i], y[i]))
            ok = base.reset(x[i], y[i], yaw, z[i] if z_hint is not None or pts.shape[1] > 2 else base.ground.z0)
    return {"length_m": round(float(s[-1]), 1), "samples": int(n - 1), "events": counts,
            "fail_fraction": round(sum(counts.values()) / (n - 1), 4),
            "unsupported_foot_fraction": round(unsupported / (n - 1), 4), "clusters": cluster_events(events)}


def load_poses(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 4:
                try:
                    rows.append([float(t[1]), float(t[2]), float(t[3])])
                except ValueError:
                    continue
    return np.array(rows)


def load_waypoints(path: str) -> np.ndarray:
    with open(path) as f:
        d = yaml.safe_load(f)
    return np.array([[w["x"], w["y"], w.get("z", 0.0)] for w in d["waypoints"]], dtype=np.float64)


# -- main --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--pcd", help="source PCD: stair coverage and --find-stairs need it")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stairs-box", action="append", default=[], help="xmin,xmax,ymin,ymax[,zmin,zmax]; repeatable")
    ap.add_argument("--stairs-axis", type=float, default=None, help="profile axis (deg); default: fitted")
    ap.add_argument("--find-stairs", action="store_true", help="scan the PCD for staircases and print boxes")
    ap.add_argument("--auto", action="store_true", help="walk every reachable floor cell from the spawn")
    ap.add_argument("--lattice", type=int, default=3, help="walk every n-th cell in x and y (default 3 = 0.3 m)")
    ap.add_argument("--poses", help="poses.txt (SLAM keyframes) to replay")
    ap.add_argument("--waypoints", help="waypoints yaml to replay as straight legs")
    ap.add_argument("--z-floor", type=float, default=None, help="floor z hint for --waypoints (default: z - 0.4)")
    ap.add_argument("--spawn", help="x,y,yaw,z_floor (default: config)")
    ap.add_argument("--climb", default="robot.max_step_up=0.22,robot.max_slope_deg=45",
                    help="comma-separated robot overrides for the second stair replay ('' = skip)")
    ap.add_argument("--json", default=None, help="report path (default <scene>/eval.json)")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    cfg["scene"] = a.scene
    if a.spawn:
        cfg["spawn"] = [float(t) for t in a.spawn.split(",")]
    t0 = time.time()
    sim = Simulator(cfg)
    report = {"scene": a.scene, "config_spawn": cfg["spawn"], "overrides": a.overrides}
    meta = sim.meta
    report["ground"] = meta.get("ground")
    report["build_args"] = meta.get("build_args")
    report["boxes"] = meta.get("boxes")
    report["spawn"] = {"requested": list(map(float, sim.spawn_requested)), "ok": bool(sim.spawn_ok),
                       "moved": bool(sim.spawn_moved), "floor_z": round(float(sim.spawn_floor_z), 3),
                       "pose": [round(sim.base.x, 3), round(sim.base.y, 3)]}
    print(f"scene {a.scene}: {meta.get('boxes')} boxes, load {time.time() - t0:.1f}s")
    print(f"ground: {json.dumps(meta.get('ground'))}")
    print(f"spawn: {report['spawn']}")

    ground_path = os.path.join(a.scene, "ground.npz")
    slabs = np.load(ground_path)["slabs"] if os.path.exists(ground_path) else None
    cells = SlabCells(slabs, sim.voxels.origin, sim.voxels.voxel)
    boxes = [parse_box(s) for s in a.stairs_box]

    if a.find_stairs:
        if not a.pcd:
            ap.error("--find-stairs needs --pcd")
        rx, ry, rz, _ = raw_surface(a.pcd, sim.voxels.voxel)
        cands = find_stairs(rx, ry, rz)
        report["stair_candidates"] = cands
        print(f"stair candidates ({len(cands)}), largest first:")
        for c in cands[:20]:
            print(f"  --stairs-box={c['xmin']:.0f},{c['xmax']:.0f},{c['ymin']:.0f},{c['ymax']:.0f},"
                  f"{c['zmin'] - 0.3:.1f},{c['zmax'] + 0.3:.1f}   {c['cells_1m']} m2, {c['n_levels']} levels, "
                  f"z {c['zmin']:.2f}..{c['zmax']:.2f}, axis {c['axis_deg']} deg")

    raw = raw_surface(a.pcd, sim.voxels.voxel, boxes) if (a.pcd and boxes) else None
    profiles = []
    report["stairs"] = []
    for b in boxes:
        m = stair_metrics(cells, b, raw)
        prof = ascii_profile(cells, b, a.stairs_axis)
        profiles.append(prof)
        report["stairs"].append(m)
        print(f"stairs {b['xmin']:.0f},{b['xmax']:.0f},{b['ymin']:.0f},{b['ymax']:.0f}: levels {m.get('n_levels')} "
              f"rises {m.get('n_rises')} x {m.get('rise_mean')}±{m.get('rise_std')} [{m.get('rise_min')},{m.get('rise_max')}] "
              f"off-level {m.get('off_level_fraction')} ramp {m.get('ramp_fraction')} coverage {m.get('coverage')} "
              f"res p95 {m.get('residual_p95')} buried {m.get('buried_fraction')} cells {m.get('slab_cells')}")
        print(prof)

    base = sim.base
    if a.auto:
        w = walk_auto(sim, cells, base, a.lattice, label="walk")
        report["walk"] = w
        print(f"walk: reachable {w.get('reachable_m2')} m2 ({w.get('reachable_cells')} of {w.get('total_cells')} cells), "
              f"samples {w.get('samples')} (blocked {w.get('blocked_samples')}, edge {w.get('edge_samples')} = "
              f"{w.get('edge_fraction')}, valid {w.get('valid_samples')}), fail {w.get('fail_fraction')}, "
              f"events {w.get('events')}, foot-over-nothing {w.get('unsupported_foot_fraction')} ({w.get('seconds')}s)")
        for c in w.get("clusters", []):
            print(f"   {c['reason']:12s} x {c['x']:7.1f} y {c['y']:7.1f}  {c['count']}")
        if boxes:
            xy = cells.xy()
            report["stairs_walk"] = []
            for b in boxes:
                mask = in_box(b, xy, cells.z)
                res = {"box": b, "config": walk_auto(sim, cells, base, 1, mask=mask)}
                if a.climb:
                    cb = make_base(sim, [t for t in a.climb.split(",") if t], collisions=False)
                    res["climb"] = walk_auto(sim, cells, cb, 1, mask=mask)
                    res["climb_overrides"] = a.climb
                report["stairs_walk"].append(res)
                for k in ("config", "climb"):
                    if k in res:
                        r = res[k]
                        print(f"stairs-walk {b['xmin']:.0f},{b['xmax']:.0f},{b['ymin']:.0f},{b['ymax']:.0f} [{k}]: "
                              f"valid {r.get('valid_samples')} (edge {r.get('edge_fraction')}) fail {r.get('fail_fraction')} "
                              f"events {r.get('events')} foot-over-nothing {r.get('unsupported_foot_fraction')}")
    if a.poses:
        pts = load_poses(a.poses)
        pts[:, 2] -= 0.4 if a.z_floor is None else 0.0
        if a.z_floor is not None:
            pts[:, 2] = a.z_floor
        w = walk_polyline(sim, base, pts, None)
        report["poses_walk"] = w
        print(f"poses: {w['length_m']} m, {w['samples']} samples, fail {w['fail_fraction']}, events {w['events']}, "
              f"foot-over-nothing {w['unsupported_foot_fraction']}")
        for c in w["clusters"]:
            print(f"   {c['reason']:12s} x {c['x']:7.1f} y {c['y']:7.1f}  {c['count']}")
    if a.waypoints:
        pts = load_waypoints(a.waypoints)
        pts[:, 2] = (pts[:, 2] - 0.4) if a.z_floor is None else a.z_floor
        w = walk_polyline(sim, base, pts, None)
        report["waypoints_walk"] = w
        print(f"waypoints: {w['length_m']} m, fail {w['fail_fraction']}, events {w['events']}, "
              f"foot-over-nothing {w['unsupported_foot_fraction']}")

    out = a.json or os.path.join(a.scene, "eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    if profiles:
        with open(os.path.join(a.scene, "eval_profile.txt"), "w") as f:
            f.write("\n".join(profiles))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
