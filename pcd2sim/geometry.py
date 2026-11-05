"""Robot-free geometry metrics of a built scene: floor reachability, holes, floor under a trajectory.

Everything works on the scene's own voxel grid (voxels.npz): a floor cell is a standable voxel (free for
0.8 m above, a roof further up), or a flattened slab cell when the scene has ground.npz. No robot model,
no step/slope limits, so the numbers describe the geometry build_scene produced, not what a particular
robot makes of it (scripts/eval_scene.py --geometry).

  load_grid         voxels.npz -> VoxelGrid sorted by code
  floor_cells       standable voxels as (ix, iy, kz) + top height, sorted by column
  reachable         8-connected flood from a start cell with |dz| <= dz_max (stairs included)
  holes             empty columns next to the reachable floor: nothing at all within [-0.3, +0.8] m of the
                    neighbouring floor top, grown ring by ring up to a radius; components that reach the
                    radius are open edges (map boundary, unscanned rooms), the rest are holes
  trajectory_floor  floor under samples of a trajectory (SLAM keyframes): missing floor, obstacles, and the
                    height residual against a reference floor (the raw cloud)
"""
from __future__ import annotations

import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .voxel import VoxelGrid, contains, encode, standable_mask

_XY8 = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)], dtype=np.int64)
_XY8Z = np.c_[_XY8, np.zeros(len(_XY8), np.int64)]
SIZE_BUCKETS = ((1, 1), (2, 4), (5, 20), (21, 100), (101, 10 ** 9))


def load_grid(scene: str) -> VoxelGrid:
    """The scene's voxel grid, sorted by code (make_stairs appends its voxels unsorted)."""
    d = np.load(os.path.join(scene, "voxels.npz"))
    keys = d["keys"].astype(np.int32)
    dims = d["dims"].astype(np.int64)
    order = np.argsort(encode(keys, dims), kind="stable")
    return VoxelGrid(keys[order], np.ones(len(keys), np.int32), d["intensity"][order], d["zmean"][order],
                     d["origin"].astype(np.float64), float(d["voxel"]), dims)


class FloorCells:
    """Floor cells (ix, iy, kz) with a top height, sorted by (column, z); several per column on multi-storey
    maps. kz is the voxel index whose top face is the floor (slab cells: the voxel just below the slab top)."""

    def __init__(self, grid: VoxelGrid, keys: np.ndarray, z: np.ndarray):
        k = np.asarray(keys, dtype=np.int64)
        z = np.asarray(z, dtype=np.float64)
        self.v = float(grid.voxel)
        self.origin = grid.origin.astype(np.float64)
        self.dims = grid.dims.astype(np.int64)
        col = k[:, 0] * int(self.dims[1]) + k[:, 1]
        order = np.lexsort((k[:, 2], col))
        self.k, self.z, self.col = k[order], z[order], col[order]
        self.n = len(self.k)

    def __len__(self):
        return self.n

    def xy(self, idx=slice(None)) -> np.ndarray:
        return self.origin[:2] + (self.k[idx, :2] + 0.5) * self.v

    def column_range(self, col: np.ndarray):
        return np.searchsorted(self.col, col, side="left"), np.searchsorted(self.col, col, side="right")

    def nearest(self, x: float, y: float, z: float, r: float = 0.5, dz: float = 0.6) -> int | None:
        xy = self.xy()
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        ok = (d <= r) & (np.abs(self.z - z) <= dz)
        if not ok.any():
            return None
        cand = np.flatnonzero(ok)
        return int(cand[np.argmin(d[cand] + np.abs(self.z[cand] - z))])

    def edges(self, dz_max: float):
        rows, cols = [], []
        depth = int(np.max(np.bincount(self.col - self.col.min()))) if self.n else 0
        for dx, dy in _XY8:
            nc = (self.k[:, 0] + dx) * int(self.dims[1]) + (self.k[:, 1] + dy)
            lo, hi = self.column_range(nc)
            for s in range(depth):
                j = lo + s
                f = j < hi
                i = np.flatnonzero(f)
                j = j[f]
                close = np.abs(self.z[i] - self.z[j]) <= dz_max
                rows.append(i[close])
                cols.append(j[close])
        return np.concatenate(rows), np.concatenate(cols)


def floor_cells(grid: VoxelGrid, roof_cell: float | None = None, use_zmean: bool = True) -> FloorCells:
    """Standable voxels of the scene grid. With use_zmean the top is the voxel's mean point z when finite
    (what the terrain probe sees), the voxel top face otherwise."""
    stand = standable_mask(grid, roof_cell=roof_cell)
    face = grid.origin[2] + (grid.keys[:, 2] + 1) * grid.voxel
    top = np.where(np.isfinite(grid.zmean), grid.zmean, face) if use_zmean else face
    return FloorCells(grid, grid.keys[stand], top[stand])


def slab_floor_cells(grid: VoxelGrid, slab_keys: np.ndarray, slab_z: np.ndarray) -> FloorCells:
    """Flattened scenes: floor cells from rasterised slab cells (ix, iy) with their top height."""
    kz = np.floor((slab_z - grid.origin[2]) / grid.voxel - 1e-6).astype(np.int64) - 1
    keys = np.column_stack([slab_keys[:, 0], slab_keys[:, 1], kz])
    return FloorCells(grid, keys, slab_z)


def reachable(cells: FloorCells, start: int, dz_max: float = 0.2) -> np.ndarray:
    """Bool mask of the cells reachable from `start` over 8-neighbour steps of at most dz_max."""
    rows, cols = cells.edges(dz_max)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(cells.n,) * 2).tocsr()
    order = breadth_first_order(graph, start, directed=False, return_predecessors=False)
    reach = np.zeros(cells.n, dtype=bool)
    reach[order] = True
    return reach


def _in_grid(keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return np.all((keys >= 0) & (keys < dims), axis=1)


def holes(grid: VoxelGrid, cells: FloorCells, reach: np.ndarray, below: float = 0.3, above: float = 0.8,
          max_radius: float = 1.5, extra_occupied: np.ndarray | None = None) -> dict:
    """Empty columns next to the reachable floor.

    A column is empty at reference height kz (the neighbouring floor cell's index) when the scene has no
    voxel at (column, kz + d) for d in [-below, +above] voxels, and no `extra_occupied` key (flattened slab
    cells as (ix, iy, kz)) either. Holes grow ring by ring from the floor boundary; a component still
    growing at `max_radius` is an open edge (map boundary, unscanned space) and is reported separately.
    Returns counts, a size distribution and the largest holes with their centres (metres).
    """
    v = grid.voxel
    d1, d2 = int(grid.dims[1]), int(grid.dims[2])
    nb, na = int(np.ceil(below / v)), int(np.ceil(above / v))
    rings = int(np.ceil(max_radius / v))
    occ = grid.codes
    if extra_occupied is not None and len(extra_occupied):
        ek = np.asarray(extra_occupied, dtype=np.int64)
        ek = ek[_in_grid(ek, grid.dims)]
        occ = np.union1d(occ, encode(ek, grid.dims))

    def empty(keys: np.ndarray) -> np.ndarray:
        out = np.ones(len(keys), dtype=bool)
        for d in range(-nb, na + 1):
            out &= ~contains(occ, keys + np.array([0, 0, d], dtype=np.int32), grid.dims)
        return out

    def grow(frontier: np.ndarray) -> np.ndarray:
        nxt = (frontier[:, None, :].astype(np.int64) + _XY8Z[None]).reshape(-1, 3)
        nxt = nxt[_in_grid(nxt, grid.dims)]
        return np.unique(nxt, axis=0).astype(np.int32)

    cand = grow(cells.k[reach].astype(np.int32))
    cand = cand[empty(cand)]
    if len(cand) == 0:
        return {"cells": 0, "m2": 0.0, "components": 0, "open_edge_components": 0, "open_edge_cells": 0,
                "by_size": {}, "largest": []}
    seen = np.sort(encode(cand, grid.dims))
    all_keys, ring_of = [cand], [np.zeros(len(cand), np.int32)]
    frontier = cand
    for r in range(1, rings + 1):
        nxt = grow(frontier)
        nc = encode(nxt, grid.dims)
        nxt, nc = nxt[~np.isin(nc, seen)], nc[~np.isin(nc, seen)]
        if len(nxt) == 0:
            break
        ok = empty(nxt)
        nxt, nc = nxt[ok], nc[ok]
        if len(nxt) == 0:
            break
        seen = np.union1d(seen, nc)
        all_keys.append(nxt)
        ring_of.append(np.full(len(nxt), r, np.int32))
        frontier = nxt
    keys = np.concatenate(all_keys).astype(np.int64)
    ring = np.concatenate(ring_of)
    code = encode(keys, grid.dims)
    order = np.argsort(code)
    code_s = code[order]
    rows, cols = [], []
    for dx, dy in _XY8:
        for dz in range(-nb, nb + 1):
            q = ((keys[:, 0] + dx) * d1 + (keys[:, 1] + dy)) * d2 + keys[:, 2] + dz
            j = np.minimum(np.searchsorted(code_s, q), len(code_s) - 1)
            hit = code_s[j] == q
            rows.append(np.flatnonzero(hit))
            cols.append(order[j[hit]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n_comp, lab = connected_components(coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                                                  shape=(len(keys),) * 2), directed=False)
    size = np.bincount(lab, minlength=n_comp)
    open_edge = np.zeros(n_comp, dtype=bool)
    open_edge[lab[ring >= rings]] = True
    hole = ~open_edge
    by_size = {}
    for lo, hi in SIZE_BUCKETS:
        m = hole & (size >= lo) & (size <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo - 1}")
        by_size[label] = {"holes": int(m.sum()), "cells": int(size[m].sum())}
    largest = []
    for c in np.argsort(-(size * hole))[:10]:
        if not hole[c] or size[c] == 0:
            break
        m = lab == c
        xy = grid.origin[:2] + (keys[m, :2] + 0.5) * v
        z = grid.origin[2] + (keys[m, 2] + 1) * v
        largest.append({"cells": int(size[c]), "m2": round(float(size[c]) * v * v, 2),
                        "x": round(float(xy[:, 0].mean()), 1), "y": round(float(xy[:, 1].mean()), 1),
                        "z": round(float(z.mean()), 2)})
    return {"cells": int(size[hole].sum()), "m2": round(float(size[hole].sum()) * v * v, 2),
            "components": int(hole.sum()), "open_edge_components": int(open_edge.sum()),
            "open_edge_cells": int(size[open_edge].sum()), "by_size": by_size, "largest": largest}


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` metres of arc length along a polyline [N,>=2] (other columns interpolated)."""
    seg = np.diff(pts[:, :2], axis=0)
    s = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, pts[:, j]) for j in range(pts.shape[1])])


def _nearest_in_columns(grid: VoxelGrid, base: np.ndarray, col_sorted: np.ndarray, z_sorted: np.ndarray,
                        z_hint: np.ndarray, r: int, dz_max: float = 0.5) -> np.ndarray:
    """For each sample column `base` [N,2], the z among (col_sorted, z_sorted) within r cells in xy whose
    value is nearest z_hint (and within dz_max); NaN when none."""
    n = len(base)
    out = np.full(n, np.nan)
    best = np.full(n, np.inf)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            col = (base[:, 0] + dx) * int(grid.dims[1]) + (base[:, 1] + dy)
            lo = np.searchsorted(col_sorted, col, side="left")
            hi = np.searchsorted(col_sorted, col, side="right")
            depth = int((hi - lo).max()) if n else 0
            for s in range(depth):
                j = lo + s
                f = j < hi
                i = np.flatnonzero(f)
                d = np.abs(z_sorted[j[f]] - z_hint[i])
                better = (d < best[i]) & (d <= dz_max)
                out[i[better]] = z_sorted[j[f][better]]
                best[i[better]] = d[better]
    return out


def trajectory_floor(grid: VoxelGrid, cells: FloorCells, samples: np.ndarray, z_hint: np.ndarray,
                     reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                     search: float = 0.15, clearance: float = 0.3) -> dict:
    """Floor under each sample (x, y) given a floor-height hint: the floor cell within `search` metres in xy
    whose top is nearest the hint (within 0.5 m). Reports missing floor, obstacles (a scene voxel between
    floor top + `clearance` and hint + 0.3, i.e. something standing where the robot was), the spread of
    floor - hint (floor noise along the path when the sensor height was constant), and the residual against
    a reference floor given as (x, y, z) arrays, e.g. the raw cloud's standable columns."""
    v = grid.voxel
    n = len(samples)
    r = int(np.ceil(search / v))
    base = np.floor((samples[:, :2] - grid.origin[:2]) / v).astype(np.int64)
    floor_z = _nearest_in_columns(grid, base, cells.col, cells.z, z_hint, r)
    has = np.isfinite(floor_z)
    out = {"samples": int(n), "no_floor": int((~has).sum()), "no_floor_fraction": round(float((~has).mean()), 4)}
    obst = np.zeros(n, dtype=bool)
    if has.any():
        kz0 = np.floor((floor_z[has] + clearance - grid.origin[2]) / v).astype(np.int64)
        kz1 = np.floor((z_hint[has] + 0.3 - grid.origin[2]) / v).astype(np.int64)
        codes = grid.codes
        o = np.zeros(int(has.sum()), dtype=bool)
        for d in range(max(0, int((kz1 - kz0).max())) + 1):
            kz = kz0 + d
            keys = np.column_stack([base[has], kz]).astype(np.int32)
            o |= (kz <= kz1) & contains(codes, keys, grid.dims)
        obst[has] = o
        rel = floor_z[has] - z_hint[has]
        out["hint_offset_median"] = round(float(np.median(rel)), 3)
        out["hint_offset_p95_abs_dev"] = round(float(np.percentile(np.abs(rel - np.median(rel)), 95)), 3)
    out["obstacle"] = int(obst.sum())
    out["obstacle_fraction"] = round(float(obst.mean()), 4)
    if reference is not None and has.any():
        rx, ry, rz = (np.asarray(a, dtype=np.float64) for a in reference)
        rb = np.floor((np.column_stack([rx, ry]) - grid.origin[:2]) / v).astype(np.int64)
        rcol = rb[:, 0] * int(grid.dims[1]) + rb[:, 1]
        order = np.lexsort((rz, rcol))
        ref_z = _nearest_in_columns(grid, base, rcol[order], rz[order], z_hint, r)
        both = has & np.isfinite(ref_z)
        out["reference_samples"] = int(both.sum())
        out["reference_missing"] = int((~np.isfinite(ref_z)).sum())
        if both.any():
            res = floor_z[both] - ref_z[both]
            out["residual_median"] = round(float(np.median(res)), 3)
            out["residual_p95_abs"] = round(float(np.percentile(np.abs(res), 95)), 3)
            out["residual_max_abs"] = round(float(np.abs(res).max()), 3)
    return out
