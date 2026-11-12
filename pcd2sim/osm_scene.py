"""OSM buildings -> a quadsim scene, without a point cloud in between.

A 2 x 2 km area at the 0.1-0.2 m voxels of pcd2sim.build_scene would not fit in memory, and a building from OSM is
already what a box scene wants: an outline extruded to a height. So each footprint is rasterised on a grid aligned
with the map axes (cell centres inside the outline), every cell is a column from the ground to the roof, and the
columns are merged into boxes with pcd2sim.voxel.merge_boxes (runs along x, then equal runs along y).
The ground is flat: square tiles whose tops sit at ground_z, all in the contact set.
"""
from __future__ import annotations

import numpy as np

from .osm_buildings import Building
from .voxel import merge_boxes

Clip = tuple[np.ndarray, np.ndarray]             # (lo_xy, hi_xy) of the scene square in map metres


def footprint_cells(fp: np.ndarray, cell: float, clip: Clip | None = None) -> np.ndarray:
    """[K,2] int grid indices whose cell centres lie inside the footprint (even-odd rule) and inside clip."""
    lo, hi = fp.min(axis=0), fp.max(axis=0)
    if clip is not None:
        lo, hi = np.maximum(lo, clip[0]), np.minimum(hi, clip[1])
    # the cells whose centre (i + 0.5) * cell lies in [lo, hi]
    i0 = np.ceil(lo / cell - 0.5).astype(np.int64)
    i1 = np.floor(hi / cell - 0.5).astype(np.int64) + 1
    if np.any(i1 <= i0):
        return np.zeros((0, 2), dtype=np.int64)
    ix, iy = np.meshgrid(np.arange(i0[0], i1[0]), np.arange(i0[1], i1[1]), indexing="ij")
    keys = np.column_stack([ix.ravel(), iy.ravel()])
    px, py = (keys[:, 0] + 0.5) * cell, (keys[:, 1] + 0.5) * cell
    inside = np.zeros(len(keys), dtype=bool)
    a, b = fp, np.roll(fp, -1, axis=0)
    for (ax, ay), (bx, by) in zip(a, b):
        crosses = (ay > py) != (by > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (px < x_at)
    return keys[inside]


def _columns_to_boxes(keys: np.ndarray, cell: float, z0: float, z1: float) -> np.ndarray:
    runs = merge_boxes(np.column_stack([keys, np.zeros(len(keys), dtype=np.int64)]))
    x0, x1 = runs[:, 0] * cell, (runs[:, 1] + 1) * cell
    y0, y1 = runs[:, 2] * cell, (runs[:, 3] + 1) * cell
    n = len(runs)
    return np.column_stack([(x0 + x1) / 2, (y0 + y1) / 2, np.full(n, (z0 + z1) / 2),
                            (x1 - x0) / 2, (y1 - y0) / 2, np.full(n, (z1 - z0) / 2)])


def buildings_to_boxes(buildings: list[Building], cell: float, ground_z: float,
                       clip: Clip | None = None) -> np.ndarray:
    """float32 [M,6] (cx, cy, cz, hx, hy, hz): every building as merged full-height columns on a cell grid."""
    parts = []
    for b in buildings:
        keys = footprint_cells(b.footprint, cell, clip)
        if len(keys):
            parts.append(_columns_to_boxes(keys, cell, ground_z, ground_z + b.height_m))
    if not parts:
        return np.zeros((0, 6), dtype=np.float32)
    return np.vstack(parts).astype(np.float32)


def ground_tiles(lo: np.ndarray, hi: np.ndarray, tile: float, z: float, thickness: float = 0.2) -> np.ndarray:
    """float32 [K,6] square tiles covering [lo, hi] (the last row / column shortened to fit), tops at z."""
    xs = np.append(np.arange(lo[0], hi[0], tile), hi[0])
    ys = np.append(np.arange(lo[1], hi[1], tile), hi[1])
    x0, y0 = np.meshgrid(xs[:-1], ys[:-1], indexing="ij")
    x1, y1 = np.meshgrid(xs[1:], ys[1:], indexing="ij")
    x0, y0, x1, y1 = (v.ravel() for v in (x0, y0, x1, y1))
    keep = (x1 - x0 > 1e-9) & (y1 - y0 > 1e-9)
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    n = len(x0)
    return np.column_stack([(x0 + x1) / 2, (y0 + y1) / 2, np.full(n, z - thickness / 2),
                            (x1 - x0) / 2, (y1 - y0) / 2, np.full(n, thickness / 2)]).astype(np.float32)
