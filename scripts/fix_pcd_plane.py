#!/usr/bin/env python3
"""Lift a synthetic ground plane in a PCD to the height of the real floor.

Some exported maps carry a generated ground plane (a dense, perfectly flat layer at one z,
e.g. z = -0.95 in Maps/dp_1012.pcd) that lies below the real, scanned floor. Built as is,
every scan hole in the real floor becomes a pit down to that plane. This script keeps the
plane only where the real floor has no points (so it fills the holes) and lifts those points
to the floor height measured from the real floor points nearby; everywhere else the plane is
dropped. Points that are not part of the plane are copied unchanged.

Usage:
  python3 scripts/fix_pcd_plane.py Maps/dp_1012.pcd -o Maps/dp_1012_fixed.pcd
  python3 scripts/fix_pcd_plane.py Maps/dp_1012.pcd --plane-z -0.95 --dry-run      # only report

The plane is found automatically (the most populated 2 cm z bin whose layer is unusually flat
and dense) unless --plane-z is given. Pure numpy/scipy: runs on the host or in the container.

Maps/dp_1012.pcd (2026-10-15 export): plane at z = -0.952 (3.17 M points, 24.7 %); the real
floor is snapped to flat levels (-0.52 main, -0.35, -0.58, -0.72) with a sunken area near
-0.8/-1.0 at x 15-37, y 25-55. 1.26 M plane points fill holes, 1.87 M are dropped.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pcd2sim.pcd_io import read_pcd, write_pcd  # noqa: E402


def detect_plane_z(z: np.ndarray, bin_m: float = 0.02) -> float | None:
    """z of the densest 2 cm layer if it holds > 5 % of all points and is flat: at least 80 % of the
    points within +-10 cm of it lie within +-2 cm (a scanned floor scatters a few cm: ~35 %)."""
    lo, hi = np.floor(z.min()), np.ceil(z.max())
    h, e = np.histogram(z, bins=np.arange(lo, hi + bin_m, bin_m))
    k = int(np.argmax(h))
    if h[k] < 0.05 * len(z):
        return None
    sel = (z >= e[k] - bin_m) & (z < e[k + 1] + bin_m)
    zp = float(np.median(z[sel]))
    tight = np.count_nonzero(np.abs(z - zp) < 0.02)
    wide = np.count_nonzero(np.abs(z - zp) < 0.10)
    return zp if tight > 0.8 * wide else None


def floor_height_map(xyz: np.ndarray, cell: float, z_max: float, min_pts: int, origin: np.ndarray,
                     bin_m: float = 0.05, half: float = 0.07):
    """Per-cell height of the lowest surface among points with z < z_max.

    Returns (cell_ids int64 [C] sorted, floor_h [C]). A cell has a floor when it has a 5 cm z bin
    with >= min_pts points; the lowest bin holding at least 30 % of the cell's busiest bin is the
    floor (so a few noise points under the floor do not pull it down) and the height is the mean z
    within +-half of that bin.
    """
    pts = xyz[xyz[:, 2] < z_max]
    if len(pts) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    ij = np.floor((pts[:, :2] - origin[:2]) / cell).astype(np.int64)
    cid = ij[:, 0] * 1_000_003 + ij[:, 1]
    zb = np.floor((pts[:, 2] - origin[2]) / bin_m).astype(np.int64)
    pair = cid * 100_000 + zb                             # zb < 100000 for any sane map
    up, cnt = np.unique(pair, return_counts=True)
    pc_all = up // 100_000
    cells_all, inv = np.unique(pc_all, return_inverse=True)
    peak = np.zeros(len(cells_all), dtype=np.int64)
    np.maximum.at(peak, inv, cnt)
    up = up[(cnt >= min_pts) & (cnt >= 0.3 * peak[inv])]
    if len(up) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    pc = up // 100_000
    cells, first = np.unique(pc, return_index=True)       # sorted by pair -> first = lowest bin
    zc = origin[2] + (up[first] % 100_000 + 0.5) * bin_m
    pos = np.searchsorted(cells, cid)
    pos = np.clip(pos, 0, len(cells) - 1)
    hit = cells[pos] == cid
    near = hit & (np.abs(pts[:, 2] - zc[np.clip(pos, 0, len(cells) - 1)]) <= half)
    s = np.bincount(pos[near], weights=pts[near, 2], minlength=len(cells))
    n = np.bincount(pos[near], minlength=len(cells))
    floor_h = np.where(n > 0, s / np.maximum(n, 1), zc)
    return cells, floor_h


def fix_plane(xyz: np.ndarray, plane_z: float, plane_band: float = 0.10, floor_zmax: float = -0.1,
              cell: float = 0.25, radius: float = 3.0, min_floor_pts: int = 5, k_nb: int = 24):
    """Return (keep mask over xyz, new z for kept points, stats dict)."""
    z = xyz[:, 2]
    plane = np.abs(z - plane_z) < plane_band
    origin = xyz.min(axis=0) - 1e-6
    cells, floor_h = floor_height_map(xyz[~plane], cell, floor_zmax, min_floor_pts, origin)
    stats = {"points": int(len(xyz)), "plane_points": int(plane.sum()), "floor_cells": int(len(cells))}
    new_z = z.copy()
    keep = ~plane
    if plane.sum() == 0 or len(cells) == 0:
        stats.update(lifted=0, dropped_floor_present=0, dropped_no_floor_nearby=int(plane.sum()))
        return keep, new_z, stats

    pi = np.flatnonzero(plane)
    ij = np.floor((xyz[pi, :2] - origin[:2]) / cell).astype(np.int64)
    cid = ij[:, 0] * 1_000_003 + ij[:, 1]
    pos = np.clip(np.searchsorted(cells, cid), 0, len(cells) - 1)
    has_floor = cells[pos] == cid

    # cells of plane points without a real floor: interpolate the floor height from floor cells nearby
    ucells, inv = np.unique(cid[~has_floor], return_inverse=True)
    uij = np.column_stack([ucells // 1_000_003, ucells % 1_000_003])
    centres = origin[:2] + (uij + 0.5) * cell
    fij = np.column_stack([cells // 1_000_003, cells % 1_000_003])
    tree = cKDTree(origin[:2] + (fij + 0.5) * cell)
    k = min(k_nb, len(cells))
    dist, idx = tree.query(centres, k=k, distance_upper_bound=radius)
    dist = dist.reshape(len(centres), -1)
    idx = idx.reshape(len(centres), -1)
    ok = np.isfinite(dist)
    w = np.where(ok, 1.0 / np.maximum(dist, 0.5 * cell) ** 2, 0.0)
    hsum = (w * np.where(ok, floor_h[np.minimum(idx, len(cells) - 1)], 0.0)).sum(axis=1)
    wsum = w.sum(axis=1)
    cell_h = np.where(wsum > 0, hsum / np.maximum(wsum, 1e-12), np.nan)

    target = np.full(len(pi), np.nan)
    target[~has_floor] = cell_h[inv]
    lift = np.isfinite(target)
    new_z[pi[lift]] = target[lift]
    keep[pi[lift]] = True
    stats.update(lifted=int(lift.sum()), dropped_floor_present=int(has_floor.sum()),
                 dropped_no_floor_nearby=int((~has_floor & ~lift).sum()),
                 lifted_z_min=float(new_z[pi[lift]].min()) if lift.any() else None,
                 lifted_z_max=float(new_z[pi[lift]].max()) if lift.any() else None)
    return keep, new_z, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcd")
    ap.add_argument("-o", "--out", help="output PCD (required unless --dry-run)")
    ap.add_argument("--plane-z", type=float, default=None, help="z of the synthetic plane (default: detect)")
    ap.add_argument("--plane-band", type=float, default=0.10, help="|z - plane_z| below this is plane (m)")
    ap.add_argument("--floor-zmax", type=float, default=-0.1,
                    help="real-floor search uses points below this z (keeps upper storeys out)")
    ap.add_argument("--cell", type=float, default=0.25, help="grid cell of the floor height map (m)")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="plane points with no real-floor cell within this distance are dropped (m)")
    ap.add_argument("--min-floor-pts", type=int, default=5, help="points in a 5 cm z bin that make a floor")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.out:
        ap.error("-o/--out is required unless --dry-run")

    xyz, inten = read_pcd(a.pcd)
    plane_z = a.plane_z if a.plane_z is not None else detect_plane_z(xyz[:, 2])
    if plane_z is None:
        print(f"{a.pcd}: no synthetic plane found ({len(xyz)} points); nothing to do")
        return 1
    keep, new_z, st = fix_plane(xyz, plane_z, a.plane_band, a.floor_zmax, a.cell, a.radius, a.min_floor_pts)
    print(f"{a.pcd}: {st['points']} points, plane at z={plane_z:.3f}: {st['plane_points']} points "
          f"({100 * st['plane_points'] / st['points']:.1f}%); real-floor cells {st['floor_cells']}")
    print(f"  dropped (real floor present) {st['dropped_floor_present']}, "
          f"dropped (no floor within {a.radius} m) {st['dropped_no_floor_nearby']}, lifted {st['lifted']}")
    if st["lifted"]:
        lz = new_z[keep & (np.abs(xyz[:, 2] - plane_z) < a.plane_band)]
        h, e = np.histogram(lz, bins=np.arange(np.floor(lz.min() * 10) / 10, lz.max() + 0.1, 0.1))
        print("  lifted points by z: " + ", ".join(f"{e[i]:.1f}:{h[i]}" for i in range(len(h)) if h[i]))
    if a.dry_run:
        return 0
    out = xyz[keep].copy()
    out[:, 2] = new_z[keep]
    write_pcd(a.out, out.astype(np.float32), None if inten is None else inten[keep])
    print(f"  wrote {a.out}: {len(out)} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
