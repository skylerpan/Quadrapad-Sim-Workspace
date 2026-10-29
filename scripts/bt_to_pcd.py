#!/usr/bin/env python3
"""Sample an OctoMap (.bt / .ot) back into a surface point cloud (PCD).

Every occupied voxel contributes its *exposed* faces only (faces whose neighbour voxel is
free or unknown), so walls, floors, stairs and ceilings become evenly covered surfaces and
nothing is generated inside solid blocks. The output is a binary PCD with float32 x y z in
the octree's own frame, which makes a hand-edited octomap usable as the static map of
map_server_3d / the local costmap without going through a scan export.

Usage:
  python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt -o Maps/dp_1012_handmade_surface.pcd
  python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt -o out.pcd --points 12800000   # exact count
  python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt -o out.pcd --grid 4            # 4x4 per face
  python3 scripts/bt_to_pcd.py Maps/dp_1012_handmade.bt --dry-run                      # only report

Sampling modes (one of):
  --points N   N points in total, spread evenly over all exposed faces (each face gets
               floor(N/F) or ceil(N/F) points). Points are stratified-jittered inside the
               face so the density is uniform without a visible lattice. Default 12,800,000.
  --grid n     a regular n x n lattice per face (n=3 at 0.2 m -> 6.7 cm spacing).

Pure numpy: runs on the host or in the container.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcd2sim import octomap_io  # noqa: E402
from pcd2sim.pcd_io import write_pcd  # noqa: E402

DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)
# in-plane axes for each face direction
AXES = {0: (1, 2), 1: (1, 2), 2: (0, 2), 3: (0, 2), 4: (0, 1), 5: (0, 1)}


def pack(idx: np.ndarray) -> np.ndarray:
    """Pack integer voxel indices [N,3] into one int64 key (indices must fit in 21 bits)."""
    off = 1 << 20
    return ((idx[:, 0] + off) << 42) | ((idx[:, 1] + off) << 21) | (idx[:, 2] + off)


def exposed_faces(centers: np.ndarray, res: float, skip_down: bool):
    """Return (face_centres [F,3], face_dir [F] int) for faces without an occupied neighbour."""
    idx = np.floor(centers / res + 1e-6).astype(np.int64)
    keys = np.sort(pack(idx))
    faces = []
    dirs = []
    per_dir = {}
    for d, vec in enumerate(DIRS):
        if skip_down and d == 5:
            per_dir[d] = 0
            continue
        nb = pack(idx + vec)
        pos = np.searchsorted(keys, nb)
        pos[pos >= len(keys)] = len(keys) - 1
        occupied = keys[pos] == nb
        sel = ~occupied
        per_dir[d] = int(sel.sum())
        faces.append(centers[sel] + vec * (res / 2.0))
        dirs.append(np.full(int(sel.sum()), d, dtype=np.int8))
    return np.vstack(faces), np.concatenate(dirs), per_dir


def sample_grid(faces: np.ndarray, dirs: np.ndarray, res: float, n: int, offset: float) -> np.ndarray:
    u = (np.arange(n) + 0.5) / n - 0.5
    uu, vv = np.meshgrid(u, u, indexing="ij")
    uv = np.column_stack([uu.ravel(), vv.ravel()]) * res  # [n*n, 2]
    out = []
    for d in range(6):
        sel = dirs == d
        if not sel.any():
            continue
        a, b = AXES[d]
        f = faces[sel]
        pts = np.repeat(f, len(uv), axis=0)
        pts[:, a] += np.tile(uv[:, 0], len(f))
        pts[:, b] += np.tile(uv[:, 1], len(f))
        if offset:
            pts += DIRS[d] * offset
        out.append(pts)
    return np.vstack(out)


def sample_points(faces: np.ndarray, dirs: np.ndarray, res: float, total: int, offset: float,
                  rng: np.random.Generator) -> np.ndarray:
    nf = len(faces)
    base, extra = divmod(total, nf)
    per_face = np.full(nf, base, dtype=np.int64)
    if extra:
        per_face[rng.choice(nf, extra, replace=False)] += 1
    out = []
    for k in np.unique(per_face):
        if k == 0:
            continue
        sel_k = per_face == k
        m = int(math.ceil(math.sqrt(k)))  # m x m strata per face, pick k of them
        cells = m * m
        f = faces[sel_k]
        d = dirs[sel_k]
        nfk = len(f)
        if k == cells:
            chosen = np.tile(np.arange(cells), (nfk, 1))
        else:
            chosen = np.argsort(rng.random((nfk, cells)), axis=1)[:, :k]
        ci = chosen // m
        cj = chosen % m
        u = (ci + rng.random((nfk, k))) / m - 0.5
        v = (cj + rng.random((nfk, k))) / m - 0.5
        pts = np.repeat(f, k, axis=0)
        u = (u * res).ravel()
        v = (v * res).ravel()
        drep = np.repeat(d, k)
        for dd in range(6):
            s = drep == dd
            if not s.any():
                continue
            a, b = AXES[dd]
            pts[s, a] += u[s]
            pts[s, b] += v[s]
            if offset:
                pts[s] += DIRS[dd] * offset
        out.append(pts)
    return np.vstack(out)


def report(xyz: np.ndarray) -> None:
    z = xyz[:, 2]
    print(f"points: {len(xyz):,}   bbox x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] z[{z.min():.2f},{z.max():.2f}]")
    edges = np.arange(math.floor(z.min() * 5) / 5, z.max() + 0.2, 0.2)
    h, e = np.histogram(z, bins=edges)
    top = np.argsort(h)[::-1][:12]
    print("z histogram, 12 fullest 0.2 m bins (z_lo: count):",
          ", ".join(f"{e[i]:.1f}: {h[i]:,}" for i in sorted(top)))
    print(f"points with z < 0 (1F floor band): {(z < 0).sum():,}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("octomap", help=".bt or .ot file")
    ap.add_argument("-o", "--out", help="output PCD (required unless --dry-run)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--points", type=int, default=None, help="total number of points (default 12800000)")
    g.add_argument("--grid", type=int, default=None, help="regular n x n lattice per face instead of --points")
    ap.add_argument("--skip-down", action="store_true", help="do not sample faces that point down (ceilings)")
    ap.add_argument("--offset", type=float, default=0.0, help="push points outward along the face normal (m)")
    ap.add_argument("--occ-thres", type=float, default=0.5, help="occupancy threshold for .ot files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report voxel / face counts only")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.out:
        ap.error("-o/--out is required unless --dry-run")

    t0 = time.time()
    leaves = octomap_io.read_octomap(args.octomap, args.occ_thres)
    centers = octomap_io.expand_leaves(leaves)
    res = leaves.res
    print(f"{args.octomap}: res {res} m, occupied voxels {len(centers):,} ({time.time() - t0:.1f}s)")

    faces, dirs, per_dir = exposed_faces(centers, res, args.skip_down)
    names = ["+x", "-x", "+y", "-y", "+z", "-z"]
    print(f"exposed faces {len(faces):,} of {6 * len(centers):,}  "
          + "  ".join(f"{names[d]}: {per_dir[d]:,}" for d in range(6))
          + f"   area {len(faces) * res * res:,.0f} m^2")
    if args.dry_run:
        n = args.points or 12_800_000
        print(f"--points {n:,} -> {n / len(faces):.1f} points per face, spacing ~{math.sqrt(res * res * len(faces) / n) * 100:.1f} cm")
        for k in (3, 4, 5):
            print(f"--grid {k} -> {len(faces) * k * k:,} points, spacing {res / k * 100:.1f} cm")
        return 0

    rng = np.random.default_rng(args.seed)
    if args.grid:
        xyz = sample_grid(faces, dirs, res, args.grid, args.offset)
    else:
        xyz = sample_points(faces, dirs, res, args.points or 12_800_000, args.offset, rng)
    report(xyz)
    write_pcd(args.out, xyz.astype(np.float32))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
