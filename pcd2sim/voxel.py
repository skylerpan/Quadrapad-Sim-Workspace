"""Sparse voxel processing: voxelize, denoise, merge into boxes, floor/spawn analysis.

Everything works on sorted int64 voxel codes (no dense grid), so multi-floor
maps fit in a few hundred MB of RAM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelGrid:
    keys: np.ndarray        # [N,3] int32 voxel indices, sorted by code
    counts: np.ndarray      # [N] points per voxel
    intensity: np.ndarray   # [N] mean intensity
    zmean: np.ndarray       # [N] mean point z (sub-voxel surface height)
    origin: np.ndarray      # [3] world position of voxel (0,0,0) corner
    voxel: float
    dims: np.ndarray        # [3] grid extent in voxels

    def encode(self, keys: np.ndarray) -> np.ndarray:
        return encode(keys, self.dims)

    @property
    def codes(self) -> np.ndarray:
        return self.encode(self.keys)

    def centers(self) -> np.ndarray:
        return self.origin + (self.keys + 0.5) * self.voxel

    def subset(self, mask: np.ndarray) -> "VoxelGrid":
        return VoxelGrid(self.keys[mask], self.counts[mask], self.intensity[mask], self.zmean[mask],
                         self.origin, self.voxel, self.dims)


def encode(keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    k = keys.astype(np.int64)
    return (k[:, 0] * int(dims[1]) + k[:, 1]) * int(dims[2]) + k[:, 2]


def contains(sorted_codes: np.ndarray, keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Membership test of keys in the sorted code set (out-of-range keys -> False)."""
    inb = np.all((keys >= 0) & (keys < dims), axis=1)
    out = np.zeros(len(keys), dtype=bool)
    if inb.any():
        c = encode(keys[inb], dims)
        idx = np.searchsorted(sorted_codes, c)
        idx[idx >= len(sorted_codes)] = 0
        out[inb] = sorted_codes[idx] == c
    return out


def crop_percentile(xyz: np.ndarray, pct: float) -> np.ndarray:
    """Mask of points inside the [pct, 100-pct] percentile box on every axis."""
    if pct <= 0:
        return np.ones(len(xyz), dtype=bool)
    lo = np.percentile(xyz, pct, axis=0)
    hi = np.percentile(xyz, 100 - pct, axis=0)
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


def voxelize(xyz: np.ndarray, intensity: np.ndarray | None, voxel: float) -> VoxelGrid:
    origin = np.floor(xyz.min(axis=0) / voxel) * voxel
    keys = np.floor((xyz - origin) / voxel).astype(np.int64)
    dims = keys.max(axis=0) + 1
    codes = encode(keys, dims)
    uniq, first, inv, counts = np.unique(codes, return_index=True, return_inverse=True, return_counts=True)
    if intensity is None:
        mean_i = np.zeros(len(uniq), dtype=np.float32)
    else:
        mean_i = (np.bincount(inv, weights=intensity, minlength=len(uniq)) / counts).astype(np.float32)
    zmean = (np.bincount(inv, weights=xyz[:, 2], minlength=len(uniq)) / counts).astype(np.float32)
    return VoxelGrid(keys[first].astype(np.int32), counts.astype(np.int32), mean_i, zmean, origin, voxel, dims)


_NEIGHBORS_26 = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                          if (dx, dy, dz) != (0, 0, 0)], dtype=np.int32)


def remove_isolated(grid: VoxelGrid, min_neighbors: int = 2) -> VoxelGrid:
    """Drop voxels with fewer than min_neighbors occupied voxels in their 26-neighbourhood."""
    codes = grid.codes
    nb = np.zeros(len(codes), dtype=np.int32)
    for off in _NEIGHBORS_26:
        nb += contains(codes, grid.keys + off, grid.dims)
    return grid.subset(nb >= min_neighbors)


_NEIGHBORS_13 = _NEIGHBORS_26[[i for i, (dx, dy, dz) in enumerate(_NEIGHBORS_26)
                               if dx > 0 or (dx == 0 and (dy > 0 or (dy == 0 and dz > 0)))]]   # each pair once


def components_26(grid: VoxelGrid) -> tuple[int, np.ndarray]:
    """26-connected component labels of the occupied voxels: (n_components, label per voxel)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    codes = grid.codes
    n = len(codes)
    rows, cols = [], []
    for off in _NEIGHBORS_13:
        k2 = grid.keys + off
        hit = contains(codes, k2, grid.dims)
        if hit.any():
            rows.append(np.flatnonzero(hit))
            cols.append(np.searchsorted(codes, encode(k2[hit], grid.dims)))
    if not rows:
        return n, np.arange(n)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    return connected_components(adj, directed=False)


def remove_small_clusters(grid: VoxelGrid, min_voxels: int) -> VoxelGrid:
    """Drop 26-connected clusters with fewer than min_voxels voxels (specks, ghosts). <= 1 is a no-op."""
    if min_voxels <= 1 or len(grid.keys) == 0:
        return grid
    _, lab = components_26(grid)
    size = np.bincount(lab)[lab]
    return grid.subset(size >= min_voxels)


def smooth_surface_z(grid: VoxelGrid, radius: int = 1, tol: float = 0.06) -> np.ndarray:
    """Bilateral xy-smoothing of per-voxel mean z.

    Averages zmean over neighbours within `radius` voxels in x/y (and +-1 layer in z)
    whose height differs by less than `tol`, so a noisy SLAM floor flattens while
    real steps (> tol) keep sharp edges.
    """
    codes = grid.codes
    order = np.argsort(codes)
    sorted_codes = codes[order]
    zsum = grid.zmean.astype(np.float64).copy()
    wsum = np.ones(len(codes))
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nk = grid.keys + np.array([dx, dy, dz], dtype=np.int32)
                inb = np.all((nk >= 0) & (nk < grid.dims), axis=1)
                c = encode(nk[inb], grid.dims)
                i = np.clip(np.searchsorted(sorted_codes, c), 0, len(sorted_codes) - 1)
                found = sorted_codes[i] == c
                zn = np.full(len(codes), np.nan)
                zn[np.flatnonzero(inb)[found]] = grid.zmean[order[i[found]]]
                w = np.abs(zn - grid.zmean) < tol
                zsum[w] += zn[w]
                wsum[w] += 1.0
    return (zsum / wsum).astype(np.float32)


def _merge_axis(rows: np.ndarray, group_cols: list[int], lo_col: int, hi_col: int) -> np.ndarray:
    """Merge rows sharing group_cols whose [lo,hi] intervals on one axis are consecutive.

    rows: int array; each row's interval on the merged axis is rows[:, lo_col]..rows[:, hi_col].
    Returns merged rows (lo from the first row of a run, hi from the last).
    """
    order = np.lexsort([rows[:, lo_col]] + [rows[:, c] for c in reversed(group_cols)])
    r = rows[order]
    same_group = np.all(r[1:, group_cols] == r[:-1, group_cols], axis=1)
    consecutive = r[1:, lo_col] == r[:-1, hi_col] + 1
    start = np.ones(len(r), dtype=bool)
    start[1:] = ~(same_group & consecutive)
    starts = np.flatnonzero(start)
    ends = np.r_[starts[1:], len(r)] - 1
    out = r[starts].copy()
    out[:, hi_col] = r[ends, hi_col]
    return out


def merge_boxes(keys: np.ndarray) -> np.ndarray:
    """Greedy run-merge of voxel keys into axis-aligned boxes.

    Returns int array [M,6] = (x0, x1, y0, y1, z0, z1), inclusive voxel ranges.
    Pass 1 merges along x, pass 2 merges equal x-runs along y, pass 3 merges
    equal rectangles along z. Boxes exactly tile the input set (no overlap).
    """
    k = keys.astype(np.int64)
    rows = np.stack([k[:, 0], k[:, 0], k[:, 1], k[:, 1], k[:, 2], k[:, 2]], axis=1)
    rows = _merge_axis(rows, group_cols=[2, 4], lo_col=0, hi_col=1)        # along x
    rows = _merge_axis(rows, group_cols=[0, 1, 4], lo_col=2, hi_col=3)     # along y
    rows = _merge_axis(rows, group_cols=[0, 1, 2, 3], lo_col=4, hi_col=5)  # along z
    return rows


def boxes_to_world(boxes: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """[M,6] inclusive voxel ranges -> [M,6] float (cx, cy, cz, hx, hy, hz)."""
    lo = boxes[:, [0, 2, 4]].astype(np.float64)
    hi = boxes[:, [1, 3, 5]].astype(np.float64) + 1.0
    center = grid.origin + 0.5 * (lo + hi) * grid.voxel
    half = 0.5 * (hi - lo) * grid.voxel
    return np.hstack([center, half])


def fit_floor_normal(points: np.ndarray, iters: int = 300, thresh: float = 0.03,
                     max_tilt_deg: float = 10.0, seed: int = 0) -> tuple[np.ndarray, float, int]:
    """RANSAC for the dominant near-horizontal plane. Returns (unit normal +z-ish, d, inliers)."""
    rng = np.random.default_rng(seed)
    pts = points if len(points) <= 200_000 else points[rng.choice(len(points), 200_000, replace=False)]
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    best = (np.array([0.0, 0.0, 1.0]), 0.0, -1)
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_max:
            continue
        d = -n @ p[0]
        inl = int(np.count_nonzero(np.abs(pts @ n + d) < thresh))
        if inl > best[2]:
            best = (n, d, inl)
    n, d, _ = best
    # least-squares refine on inliers
    inl_pts = pts[np.abs(pts @ n + d) < thresh]
    if len(inl_pts) >= 3:
        c = inl_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(inl_pts - c, full_matrices=False)
        n = vt[2] if vt[2][2] > 0 else -vt[2]
        d = -n @ c
    return n, float(d), len(inl_pts)


def standable_mask(grid: VoxelGrid, clearance: float = 0.8, roof_min: float = 1.5,
                   roof_cell: float | None = None) -> np.ndarray:
    """Voxels whose top is free for `clearance` metres and that have structure (a roof) above.

    The roof test rejects the upper side of ceilings, which are also free above. It looks at the highest
    voxel in the `roof_cell` x `roof_cell` column around the voxel (default: the voxel's own column); a
    ceiling scanned sparsely has no point in most 3 cm columns, so fine builds pass 0.1-0.2 m.
    """
    codes = grid.codes
    k = grid.keys
    free = np.ones(len(k), dtype=bool)
    for dz in range(1, int(np.ceil(clearance / grid.voxel)) + 1):
        free &= ~contains(codes, k + np.array([0, 0, dz], dtype=np.int32), grid.dims)
    # highest occupied z per (x,y) column (coarsened to roof_cell)
    n = max(1, int(round((roof_cell or grid.voxel) / grid.voxel)))
    col = (k[:, 0].astype(np.int64) // n) * (int(grid.dims[1]) // n + 1) + k[:, 1] // n
    order = np.argsort(col, kind="stable")
    col_s = col[order]
    starts = np.flatnonzero(np.r_[True, col_s[1:] != col_s[:-1]])
    col_max = np.maximum.reduceat(k[order, 2], starts)
    top = np.empty(len(k), dtype=np.int64)
    top[order] = np.repeat(col_max, np.diff(np.r_[starts, len(k)]))
    roofed = (top - k[:, 2]) * grid.voxel >= roof_min
    return free & roofed


def find_spawn(grid: VoxelGrid, stand: np.ndarray, radius: float = 0.4) -> np.ndarray | None:
    """Pick a flat, clear standable voxel near the centroid of all standable voxels.

    Returns the world position of the voxel's top surface (x, y, z_floor) or None.
    """
    sk = grid.keys[stand]
    if len(sk) == 0:
        return None
    s_codes = np.sort(grid.encode(sk))
    r = int(np.ceil(radius / grid.voxel))
    ok = np.ones(len(sk), dtype=bool)
    for dx in range(-r, r + 1, max(1, r // 4)):
        for dy in range(-r, r + 1, max(1, r // 4)):
            if dx * dx + dy * dy > r * r:
                continue
            hit = np.zeros(len(sk), dtype=bool)
            for dz in (-1, 0, 1):  # allow 1-voxel unevenness
                hit |= contains(s_codes, sk + np.array([dx, dy, dz], dtype=np.int32), grid.dims)
            ok &= hit
    cand = sk[ok]
    if len(cand) == 0:
        return None
    centroid = sk.mean(axis=0)
    best = cand[np.argmin(np.linalg.norm(cand - centroid, axis=1))]
    pos = grid.origin + (best + 0.5) * grid.voxel
    pos[2] = grid.origin[2] + (best[2] + 1) * grid.voxel
    return pos


# ---------------------------------------------------------------------------------------------------------
# Hole closing: a SLAM surface voxelized at 2-3 cm is a sieve (most voxels hold one point), so walls and
# floors are scattered over 2-3 layers with gaps between the boxes. A closing pass fills an empty cell when
# it sits inside a surface: >= min_plane of its 8 neighbours in one axis plane (xy, xz or yz) are occupied
# (a hole in a floor or wall), or >= min_nb of all 26 are (a hole in a thick, noisy surface, or an inside
# corner). A cell lying flat against a complete sheet sees only the 9 cells of that sheet, none in its own
# planes, so sheets do not thicken and edges do not grow. Big maps (30 M candidates) are processed in chunks
# by forked worker processes with the per-chunk results parked on disk, to bound the peak RSS.
# ---------------------------------------------------------------------------------------------------------

_FACES_6 = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int32)
_CLOSE_SHARED: dict = {}        # read-only arrays the forked workers inherit (no pickling, no copies)


def _decode(codes: np.ndarray, dims: np.ndarray) -> np.ndarray:
    d1, d2 = int(dims[1]), int(dims[2])
    return np.stack([codes // (d1 * d2), (codes // d2) % d1, codes % d2], axis=1).astype(np.int32)


def _count_neighbors(keys: np.ndarray, codes: np.ndarray, intensity: np.ndarray, dims: np.ndarray):
    """Per key: occupied 26-neighbour count, the largest occupied 8-neighbour count over the three axis
    planes through the cell, and the summed neighbour intensity."""
    n = np.zeros(len(keys), dtype=np.int32)
    n_plane = np.zeros((3, len(keys)), dtype=np.int32)       # xy (dz=0), xz (dy=0), yz (dx=0)
    isum = np.zeros(len(keys), dtype=np.float64)
    for off in _NEIGHBORS_26:
        nk = keys + off
        inb = np.all((nk >= 0) & (nk < dims), axis=1)
        if not inb.any():
            continue
        c = encode(nk[inb], dims)
        i = np.clip(np.searchsorted(codes, c), 0, len(codes) - 1)
        hit = codes[i] == c
        where = np.flatnonzero(inb)[hit]
        n[where] += 1
        isum[where] += intensity[i[hit]]
        for ax, p in ((2, 0), (1, 1), (0, 2)):
            if off[ax] == 0:
                n_plane[p, where] += 1
    return n, n_plane.max(axis=0), isum


def _close_chunk(job):
    """Worker: neighbour-count one chunk of candidate keys, save the accepted cells, return (path, n)."""
    cand_path, start, stop, min_nb, min_plane, out_path = job
    keys = np.load(cand_path, mmap_mode="r")[start:stop]
    keys = np.ascontiguousarray(keys)
    s = _CLOSE_SHARED
    n, n_plane, isum = _count_neighbors(keys, s["codes"], s["intensity"], s["dims"])
    ok = (n >= min_nb) | (n_plane >= min_plane)
    np.savez(out_path, keys=keys[ok], intensity=(isum[ok] / np.maximum(n[ok], 1)).astype(np.float32))
    return out_path, int(ok.sum())


def close_holes(grid: VoxelGrid, min_nb: int = 12, min_plane: int = 6, rounds: int = 2,
                workers: int | None = None, tmp_dir: str | None = None,
                chunk: int = 2_000_000) -> tuple[VoxelGrid, int]:
    """Fill empty cells inside a surface, `rounds` times: >= min_plane of the 8 neighbours in one axis plane
    occupied, or >= min_nb of all 26 (a cell against a complete flat sheet sees 9, so 12 never thickens it).

    Candidates are the empty 6-face neighbours of occupied voxels. An added voxel gets counts 1, the mean
    intensity of its occupied neighbours and zmean at its centre. Returns (grid, voxels added). rounds <= 0
    returns the input grid. Work is chunked (`chunk` keys per job) over `workers` forked processes
    (default os.cpu_count()); candidate and per-chunk files live in `tmp_dir` (a fresh temporary
    directory by default) and are removed before the function returns.
    """
    import multiprocessing as mp
    import os
    import shutil
    import tempfile

    if rounds <= 0 or len(grid.keys) == 0:
        return grid, 0
    workers = max(1, workers or os.cpu_count() or 1)
    own_tmp = tmp_dir is None or not os.path.isdir(tmp_dir)     # created here -> removed here
    tmp_dir = tempfile.mkdtemp(prefix="close_") if tmp_dir is None else tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    total_added = 0
    try:
        for rnd in range(rounds):
            codes = grid.codes
            dims = grid.dims
            # 1) candidates: empty face neighbours, built one face at a time, parked on disk
            cand_files = []
            for f, off in enumerate(_FACES_6):
                nk = grid.keys + off
                inb = np.all((nk >= 0) & (nk < dims), axis=1)
                c = np.unique(encode(nk[inb], dims))
                p = os.path.join(tmp_dir, f"cand_{f}.npy")
                np.save(p, c)
                cand_files.append(p)
                del nk, inb, c
            cc = np.unique(np.concatenate([np.load(p) for p in cand_files]))
            for p in cand_files:
                os.remove(p)
            cc = cc[~np.isin(cc, codes, assume_unique=True)]
            cand_path = os.path.join(tmp_dir, "cand.npy")
            np.save(cand_path, _decode(cc, dims))
            n_cand = len(cc)
            del cc
            # 2) neighbour counts per chunk, in forked workers sharing the sorted codes read-only
            _CLOSE_SHARED.update(codes=codes, intensity=grid.intensity, dims=dims)
            jobs = [(cand_path, s, min(s + chunk, n_cand), min_nb, min_plane, os.path.join(tmp_dir, f"add_{i}.npz"))
                    for i, s in enumerate(range(0, n_cand, chunk))]
            results = []
            try:
                if workers == 1 or len(jobs) <= 1:
                    results = [_close_chunk(j) for j in jobs]
                else:
                    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
                        results = pool.map(_close_chunk, jobs)
                added_keys = [np.load(p)["keys"] for p, _ in results]
                added_int = [np.load(p)["intensity"] for p, _ in results]
            finally:
                _CLOSE_SHARED.clear()
                for p, _ in results:
                    if os.path.exists(p):
                        os.remove(p)
                os.remove(cand_path)
            n_add = sum(n for _, n in results)
            if n_add == 0:
                break
            ak = np.concatenate(added_keys).astype(np.int32)
            ai = np.concatenate(added_int).astype(np.float32)
            az = (grid.origin[2] + (ak[:, 2] + 0.5) * grid.voxel).astype(np.float32)
            keys = np.vstack([grid.keys, ak])
            order = np.argsort(encode(keys, dims), kind="stable")
            grid = VoxelGrid(keys[order],
                             np.concatenate([grid.counts, np.ones(n_add, np.int32)])[order],
                             np.concatenate([grid.intensity, ai])[order],
                             np.concatenate([grid.zmean, az])[order],
                             grid.origin, grid.voxel, dims)
            total_added += n_add
    finally:
        if own_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return grid, total_added
