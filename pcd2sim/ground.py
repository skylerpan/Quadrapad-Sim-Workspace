"""Ground flattening: noisy SLAM floor voxels -> planar triangle slabs.

A 10 cm voxel floor built from a SLAM map is grainy: the floor points scatter a
few cm vertically, and the floor itself is slightly tilted (SLAM drift), which
axis-aligned boxes can only represent as terraces. Here the walkable surface
cells are grouped into regions of continuous height, each region (or 4 m tile
of it) is fitted with a plane, and the floor voxels around that height are
replaced by thick slabs whose top faces lie exactly on the plane. Real height
changes larger than `tol` (stairs, platforms) split regions and keep their edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .voxel import VoxelGrid, _merge_axis, encode

_XY8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
_NB26 = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                  if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)


@dataclass
class Ground:
    remove: np.ndarray          # [N] bool over grid voxels replaced by slabs
    cell_keys: np.ndarray       # [C,3] voxel key of each ground cell (its top floor voxel)
    cell_h: np.ndarray          # [C] flattened surface height
    slabs: np.ndarray           # [K,8,3] float32 slab corners, corner index = 4*ix + 2*iy + iz
    stats: dict = field(default_factory=dict)


def _lookup(sorted_codes: np.ndarray, keys: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Index into sorted_codes for each key, -1 where absent."""
    out = np.full(len(keys), -1, dtype=np.int64)
    inb = np.all((keys >= 0) & (keys < dims), axis=1)
    if inb.any() and len(sorted_codes):
        c = encode(keys[inb], dims)
        i = np.clip(np.searchsorted(sorted_codes, c), 0, len(sorted_codes) - 1)
        out[np.flatnonzero(inb)[sorted_codes[i] == c]] = i[sorted_codes[i] == c]
    return out


def _fit_plane(x, y, z, iters: int = 3):
    """Robust z = a*x + b*y + c. Returns (coef, inlier RMS, inlier mask)."""
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    ok = np.ones(len(z), dtype=bool)
    coef = np.zeros(3)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[ok], z[ok], rcond=None)
        r = z - A @ coef
        sigma = 1.4826 * np.median(np.abs(r[ok])) + 1e-6
        new_ok = np.abs(r) < max(2.5 * sigma, 0.01)
        if new_ok.sum() < 3 or (new_ok == ok).all():
            break
        ok = new_ok
    r = (z - A @ coef)[ok]
    return coef, float(np.sqrt(np.mean(r * r))), ok


def _weighted_median(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted median of z [C,L] with weights w [C,L] (w = 0 marks missing entries)."""
    order = np.argsort(z, axis=1)
    zs = np.take_along_axis(z, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    cum = np.cumsum(ws, axis=1)
    half = 0.5 * cum[:, -1:]
    k = np.argmax(cum >= half, axis=1)
    return zs[np.arange(len(z)), k]


def _components(n: int, rows: np.ndarray, cols: np.ndarray, members: np.ndarray) -> list[np.ndarray]:
    """Connected components of the sub-graph induced by `members` (edge lists over n nodes)."""
    inset = np.zeros(n, dtype=bool)
    inset[members] = True
    e = inset[rows] & inset[cols]
    local = np.full(n, -1, dtype=np.int64)
    local[members] = np.arange(len(members))
    adj = coo_matrix((np.ones(int(e.sum()), dtype=np.int8), (local[rows[e]], local[cols[e]])),
                     shape=(len(members), len(members)))
    _, lab = connected_components(adj, directed=False)
    order = np.argsort(lab, kind="stable")
    bounds = np.flatnonzero(np.r_[True, lab[order][1:] != lab[order][:-1], True])
    return [members[order[bounds[k]:bounds[k + 1]]] for k in range(len(bounds) - 1)]


def _in_sorted(sorted_vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Membership of `q` in the sorted unique array `sorted_vals` (np.isin re-sorts both on every call)."""
    if len(sorted_vals) == 0:
        return np.zeros(len(q), dtype=bool)
    pos = np.clip(np.searchsorted(sorted_vals, q), 0, len(sorted_vals) - 1)
    return sorted_vals[pos] == q


def extract_ground(grid: VoxelGrid, stand: np.ndarray, tol: float = 0.06, max_rms: float = 0.035,
                   tile: float = 4.0, min_cells: int = 30, band_below: float = 0.2, band_above: float = 0.1,
                   thickness: float = 0.2, layers: int = 2, grow: int = 3, clearance: float = 0.8,
                   max_above: int = 2, spike_pts: int = 3, floor_tol: float = 0.15, fill: int = 12, fill_nb: int = 3,
                   float_height: float = 0.5, float_max_voxels: int = 4, sheet_pts: int = 2,
                   fill_guard: float = 0.0, surface: str = "mean", peel: int = 0, small_flat: int = 0,
                   column_gap: float = 0.0, link_height: float = 0.1) -> Ground:
    """Flatten the walkable surface of `grid` (raw per-voxel zmean, before smoothing).

    stand: [N] mask of walkable top-surface voxels (see voxel.standable_mask).
    A cell's raw height is the count-weighted mean (`surface="mean"`) or weighted median
    (`surface="median"`, robust to a sparse noise voxel on top and to riser points below a stair
    tread) of its top voxel and `layers` voxels below. Cells whose heights differ by < tol link
    into regions; a region that is not one plane (rms >= max_rms) is fitted per 4 m tile, or, with
    `peel` > 0, split first: the robust fit's inliers become one plane and the remaining cells are
    re-split into connected pieces, up to `peel` levels deep. That separates a landing from the
    treads chained to it through riser columns of intermediate height. Regions of at most
    `small_flat` cells are snapped to a horizontal plane at their median height instead of a fitted
    one: a tread is only 2-3 cells deep and its raw surface slopes by a few cm (riser points in the
    front column), which a free plane fit turns into a ramp. 0 = off; 60 (0.6 m^2) keeps real ramps.
    Regions smaller than `min_cells` are left as voxels (clutter tops); stair treads are small, so
    lower it (8-12) on maps with stairs. Columns next to the flattened floor that only failed the
    standable test because of a few noise voxels above it are grown into the floor (`grow` rings);
    their sparse noise voxels (<= spike_pts points, <= max_above per column) are removed. Interior
    holes are filled (`fill` rounds; a column with >= fill_nb floor neighbours and nothing standing
    on it becomes floor, so gaps left by sparse returns or flattening seams close instead of
    swallowing a foot). With fill_guard > 0 a column is neither grown nor filled when it has a solid
    voxel (> spike_pts points) between max(voxel, band_above) and fill_guard metres above the proposed height:
    that column has a surface of its own (a stair tread next to a landing) and must not be buried.
    Cells link into regions (and share the 5x5 median) when their top voxels lie within `link_height` of
    each other (one layer at 10 cm voxels). Growing and filling skip columns that already hold a ground cell; with `column_gap` > 0 only a cell
    within column_gap..2*column_gap of the proposed height blocks, so on a multi-storey map the floor under
    an upper storey can still be filled (default 0: any cell in the column blocks, as before).
    Clusters of <= float_max_voxels voxels floating within float_height above the floor without
    touching anything else are dropped as noise, as are sparse (<= sheet_pts points) voxels in the
    first layer above the band with nothing on top (the upper tail of the floor noise).
    """
    v = grid.voxel
    k_floor = int(np.ceil(floor_tol / v)) + 1
    # neighbouring cells link (regions, median) when their top voxels are within link_height: one layer at
    # 10 cm voxels, where this was tuned; at 3 cm the floor scatter alone puts neighbours 2-3 layers apart
    k_link = max(1, int(np.ceil(link_height / v)))
    link_dz = range(-k_link, k_link + 1)
    codes = grid.codes                      # grid keys are sorted by code
    keys = grid.keys.astype(np.int64)
    ck = keys[stand]
    C = len(ck)
    empty = Ground(np.zeros(len(keys), bool), ck[:0], np.zeros(0), np.zeros((0, 8, 3), np.float32), {"cells": 0})
    if C == 0:
        return empty

    # 1) surface height per cell: count-weighted mean (or median) z of the top voxel and `layers` below
    if surface not in ("mean", "median"):
        raise ValueError(f"surface={surface!r}: expected 'mean' or 'median'")
    zl = np.zeros((C, layers + 1))
    wl = np.zeros((C, layers + 1))
    for dz in range(0, layers + 1):
        i = _lookup(codes, ck - np.array([0, 0, dz]), grid.dims)
        f = i >= 0
        wl[f, dz] = grid.counts[i[f]]
        zl[f, dz] = grid.zmean[i[f]]
    h_raw = (zl * wl).sum(axis=1) / wl.sum(axis=1) if surface == "mean" else _weighted_median(zl, wl)

    # 2) regions: 8-neighbour cells (+-1 voxel layer) whose heights differ by < tol
    ccodes = encode(ck, grid.dims)          # sorted, since ck is a subset of sorted keys
    rows, cols = [], []
    for dx, dy in _XY8:
        for dz in link_dz:
            j = _lookup(ccodes, ck + np.array([dx, dy, dz]), grid.dims)
            f = j >= 0
            i = np.flatnonzero(f)
            j = j[f]
            close = np.abs(h_raw[i] - h_raw[j]) < tol
            rows.append(i[close])
            cols.append(j[close])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(C, C))
    _, label = connected_components(adj, directed=False)
    sizes = np.bincount(label)
    keep = sizes[label] >= min_cells         # tiny regions are clutter tops: leave them as voxels
    ck, h_raw, label, ccodes = ck[keep], h_raw[keep], label[keep], ccodes[keep]
    C = len(ck)
    if C == 0:
        return empty
    xy = grid.origin[:2] + (ck[:, :2] + 0.5) * v

    # 3) planes per region, else per tile; leftovers get a region-local median
    h = h_raw.copy()
    group = np.full(C, -1, dtype=np.int64)  # plane group id (planar cells), -1 = not planar
    planes = []
    rms_all = []

    def regions_of(lab):
        order = np.argsort(lab, kind="stable")
        bounds = np.flatnonzero(np.r_[True, lab[order][1:] != lab[order][:-1], True])
        return [order[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]

    if peel > 0:                             # 2b) split non-planar regions: dominant plane + the rest
        new = np.cumsum(keep) - 1
        e = keep[rows] & keep[cols]
        krows, kcols = new[rows[e]], new[cols[e]]
        next_label = int(label.max()) + 1
        stack = [(idx, 0) for idx in regions_of(label)]
        while stack:
            idx, depth = stack.pop()
            if len(idx) < min_cells or depth >= peel:
                continue
            c0 = xy[idx].mean(axis=0)
            _, rms, inl = _fit_plane(xy[idx, 0] - c0[0], xy[idx, 1] - c0[1], h_raw[idx])
            if (rms < max_rms and inl.mean() > 0.95) or inl.sum() < min_cells:
                continue
            label[idx[inl]] = next_label
            next_label += 1
            for comp in _components(C, krows, kcols, idx[~inl]):
                label[comp] = next_label
                next_label += 1
                stack.append((comp, depth + 1))

    def fit_groups(ids_per_group):
        for idx in ids_per_group:
            if len(idx) < min_cells:
                continue
            c0 = xy[idx].mean(axis=0)
            if small_flat and len(idx) <= small_flat:        # a tread: horizontal at the median
                coef = np.array([0.0, 0.0, float(np.median(h_raw[idx]))])
                r = h_raw[idx] - coef[2]
                rms = float(np.sqrt(np.mean(r * r)))
                group[idx] = len(planes)
                planes.append((coef, c0))
                rms_all.append((rms, len(idx)))
                continue
            coef, rms, _ = _fit_plane(xy[idx, 0] - c0[0], xy[idx, 1] - c0[1], h_raw[idx])
            if rms < max_rms:
                group[idx] = len(planes)
                planes.append((coef, c0))
                rms_all.append((rms, len(idx)))

    fit_groups(regions_of(label))
    rest = np.flatnonzero(group < 0)
    if len(rest):
        t = np.floor(xy[rest] / tile).astype(np.int64)
        tkey = (label[rest].astype(np.int64) * 1_000_003 + t[:, 0]) * 1_000_003 + t[:, 1]
        o = np.argsort(tkey, kind="stable")
        b = np.flatnonzero(np.r_[True, tkey[o][1:] != tkey[o][:-1], True])
        fit_groups(rest[o[b[k]:b[k + 1]]] for k in range(len(b) - 1))
    planar = group >= 0
    for g, (coef, c0) in enumerate(planes):
        idx = np.flatnonzero(group == g)
        h[idx] = coef[0] * (xy[idx, 0] - c0[0]) + coef[1] * (xy[idx, 1] - c0[1]) + coef[2]
    rest = np.flatnonzero(~planar)
    if len(rest):                            # 5x5 median within the same region, keeps edges sharp
        nb = np.full((len(rest), 25 * len(link_dz)), np.nan)
        col = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in link_dz:
                    j = _lookup(ccodes, ck[rest] + np.array([dx, dy, dz]), grid.dims)
                    f = j >= 0
                    f[f] = label[j[f]] == label[rest[f]]
                    nb[f, col] = h_raw[j[f]]
                    col += 1
        h[rest] = np.nanmedian(nb, axis=1)

    raw_std = float(np.std(h_raw - h))
    # 3b) grow into neighbouring floor columns that a few noise voxels above made "not standable"
    #     (floor voxels near the plane height, at most `max_above` voxels up to `clearance` above),
    #     then fill small interior holes (>= fill_nb of 8 neighbours are floor, no floor points needed)
    zc_all = grid.origin[2] + (keys[:, 2] + 0.5) * v
    n_above_k = int(np.ceil(clearance / v)) + 1
    spikes = np.zeros(len(keys), dtype=bool)
    n_grown = n_filled = n_guarded = 0
    frontier = np.arange(C)
    # a, b, c, x0, y0 per plane, indexed once per round (a Python loop over every proposal took minutes
    # per round on a 0.03 m map with millions of cells)
    plane_arr = np.array([np.r_[coef, c0] for coef, c0 in planes], dtype=float).reshape(-1, 5)
    for rnd in range(grow + fill):
        filling = rnd >= grow
        base = np.arange(C) if filling else frontier
        if len(base) == 0:
            continue
        cxy = (ck[base, None, :2] + np.array(_XY8)[None]).reshape(-1, 2)
        src = np.repeat(base, len(_XY8))
        cc = cxy[:, 0] * int(grid.dims[1]) + cxy[:, 1]
        inb = np.all((cxy >= 0) & (cxy < grid.dims[:2]), axis=1)
        cxy, src, cc = cxy[inb], src[inb], cc[inb]
        g = group[src]
        hc = h[src].copy()                             # proposed height: the source cell's plane, or its height
        pl = g >= 0
        if pl.any():
            p = plane_arr[g[pl]]                       # a, b, c, x0, y0
            px = grid.origin[0] + (cxy[pl, 0] + 0.5) * v
            py = grid.origin[1] + (cxy[pl, 1] + 0.5) * v
            hc[pl] = p[:, 0] * (px - p[:, 3]) + p[:, 1] * (py - p[:, 4]) + p[:, 2]
        if column_gap > 0:
            # multi-storey: only a ground cell of the same storey blocks the column. Two height bands of
            # 2 * column_gap, offset by half a band, so cells within column_gap always collide and cells
            # further than 2 * column_gap never do.
            band = 2.0 * column_gap
            ccol = ck[:, 0] * int(grid.dims[1]) + ck[:, 1]
            k1 = np.floor(h / band).astype(np.int64)
            k2 = np.floor((h + 0.5 * band) / band).astype(np.int64)
            taken = np.unique(np.r_[(ccol * 4_000_000 + k1) * 2, (ccol * 4_000_000 + k2) * 2 + 1])
            q1 = (cc * 4_000_000 + np.floor(hc / band).astype(np.int64)) * 2
            q2 = (cc * 4_000_000 + np.floor((hc + 0.5 * band) / band).astype(np.int64)) * 2 + 1
            new = ~_in_sorted(taken, q1) & ~_in_sorted(taken, q2)
            dedupe = cc * 4_000_000 + np.floor(hc / band).astype(np.int64)     # one proposal per column & band
        else:
            col_codes = np.unique(ck[:, 0] * int(grid.dims[1]) + ck[:, 1])
            pos = np.clip(np.searchsorted(col_codes, cc), 0, len(col_codes) - 1)
            new = col_codes[pos] != cc                 # column has no ground cell yet (any storey)
            dedupe = cc
        cxy, src, cc, hc, g, dedupe = cxy[new], src[new], cc[new], hc[new], g[new], dedupe[new]
        _, first, n_nb = np.unique(dedupe, return_index=True, return_counts=True)
        cxy, src, hc, g = cxy[first], src[first], hc[first], g[first]
        if len(cxy) == 0:
            continue
        kz = np.floor((hc - grid.origin[2]) / v).astype(np.int64)
        has_floor = np.zeros(len(cxy), dtype=bool)
        above = np.zeros(len(cxy), dtype=np.int64)
        solid = np.zeros(len(cxy), dtype=bool)
        for dz in range(-k_floor, n_above_k + 1):
            i = _lookup(codes, np.column_stack([cxy, kz + dz]), grid.dims)
            f = i >= 0
            zc = np.full(len(cxy), np.nan)
            zc[f] = zc_all[i[f]]
            has_floor |= f & (np.abs(zc - hc) <= floor_tol)
            above += f & (zc > hc + band_above) & (zc <= hc + clearance)
            if fill_guard > 0:
                cnt = np.zeros(len(cxy), dtype=np.int64)
                cnt[f] = grid.counts[i[f]]
                # the guard band starts above the floor noise band (band_above; one voxel at 10 cm, where
                # it was tuned), not one voxel up: at 3 cm voxels the floor's own scatter sits 1-2 voxels up
                solid |= f & (zc > hc + max(v, band_above)) & (zc <= hc + fill_guard) & (cnt > spike_pts)
        ok = (n_nb >= fill_nb) if filling else has_floor
        ok &= above <= max_above
        if fill_guard > 0:
            n_guarded += int((ok & solid).sum())
            ok &= ~solid
        if not ok.any():
            frontier = frontier[:0]
            continue
        new_keys = np.column_stack([cxy[ok], kz[ok]])
        new_idx = np.arange(C, C + int(ok.sum()))
        ck = np.vstack([ck, new_keys])
        h = np.r_[h, hc[ok]]
        h_raw = np.r_[h_raw, hc[ok]]
        group = np.r_[group, g[ok]]
        label = np.r_[label, label[src[ok]]]
        planar = group >= 0
        C = len(ck)
        if filling:
            n_filled += int(ok.sum())
        else:
            n_grown += int(ok.sum())
        frontier = new_idx
        # sparse voxels above the band in these columns are SLAM noise (a real object has more)
        for dz in range(1, n_above_k + 1):
            i = _lookup(codes, new_keys + np.array([0, 0, dz]), grid.dims)
            f = i >= 0
            zc = zc_all[i[f]]
            hn = hc[ok][f]
            noisy = (zc > hn + band_above) & (zc <= hn + clearance) & (grid.counts[i[f]] <= spike_pts)
            spikes[i[f][noisy]] = True

    # 4) remove floor voxels in [h - band_below, h + band_above] of every cell's column
    remove = spikes.copy()
    k_lo = int(np.ceil(band_below / v)) + 1
    k_hi = int(np.ceil(band_above / v)) + 1
    for dz in range(-k_lo, k_hi + 1):
        nk = ck + np.array([0, 0, dz])
        i = _lookup(codes, nk, grid.dims)
        f = i >= 0
        zc = grid.origin[2] + (nk[f, 2] + 0.5) * v
        inband = (zc >= h[f] - band_below) & (zc <= h[f] + band_above)
        remove[i[f][inband]] = True

    # 4b) small clusters left floating just above the floor (not touching walls/furniture) are noise.
    #     Search from below the cell voxel: a noise voxel can itself be the cell's top voxel (it made the
    #     column standable), with more floor noise just under it; zc > h + band_above does the filtering.
    low = np.zeros(len(keys), dtype=bool)
    for dz in range(-k_lo, int(np.ceil(float_height / v)) + 2):
        i = _lookup(codes, ck + np.array([0, 0, dz]), grid.dims)
        f = i >= 0
        zc = zc_all[i[f]]
        low[i[f][(zc > h[f] + band_above) & (zc <= h[f] + float_height)]] = True
    low &= ~remove
    li = np.flatnonzero(low)
    n_float = 0
    if len(li):
        lcodes = codes[li]                      # sorted (subset of sorted codes)
        attached = np.zeros(len(li), dtype=bool)
        rows, cols = [], []
        for off in _NB26:
            nb = _lookup(codes, keys[li] + off, grid.dims)
            f = nb >= 0
            f[f] = ~remove[nb[f]]
            j = _lookup(lcodes, keys[li] + off, grid.dims)
            attached |= f & (j < 0)             # touches a solid voxel that is not low clutter
            m = j >= 0
            rows.append(np.flatnonzero(m))
            cols.append(j[m])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        n_c, lab = connected_components(coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                                                   shape=(len(li), len(li))), directed=False)
        size = np.bincount(lab, minlength=n_c)
        anchored = np.bincount(lab, weights=attached, minlength=n_c) > 0
        drop = (size[lab] <= float_max_voxels) & ~anchored[lab]
        # the upper tail of the floor noise (points up to ~3 sigma above the plane) forms connected sheets
        # of 1-2 point voxels one layer above the band: drop those too unless something stands on them
        zl = zc_all[li]
        hl = np.full(len(li), np.nan)
        for dz in range(-k_lo, int(np.ceil(float_height / v)) + 2):     # plane height of each low voxel
            j = _lookup(codes, ck + np.array([0, 0, dz]), grid.dims)
            pos = np.searchsorted(li, j)
            hit = (j >= 0) & (pos < len(li))
            hit[hit] = li[pos[hit]] == j[hit]
            hl[pos[hit]] = h[hit]
        above = _lookup(codes, keys[li] + np.array([0, 0, 1]), grid.dims)
        covered = above >= 0
        covered[covered] = ~remove[above[covered]]
        sheet = (zl <= hl + band_above + v) & (grid.counts[li] <= sheet_pts) & ~covered
        drop |= sheet
        remove[li[drop]] = True
        n_float = int(drop.sum())

    # 5) slabs: planar groups run-merged into rectangles (sloped tops), other cells one slab each
    gid = np.where(planar, group, len(planes) + np.arange(C))
    r = np.stack([ck[:, 0], ck[:, 0], ck[:, 1], ck[:, 1], gid], axis=1)
    r = _merge_axis(r, group_cols=[2, 4], lo_col=0, hi_col=1)
    r = _merge_axis(r, group_cols=[0, 1, 4], lo_col=2, hi_col=3)
    flat_h = dict(zip(gid[~planar].tolist(), h[~planar].tolist()))
    x0 = grid.origin[0] + r[:, 0] * v
    x1 = grid.origin[0] + (r[:, 1] + 1) * v
    y0 = grid.origin[1] + r[:, 2] * v
    y1 = grid.origin[1] + (r[:, 3] + 1) * v
    slabs = np.zeros((len(r), 8, 3))
    for n, (sx, sy) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):   # corner = 4*sx + 2*sy + sz
        cx = np.where(sx, x1, x0)
        cy = np.where(sy, y1, y0)
        top = np.empty(len(r))
        is_plane = r[:, 4] < len(planes)
        if is_plane.any():
            coefs = np.array([planes[g][0] for g in r[is_plane, 4]])
            c0s = np.array([planes[g][1] for g in r[is_plane, 4]])
            top[is_plane] = coefs[:, 0] * (cx[is_plane] - c0s[:, 0]) + coefs[:, 1] * (cy[is_plane] - c0s[:, 1]) \
                + coefs[:, 2]
        if (~is_plane).any():
            top[~is_plane] = [flat_h[g] for g in r[~is_plane, 4]]
        ci = 4 * sx + 2 * sy
        slabs[:, ci] = np.stack([cx, cy, top - thickness], axis=1)
        slabs[:, ci + 1] = np.stack([cx, cy, top], axis=1)

    rms_w = sum(rm * n for rm, n in rms_all) / max(1, sum(n for _, n in rms_all))
    stats = {
        "cells": int(C),
        "regions": int(len(np.unique(label))),
        "planes": len(planes),
        "planar_fraction": round(float(planar.mean()), 4),
        "raw_surface_std_m": round(raw_std, 4),
        "plane_rms_m": round(float(rms_w), 4),
        "cells_grown": n_grown,
        "cells_filled": n_filled,
        "cells_guarded": n_guarded,
        "noise_voxels_removed": int(spikes.sum()),
        "floating_voxels_removed": n_float,
        "voxels_removed": int(remove.sum()),
        "slabs": int(len(slabs)),
    }
    return Ground(remove, ck, h, slabs.astype(np.float32), stats)
