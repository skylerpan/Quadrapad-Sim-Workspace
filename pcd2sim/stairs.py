"""Draft a make_stairs spec from a scene's floor: find the flights and landings and describe them.

The floor cells (pcd2sim.geometry.FloorCells: the scene's standable voxels, or its slab cells) are coarsened
to a 0.1 m grid. Every cell gets the gradient of a plane fitted to the floor within 0.3 m of it. Cells whose
floor climbs between `slope` = 0.3 and 1.2 (stairs are 0.5-0.7: a 0.16 m riser per 0.25-0.3 m tread) and
whose neighbours climb the same way (direction within 35 deg, |dz| <= 0.3) form one *flight*, whether the
scan shows it as steps, as 0.1/0.2 m quantised levels (a 1:1 octree) or as a ramp (a flattened floor that
merged the treads). The flight axis is the mean gradient direction, snapped to x or y when within
`snap_deg`; the climb is a straight line fitted to height against distance along the axis. z0 is the floor
level found just before the foot and z1 the level just past the head (a rise below / above the fitted
ends when none is found); the number of steps is round((z1 - z0) / riser), as make_stairs builds it. The
line gives the foot (where it passes z0 + rise/2) and the head (z1 + rise/2); the width is the across
extent of the flight cells. Landings are the floor levels at the ends grown to their full extent, as
axis-aligned boxes, unless they are bigger than 25 m2 (open floor). Flights sharing a landing or ending
within 2.5 m of each other are written as one staircase. Every number comes with its evidence as a YAML
comment and weak spots are flagged CHECK: a human confirms the draft before make_stairs runs on it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import FloorCells
from .voxel import VoxelGrid

COARSE = 0.1
_XY8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def _grid_like(cells: FloorCells, origin=None, size=None, dims=None) -> VoxelGrid:
    return VoxelGrid(np.zeros((0, 3), np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32),
                     np.zeros(0, np.float32), cells.origin if origin is None else origin,
                     cells.v if size is None else size, cells.dims if dims is None else dims)


def coarsen(cells: FloorCells, size: float = COARSE) -> FloorCells:
    """Floor cells on a coarser xy grid: one cell per (column, 5 cm height bin) with the mean height."""
    if cells.v >= size - 1e-9:
        return cells
    xy = cells.xy()
    o = cells.origin
    cx = np.floor((xy[:, 0] - o[0]) / size).astype(np.int64)
    cy = np.floor((xy[:, 1] - o[1]) / size).astype(np.int64)
    dims = np.array([max(int(np.ceil(cells.dims[0] * cells.v / size)), int(cx.max()) + 1) + 2,
                     max(int(np.ceil(cells.dims[1] * cells.v / size)), int(cy.max()) + 1) + 2,
                     int(np.ceil(cells.dims[2] * cells.v / size)) + 2])
    col = cx * int(dims[1]) + cy
    order = np.lexsort((cells.z, col))
    col_s, z_s = col[order], cells.z[order]
    # one coarse cell per run of heights within `size` of each other in a column (a tread edge crossing a
    # coarse cell gives one cell at the mean height, not two)
    new = np.r_[True, (col_s[1:] != col_s[:-1]) | (np.diff(z_s) > size)]
    cid = np.cumsum(new) - 1
    n = int(cid[-1]) + 1 if len(cid) else 0
    z = np.bincount(cid, weights=z_s, minlength=n) / np.bincount(cid, minlength=n)
    first = np.flatnonzero(new)
    cxy = np.column_stack([cx[order][first], cy[order][first]])
    kz = np.floor((z - o[2]) / size - 1e-6).astype(np.int64) - 1
    return FloorCells(_grid_like(cells, size=size, dims=dims), np.column_stack([cxy, kz]), z)


def _neighbours(cells: FloorCells, radius_cells: int, dz_max: float):
    """(i, j) index pairs of cells within radius_cells in xy (excluding self) with |dz| <= dz_max, plus the
    xy offset of j relative to i in metres."""
    depth = int(np.max(np.bincount(cells.col - cells.col.min()))) if cells.n else 0
    ii, jj, off = [], [], []
    r = radius_cells
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if (dx == 0 and dy == 0) or dx * dx + dy * dy > r * r:
                continue
            nc = (cells.k[:, 0] + dx) * int(cells.dims[1]) + (cells.k[:, 1] + dy)
            lo, hi = cells.column_range(nc)
            for s in range(depth):
                j = lo + s
                f = j < hi
                i = np.flatnonzero(f)
                j = j[f]
                ok = np.abs(cells.z[j] - cells.z[i]) <= dz_max
                ii.append(i[ok])
                jj.append(j[ok])
                off.append(np.tile([dx * cells.v, dy * cells.v], (int(ok.sum()), 1)))
    if not ii:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(off)


def local_gradient(cells: FloorCells, radius: float = 0.3, dz_max: float = 0.35, min_nb: int = 6):
    """Per cell: gradient (gx, gy) of the least-squares plane through the floor cells within `radius`
    (|dz| <= dz_max), NaN with fewer than min_nb neighbours."""
    i, j, off = _neighbours(cells, int(np.ceil(radius / cells.v)), dz_max)
    dz = cells.z[j] - cells.z[i]
    n = cells.n
    sxx = np.bincount(i, weights=off[:, 0] ** 2, minlength=n)
    sxy = np.bincount(i, weights=off[:, 0] * off[:, 1], minlength=n)
    syy = np.bincount(i, weights=off[:, 1] ** 2, minlength=n)
    sxz = np.bincount(i, weights=off[:, 0] * dz, minlength=n)
    syz = np.bincount(i, weights=off[:, 1] * dz, minlength=n)
    cnt = np.bincount(i, minlength=n)
    det = sxx * syy - sxy * sxy
    ok = (cnt >= min_nb) & (np.abs(det) > 1e-9)
    g = np.full((n, 2), np.nan)
    g[ok, 0] = (syy[ok] * sxz[ok] - sxy[ok] * syz[ok]) / det[ok]
    g[ok, 1] = (sxx[ok] * syz[ok] - sxy[ok] * sxz[ok]) / det[ok]
    return g


def flight_components(cells: FloorCells, g: np.ndarray, slope: tuple[float, float] = (0.3, 1.2),
                      angle_deg: float = 35.0, dz_max: float = 0.3) -> np.ndarray:
    """Component label per cell (-1 outside): climbing cells joined over 8-neighbours that climb the same way."""
    mag = np.hypot(g[:, 0], g[:, 1])
    mask = np.isfinite(mag) & (mag >= slope[0]) & (mag <= slope[1])
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.full(cells.n, -1, dtype=np.int64)
    sub = FloorCells(_grid_like(cells), cells.k[idx], cells.z[idx])
    order = np.lexsort((cells.k[idx, 2], cells.col[idx]))          # sub[i] == cells[idx[order[i]]]
    gsub = g[idx[order]]
    theta = np.arctan2(gsub[:, 1], gsub[:, 0])
    i, j, _ = _neighbours(sub, 1, dz_max)
    d = np.abs((theta[i] - theta[j] + np.pi) % (2 * np.pi) - np.pi)
    keep = d <= math.radians(angle_deg)
    n, lab = connected_components(coo_matrix((np.ones(int(keep.sum()), np.int8), (i[keep], j[keep])),
                                             shape=(sub.n,) * 2), directed=False)
    out = np.full(cells.n, -1, dtype=np.int64)
    out[idx[order]] = lab
    return out


def grow_level(cells: FloorCells, seeds: np.ndarray, z: float, tol: float = 0.05, radius: float = 3.0) -> np.ndarray:
    """Indices of the cells at height z (+-tol) connected to `seeds` over 8-neighbours, grown ring by ring up
    to `radius` metres: the full extent of a landing, capped so an open floor does not swallow the map."""
    rings = int(np.ceil(radius / cells.v))
    depth = int(np.max(np.bincount(cells.col - cells.col.min()))) if cells.n else 0
    seen = np.zeros(cells.n, dtype=bool)
    seen[seeds] = True
    frontier = np.asarray(seeds, dtype=np.int64)
    for _ in range(rings):
        new = []
        for dx, dy in _XY8:
            nc = (cells.k[frontier, 0] + dx) * int(cells.dims[1]) + (cells.k[frontier, 1] + dy)
            lo, hi = cells.column_range(nc)
            for s in range(depth):
                j = lo + s
                j = j[j < hi]
                ok = (np.abs(cells.z[j] - z) <= tol) & ~seen[j]
                new.append(j[ok])
        frontier = np.unique(np.concatenate(new)) if new else np.zeros(0, np.int64)
        if len(frontier) == 0:
            break
        seen[frontier] = True
    return np.flatnonzero(seen)


def _level_near(cells: FloorCells, flat: np.ndarray, p: np.ndarray, u: np.ndarray, w: np.ndarray,
                along: tuple[float, float], half_width: float, z_lo: float, z_hi: float) -> np.ndarray:
    """Indices of flat cells (not climbing) in the box [along] metres from p along u, within half_width
    across, z in [z_lo, z_hi]: the floor or landing level at a flight end."""
    rel = cells.xy() - p
    s, t = rel @ u, rel @ w
    m = flat & (s >= along[0]) & (s <= along[1]) & (np.abs(t) <= half_width) & (cells.z >= z_lo) & (cells.z <= z_hi)
    return np.flatnonzero(m)


def _flight(cells: FloorCells, idx: np.ndarray, g: np.ndarray, flat: np.ndarray, snap_deg: float,
            riser: float) -> dict | None:
    v = cells.v
    # trim the component to the cells that climb like its core: the 0.3 m gradient window reaches one cell
    # past the risers into the floor and the landing, and sideways onto the floor beside the flight
    gi = g[idx]
    mag = np.hypot(gi[:, 0], gi[:, 1])
    u0 = np.nanmean(gi, axis=0)
    u0 /= max(1e-9, np.hypot(*u0))
    along = gi @ u0
    core = (mag >= 0.7 * np.nanmedian(mag)) & (along >= math.cos(math.radians(20)) * mag)
    idx_all = idx
    if core.sum() >= 6:
        idx = idx[core]
        gi = gi[core]
    xy = cells.xy(idx)
    z = cells.z[idx]
    u = np.nanmean(gi, axis=0)
    u /= max(1e-9, np.hypot(*u))
    yaw = math.degrees(math.atan2(u[1], u[0]))
    snapped = False
    for ax_yaw in (0.0, 90.0, 180.0, -90.0, -180.0):
        if abs((yaw - ax_yaw + 180) % 360 - 180) <= snap_deg:
            u = np.array([math.cos(math.radians(ax_yaw)), math.sin(math.radians(ax_yaw))])
            yaw = ax_yaw if ax_yaw != -180.0 else 180.0
            snapped = True
            break
    w = np.array([-u[1], u[0]])
    s, t = xy @ u, xy @ w
    A = np.column_stack([s, np.ones(len(s))])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    slope, icpt = float(coef[0]), float(coef[1])
    if slope <= 0.2:
        return None
    rms = float(np.sqrt(np.mean((A @ coef - z) ** 2)))
    s_lo, s_hi = float(np.percentile(s, 2)), float(np.percentile(s, 98))
    t_c = float(np.median(t))
    # width across the whole component, but only cells on the climb (the floor beside an open-sided flight
    # is far below the fitted line and drops out)
    xa, za = cells.xy(idx_all), cells.z[idx_all]
    sa, ta = xa @ u, xa @ w
    on = np.abs(za - (icpt + slope * sa)) <= 0.15
    ta = ta[on] if on.sum() >= 4 else t
    width = float(np.percentile(ta, 98) - np.percentile(ta, 2)) + v
    z_lo, z_hi = icpt + slope * s_lo, icpt + slope * s_hi             # fitted heights at the ends
    p_lo, p_hi = u * s_lo + w * t_c, u * s_hi + w * t_c
    notes = []
    foot = _level_near(cells, flat, p_lo, u, w, (-0.8, 0.25), width / 2 + 0.15, z_lo - 0.45, z_lo + 0.1)
    if len(foot) >= 3:
        z0 = float(np.median(cells.z[foot]))
    else:
        z0 = z_lo - riser / 2
        notes.append(f"no floor found before the foot: z0 = fitted start height - {riser / 2:.2f}")
    head = _level_near(cells, flat, p_hi, u, w, (-0.25, 0.8), width / 2 + 0.15, z_hi - 0.15, z_hi + 0.45)
    if len(head) >= 3:
        z1 = float(np.median(cells.z[head]))
    else:
        z1 = z_hi + riser / 2
        notes.append(f"no landing found past the head: z1 = fitted end height + {riser / 2:.2f}")
    if z1 - z0 < 0.3:
        return None
    n = max(1, int(round((z1 - z0) / riser)))
    rise = (z1 - z0) / n
    s0 = (z0 + rise / 2 - icpt) / slope                               # the line passes z0 + rise/2 at the foot
    s1 = (z1 + rise / 2 - icpt) / slope
    length = s1 - s0
    tread = length / n
    start, end = u * s0 + w * t_c, u * s1 + w * t_c
    if rms > 0.08:
        notes.append(f"climb fit rms {rms:.2f} m")
    if not snapped:
        notes.append(f"not axis-aligned ({yaw:.1f} deg): make_stairs falls back to raster boxes")
    if tread < 0.2 or tread > 0.45:
        notes.append(f"tread {tread:.2f} m is unusual for {n} steps of {rise:.3f} m")
    if len(idx) * v * v < 1.0:
        notes.append(f"small flight ({len(idx) * v * v:.1f} m2 of climbing floor)")
    return {"start": [round(float(start[0]), 2), round(float(start[1]), 2)],
            "end": [round(float(end[0]), 2), round(float(end[1]), 2)], "width": round(width, 2),
            "z0": round(z0, 3), "z1": round(z1, 3), "steps": n, "rise": round(rise, 4), "tread": round(tread, 3),
            "slope": round(slope, 3), "fit_rms": round(rms, 3), "yaw_deg": round(yaw, 1),
            "cells": int(len(idx)), "notes": notes, "_foot": foot, "_head": head, "_p": (start, end)}


def _merge_collinear(cells: FloorCells, flights: list[dict], riser: float) -> list[dict]:
    """Join two flights that are one flight broken by a scan hole: same axis, across offset < 0.3 m, the
    second starts (z and position) within 0.25 m / 0.8 m of where the first ends. Repeats until stable."""
    changed = True
    while changed:
        changed = False
        for a in range(len(flights)):
            for b in range(len(flights)):
                if a == b:
                    continue
                fa, fb = flights[a], flights[b]
                if fa["yaw_deg"] != fb["yaw_deg"]:
                    continue
                u = np.array([math.cos(math.radians(fa["yaw_deg"])), math.sin(math.radians(fa["yaw_deg"]))])
                w = np.array([-u[1], u[0]])
                ea, sb = np.array(fa["end"]), np.array(fb["start"])
                if abs((sb - ea) @ w) > 0.3 or not (-0.3 <= (sb - ea) @ u <= 0.8) or abs(fb["z0"] - fa["z1"]) > 0.25:
                    continue
                z0, z1 = fa["z0"], fb["z1"]
                n = max(1, int(round((z1 - z0) / riser)))
                start, end = np.array(fa["start"]), np.array(fb["end"])
                t_c = 0.5 * ((start @ w) + (end @ w))
                start, end = u * (start @ u) + w * t_c, u * (end @ u) + w * t_c
                length = float((end - start) @ u)
                merged = dict(fa)
                merged.update(start=[round(float(start[0]), 2), round(float(start[1]), 2)],
                              end=[round(float(end[0]), 2), round(float(end[1]), 2)],
                              width=round(0.5 * (fa["width"] + fb["width"]), 2), z0=z0, z1=z1, steps=n,
                              rise=round((z1 - z0) / n, 4), tread=round(length / n, 3),
                              cells=fa["cells"] + fb["cells"], _head=fb["_head"], _p=(start, end),
                              notes=[n_ for n_ in fa["notes"] if "landing" not in n_] +
                                    [n_ for n_ in fb["notes"] if "floor" not in n_] +
                                    [f"merged two pieces (break at z {fa['z1']:.2f})"])
                flights = [f for k, f in enumerate(flights) if k not in (a, b)] + [merged]
                changed = True
                break
            if changed:
                break
    return flights


def _overlap(a: list, b: list) -> float:
    """Intersection over the smaller rect's area, rects as [xmin, xmax, ymin, ymax]."""
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    small = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
    return ix * iy / small if small > 0 else 0.0


def _landing(cells: FloorCells, seeds: np.ndarray, z: float, max_m2: float = 25.0) -> dict | None:
    if len(seeds) == 0:
        return None
    grown = grow_level(cells, seeds, z)
    v = cells.v
    area = len(grown) * v * v
    if area > max_m2:
        return None
    p = cells.xy(grown)
    return {"z": round(float(np.median(cells.z[grown])), 3),
            "rect": [round(float(p[:, 0].min() - v / 2), 2), round(float(p[:, 0].max() + v / 2), 2),
                     round(float(p[:, 1].min() - v / 2), 2), round(float(p[:, 1].max() + v / 2), 2)],
            "cells": int(len(grown)), "m2": round(area, 2)}


def draft(cells: FloorCells, riser: float = 0.16, snap_deg: float = 12.0, min_m2: float = 0.4,
          min_climb: float = 0.3, group_dist: float = 2.5) -> list[dict]:
    """Staircase drafts: [{name, flights: [...], landings: [...]}] in make_stairs spec terms."""
    cc = coarsen(cells)
    v = cc.v
    g = local_gradient(cc)
    lab = flight_components(cc, g)
    mag = np.hypot(g[:, 0], g[:, 1])
    flat = (lab < 0) & (~np.isfinite(mag) | (mag < 0.2))            # floor / landing cells
    flights = []
    for c in range(lab.max() + 1 if lab.size else 0):
        idx = np.flatnonzero(lab == c)
        if len(idx) * v * v < min_m2 or cc.z[idx].max() - cc.z[idx].min() < min_climb:
            continue
        f = _flight(cc, idx, g, flat, snap_deg, riser)
        if f is not None:
            flights.append(f)
    if not flights:
        return []
    flights = _merge_collinear(cc, flights, riser)
    landings = []
    for f in flights:
        f["_landings"] = []
        for seeds, z in ((f["_foot"], f["z0"]), (f["_head"], f["z1"])):
            l = _landing(cc, seeds, z)
            if l is None:
                continue
            for k, other in enumerate(landings):
                if abs(other["z"] - l["z"]) <= 0.12 and _overlap(other["rect"], l["rect"]) >= 0.5:
                    if l["m2"] > other["m2"]:                     # same landing seen from another flight end
                        landings[k] = l
                    f["_landings"].append(k)
                    break
            else:
                landings.append(l)
                f["_landings"].append(len(landings) - 1)
    parent = list(range(len(flights)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(flights)):
        for b in range(a + 1, len(flights)):
            fa, fb = flights[a], flights[b]
            shared = set(fa["_landings"]) & set(fb["_landings"])
            near = min(np.hypot(*(pa - pb)) for pa in fa["_p"] for pb in fb["_p"]) <= group_dist
            if shared or near:
                parent[find(a)] = find(b)
    groups = {}
    for i in range(len(flights)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        members.sort(key=lambda i: flights[i]["z0"])
        fl, ld, seen = [], [], set()
        for n, i in enumerate(members):
            f = dict(flights[i])
            for k in f["_landings"]:
                if k not in seen:
                    seen.add(k)
                    ld.append(dict(landings[k], name=f"L{len(ld) + 1}"))
            for key in [k for k in f if k.startswith("_")]:
                del f[key]
            f["name"] = f"F{n + 1}"
            fl.append(f)
        pts = np.array([p for i in members for p in flights[i]["_p"]])
        out.append({"name": "", "x": round(float(pts[:, 0].mean()), 1), "y": round(float(pts[:, 1].mean()), 1),
                    "z0": min(f["z0"] for f in fl), "z1": max(f["z1"] for f in fl), "flights": fl, "landings": ld})
    out.sort(key=lambda st: -(st["z1"] - st["z0"]))
    for n, st in enumerate(out):
        st["name"] = f"stairs_{n + 1}"
    return out


def to_yaml(drafts: list[dict], scene: str, riser: float = 0.16, thickness: float = 0.30,
            clear=(0.35, 1.0)) -> str:
    """make_stairs spec text with the evidence as comments. Flights with notes are marked CHECK."""
    lines = [f"# DRAFT stair spec from {scene} (scripts/make_stairs.py --draft). Check every flight against the",
             "# map before running make_stairs: footprints are 0.1 m estimates from the scanned floor, z0 / z1 are",
             "# the floor levels found at the foot and the head. Flights without a CHECK still deserve a look.",
             f"riser: {riser}", f"thickness: {thickness}", f"clear: [{clear[0]}, {clear[1]}]", "", "stairs:"]
    for st in drafts:
        lines.append(f"  - name: {st['name']}    # around x {st['x']} y {st['y']}, {st['z0']} -> {st['z1']}")
        lines.append("    flights:")
        for f in st["flights"]:
            tag = "  # CHECK: " + "; ".join(f["notes"]) if f["notes"] else ""
            lines.append(f"      - {{name: {f['name']}, start: [{f['start'][0]}, {f['start'][1]}], "
                         f"end: [{f['end'][0]}, {f['end'][1]}], width: {f['width']}, z0: {f['z0']}, z1: {f['z1']}}}"
                         f"   # {f['steps']} x {f['rise']} m, tread {f['tread']} m, yaw {f['yaw_deg']} deg, "
                         f"slope {f['slope']} (fit rms {f['fit_rms']}), {f['cells']} cells{tag}")
        if st["landings"]:
            lines.append("    landings:")
            for l in st["landings"]:
                lines.append(f"      - {{name: {l['name']}, rect: [{l['rect'][0]}, {l['rect'][1]}, {l['rect'][2]}, "
                             f"{l['rect'][3]}], z: {l['z']}}}   # {l['m2']} m2, {l['cells']} cells")
        else:
            lines.append("    # landings: none within 25 m2 (the flight ends on the open floor); add an approach "
                         "pad if the floor there is noisy")
    if not drafts:
        lines.append("  []   # no staircase found")
    return "\n".join(lines) + "\n"
