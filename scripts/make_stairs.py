"""Replace scanned staircases in a built scene with clean synthetic steps (flat treads, solid risers).

Usage (inside the container):
  python3 scripts/make_stairs.py --scene scenes/dp_full_003 --out scenes/dp_full_003_s16 --spec config/stairs_dp_full.yaml
  python3 scripts/make_stairs.py --scene scenes/dp_full_003_s16 --spec config/stairs_dp_full.yaml --map east   # inspect only

The spec (YAML) lists staircases made of flights and landings:
  riser: 0.16            target riser; a flight with rise R gets n = round(R / riser) steps of R / n
  thickness: 0.30        solid depth below every tread top (> riser, so the risers are solid)
  clear: [0.35, 1.0]     original geometry in [h - 0.35, h + 1.0] of the target height h is removed (boxes only when
                         they lie entirely inside the band, so wall columns survive); any element may override `clear`
  stairs:
    - name: east
      flights:  [{name, start: [x, y], end: [x, y], width, z0, z1}, ...]   axis from the foot to the head, any yaw
      landings: [{name, rect: [xmin, xmax, ymin, ymax], z}, ...]            horizontal, axis-aligned

Inside each footprint the scene boxes, floor slabs, voxels and map_ds points in the clearing band are dropped and the
steps are rasterised on the scene's own voxel grid, merged into axis-aligned boxes (pcd2sim.voxel.merge_boxes) and
written back to boxes.npy / voxels.npz / ground.npz / map_ds.pcd. Tread tops keep their exact height in voxels.npz
(zmean), so the ideal base sees the true riser even though the boxes are 3 cm-quantised.

Exact-box mode (--exact-boxes; the default for scenes without ground.npz, i.e. 1:1 octree builds made with
--no-flatten): a coarse grid (0.1-0.2 m) cannot carry 0.16 m risers in rasterised boxes, so every tread and landing
becomes one exact AABB [h - thickness, h] instead, and the original boxes are re-merged from the voxels that survive
the clearing (a 1:1 build's boxes.npy is exactly merge_boxes(voxels), so the footprint edges come out clean: no gaps
outside, no leftovers inside). The synthetic voxels are still written so VoxelLookup / map_ds stay consistent, with
the voxel under the terrain probe carrying the exact height. Flights that are not axis-aligned fall back to raster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pcd2sim.pcd_io import read_pcd, write_pcd  # noqa: E402
from pcd2sim.voxel import VoxelGrid, _merge_axis, boxes_to_world, encode, merge_boxes  # noqa: E402

SLAB_THICKNESS = 0.2


# -- spec ---------------------------------------------------------------------------------------------
class Element:
    """One flight or landing: a footprint in xy with a target top height h(x, y)."""

    def __init__(self, stair: str, spec: dict, riser: float, clear: tuple[float, float]):
        self.stair, self.name = stair, spec["name"]
        self.clear = tuple(float(v) for v in spec.get("clear", clear))   # per-element override of the band
        if "rect" in spec:                                   # landing
            self.kind = "landing"
            self.rect = [float(v) for v in spec["rect"]]
            self.z = float(spec["z"])
            self.n, self.rise = 1, 0.0
            self.z_lo = self.z_hi = self.z
        else:                                                # flight
            self.kind = "flight"
            self.start = np.asarray(spec["start"], dtype=np.float64)
            self.end = np.asarray(spec["end"], dtype=np.float64)
            self.width = float(spec["width"])
            self.z0, self.z1 = float(spec["z0"]), float(spec["z1"])
            d = self.end - self.start
            self.length = float(np.hypot(*d))
            if self.length <= 0 or self.z1 <= self.z0:
                raise ValueError(f"{self.name}: a flight needs end != start and z1 > z0")
            self.u = d / self.length                         # along the climb
            self.v = np.array([-self.u[1], self.u[0]])       # across
            self.n = max(1, int(round((self.z1 - self.z0) / riser)))
            self.rise = (self.z1 - self.z0) / self.n
            self.tread = self.length / self.n
            self.z_lo, self.z_hi = self.z0 + self.rise, self.z1
            self.yaw_deg = math.degrees(math.atan2(self.u[1], self.u[0]))

    def bbox(self) -> tuple[float, float, float, float]:
        if self.kind == "landing":
            return self.rect[0], self.rect[1], self.rect[2], self.rect[3]
        c = np.array([self.start, self.end, self.start, self.end]) + \
            np.array([self.v, self.v, -self.v, -self.v]) * (self.width / 2)
        return float(c[:, 0].min()), float(c[:, 0].max()), float(c[:, 1].min()), float(c[:, 1].max())

    def height(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(inside mask, target top height h, step index 1..n) for xy [N,2]; h / step are 0 outside."""
        if self.kind == "landing":
            x0, x1, y0, y1 = self.rect
            m = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            return m, np.where(m, self.z, 0.0), m.astype(np.int64)
        rel = xy - self.start
        s = rel @ self.u
        t = rel @ self.v
        m = (s >= 0.0) & (s < self.length) & (np.abs(t) <= self.width / 2)
        k = np.clip(np.floor(s / self.tread).astype(np.int64) + 1, 1, self.n)
        return m, np.where(m, self.z0 + k * self.rise, 0.0), np.where(m, k, 0)

    @property
    def axis_aligned(self) -> bool:
        return self.kind == "landing" or min(abs(self.u[0]), abs(self.u[1])) < 1e-9

    def exact_boxes(self, thickness: float) -> np.ndarray:
        """One AABB (cx, cy, cz, hx, hy, hz) per tread / landing, tops at the exact heights. Axis-aligned only."""
        if not self.axis_aligned:
            raise ValueError(f"{self.name}: exact boxes need an axis-aligned flight (yaw {self.yaw_deg:.1f} deg)")
        if self.kind == "landing":
            x0, x1, y0, y1 = self.rect
            return np.array([[0.5 * (x0 + x1), 0.5 * (y0 + y1), self.z - thickness / 2,
                              0.5 * (x1 - x0), 0.5 * (y1 - y0), thickness / 2]], dtype=np.float64)
        out = np.zeros((self.n, 6))
        for k in range(1, self.n + 1):
            a = self.start + self.u * ((k - 1) * self.tread)
            b = self.start + self.u * (k * self.tread)
            corners = np.array([a + self.v * (self.width / 2), a - self.v * (self.width / 2),
                                b + self.v * (self.width / 2), b - self.v * (self.width / 2)])
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            h = self.z0 + k * self.rise
            out[k - 1] = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), h - thickness / 2,
                          0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), thickness / 2]
        return out

    def describe(self) -> str:
        if self.kind == "landing":
            return f"{self.stair}/{self.name}: landing z={self.z:.3f} rect={self.rect}"
        return (f"{self.stair}/{self.name}: {self.n} steps, riser {self.rise:.4f} m, tread {self.tread:.3f} m, "
                f"width {self.width}, {self.z0:.2f} -> {self.z1:.2f}, yaw {self.yaw_deg:.1f} deg")


def load_spec(path: str) -> dict:
    with open(path) as f:
        spec = yaml.safe_load(f)
    spec.setdefault("riser", 0.16)
    spec.setdefault("thickness", 0.30)
    spec.setdefault("clear", [0.35, 1.0])
    clear = tuple(float(v) for v in spec["clear"])
    elements = []
    for st in spec["stairs"]:
        for fl in st.get("flights", []) or []:
            elements.append(Element(st["name"], fl, spec["riser"], clear))
        for ld in st.get("landings", []) or []:
            elements.append(Element(st["name"], ld, spec["riser"], clear))
    spec["elements"] = elements
    return spec


# -- scene I/O ----------------------------------------------------------------------------------------
def load_scene(scene: str):
    boxes = np.load(os.path.join(scene, "boxes.npy"))
    d = np.load(os.path.join(scene, "voxels.npz"))
    grid = VoxelGrid(d["keys"].astype(np.int32), np.ones(len(d["keys"]), np.int32), d["intensity"], d["zmean"],
                     d["origin"].astype(np.float64), float(d["voxel"]), d["dims"].astype(np.int64))
    gp = os.path.join(scene, "ground.npz")
    slabs = np.load(gp)["slabs"] if os.path.exists(gp) else None
    with open(os.path.join(scene, "meta.json")) as f:
        meta = json.load(f)
    return boxes, grid, slabs, meta


# -- geometry -----------------------------------------------------------------------------------------
def rasterize(el: Element, grid: VoxelGrid, thickness: float, exact: bool = False):
    """Voxel keys of the solid steps of one element: (keys [K,3] int64, is_top [K] bool, h [K] exact top, step [K]).
    Raster mode puts the top voxel's face on the grid plane nearest h (the boxes are merged from these voxels).
    Exact mode (the boxes are exact AABBs with their top at h) gives every voxel that reaches into [h - v/2, h] the
    exact zmean h, so the terrain probe at h - v/4 (quadsim.base: the ray hit is probed a quarter voxel down) finds
    the exact height whichever cell it rounds into."""
    v, o = grid.voxel, grid.origin
    x0, x1, y0, y1 = el.bbox()
    ix = np.arange(int(np.floor((x0 - o[0]) / v)), int(np.ceil((x1 - o[0]) / v)) + 1)
    iy = np.arange(int(np.floor((y0 - o[1]) / v)), int(np.ceil((y1 - o[1]) / v)) + 1)
    gx, gy = np.meshgrid(ix, iy, indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    xy = o[:2] + (cells + 0.5) * v
    m, h, step = el.height(xy)
    cells, h, step = cells[m], h[m], step[m]
    if exact:
        # column from the cell that contains h (h exactly on a face -> the cell below it) down through the
        # thickness; every cell that reaches into [h - v/2, h] is a "top" cell (zmean = h), so the probe at
        # h - v/4 finds the exact height even when it sits on a cell face and rounds either way
        kz_top = np.floor((h - o[2]) / v - 1e-9).astype(np.int64)
        kz_lo = np.floor((h - 0.5 * v - o[2]) / v).astype(np.int64)
        n_layers = int(np.ceil(thickness / v)) + 1
        layers = np.arange(n_layers)
        kz = kz_top[:, None] - layers[None]
        keys = np.column_stack([np.repeat(cells, n_layers, axis=0), kz.ravel()])
        is_top = (kz >= kz_lo[:, None]).ravel()
        return keys, is_top, np.repeat(h, n_layers), np.repeat(step, n_layers)
    kz_top = np.rint((h - o[2]) / v).astype(np.int64) - 1          # top face nearest the exact height
    n_layers = int(np.ceil(thickness / v))
    layers = np.arange(n_layers)
    keys = np.column_stack([np.repeat(cells, n_layers, axis=0),
                            (kz_top[:, None] - layers[None]).ravel()])
    is_top = np.tile(layers == 0, len(cells))
    return keys, is_top, np.repeat(h, n_layers), np.repeat(step, n_layers)


def _slab_from_plane(x0, x1, y0, y1, plane, thickness) -> np.ndarray:
    """One slab [8,3] over [x0,x1]x[y0,y1] whose top follows plane (z1, ax, ay, px, py): z = z1 + ax(x-px) + ay(y-py)."""
    z1, ax, ay, px, py = plane
    s = np.zeros((8, 3))
    for sx, sy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        cx, cy = (x1 if sx else x0), (y1 if sy else y0)
        top = z1 + ax * (cx - px) + ay * (cy - py)
        ci = 4 * sx + 2 * sy
        s[ci] = [cx, cy, top - thickness]
        s[ci + 1] = [cx, cy, top]
    return s


def clip_slabs(el: Element, slabs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slabs whose centre is outside the element but which reach into its bounding rectangle with a top inside
    the clearing band: cut the part inside the rectangle away (the outside remainder stays as up to 4 slabs).
    Returns (mask of slabs replaced, their replacement slabs). Clipping is against the footprint's bbox, exact
    for axis-aligned flights and landings."""
    if slabs is None or len(slabs) == 0:
        return np.zeros(0, bool), np.zeros((0, 8, 3), np.float32)
    bx0, bx1, by0, by1 = el.bbox()
    lo, hi = el.clear
    x0, y0 = slabs[:, 0, 0].astype(np.float64), slabs[:, 0, 1].astype(np.float64)
    x1, y1 = slabs[:, 7, 0].astype(np.float64), slabs[:, 7, 1].astype(np.float64)
    overlap = (x1 > bx0) & (x0 < bx1) & (y1 > by0) & (y0 < by1)
    centre_in = el.height(0.5 * np.column_stack([x0 + x1, y0 + y1]))[0]
    cand = np.flatnonzero(overlap & ~centre_in)
    if len(cand) == 0:
        return np.zeros(len(slabs), bool), np.zeros((0, 8, 3), np.float32)
    # band test at the centre of the overlap rectangle (where the slab actually pokes in)
    ox = 0.5 * (np.maximum(x0[cand], bx0) + np.minimum(x1[cand], bx1))
    oy = 0.5 * (np.maximum(y0[cand], by0) + np.minimum(y1[cand], by1))
    m, h, _ = el.height(np.column_stack([ox, oy]))
    top = slabs[cand, 1, 2].astype(np.float64)
    cut = m & (top >= h - lo) & (top <= h + hi)
    replaced = np.zeros(len(slabs), bool)
    out = []
    for i in cand[cut]:
        s = slabs[i].astype(np.float64)
        sx0, sy0, sx1, sy1 = s[0, 0], s[0, 1], s[7, 0], s[7, 1]
        z1, z3, z5 = s[1, 2], s[3, 2], s[5, 2]
        ax = (z5 - z1) / (sx1 - sx0) if sx1 > sx0 else 0.0
        ay = (z3 - z1) / (sy1 - sy0) if sy1 > sy0 else 0.0
        plane = (z1, ax, ay, sx0, sy0)
        th = s[1, 2] - s[0, 2]
        pieces = []
        if sx0 < bx0:
            pieces.append((sx0, bx0, sy0, sy1))
        if sx1 > bx1:
            pieces.append((bx1, sx1, sy0, sy1))
        mx0, mx1 = max(sx0, bx0), min(sx1, bx1)
        if sy0 < by0:
            pieces.append((mx0, mx1, sy0, by0))
        if sy1 > by1:
            pieces.append((mx0, mx1, by1, sy1))
        replaced[i] = True
        out += [_slab_from_plane(*p, plane, th) for p in pieces if p[1] - p[0] > 1e-6 and p[3] - p[2] > 1e-6]
    return replaced, (np.asarray(out, dtype=np.float32).reshape(-1, 8, 3) if out else np.zeros((0, 8, 3), np.float32))


def tread_slabs(el: Element, grid: VoxelGrid) -> np.ndarray:
    """One flat slab per tread (per landing): rasterised cells run-merged into rectangles, top = exact h."""
    v, o = grid.voxel, grid.origin
    x0, x1, y0, y1 = el.bbox()
    ix = np.arange(int(np.floor((x0 - o[0]) / v)), int(np.ceil((x1 - o[0]) / v)) + 1)
    iy = np.arange(int(np.floor((y0 - o[1]) / v)), int(np.ceil((y1 - o[1]) / v)) + 1)
    gx, gy = np.meshgrid(ix, iy, indexing="ij")
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    m, h, step = el.height(o[:2] + (cells + 0.5) * v)
    cells, h, step = cells[m], h[m], step[m]
    if len(cells) == 0:
        return np.zeros((0, 8, 3), np.float32)
    rows = np.stack([cells[:, 0], cells[:, 0], cells[:, 1], cells[:, 1], step], axis=1)
    rows = _merge_axis(rows, group_cols=[2, 4], lo_col=0, hi_col=1)
    rows = _merge_axis(rows, group_cols=[0, 1, 4], lo_col=2, hi_col=3)
    top = np.array([el.z if el.kind == "landing" else el.z0 + k * el.rise for k in rows[:, 4]])
    xa, xb = o[0] + rows[:, 0] * v, o[0] + (rows[:, 1] + 1) * v
    ya, yb = o[1] + rows[:, 2] * v, o[1] + (rows[:, 3] + 1) * v
    slabs = np.zeros((len(rows), 8, 3))
    for sx, sy in [(0, 0), (0, 1), (1, 0), (1, 1)]:                   # corner = 4*sx + 2*sy + sz
        cx, cy = (xb if sx else xa), (yb if sy else ya)
        ci = 4 * sx + 2 * sy
        slabs[:, ci] = np.stack([cx, cy, top - SLAB_THICKNESS], axis=1)
        slabs[:, ci + 1] = np.stack([cx, cy, top], axis=1)
    return slabs.astype(np.float32)


def clear_masks(el: Element, boxes, grid: VoxelGrid, slabs, pts):
    """Boolean masks (boxes, voxels, slabs, points) of original geometry to drop for one element.
    A box goes only when it lies entirely inside the band: stair treads, risers, rails and clutter are short
    boxes, while a wall column merged over its whole height sticks out of the band and survives."""
    lo, hi = el.clear
    m_b, h_b, _ = el.height(boxes[:, :2].astype(np.float64))
    zb0, zb1 = boxes[:, 2] - boxes[:, 5], boxes[:, 2] + boxes[:, 5]
    drop_b = m_b & (zb0 >= h_b - lo - 0.05) & (zb1 <= h_b + hi + 0.05)
    c = grid.centers()
    m_v, h_v, _ = el.height(c[:, :2])
    drop_v = m_v & (c[:, 2] >= h_v - lo) & (c[:, 2] <= h_v + hi)
    drop_s = None
    if slabs is not None and len(slabs):
        sc = 0.5 * (slabs[:, 0, :2] + slabs[:, 7, :2]).astype(np.float64)
        st = slabs[:, 1, 2].astype(np.float64)
        m_s, h_s, _ = el.height(sc)
        drop_s = m_s & (st >= h_s - lo) & (st <= h_s + hi)
    drop_p = None
    if pts is not None:
        m_p, h_p, _ = el.height(pts[:, :2].astype(np.float64))
        drop_p = m_p & (pts[:, 2] >= h_p - lo) & (pts[:, 2] <= h_p + hi)
    return drop_b, drop_v, drop_s, drop_p


# -- inspection ---------------------------------------------------------------------------------------
def band_map(slabs: np.ndarray, x0, x1, y0, y1, z0, z1, cell: float = 0.2) -> str:
    """Per cell: area-weighted median slab top (dm) within the z band; '.' = no slab."""
    lo, hi = slabs[:, 0, :2].astype(np.float64), slabs[:, 7, :2].astype(np.float64)
    top = slabs[:, 1, 2].astype(np.float64)
    m = (hi[:, 0] > x0) & (lo[:, 0] < x1) & (hi[:, 1] > y0) & (lo[:, 1] < y1) & (top > z0) & (top < z1)
    lo, hi, top = lo[m], hi[m], top[m]
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    acc: dict[tuple[int, int], list] = {}
    for a, b, t in zip(lo, hi, top):
        i0, i1 = max(0, int((a[0] - x0) / cell)), min(nx - 1, int((b[0] - x0 - 1e-9) / cell))
        j0, j1 = max(0, int((a[1] - y0) / cell)), min(ny - 1, int((b[1] - y0 - 1e-9) / cell))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                w = (min(b[0], x0 + (i + 1) * cell) - max(a[0], x0 + i * cell)) * \
                    (min(b[1], y0 + (j + 1) * cell) - max(a[1], y0 + j * cell))
                if w > 0:
                    acc.setdefault((i, j), []).append((t, w))
    lines = [f"slab tops (dm) for z in [{z0:.2f}, {z1:.2f}], cell {cell} m; rows y (north up), cols x",
             "        " + "".join(f"{x0 + (i + 0.5) * cell:6.1f}"[-3:] if i % 2 == 0 else "   " for i in range(nx))]
    for j in range(ny - 1, -1, -1):
        row = ""
        for i in range(nx):
            v = acc.get((i, j))
            if not v:
                row += "  ."
                continue
            t = np.array([a for a, _ in v])
            w = np.array([b for _, b in v])
            o = np.argsort(t)
            cw = np.cumsum(w[o])
            row += f"{int(round(t[o][np.searchsorted(cw, cw[-1] / 2)] * 10)):3d}"
        lines.append(f"{y0 + (j + 0.5) * cell:7.1f}|{row}")
    return "\n".join(lines)


def voxel_band_map(grid: VoxelGrid, x0, x1, y0, y1, z0, z1, cell: float) -> str:
    """Per cell: highest voxel top (dm) within the z band; '.' = no voxel. For scenes without ground.npz."""
    c = grid.centers()
    top = c[:, 2] + grid.voxel / 2
    m = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (top > z0) & (top < z1)
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    i = np.clip(((c[m, 0] - x0) / cell).astype(int), 0, nx - 1)
    j = np.clip(((c[m, 1] - y0) / cell).astype(int), 0, ny - 1)
    hmap = np.full((nx, ny), -np.inf)
    np.maximum.at(hmap, (i, j), top[m])
    lines = [f"voxel tops (dm) for z in [{z0:.2f}, {z1:.2f}], cell {cell} m; rows y (north up), cols x",
             "        " + "".join(f"{x0 + (i + 0.5) * cell:6.1f}"[-3:] if i % 2 == 0 else "   " for i in range(nx))]
    for j in range(ny - 1, -1, -1):
        row = "".join("  ." if not np.isfinite(hmap[i, j]) else f"{int(round(hmap[i, j] * 10)):3d}" for i in range(nx))
        lines.append(f"{y0 + (j + 0.5) * cell:7.1f}|{row}")
    return "\n".join(lines)


def print_maps(scene: str, spec: dict, stair: str, margin: float = 0.4):
    gp = os.path.join(scene, "ground.npz")
    slabs = np.load(gp)["slabs"] if os.path.exists(gp) else None
    grid = None if slabs is not None else load_scene(scene)[1]
    for el in spec["elements"]:
        if el.stair != stair and stair != "*":
            continue
        x0, x1, y0, y1 = el.bbox()
        z0, z1 = el.z_lo - el.clear[0] - 0.1, el.z_hi + el.clear[1] + 0.1
        print(f"\n== {el.describe()}")
        if slabs is not None:
            print(band_map(slabs, x0 - margin, x1 + margin, y0 - margin, y1 + margin, z0, z1))
        else:
            print(voxel_band_map(grid, x0 - margin, x1 + margin, y0 - margin, y1 + margin, z0, z1,
                                 cell=grid.voxel if grid.voxel >= 0.1 else 0.2))


# -- main ---------------------------------------------------------------------------------------------
def apply(scene: str, out: str, spec: dict, log=print, exact: bool | None = None) -> dict:
    """exact=None: exact-box mode when the scene has no ground.npz (a 1:1 octree build), raster mode otherwise."""
    if os.path.abspath(scene) == os.path.abspath(out):
        raise SystemExit("--out must differ from --scene (the original scene is kept)")
    t0 = time.time()
    boxes, grid, slabs, meta = load_scene(scene)
    if exact is None:
        exact = slabs is None
    if exact and slabs is not None:
        raise SystemExit("--exact-boxes needs a 1:1 build (no ground.npz): a flattened scene's boxes are not "
                         "merge_boxes(voxels), so they cannot be rebuilt from the surviving voxels")
    pcd_path = os.path.join(scene, "map_ds.pcd")
    pts, inten = read_pcd(pcd_path) if os.path.exists(pcd_path) else (None, None)
    log(f"loaded {scene}: {len(boxes)} boxes, {len(grid.keys)} voxels, "
        f"{0 if slabs is None else len(slabs)} slabs, {0 if pts is None else len(pts)} map points ({time.time() - t0:.1f}s); "
        f"mode {'exact boxes' if exact else 'raster'}")
    exact_el = []
    drop_b = np.zeros(len(boxes), bool)
    drop_v = np.zeros(len(grid.keys), bool)
    drop_s = np.zeros(0 if slabs is None else len(slabs), bool)
    drop_p = np.zeros(0 if pts is None else len(pts), bool)
    new_keys, new_top, new_h, new_slabs, report = [], [], [], [], []
    n_clipped = 0
    for el in spec["elements"]:
        b, vx, s, p = clear_masks(el, boxes, grid, slabs, pts)
        drop_b |= b
        drop_v |= vx
        if s is not None:
            drop_s |= s
            replaced, remainder = clip_slabs(el, slabs)
            replaced &= ~drop_s
            drop_s |= replaced
            n_clipped += int(replaced.sum())
            new_slabs.append(remainder)
        if p is not None:
            drop_p |= p
        el_exact = exact and el.axis_aligned
        if exact and not el_exact:
            log(f"warning: {el.describe()} is not axis-aligned; its boxes are rasterised on the grid")
        exact_el.append(el_exact)
        keys, is_top, h, _ = rasterize(el, grid, float(spec["thickness"]), exact=el_exact)
        new_keys.append(keys)
        new_top.append(is_top)
        new_h.append(h)
        new_slabs.append(tread_slabs(el, grid))
        info = {"stair": el.stair, "name": el.name, "kind": el.kind, "steps": el.n, "riser": round(el.rise, 4),
                "clear": list(el.clear), "boxes_removed": int(b.sum()), "voxels_removed": int(vx.sum()),
                "voxels_added": int(len(keys)), "boxes": "exact" if el_exact else "raster"}
        if el.kind == "flight":
            info.update({"tread": round(el.tread, 4), "length": round(el.length, 3), "width": el.width,
                         "z0": el.z0, "z1": el.z1, "yaw_deg": round(el.yaw_deg, 2)})
        else:
            info.update({"z": el.z, "rect": el.rect})
        report.append(info)
        log(f"{el.describe()}: removed {int(b.sum())} boxes / {int(vx.sum())} voxels"
            f"{'' if s is None else f' / {int(s.sum())} slabs'}; added {len(keys)} voxels")

    keys = np.concatenate(new_keys)
    is_top = np.concatenate(new_top)
    h_exact = np.concatenate(new_h)
    # the scene grid must contain the new voxels (VoxelLookup drops keys outside dims)
    dims = np.maximum(grid.dims, keys.max(axis=0) + 1)
    if (keys < 0).any():
        raise SystemExit("synthetic stairs fall below the scene grid origin; move the spec inside the map")
    codes = encode(keys, dims)
    _, first = np.unique(codes, return_index=True)                     # overlapping elements: keep one
    keys, is_top, h_exact = keys[first], is_top[first], h_exact[first]
    kept_inten = grid.intensity[~drop_v]
    fill_i = float(np.median(grid.intensity[drop_v])) if drop_v.any() else float(np.median(kept_inten)) if len(kept_inten) else 0.0
    centers_new = grid.origin + (keys + 0.5) * grid.voxel
    zmean_new = np.where(is_top, h_exact, centers_new[:, 2]).astype(np.float32)
    # new voxels must not duplicate kept originals (the clearing band covers [h - thickness, h] whenever
    # clear[0] >= thickness; otherwise drop the originals that collide)
    old_codes = encode(grid.keys[~drop_v].astype(np.int64), dims)
    collide = np.isin(old_codes, codes[first])
    keep_old = np.flatnonzero(~drop_v)[~collide]
    out_keys = np.concatenate([grid.keys[keep_old].astype(np.int32), keys.astype(np.int32)])
    out_inten = np.concatenate([grid.intensity[keep_old], np.full(len(keys), fill_i, np.float32)])
    out_zmean = np.concatenate([grid.zmean[keep_old], zmean_new])

    if exact:
        # the surviving original geometry is re-merged from its voxels (a 1:1 build's boxes are exactly
        # merge_boxes(voxels)); the synthetic elements are exact AABBs, raster fallbacks are merged from their voxels
        old_boxes = boxes_to_world(merge_boxes(grid.keys[keep_old].astype(np.int64)), grid).astype(np.float32)
        parts = [el.exact_boxes(float(spec["thickness"])) for el, ex in zip(spec["elements"], exact_el) if ex]
        raster_keys = [k for k, ex in zip(new_keys, exact_el) if not ex]
        if raster_keys:
            rk = np.concatenate(raster_keys)
            rk = rk[np.unique(encode(rk, dims), return_index=True)[1]]
            parts.append(boxes_to_world(merge_boxes(rk), grid))
        new_boxes = np.concatenate(parts).astype(np.float32)
        out_boxes = np.concatenate([old_boxes, new_boxes])
        log(f"boxes: {len(boxes)} -> {len(old_boxes)} re-merged from the surviving voxels + {len(new_boxes)} synthetic "
            f"= {len(out_boxes)}; voxels: {len(grid.keys)} - {int(drop_v.sum()) + int(collide.sum())} + {len(keys)} = {len(out_keys)}")
    else:
        new_boxes = boxes_to_world(merge_boxes(keys), grid).astype(np.float32)
        out_boxes = np.concatenate([boxes[~drop_b], new_boxes])
        log(f"boxes: {len(boxes)} - {int(drop_b.sum())} + {len(new_boxes)} = {len(out_boxes)}; "
            f"voxels: {len(grid.keys)} - {int(drop_v.sum()) + int(collide.sum())} + {len(keys)} = {len(out_keys)}")

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "boxes.npy"), out_boxes)
    np.savez_compressed(os.path.join(out, "voxels.npz"), keys=out_keys, intensity=out_inten, zmean=out_zmean,
                        origin=grid.origin, voxel=grid.voxel, dims=dims)
    if slabs is not None:
        out_slabs = np.concatenate([slabs[~drop_s]] + new_slabs).astype(np.float32)
        np.savez_compressed(os.path.join(out, "ground.npz"), slabs=out_slabs)
        log(f"slabs: {len(slabs)} - {int(drop_s.sum())} ({n_clipped} of them clipped at a footprint edge) "
            f"+ {sum(len(s) for s in new_slabs)} = {len(out_slabs)}")
    if pts is not None:
        out_pts = np.concatenate([pts[~drop_p], centers_new]).astype(np.float32)
        out_i = np.concatenate([inten[~drop_p] if inten is not None else np.zeros(int((~drop_p).sum()), np.float32),
                                np.full(len(keys), fill_i, np.float32)])
        write_pcd(os.path.join(out, "map_ds.pcd"), out_pts, out_i)
        log(f"map points: {len(pts)} - {int(drop_p.sum())} + {len(keys)} = {len(out_pts)}")
    for extra in ("eval.json", "eval_profile.txt"):
        if os.path.exists(os.path.join(out, extra)):
            os.remove(os.path.join(out, extra))              # stale evaluations of the source scene
    meta = dict(meta)
    meta["boxes"] = int(len(out_boxes))
    meta["voxels"] = int(len(out_keys))
    meta["synthetic_stairs"] = {"source_scene": os.path.abspath(scene), "mode": "exact_boxes" if exact else "raster",
                                "riser": spec["riser"], "thickness": spec["thickness"], "clear": list(spec["clear"]),
                                "elements": report}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {out} ({time.time() - t0:.1f}s)")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="source scene directory")
    ap.add_argument("--spec", required=True, help="stairs spec yaml")
    ap.add_argument("--out", help="output scene directory (required unless --map)")
    ap.add_argument("--map", metavar="STAIR", help="print height maps (slab tops, or voxel tops when the scene has no "
                                                   "ground.npz) around every element of this stair (name, or * for all) "
                                                   "and exit without writing")
    ap.add_argument("--exact-boxes", action="store_true", default=None,
                    help="exact AABB per tread, original boxes re-merged from the surviving voxels "
                         "(default for scenes without ground.npz; refused for flattened scenes)")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    for el in spec["elements"]:
        print(el.describe())
    if a.map:
        print_maps(a.scene, spec, a.map)
        return
    if not a.out:
        ap.error("--out is required (or use --map)")
    apply(a.scene, a.out, spec, exact=a.exact_boxes)


if __name__ == "__main__":
    main()
