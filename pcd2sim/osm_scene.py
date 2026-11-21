"""OSM buildings -> a quadsim scene, without a point cloud in between.

A 2 x 2 km area at the 0.1-0.2 m voxels of pcd2sim.build_scene would not fit in memory, and a building from OSM is
already what a box scene wants: an outline extruded to a height. So each footprint is rasterised on a grid aligned
with the map axes (cell centres inside the outline), every cell is a column from the ground to the roof, and the
columns are merged into boxes with pcd2sim.voxel.merge_boxes (runs along x, then equal runs along y).
The ground is flat: square tiles whose tops sit at ground_z, all in the contact set. Around a base scan (--base) it
starts at the scan's edge and eases from the edge's height to ground_z over --blend metres (Apron).

  python3 -m pcd2sim.osm_scene --url "https://streets.gl/#24.95919,121.41896,45.00,0.00,586.00" --radius 1000 \
      --out scenes/osm_tucheng                       # Overpass answer cached in Maps/osm_tucheng/overpass.json

The scene is georeferenced: meta.json `geo` is its datum (the centre, heading 0, so map = ENU), which
quadsim.gnss.resolve_datum prefers over config gnss.datum. Run it with `spawn=null` so the robot starts at the
scene's own spawn (the free spot nearest the centre) instead of config's dp spawn. Map data (c) OpenStreetMap
contributors, ODbL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from quadsim.config import load_config, load_scene_meta
from quadsim.gnss import Datum, enu_to_lla, enu_to_map, lla_to_enu, map_to_enu, resolve_datum

from . import SCENE_FORMAT
from .clearing import clear_boxes, parse_rects, points_kept, rect_boxes
from .osm_buildings import Building, parse_overpass, parse_streets_gl_url
from .osm_colors import GROUND_RGB, GroundClass
from .osm_ground import overpass_ground_query, paint_ground, parse_overpass_ground
from .pcd_io import read_pcd, write_pcd
from .voxel import merge_boxes

OVERPASS_URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
SPAWN_CLEARANCE = 1.0                            # m between the spawn and the nearest building box

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


@dataclass(frozen=True)
class ColouredBoxes:
    """Scene boxes and their scene colours, row for row: float32 [M,6] (cx, cy, cz, hx, hy, hz) and uint8 [M,4]
    RGBA (A = 255: a colour from OSM, A = 0: none). Built only together, so the two can never fall out of step."""
    boxes: np.ndarray
    rgba: np.ndarray

    def __post_init__(self):
        if self.boxes.shape != (len(self.boxes), 6) or self.rgba.shape != (len(self.boxes), 4):
            raise ValueError(f"boxes {self.boxes.shape} and rgba {self.rgba.shape} do not pair up")

    @classmethod
    def of(cls, boxes: np.ndarray, colour, alpha: int = 255) -> "ColouredBoxes":
        """Boxes that all take one colour."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 6)
        return cls(boxes, np.tile(np.array([*colour, alpha], np.uint8), (len(boxes), 1)))

    @classmethod
    def concat(cls, parts: list["ColouredBoxes"]) -> "ColouredBoxes":
        if not parts:
            return cls(np.zeros((0, 6), np.float32), np.zeros((0, 4), np.uint8))
        return cls(np.vstack([p.boxes for p in parts]), np.vstack([p.rgba for p in parts]))


def buildings_to_boxes(buildings: list[Building], cell: float, ground_z: float,
                       clip: Clip | None = None, cut: "Cut | None" = None) -> ColouredBoxes:
    """Every building as merged full-height columns on a cell grid, in its colour, without the cells a base scene
    keeps (cut)."""
    parts = []
    for b in buildings:
        keys = footprint_cells(b.footprint, cell, clip)
        if cut is not None:
            keys = keys[cut.keep(keys)]
        if len(keys):
            parts.append(ColouredBoxes.of(_columns_to_boxes(keys, cell, ground_z, ground_z + b.height_m), b.colour))
    return ColouredBoxes.concat(parts)


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


# -- Overpass ---------------------------------------------------------------------------------------------------
def overpass_bbox(frame: Datum, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float, float, float]:
    """(south, west, north, east) around the scene square's corners (the square may be turned against ENU)."""
    corners = [map_to_enu(np.array([x, y, 0.0]), frame) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    lla = np.array([enu_to_lla(c, frame)[:2] for c in corners])
    return float(lla[:, 0].min()), float(lla[:, 1].min()), float(lla[:, 0].max()), float(lla[:, 1].max())


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    b = ",".join(f"{v:.6f}" for v in bbox)
    return (f'[out:json][timeout:180];(way["building"]({b});'
            f'relation["building"]["type"="multipolygon"]({b}););out tags geom;')


def fetch_overpass(path: str, query: str, urls=OVERPASS_URLS) -> None:
    """Download an Overpass query once: an existing file is the cache (delete it to refetch). Written whole or not
    at all."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for url in urls:
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": "quadsim-osm-scene"})
            with urllib.request.urlopen(req, timeout=240) as r:
                raw = r.read()
            json.loads(raw)                                      # an error page is not a cache
        except (OSError, ValueError) as e:
            errors.append(f"{url}: {e}")
            continue
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, path)
        return
    raise RuntimeError("Overpass download failed: " + "; ".join(errors))


# -- base scene ------------------------------------------------------------------------------------------------
class Cut:
    """The grid cells OSM leaves to a base scene: every base box's xy extent grown by margin, rasterised by cell
    centre on the same grid as the OSM footprints."""

    def __init__(self, base_boxes: np.ndarray, cell: float, margin: float):
        lo = base_boxes[:, :2] - base_boxes[:, 3:5] - margin
        hi = base_boxes[:, :2] + base_boxes[:, 3:5] + margin
        c0 = np.ceil(lo / cell - 0.5).astype(np.int64)            # cells whose centre lies in [lo, hi]
        c1 = np.floor(hi / cell - 0.5).astype(np.int64) + 1
        self.i0 = c0.min(axis=0)
        self.mask = np.zeros(c1.max(axis=0) - self.i0, dtype=bool)
        for (a0, b0), (a1, b1) in zip(c0 - self.i0, c1 - self.i0):
            self.mask[a0:a1, b0:b1] = True
        self.lo_m, self.hi_m = self.i0 * cell, (self.i0 + np.array(self.mask.shape)) * cell

    def keep(self, keys: np.ndarray) -> np.ndarray:
        k = keys - self.i0
        inside = np.all((k >= 0) & (k < self.mask.shape), axis=1)
        cut = np.zeros(len(keys), dtype=bool)
        cut[inside] = self.mask[k[inside, 0], k[inside, 1]]
        return ~cut


@dataclass(frozen=True)
class MapPoints:
    """A scene's map_ds.pcd and its colours, row for row (A = 0: none)."""
    xyz: np.ndarray
    intensity: np.ndarray
    rgba: np.ndarray


@dataclass(frozen=True)
class Base:
    """A scanned scene to lay the OSM area around, kept as it is but for its clear rectangles (pcd2sim.clearing).
    boxes, contact and rgba are row for row."""
    path: str
    datum: Datum
    boxes: np.ndarray
    contact: np.ndarray
    rgba: np.ndarray                             # uint8 [N, 4], the scan's own colours or none (A = 0)
    voxel: float
    spawn: list[float]                           # [x, y, floor z]
    points: MapPoints | None                     # None: the base has no map_ds.pcd
    clear: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    @property
    def floor_z(self) -> float:
        return float(self.spawn[2])

    def cleared(self, rects) -> "Base":
        """This scan with what stands on the ground (below the floor + Apron.LOW_BAND) cut out of the rectangles."""
        rects = parse_rects(rects)
        below = self.floor_z + Apron.LOW_BAND
        boxes, rows = clear_boxes(self.boxes, rects, below)
        points = self.points
        if points is not None:
            k = points_kept(points.xyz, rects, below)
            points = MapPoints(points.xyz[k], points.intensity[k], points.rgba[k])
        return replace(self, boxes=boxes.astype(self.boxes.dtype), contact=self.contact[rows], rgba=self.rgba[rows],
                       points=points, clear=rects)


def _rgba_file(path: str, n: int) -> np.ndarray:
    """A scene's colours (uint8 [n, 4]), else none (A = 0)."""
    rgba = np.load(path) if os.path.exists(path) else None
    return rgba if rgba is not None and rgba.shape == (n, 4) else np.zeros((n, 4), np.uint8)


def _load_points(path: str) -> MapPoints | None:
    src = os.path.join(path, "map_ds.pcd")
    if not os.path.exists(src):
        return None
    xyz, inten = read_pcd(src)
    inten = inten.astype(np.float32) if inten is not None else np.full(len(xyz), 100.0, np.float32)
    return MapPoints(xyz.astype(np.float32), inten, _rgba_file(os.path.join(path, "map_ds_colors.npy"), len(xyz)))


def load_base(path: str, fallback_datum: dict | None) -> Base:
    meta = load_scene_meta(path)
    datum = resolve_datum(meta, fallback_datum)
    if datum is None:
        raise ValueError(f"base scene {path} has no geo and no fallback datum: OSM cannot be placed on it")
    if not meta.get("spawn"):
        raise ValueError(f"base scene {path} has no spawn: its floor height is unknown")
    boxes = np.load(os.path.join(path, "boxes.npy"))
    return Base(path, datum, boxes, np.load(os.path.join(path, "contact.npy")),
                _rgba_file(os.path.join(path, "colors.npy"), len(boxes)), float(meta["voxel"]),
                [float(v) for v in meta["spawn"]], _load_points(path))


def overlay_fit(base: Base, buildings: list[Building], cell: float, min_height: float = 2.0) -> dict:
    """How well the scan's building matches OSM: the OSM outline overlapping the scan's tall cells (walls, roof:
    tops more than min_height above the floor) the most, their IoU and the distance between their centroids."""
    tall = base.boxes[base.boxes[:, 2] + base.boxes[:, 5] > base.floor_z + min_height]
    if not len(tall):
        return {"osm_id": None}
    scan = Cut(tall, cell, 0.0)
    scan_keys = np.argwhere(scan.mask) + scan.i0
    best = None
    for b in buildings:
        keys = footprint_cells(b.footprint, cell)
        inter = int((~scan.keep(keys)).sum()) if len(keys) else 0
        if inter and (best is None or inter > best[1]):
            best = (b, inter, keys)
    if best is None:
        return {"osm_id": None}
    b, inter, keys = best
    offset = (keys.mean(axis=0) - scan_keys.mean(axis=0)) * cell
    return {"osm_id": b.osm_id, "iou": round(inter / (len(keys) + len(scan_keys) - inter), 3),
            "offset_m": round(float(np.hypot(*offset)), 2), "offset_xy": [round(float(v), 2) for v in offset]}


def register_base(base: Base, buildings: list[Building], cell: float, window: float = 80.0, min_height: float = 2.0,
                  trim: float = 0.6, iters: int = 200) -> dict:
    """The shift (map metres, heading kept) that puts the scan's building onto its OSM outlines: OSM is the
    reference, the scan's datum is what is off. The scan's tall cells (tops > min_height above the floor) are closed,
    opened, cut to their largest blob and hole-filled (the building, not trees or speckle); a cross-correlation
    of that area with the OSM footprints within `window` gives the start, a trimmed ICP of its outline against
    the OSM outlines (translation only, the `trim` nearest fraction each round) refines it."""
    from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, binary_opening, label
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
    tall = base.boxes[base.boxes[:, 2] + base.boxes[:, 5] > base.floor_z + min_height]
    if not len(tall):
        return {"shift_m": [0.0, 0.0], "ok": False, "why": "no tall cells in the scan"}
    pad = int(np.ceil(window / cell))
    scan = Cut(tall, cell, 0.0)
    i0 = scan.i0 - pad
    S = np.zeros(np.array(scan.mask.shape) + 2 * pad, bool)
    S[pad:pad + scan.mask.shape[0], pad:pad + scan.mask.shape[1]] = scan.mask
    O = np.zeros_like(S)
    for b in buildings:
        k = footprint_cells(b.footprint, cell) - i0
        k = k[np.all((k >= 0) & (k < O.shape), axis=1)]
        O[k[:, 0], k[:, 1]] = True
    if not O.any():
        return {"shift_m": [0.0, 0.0], "ok": False, "why": f"no OSM building within {window} m of the scan"}
    S = binary_opening(binary_closing(S, np.ones((7, 7))), np.ones((3, 3)))
    lab, n = label(S)
    if n == 0:
        return {"shift_m": [0.0, 0.0], "ok": False, "why": "the scan's tall cells are all speckle"}
    S = binary_fill_holes(lab == np.bincount(lab.ravel())[1:].argmax() + 1)
    O = binary_fill_holes(O)
    corr = fftconvolve(O.astype(np.float32), S[::-1, ::-1].astype(np.float32), mode="same")
    k = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    t = (k - np.array(S.shape) // 2) * cell                     # start: the best area overlap
    start = t.copy()
    edge = lambda m: m & ~binary_erosion(m, np.ones((3, 3)))
    P = (np.argwhere(edge(S)) + i0 + 0.5) * cell
    Q = (np.argwhere(edge(O)) + i0 + 0.5) * cell
    tree = cKDTree(Q)
    for _ in range(iters):
        d, j = tree.query(P + t)
        keep = d <= np.quantile(d, trim)
        step = np.mean(Q[j[keep]] - (P[keep] + t), axis=0)
        t = t + step
        if np.linalg.norm(step) < 1e-4:
            break
    d0, _ = tree.query(P)
    d, _ = tree.query(P + t)
    return {"shift_m": [round(float(v), 2) for v in t], "ok": True, "start_m": [round(float(v), 2) for v in start],
            "outline_median_m": round(float(np.median(d)), 2), "outline_within_2m": round(float(np.mean(d < 2.0)), 3),
            "outline_median_before_m": round(float(np.median(d0)), 2)}


def shifted_datum(datum: Datum, shift_xy) -> Datum:
    """The datum under which map point p lands where map point p + shift_xy did: the origin moved by the shift."""
    lat, lon, _ = enu_to_lla(map_to_enu(np.array([shift_xy[0], shift_xy[1], 0.0]), datum), datum)
    return Datum(float(lat), float(lon), datum.alt, datum.yaw_deg)


# -- scene ------------------------------------------------------------------------------------------------------
class Apron:
    """Where OSM ground meets a base scan, on a cell grid around the scan padded by the blend distance: the cells
    the scan covers near the ground (boxes starting below floor_z + LOW_BAND; OSM lays no ground there) and the
    scan's ground height by cell (the highest box top within FLOOR_BAND of the floor). The OSM ground between is
    the harmonic (Laplace) fill from the scan's ground heights to ground_z at `blend` metres out: it starts at the
    scan's edge at the edge's height and eases to the flat ground as smoothly as the edge allows, so the robot
    walks out of the scan onto the streets."""
    LOW_BAND = 1.5                               # m above the floor: boxes starting lower stand on the ground
    FLOOR_BAND = (-1.5, 0.8)                     # m around the floor: box tops in it are ground
    TOL, MAX_ITER = 1e-4, 20000                  # Laplace fill: stop when no cell moves more than TOL (m)

    def __init__(self, base_boxes: np.ndarray, floor_z: float, cell: float, blend: float, ground_z: float):
        from scipy.ndimage import distance_transform_edt
        self.cell, self.blend, self.ground_z = cell, blend, ground_z
        low = base_boxes[base_boxes[:, 2] - base_boxes[:, 5] < floor_z + self.LOW_BAND]
        if not len(low):
            self.i0 = np.zeros(2, np.int64)
            self.covered, self.has, self.z = np.zeros((0, 0), bool), np.zeros((0, 0), bool), np.zeros((0, 0))
            return
        pad = int(np.ceil(blend / cell)) + 2
        ltop = low[:, 2] + low[:, 5]
        c0 = np.ceil((low[:, :2] - low[:, 3:5]) / cell - 0.5).astype(np.int64)
        c1 = np.floor((low[:, :2] + low[:, 3:5]) / cell - 0.5).astype(np.int64) + 1
        self.i0 = c0.min(axis=0) - pad
        shape = c1.max(axis=0) + pad - self.i0
        self.covered = np.zeros(shape, bool)
        surf = np.full(shape, -np.inf)
        ground = (ltop >= floor_z + self.FLOOR_BAND[0]) & (ltop <= floor_z + self.FLOOR_BAND[1])
        for (a0, b0), (a1, b1), t, g in zip(c0 - self.i0, c1 - self.i0, ltop, ground):
            self.covered[a0:a1, b0:b1] = True
            if g:
                np.maximum(surf[a0:a1, b0:b1], t, out=surf[a0:a1, b0:b1])
        self.has = has = np.isfinite(surf)                 # cells where the scan has ground
        far = distance_transform_edt(~has) * cell >= blend if has.any() else np.ones(shape, bool)
        far[[0, -1], :] = far[:, [0, -1]] = True             # the padded border is always past the blend
        fixed = has | far
        z = np.where(has, surf, ground_z)
        for _ in range(self.MAX_ITER):                       # Jacobi on the free cells
            avg = 0.25 * (z[:-2, 1:-1] + z[2:, 1:-1] + z[1:-1, :-2] + z[1:-1, 2:])
            free = ~fixed[1:-1, 1:-1]
            step = float(np.abs(avg - z[1:-1, 1:-1])[free].max()) if free.any() else 0.0
            z[1:-1, 1:-1] = np.where(free, avg, z[1:-1, 1:-1])
            if step < self.TOL:
                break
        self.z = z

    @property
    def lo_m(self) -> np.ndarray:
        return self.i0 * self.cell

    @property
    def hi_m(self) -> np.ndarray:
        return (self.i0 + np.array(self.covered.shape)) * self.cell

    def ground(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(keep, z) for cells: no OSM ground under the scan, the eased height near it, ground_z elsewhere."""
        k = keys - self.i0
        inside = np.all((k >= 0) & (k < self.covered.shape), axis=1)
        keep = np.ones(len(keys), bool)
        z = np.full(len(keys), self.ground_z)
        keep[inside] = ~self.covered[k[inside, 0], k[inside, 1]]
        z[inside] = self.z[k[inside, 0], k[inside, 1]]
        return keep, z


def _cell_boxes(keys: np.ndarray, cls: np.ndarray, z: np.ndarray, cell: float, thickness: float) -> ColouredBoxes:
    """Ground cells -> boxes merged per (class, height), each in its class colour."""
    parts = []
    for c in np.unique(cls):
        for h in np.unique(z[cls == c]):
            sel = (cls == c) & (z == h)
            parts.append(ColouredBoxes.of(_columns_to_boxes(keys[sel], cell, h - thickness, h),
                                          GROUND_RGB[GroundClass(int(c))]))
    return ColouredBoxes.concat(parts)


def ground_boxes(lo: np.ndarray, hi: np.ndarray, tile: float, z: float, cell: float, apron: Apron | None,
                 paint: np.ndarray | None = None, thickness: float = 0.2, z_step: float = 0.01) -> ColouredBoxes:
    """The ground over the square in its OSM colours. A tile of one ground class (paint: uint8 class per cell of the
    square, osm_ground.paint_ground; None = all bare) away from the apron stays one tile; any other tile is rebuilt
    from its cells, merged per class and height: no ground under the scan, heights easing to the scan's edge
    (rounded to z_step so equal heights merge)."""
    tiles = ground_tiles(lo, hi, tile, z, thickness)
    i0 = np.round(np.asarray(lo) / cell).astype(np.int64)                  # the square's first cell on the grid
    if paint is None:
        paint = np.full(np.round((np.asarray(hi) - lo) / cell).astype(np.int64), GroundClass.BARE, np.uint8)
    tlo, thi = tiles[:, :2] - tiles[:, 3:5], tiles[:, :2] + tiles[:, 3:5]
    near = np.zeros(len(tiles), bool)
    if apron is not None and apron.covered.size:
        near = np.all((tlo < apron.hi_m) & (thi > apron.lo_m), axis=1)
    c0 = np.round(tlo / cell).astype(np.int64) - i0
    c1 = np.round(thi / cell).astype(np.int64) - i0
    whole, parts = [], []
    for k in range(len(tiles)):
        block = paint[c0[k, 0]:c1[k, 0], c0[k, 1]:c1[k, 1]]
        if not near[k] and block.min() == block.max():
            whole.append(ColouredBoxes.of(tiles[k:k + 1], GROUND_RGB[GroundClass(int(block.flat[0]))]))
            continue
        ix, iy = np.meshgrid(np.arange(c0[k, 0], c1[k, 0]), np.arange(c0[k, 1], c1[k, 1]), indexing="ij")
        keys = np.column_stack([ix.ravel(), iy.ravel()]) + i0
        cls = block.ravel()
        if near[k]:
            keep, zc = apron.ground(keys)
            keys, cls, zc = keys[keep], cls[keep], np.round(zc[keep] / z_step) * z_step
        else:
            zc = np.full(len(keys), z)
        if len(keys):
            parts.append(_cell_boxes(keys, cls, zc, cell, thickness))
    return ColouredBoxes.concat(whole + parts)


def spawn_point(building_boxes: np.ndarray, centre: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                step: float = 1.0) -> np.ndarray:
    """The grid point nearest the centre that is SPAWN_CLEARANCE away from every building box."""
    xs, ys = np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[1], hi[1] + 1e-9, step)
    pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    pts = pts[np.argsort(np.linalg.norm(pts - centre, axis=1), kind="stable")]
    if len(building_boxes) == 0:
        return pts[0]
    blo = building_boxes[:, :2] - building_boxes[:, 3:5] - SPAWN_CLEARANCE
    bhi = building_boxes[:, :2] + building_boxes[:, 3:5] + SPAWN_CLEARANCE
    for chunk in np.array_split(pts, max(1, len(pts) // 1024)):          # nearest first, in chunks
        inside = (chunk[:, None, :] >= blo[None]) & (chunk[:, None, :] <= bhi[None])
        free = np.flatnonzero(~np.any(np.all(inside, axis=2), axis=1))
        if len(free):
            return chunk[free[0]]
    raise RuntimeError("no free spawn point in the scene square")


def _write_voxels(out: str, voxel: float, base: Base | None) -> None:
    dst = os.path.join(out, "voxels.npz")
    if base is not None:                         # the scan's LiDAR intensity and surface heights
        shutil.copyfile(os.path.join(base.path, "voxels.npz"), dst)
        return
    np.savez_compressed(dst, keys=np.zeros((0, 3), np.int32), intensity=np.zeros(0, np.float32),
                        zmean=np.zeros(0, np.float32), origin=np.zeros(3), voxel=voxel, dims=np.ones(3, np.int64),
                        contact=np.zeros(0, bool))


def _write_map_points(out: str, osm: ColouredBoxes, base: Base | None) -> None:
    """map_ds.pcd (the base's points, then every OSM box's top centre) and map_ds_colors.npy row for row."""
    xyz = np.column_stack([osm.boxes[:, :2], osm.boxes[:, 2] + osm.boxes[:, 5]]).astype(np.float32)
    inten = np.full(len(xyz), 100.0, np.float32)
    rgba = osm.rgba
    if base is not None and base.points is not None:
        xyz = np.vstack([base.points.xyz, xyz])
        inten = np.concatenate([base.points.intensity, inten])
        rgba = np.vstack([base.points.rgba, rgba])
    write_pcd(os.path.join(out, "map_ds.pcd"), xyz, inten)
    np.save(os.path.join(out, "map_ds_colors.npy"), rgba)


def write_scene(out: str, osm: ColouredBoxes, base: Base | None, voxel: float, spawn: list[float],
                meta_extra: dict) -> dict:
    """The scene files quadsim.sim reads: the base scene's boxes first and unchanged, then the OSM boxes (all
    contact), and colors.npy row for row (the base's own colours or none, then OSM's). Without a base, voxels.npz
    is empty: OSM has no LiDAR intensity or surface detail."""
    os.makedirs(out, exist_ok=True)
    if base is None:
        boxes, rgba, contact = osm.boxes, osm.rgba, np.ones(len(osm.boxes), bool)
    else:
        boxes = np.vstack([base.boxes, osm.boxes])
        rgba = np.vstack([base.rgba, osm.rgba])
        contact = np.concatenate([base.contact, np.ones(len(osm.boxes), bool)])
    np.save(os.path.join(out, "boxes.npy"), boxes.astype(np.float32))
    np.save(os.path.join(out, "contact.npy"), contact)
    np.save(os.path.join(out, "colors.npy"), rgba.astype(np.uint8))
    stale = os.path.join(out, "ground.npz")
    if os.path.exists(stale):
        os.remove(stale)
    _write_voxels(out, voxel, base)
    _write_map_points(out, osm, base)
    lo, hi = boxes[:, :3] - boxes[:, 3:], boxes[:, :3] + boxes[:, 3:]
    meta = {"format_version": SCENE_FORMAT, "source_format": "osm", "voxel": voxel, "boxes": int(len(boxes)),
            "contact": {"boxes": int(contact.sum())}, "ground": None,
            "bbox_min": lo.min(axis=0).tolist(), "bbox_max": hi.max(axis=0).tolist(),
            "gravity": [0.0, 0.0, -9.81], "spawn": [float(v) for v in spawn], **meta_extra}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def scene_square(centre: np.ndarray, radius: float, cell: float) -> Clip:
    """[centre - radius, centre + radius] snapped outward to the cell grid, so tiles and cells line up."""
    return np.floor((centre - radius) / cell) * cell, np.ceil((centre + radius) / cell) * cell


def _square(lat: float, lon: float, radius: float, cell: float, frame: Datum) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scene centre in the map frame and its square [lo, hi] on the cell grid."""
    centre = enu_to_map(lla_to_enu(lat, lon, frame.alt, frame), frame)[:2]
    lo, hi = scene_square(centre, radius, cell)
    return centre, lo, hi


def _register(b: Base, buildings: list[Building], frame: Datum, cell: float) -> tuple[Datum, dict]:
    """OSM is the reference: the frame with the scan's datum moved onto its OSM outline, and the registration."""
    registration = register_base(b, buildings, cell)
    if not registration["ok"]:
        return frame, registration
    registration["datum_before"] = {"lat": b.datum.lat, "lon": b.datum.lon}
    return shifted_datum(frame, registration["shift_m"]), registration


def _ground_paint(ground_path: str | None, frame: Datum, lo: np.ndarray, hi: np.ndarray,
                  cell: float) -> tuple[np.ndarray | None, dict | None]:
    """The ground class per cell from the OSM ground answer (downloaded once when missing) and its meta.json
    record; (None, None) without one: bare ground."""
    if not ground_path:
        return None, None
    fetch_overpass(ground_path, overpass_ground_query(overpass_bbox(frame, lo, hi)))
    with open(ground_path) as f:
        features = parse_overpass_ground(json.load(f), frame)
    paint = paint_ground(features, lo, hi, cell)
    counts = Counter(x.cls.name for x in features)
    return paint, {"file": os.path.abspath(ground_path), "features": dict(sorted(counts.items())),
                   "cells": {GroundClass(int(k)).name: int(v) for k, v in zip(*np.unique(paint, return_counts=True))}}


def _osm_meta(osm_path: str, buildings: list[Building], rejected: int, n_building_boxes: int, ground_meta,
              **params) -> dict:
    sources = Counter(x.height_source for x in buildings)
    return {"file": os.path.abspath(osm_path), "buildings": len(buildings), "rejected": rejected,
            "height_source": {k: sources.get(k, 0) for k in ("tag", "levels", "default")}, **params,
            "building_boxes": n_building_boxes, "ground": ground_meta,
            "attribution": "(c) OpenStreetMap contributors, ODbL"}


def _geo_meta(frame: Datum, b: Base | None, registration: dict | None, radius: float, lat: float, lon: float) -> dict:
    source = "osm" if b is None else f"osm+{os.path.basename(os.path.normpath(b.path))}"
    if registration and registration["ok"]:
        source += " registered"
    return {"lat": frame.lat, "lon": frame.lon, "alt": frame.alt, "yaw_deg": frame.yaw_deg, "source": source,
            "radius_m": radius, "centre": [lat, lon]}


def _ground_z(ground_z: float | None, b: Base | None) -> float:
    """The ground top: as given, else the base's floor, else 0."""
    if ground_z is not None:
        return ground_z
    return b.floor_z if b else 0.0


def _cut(b: Base | None, cell: float, margin: float) -> Cut | None:
    """Where OSM lays no building: around the scan and its clear rectangles."""
    return Cut(np.vstack([b.boxes, rect_boxes(b.clear)]), cell, margin) if b else None


def _apron(b: Base | None, cell: float, blend: float, gz: float) -> Apron | None:
    return Apron(b.boxes, b.floor_z, cell, blend, gz) if b else None


def _base_meta(b: Base | None, buildings: list[Building], cell: float, registration: dict | None,
               margin: float, blend: float) -> dict:
    """meta.json records of a base scene: where it came from, how well OSM fits it, its registration."""
    if b is None:
        return {}
    meta = {"base": {"path": os.path.abspath(b.path), "boxes": int(len(b.boxes)), "margin": margin, "blend": blend,
                     "clear": b.clear.tolist()},
            "overlay": overlay_fit(b, buildings, cell)}
    if registration is not None:
        meta["registration"] = registration
    return meta


def build(lat: float, lon: float, radius: float, out: str, osm_path: str, cell: float = 0.5, tile: float = 20.0,
          default_height: float = 6.0, level_height: float = 3.0, alt: float = 0.0, ground_z: float | None = None,
          base: str | None = None, fallback_datum: dict | None = None, margin: float = 2.0,
          blend: float = 10.0, register: bool = False, ground_path: str | None = None, clear=None) -> dict:
    """An OSM scene around (lat, lon). Stand-alone: map = ENU there, ground at ground_z (default 0). With a base
    scene: its map frame (datum + heading), its floor height and spawn, OSM buildings cut out `margin` around its
    boxes, OSM ground up to the scan's edge and easing from the edge's height to ground_z over `blend` metres.
    register: OSM is the reference, the base's datum is moved so its building lands on its OSM outline (the map
    coordinates of the scan stay as they are; meta.json geo is the corrected datum).
    ground_path: an Overpass ground answer (osm_ground; downloaded once when missing) paints the ground in its OSM
    colours; None = bare ground.
    clear: rectangles [x0, y0, x1, y1] of the base's map frame (a gate) where the scan's ground-standing boxes and
    points are cut away (after registration, which fits the whole scan): the OSM ground runs through them, eased to
    the scan's edge heights, and no OSM building comes within margin of them."""
    b = load_base(base, fallback_datum) if base else None
    frame = b.datum if b else Datum(lat, lon, alt, 0.0)            # stand-alone: map = ENU at the centre
    centre, lo, hi = _square(lat, lon, radius, cell, frame)
    gz = _ground_z(ground_z, b)
    fetch_overpass(osm_path, overpass_query(overpass_bbox(frame, lo, hi)))
    with open(osm_path) as f:
        answer = json.load(f)
    buildings, rejected = parse_overpass(answer, frame, default_height, level_height)
    registration = None
    if b and register:
        frame, registration = _register(b, buildings, frame, cell)
        if registration["ok"]:
            centre, lo, hi = _square(lat, lon, radius, cell, frame)
            buildings, rejected = parse_overpass(answer, frame, default_height, level_height)
    if b and clear is not None:
        b = b.cleared(clear)
    blds = buildings_to_boxes(buildings, cell, gz, clip=(lo, hi), cut=_cut(b, cell, margin))
    paint, ground_meta = _ground_paint(ground_path, frame, lo, hi, cell)
    osm = ColouredBoxes.concat([ground_boxes(lo, hi, tile, gz, cell, _apron(b, cell, blend, gz), paint), blds])
    extra = {"geo": _geo_meta(frame, b, registration, radius, lat, lon),
             "osm": _osm_meta(osm_path, buildings, rejected, len(blds.boxes), ground_meta,
                              default_height=default_height, level_height=level_height, cell=cell, tile=tile),
             **_base_meta(b, buildings, cell, registration, margin=margin, blend=blend)}
    if b is None:
        return write_scene(out, osm, None, cell, [*spawn_point(blds.boxes, centre, lo, hi), gz], extra)
    return write_scene(out, osm, b, b.voxel, b.spawn, extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="a streets.gl view URL: its centre is the scene centre")
    where.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--radius", type=float, default=1000.0, help="half side of the scene square (m)")
    ap.add_argument("--out", required=True, help="scene directory, e.g. scenes/osm_tucheng")
    ap.add_argument("--osm", default=None, help="Overpass cache (default Maps/<out name>/overpass.json)")
    ap.add_argument("--osm-ground", default=None,
                    help="Overpass ground cache: roads, parks, water (default Maps/<out name>/overpass_ground.json)")
    ap.add_argument("--no-ground", action="store_true", help="bare ground: no ground query, no ground colours")
    ap.add_argument("--cell", type=float, default=0.5, help="footprint raster cell (m)")
    ap.add_argument("--tile", type=float, default=20.0, help="ground tile side (m)")
    ap.add_argument("--default-height", type=float, default=6.0, help="buildings with no height or levels (m)")
    ap.add_argument("--level-height", type=float, default=3.0, help="metres per building:levels")
    ap.add_argument("--alt", type=float, default=0.0, help="stand-alone: ellipsoid height of the ground (m)")
    ap.add_argument("--ground-z", type=float, default=None, help="ground top in map z (default 0, or the base floor)")
    ap.add_argument("--base", default=None,
                    help="a scanned scene (e.g. scenes/dp) to lay the OSM area around, in its map frame; its datum "
                         "is its meta.json geo, else config gnss.datum")
    ap.add_argument("--margin", type=float, default=2.0, help="OSM buildings keep this far from any base box (m)")
    ap.add_argument("--blend", type=float, default=10.0,
                    help="OSM ground eases from the scan edge's height to the flat ground over this distance (m)")
    ap.add_argument("--register", action="store_true",
                    help="OSM is the reference: move the base's datum so its building lands on its OSM outline")
    ap.add_argument("--clear", nargs=4, type=float, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                    help="with --base: a rectangle of its map frame (a gate) where the scan's ground-standing boxes "
                         "and points are cut away so the OSM ground runs through; repeatable")
    ap.add_argument("--config", default=None, help="config with the fallback gnss.datum (default config/sim.yaml)")
    a = ap.parse_args(argv)
    lat, lon = parse_streets_gl_url(a.url) if a.url else a.center
    osm = a.osm or os.path.join("Maps", os.path.basename(os.path.normpath(a.out)), "overpass.json")
    ground = None if a.no_ground else (a.osm_ground or os.path.join(os.path.dirname(osm), "overpass_ground.json"))
    fallback = (load_config(a.config, []).get("gnss") or {}).get("datum") if a.base else None
    meta = build(lat, lon, a.radius, a.out, osm, a.cell, a.tile, a.default_height, a.level_height, a.alt,
                 a.ground_z, a.base, fallback, a.margin, a.blend, a.register, ground, a.clear)
    o = meta["osm"]
    print(f"wrote {a.out}: {o['buildings']} buildings ({o['height_source']}, {o['rejected']} refused) -> "
          f"{o['building_boxes']} boxes + ground, spawn {meta['spawn']}")
    if "registration" in meta:
        print(f"registered the base onto OSM: {meta['registration']}")
        print(f"corrected datum: lat {meta['geo']['lat']:.7f} lon {meta['geo']['lon']:.7f} yaw {meta['geo']['yaw_deg']}")
    if "overlay" in meta:
        print(f"overlay fit against the base scan: {meta['overlay']}")


if __name__ == "__main__":
    main()
