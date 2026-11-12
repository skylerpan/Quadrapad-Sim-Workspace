"""OSM buildings -> a quadsim scene, without a point cloud in between.

A 2 x 2 km area at the 0.1-0.2 m voxels of pcd2sim.build_scene would not fit in memory, and a building from OSM is
already what a box scene wants: an outline extruded to a height. So each footprint is rasterised on a grid aligned
with the map axes (cell centres inside the outline), every cell is a column from the ground to the roof, and the
columns are merged into boxes with pcd2sim.voxel.merge_boxes (runs along x, then equal runs along y).
The ground is flat: square tiles whose tops sit at ground_z, all in the contact set.

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
import urllib.parse
import urllib.request
from collections import Counter

import numpy as np

from quadsim.gnss import Datum, enu_to_lla, map_to_enu

from . import SCENE_FORMAT
from .osm_buildings import Building, parse_overpass, parse_streets_gl_url
from .pcd_io import write_pcd
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


def fetch_overpass(path: str, bbox: tuple[float, float, float, float], urls=OVERPASS_URLS) -> None:
    """Download once: an existing file is the cache (delete it to refetch). Written whole or not at all."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = urllib.parse.urlencode({"data": overpass_query(bbox)}).encode()
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


# -- scene ------------------------------------------------------------------------------------------------------
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


def write_scene(out: str, boxes: np.ndarray, contact: np.ndarray, voxel: float, spawn: list[float],
                meta_extra: dict) -> dict:
    """The scene files quadsim.sim reads. voxels.npz is empty: OSM has no LiDAR intensity or surface detail."""
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "boxes.npy"), boxes.astype(np.float32))
    np.save(os.path.join(out, "contact.npy"), contact.astype(bool))
    stale = os.path.join(out, "ground.npz")
    if os.path.exists(stale):
        os.remove(stale)
    np.savez_compressed(os.path.join(out, "voxels.npz"), keys=np.zeros((0, 3), np.int32),
                        intensity=np.zeros(0, np.float32), zmean=np.zeros(0, np.float32), origin=np.zeros(3),
                        voxel=voxel, dims=np.ones(3, np.int64), contact=np.zeros(0, bool))
    tops = np.column_stack([boxes[:, :2], boxes[:, 2] + boxes[:, 5]]).astype(np.float32)
    write_pcd(os.path.join(out, "map_ds.pcd"), tops, np.full(len(tops), 100.0, np.float32))
    lo, hi = boxes[:, :3] - boxes[:, 3:], boxes[:, :3] + boxes[:, 3:]
    meta = {"format_version": SCENE_FORMAT, "source_format": "osm", "voxel": voxel, "boxes": int(len(boxes)),
            "contact": {"boxes": int(contact.sum())}, "ground": None,
            "bbox_min": lo.min(axis=0).tolist(), "bbox_max": hi.max(axis=0).tolist(),
            "gravity": [0.0, 0.0, -9.81], "spawn": [float(v) for v in spawn], **meta_extra}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build(lat: float, lon: float, radius: float, out: str, osm_path: str, cell: float = 0.5, tile: float = 20.0,
          default_height: float = 6.0, level_height: float = 3.0, alt: float = 0.0, ground_z: float = 0.0) -> dict:
    """A stand-alone OSM scene: map = ENU at (lat, lon), the square [-radius, radius]^2, flat ground at ground_z."""
    frame = Datum(lat, lon, alt, 0.0)
    lo, hi = np.array([-radius, -radius]), np.array([radius, radius])
    fetch_overpass(osm_path, overpass_bbox(frame, lo, hi))
    with open(osm_path) as f:
        buildings, rejected = parse_overpass(json.load(f), frame, default_height, level_height)
    bboxes = buildings_to_boxes(buildings, cell, ground_z, clip=(lo, hi))
    tiles = ground_tiles(lo, hi, tile, ground_z)
    boxes = np.vstack([tiles, bboxes])
    spawn = spawn_point(bboxes, np.zeros(2), lo, hi)
    sources = Counter(b.height_source for b in buildings)
    return write_scene(out, boxes, np.ones(len(boxes), bool), cell, [spawn[0], spawn[1], ground_z], {
        "geo": {"lat": lat, "lon": lon, "alt": alt, "yaw_deg": 0.0, "source": "osm", "radius_m": radius},
        "osm": {"file": os.path.abspath(osm_path), "buildings": len(buildings), "rejected": rejected,
                "height_source": {k: sources.get(k, 0) for k in ("tag", "levels", "default")},
                "default_height": default_height, "level_height": level_height, "cell": cell, "tile": tile,
                "building_boxes": int(len(bboxes)), "attribution": "(c) OpenStreetMap contributors, ODbL"},
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="a streets.gl view URL: its centre is the scene centre")
    where.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--radius", type=float, default=1000.0, help="half side of the scene square (m)")
    ap.add_argument("--out", required=True, help="scene directory, e.g. scenes/osm_tucheng")
    ap.add_argument("--osm", default=None, help="Overpass cache (default Maps/<out name>/overpass.json)")
    ap.add_argument("--cell", type=float, default=0.5, help="footprint raster cell (m)")
    ap.add_argument("--tile", type=float, default=20.0, help="ground tile side (m)")
    ap.add_argument("--default-height", type=float, default=6.0, help="buildings with no height or levels (m)")
    ap.add_argument("--level-height", type=float, default=3.0, help="metres per building:levels")
    ap.add_argument("--alt", type=float, default=0.0, help="ellipsoid height of the ground at the centre (m)")
    a = ap.parse_args(argv)
    lat, lon = parse_streets_gl_url(a.url) if a.url else a.center
    osm = a.osm or os.path.join("Maps", os.path.basename(os.path.normpath(a.out)), "overpass.json")
    meta = build(lat, lon, a.radius, a.out, osm, a.cell, a.tile, a.default_height, a.level_height, a.alt)
    o = meta["osm"]
    print(f"wrote {a.out}: {o['buildings']} buildings ({o['height_source']}, {o['rejected']} refused) -> "
          f"{o['building_boxes']} boxes + ground, spawn {meta['spawn']}")


if __name__ == "__main__":
    main()
