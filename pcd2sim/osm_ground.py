"""OpenStreetMap ground (roads, paths, parking, parks, water, forest, farmland, built-up land) -> ground features in a
map frame, and their paint: one GroundClass per ground cell, which the OSM scene turns into coloured ground boxes.

Glossary
  feature   a Line (a road or path: map polyline + width in metres) or an Area (a closed map ring), with its
            pcd2sim.osm_colors.GroundClass and OSM id. Parsed here, at the boundary, from an Overpass `out tags
            geom` answer; ways and relations whose tags say nothing about the ground are skipped.
  ring      an Area's outline. A way is a ring when it closes; a multipolygon relation's outer member ways are
            joined end to end into rings (open leftovers are dropped). Inner rings (holes) are dropped: the
            area is painted solid, like a building.
  paint     uint8 [nx, ny] class per cell of the scene square's grid (cell centres at lo + (i + 0.5) * cell),
            GroundClass.BARE where nothing is; every feature paints the cells whose centre it covers (a line:
            within width / 2 of it) and a higher class paints over a lower one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadsim.gnss import Datum, enu_to_map, lla_to_enu

from .osm_colors import GroundClass, ground_class, road_width


@dataclass(frozen=True)
class Line:
    osm_id: int
    cls: GroundClass
    xy: np.ndarray                               # [N,2] map metres, N >= 2
    width: float


@dataclass(frozen=True)
class Area:
    osm_id: int
    cls: GroundClass
    ring: np.ndarray                             # [N,2] (N >= 3) map metres, without the closing vertex


Feature = Line | Area


def overpass_ground_query(bbox: tuple[float, float, float, float]) -> str:
    b = ",".join(f"{v:.6f}" for v in bbox)
    parts = [f'way["highway"]({b});', f'way["amenity"="parking"]({b});']
    for key in ("landuse", "leisure", "natural", "waterway"):
        parts += [f'way["{key}"]({b});', f'relation["{key}"]["type"="multipolygon"]({b});']
    return "[out:json][timeout:180];(" + "".join(parts) + ");out tags geom;"


def _xy(geometry: list[dict], frame: Datum) -> np.ndarray:
    return np.array([enu_to_map(lla_to_enu(p["lat"], p["lon"], frame.alt, frame), frame)[:2] for p in geometry])


def _key(p: dict) -> tuple[float, float]:
    return p["lat"], p["lon"]


def _join_rings(segments: list[list[dict]]) -> list[list[dict]]:
    """Outer member ways joined end to end into closed rings (OSM multipolygons split outlines into ways)."""
    rings, open_ = [], [s for s in segments if len(s) >= 2]
    while open_:
        ring = list(open_.pop())
        grew = True
        while _key(ring[0]) != _key(ring[-1]) and grew:
            grew = False
            for i, s in enumerate(open_):
                if _key(s[0]) == _key(ring[-1]):
                    ring += s[1:]
                elif _key(s[-1]) == _key(ring[-1]):
                    ring += s[::-1][1:]
                else:
                    continue
                open_.pop(i)
                grew = True
                break
        if len(ring) >= 4 and _key(ring[0]) == _key(ring[-1]):
            rings.append(ring)
    return rings


def _rings(el: dict) -> list[list[dict]]:
    if el.get("type") == "way":
        g = el.get("geometry") or []
        return [g] if len(g) >= 4 and _key(g[0]) == _key(g[-1]) else []
    if el.get("type") == "relation":
        return _join_rings([m.get("geometry") or [] for m in el.get("members", [])
                            if m.get("type") == "way" and m.get("role", "outer") == "outer"])
    return []


def parse_overpass_ground(data: dict, frame: Datum) -> list[Feature]:
    """The ground features of an Overpass answer, in the map frame."""
    out: list[Feature] = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        cls = ground_class(tags)
        if cls is None or el.get("type") not in ("way", "relation"):
            continue
        osm_id = int(el.get("id", 0))
        if "highway" in tags and tags.get("area") != "yes":
            g = el.get("geometry") or []
            if el.get("type") == "way" and len(g) >= 2:
                out.append(Line(osm_id, cls, _xy(g, frame), road_width(tags)))
            continue
        out += [Area(osm_id, cls, _xy(r[:-1], frame)) for r in _rings(el)]
    return out


def _fill_ring(grid: np.ndarray, ring: np.ndarray, lo: np.ndarray, cell: float, cls: int) -> None:
    """Scanline even-odd fill: the cells whose centre lies inside the ring."""
    a, b = ring, np.roll(ring, -1, axis=0)
    ny = grid.shape[1]
    y0 = np.minimum(a[:, 1], b[:, 1])
    y1 = np.maximum(a[:, 1], b[:, 1])
    j0 = np.clip(np.ceil((y0 - lo[1]) / cell - 0.5), 0, ny).astype(np.int64)   # rows whose centre is in [y0, y1)
    j1 = np.clip(np.ceil((y1 - lo[1]) / cell - 0.5), 0, ny).astype(np.int64)
    n = j1 - j0
    if n.sum() == 0:
        return
    edge = np.repeat(np.arange(len(a)), n)
    row = np.repeat(j0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    yc = lo[1] + (row + 0.5) * cell
    ea, eb = a[edge], b[edge]
    x = ea[:, 0] + (yc - ea[:, 1]) * (eb[:, 0] - ea[:, 0]) / (eb[:, 1] - ea[:, 1])
    order = np.lexsort((x, row))
    row, x = row[order], x[order]
    starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    pos = np.arange(len(row)) - np.repeat(starts, np.diff(np.r_[starts, len(row)]))
    enter = pos % 2 == 0
    nx = grid.shape[0]
    xi0 = np.clip(np.ceil((x[enter] - lo[0]) / cell - 0.5), 0, nx).astype(np.int64)
    exit_ = np.flatnonzero(enter) + 1
    ok = exit_ < len(x)
    xi1 = np.zeros_like(xi0)
    xi1[ok] = np.clip(np.ceil((x[exit_[ok]] - lo[0]) / cell - 0.5), 0, nx).astype(np.int64)
    for r, i0, i1 in zip(row[enter][ok], xi0[ok], xi1[ok]):
        if i1 > i0:
            np.maximum(grid[i0:i1, r], cls, out=grid[i0:i1, r])


def _paint_line(grid: np.ndarray, xy: np.ndarray, width: float, lo: np.ndarray, cell: float, cls: int) -> None:
    """The cells whose centre lies within width / 2 of the polyline."""
    r = width / 2.0
    shape = np.array(grid.shape)
    for p, q in zip(xy[:-1], xy[1:]):
        i0 = np.clip(np.floor((np.minimum(p, q) - r - lo) / cell), 0, shape).astype(np.int64)
        i1 = np.clip(np.ceil((np.maximum(p, q) + r - lo) / cell) + 1, 0, shape).astype(np.int64)
        if np.any(i1 <= i0):
            continue
        cx = lo[0] + (np.arange(i0[0], i1[0]) + 0.5) * cell
        cy = lo[1] + (np.arange(i0[1], i1[1]) + 0.5) * cell
        c = np.stack(np.meshgrid(cx, cy, indexing="ij"), axis=-1)
        d = q - p
        t = np.clip(((c - p) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
        near = np.linalg.norm(c - (p + t[..., None] * d), axis=-1) <= r
        sub = grid[i0[0]:i1[0], i0[1]:i1[1]]
        sub[near] = np.maximum(sub[near], cls)


def paint_ground(features: list[Feature], lo: np.ndarray, hi: np.ndarray, cell: float) -> np.ndarray:
    """uint8 [nx, ny] GroundClass per cell of the square [lo, hi] (lo, hi on the cell grid)."""
    shape = np.round((np.asarray(hi) - np.asarray(lo)) / cell).astype(np.int64)
    grid = np.full(shape, GroundClass.BARE, np.uint8)
    lo = np.asarray(lo, np.float64)
    for f in features:
        if isinstance(f, Area):
            _fill_ring(grid, f.ring, lo, cell, int(f.cls))
        else:
            _paint_line(grid, f.xy, f.width, lo, cell, int(f.cls))
    return grid
