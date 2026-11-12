"""OpenStreetMap buildings (an Overpass `out tags geom` answer) -> footprints in a map frame, as pure functions.

Glossary
  frame        the target map frame: a quadsim.gnss.Datum (origin lat / lon / alt and the heading of map x from
               east). A stand-alone OSM scene uses its centre with heading 0 (map = ENU); a scene laid over dp
               uses dp's datum, so the buildings come out in dp's own axes.
  footprint    the outer ring of a building in map metres, counter-clockwise, without the closing vertex.
  height       metres above the ground, from the first that parses: `height` tag, `building:levels` x
               level_height, default_height. `height_source` says which one, so a scene can report how much
               of it is guessed.
Inner rings (courtyards) of multipolygons are dropped: a building is solid from its outer ring. Outlines that do
not close, cross themselves or cover less than MIN_AREA are refused here, at the boundary, and counted.
streets.gl (https://streets.gl) draws this same OSM data; its URL `#lat,lon,pitch,yaw,distance` names the centre.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

import numpy as np

from quadsim.gnss import Datum, enu_to_map, lla_to_enu

HeightSource = Literal["tag", "levels", "default"]
MIN_AREA = 1.0                                   # m2
_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class Building:
    osm_id: int
    footprint: np.ndarray                        # [N,2] map metres, CCW, N >= 3
    height_m: float
    height_source: HeightSource


def parse_streets_gl_url(url: str) -> tuple[float, float]:
    """https://streets.gl/#lat,lon,pitch,yaw,distance -> (lat, lon)."""
    parts = urlparse(url).fragment.split(",")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"not a streets.gl view URL (#lat,lon,...): {url!r}") from e
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"lat/lon out of range in {url!r}")
    return lat, lon


def _number(raw) -> float | None:
    m = _NUMBER.match(str(raw)) if raw is not None else None
    return float(m.group(1)) if m else None


def building_height(tags: dict, default_height: float, level_height: float) -> tuple[float, HeightSource]:
    h = _number(tags.get("height"))
    if h is not None and h > 0:
        return h, "tag"
    levels = _number(tags.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * level_height, "levels"
    return default_height, "default"


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _self_intersects(xy: np.ndarray) -> bool:
    """True when two non-adjacent edges of the ring cross (proper crossings only)."""
    n = len(xy)
    a, b = xy, np.roll(xy, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))            # the first and last edges share a vertex
    i, j = i[keep], j[keep]

    def orient(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    d1, d2 = orient(a[i], b[i], a[j]), orient(a[i], b[i], b[j])
    d3, d4 = orient(a[j], b[j], a[i]), orient(a[j], b[j], b[i])
    return bool(np.any((d1 * d2 < 0) & (d3 * d4 < 0)))


def ring_to_footprint(geometry: list[dict], frame: Datum) -> np.ndarray | None:
    """Overpass geometry (closed lat/lon ring) -> CCW map footprint, or None when it is not a usable outline."""
    if len(geometry) < 4 or geometry[0] != geometry[-1]:
        return None
    xy = np.array([enu_to_map(lla_to_enu(p["lat"], p["lon"], frame.alt, frame), frame)[:2] for p in geometry[:-1]])
    xy = xy[np.any(np.abs(np.diff(xy, axis=0, append=xy[:1])) > 1e-6, axis=1)]   # repeated vertices
    if len(xy) < 3 or _self_intersects(xy):
        return None
    area = _signed_area(xy)
    if abs(area) < MIN_AREA:
        return None
    return xy if area > 0 else xy[::-1].copy()


def _outer_rings(el: dict) -> list[list[dict]]:
    if el.get("type") == "way":
        return [el.get("geometry") or []]
    if el.get("type") == "relation":
        return [m.get("geometry") or [] for m in el.get("members", [])
                if m.get("type") == "way" and m.get("role", "outer") == "outer"]
    return []


def parse_overpass(data: dict, frame: Datum, default_height: float = 6.0,
                   level_height: float = 3.0) -> tuple[list[Building], int]:
    """(buildings, number of refused outlines) from an Overpass `out tags geom` answer."""
    out: list[Building] = []
    rejected = 0
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if "building" not in tags or tags.get("building") == "no":
            continue
        h, src = building_height(tags, default_height, level_height)
        for ring in _outer_rings(el):
            fp = ring_to_footprint(ring, frame)
            if fp is None:
                rejected += 1
            else:
                out.append(Building(int(el.get("id", 0)), fp, h, src))
    return out, rejected
