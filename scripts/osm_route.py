#!/usr/bin/env python3
"""A walking route between two lat/lon points along the OSM ways the scene was built from, as map-frame waypoints.

  python3 scripts/osm_route.py --scene scenes/osm_tucheng_dp --from 24.9612184,121.4217127 --to 24.9587377,121.417502 \
      -o config/route_danuan_milksha.yaml

Shortest path (Dijkstra) over the `highway` ways of Maps/osm_tucheng/overpass_ground.json that a pedestrian can
use (no motorway / trunk, no steps, nothing under construction), joined at shared coordinates. Points are converted
with the scene's datum (quadsim.gnss) and resampled every --step metres. The route is then checked against the
scene's boxes: any waypoint closer than --clearance to something taller than --obstacle-height above the local
ground is reported (and listed in the output), since OSM roads and OSM buildings do not always agree.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quadsim.config import load_config, load_scene_meta  # noqa: E402
from quadsim.gnss import enu_to_map, lla_to_enu, resolve_datum  # noqa: E402

NO_WALK = {"motorway", "motorway_link", "trunk", "trunk_link", "steps", "construction", "proposed"}


def build_graph(osm_path: str):
    """nodes: {(lat, lon)} -> index; adjacency lists with metre weights."""
    with open(osm_path) as f:
        elements = json.load(f)["elements"]
    index: dict[tuple[float, float], int] = {}
    coords: list[tuple[float, float]] = []
    adj: list[list[tuple[int, float, str]]] = []

    def node(lat, lon):
        key = (round(lat, 7), round(lon, 7))
        if key not in index:
            index[key] = len(coords)
            coords.append(key)
            adj.append([])
        return index[key]

    for e in elements:
        tags = e.get("tags", {})
        hw = tags.get("highway")
        if e["type"] != "way" or not hw or hw in NO_WALK or tags.get("foot") == "no" or "geometry" not in e:
            continue
        name = tags.get("name", hw)
        pts = e["geometry"]
        for a, b in zip(pts, pts[1:]):
            i, j = node(a["lat"], a["lon"]), node(b["lat"], b["lon"])
            d = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            adj[i].append((j, d, name))
            adj[j].append((i, d, name))
    return coords, adj


def haversine(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest(coords, lat, lon):
    best = min(range(len(coords)), key=lambda i: haversine(lat, lon, *coords[i]))
    return best, haversine(lat, lon, *coords[best])


def dijkstra(adj, src, dst):
    dist = {src: 0.0}
    prev: dict[int, tuple[int, str]] = {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w, name in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = (u, name)
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        return None, None, None
    path, names = [dst], []
    while path[-1] != src:
        u, name = prev[path[-1]]
        path.append(u)
        names.append(name)
    return path[::-1], names[::-1], dist[dst]


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = s[np.concatenate([[True], seg > 1e-6])]
    xy = xy[np.concatenate([[True], seg > 1e-6])]
    q = np.arange(0.0, keep[-1], step)
    out = np.stack([np.interp(q, keep, xy[:, 0]), np.interp(q, keep, xy[:, 1])], axis=1)
    return np.vstack([out, xy[-1]])


def obstacle_check(scene: str, xy: np.ndarray, clearance: float, obstacle_height: float):
    """Indices of waypoints with a box rising more than obstacle_height above the local ground within clearance."""
    boxes = np.load(os.path.join(scene, "boxes.npy"))            # N x 6: centre xyz, half sizes xyz
    c, h = boxes[:, :3], boxes[:, 3:6]
    lo, hi = c - h, c + h
    bad, ground = [], []
    for i, (x, y) in enumerate(xy):
        near = (lo[:, 0] < x + clearance) & (hi[:, 0] > x - clearance) & (lo[:, 1] < y + clearance) & (hi[:, 1] > y - clearance)
        under = (lo[:, 0] <= x) & (hi[:, 0] >= x) & (lo[:, 1] <= y) & (hi[:, 1] >= y)
        if not under.any():
            ground.append(float("nan"))
            bad.append(i)                                        # nothing to stand on
            continue
        g = float(hi[under, 2].min())                            # the lowest surface under the point
        ground.append(g)
        if (hi[near, 2] > g + obstacle_height).any():
            bad.append(i)
    return bad, ground


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--osm", default="Maps/osm_tucheng/overpass_ground.json")
    ap.add_argument("--from", dest="src", required=True, help="lat,lon")
    ap.add_argument("--to", dest="dst", required=True, help="lat,lon")
    ap.add_argument("--step", type=float, default=2.0, help="waypoint spacing (m)")
    ap.add_argument("--clearance", type=float, default=0.6)
    ap.add_argument("--obstacle-height", type=float, default=0.25)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    scene = a.scene or cfg["scene"]
    datum = resolve_datum(load_scene_meta(scene), cfg["gnss"].get("datum"))
    src = tuple(float(v) for v in a.src.split(","))
    dst = tuple(float(v) for v in a.dst.split(","))

    coords, adj = build_graph(a.osm)
    i, di = nearest(coords, *src)
    j, dj = nearest(coords, *dst)
    path, names, length = dijkstra(adj, i, j)
    if path is None:
        print("no walkable OSM path between the two points", file=sys.stderr)
        return 1
    xy = np.array([enu_to_map(lla_to_enu(lat, lon, datum.alt, datum), datum)[:2] for lat, lon in (coords[k] for k in path)])
    wp = resample(xy, a.step)
    bad, ground = obstacle_check(scene, wp, a.clearance, a.obstacle_height)
    streets = [n for k, n in enumerate(names) if k == 0 or n != names[k - 1]]
    print(f"route {length:.0f} m over {len(path)} OSM nodes, {len(wp)} waypoints; start node {di:.1f} m from the "
          f"requested start, end node {dj:.1f} m from the requested end")
    print("via: " + " -> ".join(streets))
    print(f"map frame: start ({wp[0][0]:.1f}, {wp[0][1]:.1f}), end ({wp[-1][0]:.1f}, {wp[-1][1]:.1f}); "
          f"ground z {np.nanmin(ground):.2f} .. {np.nanmax(ground):.2f}")
    print(f"waypoints blocked or without ground: {len(bad)}" + (f" (first at index {bad[0]}: {wp[bad[0]].round(1).tolist()})" if bad else ""))
    if a.output:
        with open(a.output, "w") as f:
            yaml.safe_dump({"scene": scene, "from": list(src), "to": list(dst), "length_m": round(float(length), 1),
                            "via": streets, "blocked": bad,
                            "waypoints": [[round(float(x), 2), round(float(y), 2)] for x, y in wp]}, f,
                           default_flow_style=None, width=200)
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
