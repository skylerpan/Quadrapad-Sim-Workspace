"""Clearing a base scan: map-frame rectangles (a gate, a doorway) where what stands on the ground is cut away.

Glossary
  clear rectangle   [x0, y0, x1, y1] in the base's map frame, x0 < x1 and y0 < y1 (checked where it comes in).
  ground-standing   a box that starts, or a point that lies, below `below` (the floor + osm_scene.Apron.LOW_BAND):
                    floors, curbs, gates, scan noise. Higher things (a canopy, a ceiling) stay.
  piece             what is left of a box outside a rectangle: up to four boxes (west, east, then south and north
                    of the rectangle's x span), each keeping its source row (its contact flag and colour).
"""
from __future__ import annotations

import numpy as np


def parse_rects(rects) -> np.ndarray:
    """Clear rectangles -> float [K, 4]; refuses one with no area."""
    r = np.asarray(list(rects or []), dtype=np.float64).reshape(-1, 4)
    bad = ~((r[:, 0] < r[:, 2]) & (r[:, 1] < r[:, 3]))
    if bad.any():
        raise ValueError(f"clear rectangle {r[bad][0].tolist()} has no area: give x0 y0 x1 y1 with x0 < x1, y0 < y1")
    return r


def _span(lo, hi, z, hz):
    """Boxes from xy corners and the z centre / half height."""
    return np.column_stack([(lo + hi) / 2, z, (hi - lo) / 2, hz])


def _subtract(boxes: np.ndarray, rows: np.ndarray, rect: np.ndarray, low: np.ndarray):
    lo, hi = boxes[:, :2] - boxes[:, 3:5], boxes[:, :2] + boxes[:, 3:5]
    hit = low & np.all((lo < rect[2:]) & (hi > rect[:2]), axis=1)
    keep = [boxes[~hit]], [rows[~hit]], [low[~hit]]
    lo, hi, z, hz, src = lo[hit], hi[hit], boxes[hit, 2], boxes[hit, 5], rows[hit]
    mid_lo, mid_hi = np.maximum(lo[:, 0], rect[0]), np.minimum(hi[:, 0], rect[2])
    cuts = (  # (where, piece lo, piece hi)
        (lo[:, 0] < rect[0], lo, np.column_stack([np.full(len(lo), rect[0]), hi[:, 1]])),
        (hi[:, 0] > rect[2], np.column_stack([np.full(len(lo), rect[2]), lo[:, 1]]), hi),
        (lo[:, 1] < rect[1], np.column_stack([mid_lo, lo[:, 1]]), np.column_stack([mid_hi, np.full(len(lo), rect[1])])),
        (hi[:, 1] > rect[3], np.column_stack([mid_lo, np.full(len(lo), rect[3])]), np.column_stack([mid_hi, hi[:, 1]])))
    for where, plo, phi in cuts:
        keep[0].append(_span(plo[where], phi[where], z[where], hz[where]))
        keep[1].append(src[where])
        keep[2].append(np.ones(int(where.sum()), bool))
    return tuple(np.concatenate(k) for k in keep)


def clear_boxes(boxes: np.ndarray, rects: np.ndarray, below: float) -> tuple[np.ndarray, np.ndarray]:
    """(boxes, source row of each) with every ground-standing box cut out of the rectangles; the rest unchanged."""
    out = np.asarray(boxes, dtype=np.float64)
    rows = np.arange(len(out))
    low = out[:, 2] - out[:, 5] < below
    for rect in rects:
        out, rows, low = _subtract(out, rows, rect, low)
    return out, rows


def points_kept(xyz: np.ndarray, rects: np.ndarray, below: float) -> np.ndarray:
    """Which points stay: all but the ground-standing ones inside a rectangle."""
    xyz = np.asarray(xyz)
    inside = np.zeros(len(xyz), bool)
    for x0, y0, x1, y1 in rects:
        inside |= (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
    return ~(inside & (xyz[:, 2] < below))


def rect_boxes(rects: np.ndarray) -> np.ndarray:
    """The rectangles as flat boxes (xy is what counts): for cutting OSM buildings out of them."""
    return _span(rects[:, :2], rects[:, 2:], np.zeros(len(rects)), np.zeros(len(rects)))
