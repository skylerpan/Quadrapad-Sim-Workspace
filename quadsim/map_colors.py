"""Map colours for both viewers (MuJoCo window and browser): the one place that decides what colour a map box or
map point gets.

Glossary
  scene colour  uint8 RGBA per box (scene colors.npy) or per map point (map_ds_colors.npy): A = 255 is a colour
                the source gave (OSM tags and ground classes, pcd2sim.osm_colors), A = 0 is none (a scan).
  map colour    what the viewers draw, by mode:
                  osm     the scene colour where there is one, else the height colour
                  height  the turbo height colormap for everything
                  plain   one grey (the MuJoCo viewer's look before colours)
"""
from __future__ import annotations

from typing import Literal, get_args

import numpy as np

MapColor = Literal["osm", "height", "plain"]
MAP_COLORS: tuple[str, ...] = get_args(MapColor)
PLAIN_RGB = np.array([140, 148, 158], np.uint8)

# turbo colormap, 9 stops
_TURBO = np.array([[48, 18, 59], [68, 90, 204], [37, 160, 253], [31, 221, 198], [103, 253, 115],
                   [194, 241, 60], [247, 183, 39], [235, 93, 11], [122, 4, 3]], dtype=np.float64) / 255.0


def parse_map_color(value) -> MapColor:
    """A map colour mode from config or the browser; anything else is refused here, at the boundary."""
    v = str(value).strip().lower()
    if v not in MAP_COLORS:
        raise ValueError(f"map_color must be one of {MAP_COLORS}, not {value!r}")
    return v  # type: ignore[return-value]


def height_range(z: np.ndarray) -> tuple[float, float]:
    """The 1st-99th percentile of the heights: the span the height colormap stretches over."""
    lo, hi = np.percentile(np.asarray(z, dtype=np.float64), [1.0, 99.0])
    return float(lo), float(hi)


def height_colors(z: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """uint8 [N,3] colours from height, turbo-like, clipped to [lo, hi] (default: 1st-99th percentile)."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if lo is None or hi is None:
        p_lo, p_hi = height_range(z)
        lo = p_lo if lo is None else lo
        hi = p_hi if hi is None else hi
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    u = np.clip((z - lo) / (hi - lo), 0.0, 1.0) * (len(_TURBO) - 1)
    i = np.minimum(u.astype(np.int64), len(_TURBO) - 2)
    f = (u - i)[:, None]
    rgb = _TURBO[i] * (1.0 - f) + _TURBO[i + 1] * f
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def map_colors(z: np.ndarray, rgba: np.ndarray | None, mode: MapColor,
               lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """uint8 [N,3] map colour per box / point from its height z and its scene colour (None = a scene without)."""
    z = np.asarray(z, dtype=np.float64)
    if mode == "plain":
        return np.tile(PLAIN_RGB, (len(z), 1))
    rgb = height_colors(z, lo, hi)
    if mode == "osm" and rgba is not None:
        own = rgba[:, 3] > 0
        rgb[own] = rgba[own, :3]
    return rgb
