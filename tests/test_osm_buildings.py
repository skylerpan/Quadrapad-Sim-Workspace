"""OSM buildings from an Overpass answer: footprints land in the target map frame (datum + heading), heights follow
height tag > building:levels > default, and broken outlines are refused at the boundary."""
import math

import numpy as np
import pytest

from pcd2sim.osm_buildings import parse_overpass, parse_streets_gl_url
from quadsim.gnss import Datum, enu_to_lla

ENU0 = Datum(lat=24.95919, lon=121.41896, alt=0.0, yaw_deg=0.0)


def _geom(enu_xy, close=True):
    pts = [enu_to_lla(np.array([e, n, 0.0]), ENU0) for e, n in enu_xy]
    if close:
        pts.append(pts[0])
    return [{"lat": la, "lon": lo} for la, lo, _ in pts]


def _square(cx, cy, half, **tags):
    ring = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    return {"type": "way", "id": 1, "tags": {"building": "yes", **tags}, "geometry": _geom(ring)}


def _area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_footprint_lands_in_enu_metres_when_the_frame_has_no_heading():
    (b,), rejected = parse_overpass({"elements": [_square(100.0, -40.0, 10.0)]}, ENU0)
    assert rejected == 0
    assert np.allclose(b.footprint.mean(axis=0), [100.0, -40.0], atol=0.01)
    assert _area(b.footprint) == pytest.approx(400.0, rel=1e-3)
    assert len(b.footprint) == 4                                  # the closing vertex is dropped


def test_footprint_is_rotated_into_a_map_frame_with_a_heading():
    frame = Datum(ENU0.lat, ENU0.lon, ENU0.alt, yaw_deg=31.0)
    (b,), _ = parse_overpass({"elements": [_square(100.0, 0.0, 5.0)]}, frame)
    a = math.radians(31.0)                                        # east is 31 deg clockwise of map x
    assert np.allclose(b.footprint.mean(axis=0), [100.0 * math.cos(a), -100.0 * math.sin(a)], atol=0.01)


def test_height_comes_from_the_tag_then_the_levels_then_the_default():
    els = [_square(0, 0, 5, height="12.5 m", **{"building:levels": "9"}),
           _square(50, 0, 5, **{"building:levels": "4"}),
           _square(100, 0, 5),
           _square(150, 0, 5, height="tall", **{"building:levels": "2;3"})]
    bs, _ = parse_overpass({"elements": els}, ENU0, default_height=7.0, level_height=3.0)
    assert [(b.height_m, b.height_source) for b in bs] == [(12.5, "tag"), (12.0, "levels"), (7.0, "default"),
                                                            (6.0, "levels")]


def test_broken_outlines_are_refused_and_counted():
    open_way = {"type": "way", "id": 2, "tags": {"building": "yes"}, "geometry": _geom([(0, 0), (10, 0), (10, 10)], close=False)}
    bowtie = {"type": "way", "id": 3, "tags": {"building": "yes"}, "geometry": _geom([(0, 0), (10, 10), (10, 0), (0, 10)])}
    sliver = _square(0, 0, 0.2)                                   # 0.16 m2: not a building at this scale
    bs, rejected = parse_overpass({"elements": [open_way, bowtie, sliver, _square(30, 30, 5)]}, ENU0)
    assert len(bs) == 1 and rejected == 3


def test_a_multipolygon_relation_contributes_its_closed_outer_rings():
    rel = {"type": "relation", "id": 9, "tags": {"building": "yes", "type": "multipolygon", "building:levels": "2"},
           "members": [{"type": "way", "role": "outer", "geometry": _square(0, 0, 10)["geometry"]},
                       {"type": "way", "role": "inner", "geometry": _square(0, 0, 3)["geometry"]}]}
    (b,), _ = parse_overpass({"elements": [rel]}, ENU0)
    assert _area(b.footprint) == pytest.approx(400.0, rel=1e-3) and b.height_m == 6.0


def test_streets_gl_url_gives_the_centre():
    assert parse_streets_gl_url("https://streets.gl/#24.95919,121.41896,45.00,0.00,586.00") == (24.95919, 121.41896)
    with pytest.raises(ValueError):
        parse_streets_gl_url("https://streets.gl/")
