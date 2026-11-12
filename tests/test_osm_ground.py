"""pcd2sim.osm_ground: an Overpass answer about the ground (roads, paths, parks, water, landuse) -> ground features in
a map frame, and the feature paint: a class per ground cell."""
import json

import numpy as np
import pytest

from pcd2sim.osm_colors import GroundClass
from pcd2sim.osm_ground import Area, Line, overpass_ground_query, paint_ground, parse_overpass_ground
from tests.osm_fixture import ENU0, geom


def _way(osm_id, xy, close, **tags):
    return {"type": "way", "id": osm_id, "tags": tags, "geometry": geom(xy, close=close)}


ROAD = _way(1, [(-50, 0), (50, 0)], False, highway="residential")                       # 6 m wide, along x
PATH = _way(2, [(0, -30), (0, 30)], False, highway="footway")                            # 2 m, along y
PARK = _way(3, [(10, 10), (40, 10), (40, 40), (10, 40)], True, leisure="park")
STOP = {"type": "node", "id": 4, "lat": 24.95919, "lon": 121.41896, "tags": {"highway": "bus_stop"}}
ANSWER = {"elements": [ROAD, PATH, PARK, STOP, _way(5, [(0, 0), (1, 1)], False, name="nothing")]}


def test_the_answer_becomes_lines_with_widths_and_areas_in_the_map_frame():
    features = parse_overpass_ground(ANSWER, ENU0)
    lines = {f.osm_id: f for f in features if isinstance(f, Line)}
    areas = {f.osm_id: f for f in features if isinstance(f, Area)}
    assert set(lines) == {1, 2} and set(areas) == {3}
    assert lines[1].cls is GroundClass.ROAD and lines[1].width == 6.0
    assert lines[2].cls is GroundClass.PATH and lines[2].width == 2.0
    assert np.allclose(lines[1].xy[[0, -1]], [[-50, 0], [50, 0]], atol=0.05)
    assert areas[3].cls is GroundClass.GRASS and len(areas[3].ring) == 4


def test_the_paint_puts_a_class_on_every_cell_higher_classes_on_top():
    lo, hi, cell = np.array([-60.0, -60.0]), np.array([60.0, 60.0]), 0.5
    grid = paint_ground(parse_overpass_ground(ANSWER, ENU0), lo, hi, cell)
    assert grid.shape == (240, 240) and grid.dtype == np.uint8

    def at(x, y):
        return GroundClass(int(grid[int((x - lo[0]) / cell), int((y - lo[1]) / cell)]))

    assert at(-40, 0.2) is GroundClass.ROAD and at(-40, 2.8) is GroundClass.ROAD          # 6 m wide: |y| <= 3
    assert at(-40, 3.4) is GroundClass.BARE
    assert at(0.2, -20) is GroundClass.PATH and at(1.4, -20) is GroundClass.BARE           # 2 m wide
    assert at(0.2, 0.2) is GroundClass.ROAD                                                # the road over the path
    assert at(25, 25) is GroundClass.GRASS and at(45, 25) is GroundClass.BARE
    assert (grid == GroundClass.ROAD).sum() * cell * cell == pytest.approx(100 * 6, rel=0.05)


def test_the_query_asks_for_roads_and_areas_with_their_geometry():
    q = overpass_ground_query((24.9, 121.4, 25.0, 121.5))
    assert 'way["highway"](24.900000,121.400000,25.000000,121.500000)' in q
    assert 'relation["landuse"]' in q and 'way["amenity"="parking"]' in q and q.endswith("out tags geom;")
    json.dumps(q)


def test_a_multipolygon_split_over_member_ways_is_joined_into_one_area():
    north = geom([(-20, 0), (-20, 10), (20, 10), (20, 0)], close=False)
    south = geom([(-20, 0), (20, 0)], close=False)[::-1]          # runs the other way: joined reversed
    rel = {"type": "relation", "id": 9, "tags": {"type": "multipolygon", "natural": "water"},
           "members": [{"type": "way", "role": "outer", "geometry": north},
                       {"type": "way", "role": "outer", "geometry": south},
                       {"type": "way", "role": "inner", "geometry": geom([(0, 2), (1, 2), (1, 3)])}]}
    (a,) = parse_overpass_ground({"elements": [rel]}, ENU0)
    assert isinstance(a, Area) and a.cls is GroundClass.WATER and len(a.ring) == 4
    grid = paint_ground([a], np.array([-30.0, -10.0]), np.array([30.0, 20.0]), 0.5)
    assert (grid == GroundClass.WATER).sum() * 0.25 == pytest.approx(400.0, rel=0.02)
