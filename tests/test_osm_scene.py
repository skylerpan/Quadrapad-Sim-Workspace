"""OSM buildings -> scene boxes: footprints rasterised on a map-aligned grid into full-height columns and merged into
few boxes, clipped to the scene square, standing on flat ground tiles."""
import math

import numpy as np
import pytest

from pcd2sim.osm_buildings import Building
from pcd2sim.osm_scene import buildings_to_boxes, ground_tiles


def _rotated_square(cx, cy, side, deg):
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    h = side / 2
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ r.T + [cx, cy]


def _bldg(fp, h=15.0):
    return Building(1, fp, h, "tag")


def _xy_area(boxes):
    return float(np.sum(4.0 * boxes[:, 3] * boxes[:, 4]))


def test_a_rotated_building_becomes_few_full_height_boxes_covering_its_footprint():
    boxes = buildings_to_boxes([_bldg(_rotated_square(20.0, -7.0, 10.0, 31.0))], cell=0.5, ground_z=0.0)
    assert boxes.dtype == np.float32 and boxes.shape[1] == 6
    assert _xy_area(boxes) == pytest.approx(100.0, rel=0.05)
    bottom, top = boxes[:, 2] - boxes[:, 5], boxes[:, 2] + boxes[:, 5]
    assert np.allclose(bottom, 0.0) and np.allclose(top, 15.0)
    assert np.allclose(boxes[:, :2].mean(axis=0), [20.0, -7.0], atol=1.0)
    assert len(boxes) <= 2 * 10.0 / 0.5                           # about one box per grid row, not one per cell


def test_boxes_stand_on_the_ground_height_and_tile_without_overlap():
    a = buildings_to_boxes([_bldg(_rotated_square(0, 0, 8.0, 0.0), h=6.0)], cell=0.5, ground_z=-0.4)
    assert np.allclose(a[:, 2] - a[:, 5], -0.4) and np.allclose(a[:, 2] + a[:, 5], 5.6)
    assert _xy_area(a) == pytest.approx(64.0, abs=1e-3)           # grid-aligned square: exact, no double counting


def test_buildings_are_clipped_to_the_scene_square():
    fp = _rotated_square(100.0, 0.0, 20.0, 0.0)                   # x 90..110, the square ends at 100
    boxes = buildings_to_boxes([_bldg(fp)], cell=0.5, ground_z=0.0, clip=(np.array([-100.0, -100.0]), np.array([100.0, 100.0])))
    assert (boxes[:, 0] + boxes[:, 3] <= 100.0 + 1e-4).all()
    assert _xy_area(boxes) == pytest.approx(200.0, abs=1e-3)


def test_a_clip_edge_off_the_grid_keeps_only_cells_whose_centre_is_inside():
    fp = _rotated_square(100.0, 0.0, 20.0, 0.0)
    boxes = buildings_to_boxes([_bldg(fp)], cell=0.5, ground_z=0.0, clip=(np.array([-100.0, -100.0]), np.array([99.6, 100.0])))
    assert (boxes[:, 0] + boxes[:, 3]).max() == pytest.approx(99.5)   # the cell centred at 99.75 is out


def test_no_buildings_give_an_empty_box_array():
    boxes = buildings_to_boxes([], cell=0.5, ground_z=0.0)
    assert boxes.shape == (0, 6) and boxes.dtype == np.float32


def test_ground_tiles_cover_the_square_with_their_tops_at_ground_height():
    t = ground_tiles(np.array([-25.0, -10.0]), np.array([25.0, 10.0]), tile=10.0, z=0.3, thickness=0.2)
    assert len(t) == 5 * 2
    assert np.allclose(t[:, 2] + t[:, 5], 0.3) and np.allclose(2 * t[:, 5], 0.2)
    assert _xy_area(t) == pytest.approx(50.0 * 20.0)
    assert (t[:, 0] - t[:, 3]).min() == pytest.approx(-25.0) and (t[:, 1] + t[:, 4]).max() == pytest.approx(10.0)
