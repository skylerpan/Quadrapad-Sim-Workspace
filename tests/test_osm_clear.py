"""osm_scene --clear: a rectangle of the base scan (a gate) where the scan's ground-standing boxes and map points are
cut away, so the OSM ground (eased to the scan's edge heights) runs through and the robot can walk out."""
import math
import os

import numpy as np
import pytest

from pcd2sim.osm_scene import build, main
from pcd2sim.pcd_io import read_pcd, write_pcd
from tests.test_osm_overlay import BASE_DATUM, FLOOR, _base, _box, _osm_at_base_xy

GATE = (10.0, -2.0, 14.0, 2.0)                    # x0, y0, x1, y1 across the scanned yard east of the building


def _gated_base(tmp_path, extra=()):
    """The overlay base, with a 0.6 m curb across its yard at x 12 (what a scanned gate leaves) and map points."""
    base, boxes = _base(tmp_path)
    boxes = np.vstack([boxes, *[[b] for b in extra],
                       [_box(11.8, 12.2, -6, 6, FLOOR, FLOOR + 0.6),                  # the curb
                        _box(6, 16, -6, 6, 2.6, 2.8)]]).astype(np.float32)            # a canopy over the yard
    np.save(os.path.join(base, "boxes.npy"), boxes)
    np.save(os.path.join(base, "contact.npy"), np.ones(len(boxes), bool))
    rgba = np.zeros((len(boxes), 4), np.uint8)
    rgba[-2] = [200, 0, 0, 255]
    np.save(os.path.join(base, "colors.npy"), rgba)
    pts = np.array([[12.0, 0.0, FLOOR + 0.5], [12.0, 0.0, 2.7], [12.0, 5.0, FLOOR + 0.5], [8.0, 0.0, FLOOR]],
                   np.float32)
    write_pcd(os.path.join(base, "map_ds.pcd"), pts, np.arange(len(pts), dtype=np.float32))
    return base, boxes


def _scene(tmp_path, clear, name="scene", osm=(), extra=()):
    base, boxes = _gated_base(tmp_path, extra)
    out = str(tmp_path / name)
    meta = build(BASE_DATUM.lat, BASE_DATUM.lon, radius=50.0, out=out, base=base, clear=clear,
                 osm_path=_osm_at_base_xy(tmp_path, *osm))
    return out, meta


def _at(boxes, x, y):
    return boxes[(np.abs(boxes[:, 0] - x) <= boxes[:, 3] + 1e-6) & (np.abs(boxes[:, 1] - y) <= boxes[:, 4] + 1e-6)]


def test_the_gate_is_cleared_down_to_the_ground_and_the_rest_of_the_scan_is_kept(tmp_path):
    out, meta = _scene(tmp_path, [GATE])
    boxes = np.load(os.path.join(out, "boxes.npy"))
    colors = np.load(os.path.join(out, "colors.npy"))
    contact = np.load(os.path.join(out, "contact.npy"))
    assert len(colors) == len(contact) == len(boxes)
    gate = _at(boxes, 12.0, 0.0)
    tops, bottoms = gate[:, 2] + gate[:, 5], gate[:, 2] - gate[:, 5]
    assert np.all((tops <= FLOOR + 0.05) | (bottoms >= FLOOR + 2.0))               # nothing a robot bumps into
    assert (np.abs(tops - FLOOR) < 0.05).any()                                       # ground to stand on
    assert (bottoms > 2.5).any()                                                     # the canopy above stays
    assert (_at(boxes, 12.0, 4.0)[:, 2] > FLOOR).any()                               # the curb outside the gate stays
    curb = (colors[:, 0] == 200) & (colors[:, 3] == 255)
    assert curb.sum() == 2 and np.all(np.abs(np.abs(boxes[curb, 1]) - 4.0) < 1e-6)   # its two ends, still red
    for x, y in ((8.0, 0.0), (15.0, 0.0), (12.0, 4.0)):                               # the yard floor around the gate
        assert (np.abs(_at(boxes, x, y)[:, 2] + _at(boxes, x, y)[:, 5] - FLOOR) < 1e-6).any(), (x, y)
    assert meta["base"]["clear"] == [list(GATE)]


def test_the_map_points_in_the_gate_are_cleared_too(tmp_path):
    out, _ = _scene(tmp_path, [GATE])
    xyz, inten = read_pcd(os.path.join(out, "map_ds.pcd"))
    base = [tuple(p) for p, i in zip(np.round(xyz, 3), inten) if i < 50]
    assert base == [(12.0, 0.0, 2.7), (12.0, 5.0, round(FLOOR + 0.5, 3)), (8.0, 0.0, FLOOR)]
    assert len(np.load(os.path.join(out, "map_ds_colors.npy"))) == len(xyz)


def test_no_osm_building_is_laid_into_a_cleared_rectangle(tmp_path):
    """A lone patch of scan noise out on the street kept an OSM building away; cleared, it still does."""
    out, _ = _scene(tmp_path, [GATE, (28.0, -2.0, 32.0, 2.0)], osm=[(30.0, 0.0, 1.0, 7)],
                    extra=[_box(29.6, 30.4, -0.4, 0.4, FLOOR, FLOOR + 0.5)])
    boxes = np.load(os.path.join(out, "boxes.npy"))
    assert np.all(_at(boxes, 30.0, 0.0)[:, 2] + _at(boxes, 30.0, 0.0)[:, 5] < FLOOR + 0.05)


def test_a_rectangle_with_no_area_is_refused(tmp_path):
    with pytest.raises(ValueError, match="clear"):
        _scene(tmp_path, [(14.0, -2.0, 10.0, 2.0)])


def test_the_cli_takes_clear_rectangles(tmp_path):
    base, _ = _gated_base(tmp_path)
    out = str(tmp_path / "cli")
    main(["--center", str(BASE_DATUM.lat), str(BASE_DATUM.lon), "--radius", "50", "--out", out, "--base", base,
          "--osm", _osm_at_base_xy(tmp_path), "--no-ground", "--clear", *map(str, GATE)])
    import json
    assert json.load(open(os.path.join(out, "meta.json")))["base"]["clear"] == [list(GATE)]


@pytest.mark.parametrize("clear, through", [([GATE], True), (None, False)])
def test_the_robot_walks_out_through_the_cleared_gate(tmp_path, clear, through):
    pytest.importorskip("mujoco")
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    out, _ = _scene(tmp_path, clear)
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=1000", "gnss.datum=null"])
    cfg["scene"] = out
    sim = Simulator(cfg)
    assert sim.set_pose(9.0, 0.0, 0.0)
    t0 = sim.d.time
    while sim.d.time - t0 < 6.0:
        sim.set_cmd(1.0, 0.0, 0.0)
        sim.control_tick()
    x = float(sim.ground_truth()[0][0])
    assert (x > 14.5) == through, x
    assert math.isfinite(x)
