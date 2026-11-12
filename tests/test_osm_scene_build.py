"""pcd2sim.osm_scene.build: an Overpass file -> a scene directory the simulator loads, georeferenced, with the robot
spawned in the open, rebuilt to the same bytes."""
import json
import os

import numpy as np
import pytest

from pcd2sim.osm_scene import build
from quadsim.gnss import enu_to_lla
from tests.osm_fixture import ENU0, square


def _osm(tmp_path, *elements):
    p = tmp_path / "overpass.json"
    p.write_text(json.dumps({"elements": list(elements)}))
    return str(p)


def _build(tmp_path, osm, **kw):
    out = str(tmp_path / "scene")
    meta = build(ENU0.lat, ENU0.lon, radius=60.0, out=out, osm_path=osm, **kw)
    return out, meta


def test_the_scene_is_georeferenced_and_spawns_outside_the_buildings(tmp_path):
    osm = _osm(tmp_path, square(0.0, 0.0, 8.0, height="20"), square(30.0, 0.0, 5.0, osm_id=2))
    out, meta = _build(tmp_path, osm)
    g = meta["geo"]
    assert (g["lat"], g["lon"], g["alt"], g["yaw_deg"]) == (ENU0.lat, ENU0.lon, 0.0, 0.0)
    assert meta["geo"]["source"] == "osm" and meta["osm"]["buildings"] == 2
    assert meta["osm"]["height_source"] == {"tag": 1, "levels": 0, "default": 1}
    boxes = np.load(os.path.join(out, "boxes.npy"))
    contact = np.load(os.path.join(out, "contact.npy"))
    assert contact.dtype == bool and contact.all() and len(contact) == len(boxes) == meta["boxes"]
    x, y, z = meta["spawn"]
    assert z == 0.0 and not (abs(x) <= 9.0 and abs(y) <= 9.0)          # out of the 16 m building and a margin
    assert (np.abs([x, y]) <= 60.0).all()


def test_a_rebuild_writes_the_same_scene(tmp_path):
    osm = _osm(tmp_path, square(10.0, 10.0, 6.0))
    out, _ = _build(tmp_path, osm)
    first = {f: open(os.path.join(out, f), "rb").read() for f in ("boxes.npy", "contact.npy")}
    _build(tmp_path, osm)
    assert first == {f: open(os.path.join(out, f), "rb").read() for f in ("boxes.npy", "contact.npy")}


def test_the_simulator_loads_it_and_reports_fixes_at_the_scene_datum(tmp_path):
    pytest.importorskip("mujoco")
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    out, meta = _build(tmp_path, _osm(tmp_path, square(0.0, 0.0, 8.0, height="20")))
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=2000", "spawn=null", "gnss.sky.mode=map"])
    cfg["scene"] = out                                                # config gnss.datum is dp's: the geo must win
    s = Simulator(cfg)
    fixes = []
    for _ in range(15):
        fixes += s.control_tick().gnss
    assert fixes and fixes[-1].fix != "none"                          # in the open next to one building
    pos, *_ = s.ground_truth()
    lat, lon, _ = enu_to_lla(np.array([pos[0], pos[1], 0.0]), ENU0)
    assert abs(fixes[-1].lat - lat) * 111e3 < 10 and abs(fixes[-1].lon - lon) * 101e3 < 10


def test_a_scene_without_voxels_answers_every_lookup_with_the_fallback(tmp_path):
    pytest.importorskip("open3d")
    from quadsim.raycast import VoxelLookup
    out, _ = _build(tmp_path, _osm(tmp_path))
    v = VoxelLookup(os.path.join(out, "voxels.npz"))
    pts = np.array([[0.0, 0.0, -0.1], [5.0, 5.0, 3.0]])
    assert np.array_equal(v.surface_z(pts, np.array([1.0, 2.0])), [1.0, 2.0])
    assert np.array_equal(v.reflectivity(pts), [0, 0])
