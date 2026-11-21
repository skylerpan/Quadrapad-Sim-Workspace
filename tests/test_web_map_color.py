"""Switching the map colouring from the browser: the dropdown's request is taken on the main thread (as sim_node does)
and both viewers follow, the MuJoCo map geoms and the browser's map cloud, without a restart."""
import json
import time

import numpy as np
import pytest

pytest.importorskip("viser")
mujoco = pytest.importorskip("mujoco")

from pcd2sim.osm_colors import BUILDING_PALETTE  # noqa: E402
from pcd2sim.osm_scene import build  # noqa: E402
from quadsim.config import load_config  # noqa: E402
from quadsim.map_colors import PLAIN_RGB  # noqa: E402
from quadsim.sim import Simulator  # noqa: E402
from quadsim.web_viewer import WebViewer  # noqa: E402
from tests.osm_fixture import ENU0, square  # noqa: E402

SCHOOL = list(BUILDING_PALETTE["school"])


def _viewers(tmp_path, port, kind="points"):
    osm = tmp_path / "overpass.json"
    osm.write_text(json.dumps({"elements": [square(15.0, 15.0, 5.0, building="school")]}))
    out = str(tmp_path / "scene")
    build(ENU0.lat, ENU0.lon, radius=30.0, out=out, osm_path=str(osm))
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=1000", "gnss.datum=null", "spawn=null",
                             f"web.port={port}", f"web.map={kind}", "viewer.map_color=osm"])
    cfg["scene"] = out
    sim = Simulator(cfg, visual_map=True)
    web = WebViewer(cfg, sim)
    t0 = time.time()
    while web.map_info in ("loading",) and time.time() - t0 < 20:
        time.sleep(0.05)
    return sim, web


def _mujoco_rgbs(sim):
    ids = [mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sim.map_paint.geoms]
    return {tuple(np.round(sim.m.geom_rgba[i][:3] * 255).astype(int)) for i in ids}


def _layer_rgbs(handle):
    c = handle.colors if hasattr(handle, "colors") else handle.batched_colors
    return {tuple(int(v) for v in x) for x in np.asarray(c).reshape(-1, 3)}


def _web_rgbs(web):
    return set().union(*(_layer_rgbs(h) for h in web.map_layers))


def _school_shaded(web):
    return tuple((np.array(SCHOOL) * 0.85 + 0.5).astype(int)) in _web_rgbs(web)


def _choose(sim, web, label):
    web.request_map_color(label)                         # what the dropdown's callback does (viser thread)
    mode = web.take_map_color()                          # sim_node's main loop, next tick
    sim.set_map_color(mode)
    web.show_map_color(mode)
    return mode


@pytest.mark.parametrize("kind, port", [("points", 8931), ("boxes", 8932)])
def test_the_browser_switches_the_map_colouring_of_both_viewers(tmp_path, kind, port):
    sim, web = _viewers(tmp_path, port, kind)
    try:
        assert web.take_map_color() is None                       # nothing asked yet
        assert tuple(SCHOOL) in _mujoco_rgbs(sim)
        assert tuple(SCHOOL) in _web_rgbs(web) or kind == "boxes" and _school_shaded(web)
        assert _choose(sim, web, "Plain") == "plain"
        shade = {tuple(PLAIN_RGB), tuple((PLAIN_RGB * 0.85 + 0.5).astype(int))}     # boxes: checkerboard shade
        assert _mujoco_rgbs(sim) == {tuple(PLAIN_RGB)} and _web_rgbs(web) <= shade
        assert _choose(sim, web, "Height") == "height"
        assert tuple(SCHOOL) not in _mujoco_rgbs(sim) and len(_web_rgbs(web)) > 1
        _choose(sim, web, "OSM")
        assert tuple(SCHOOL) in _mujoco_rgbs(sim)
        assert tuple(SCHOOL) in _web_rgbs(web) or kind == "boxes" and _school_shaded(web)
    finally:
        web.server.stop()


def test_a_bad_mode_in_the_config_is_refused_at_start(tmp_path):
    cfg = load_config(None, ["robot.type=ideal", "gnss.datum=null", "viewer.map_color=rainbow"])
    with pytest.raises(ValueError, match="map_color"):
        Simulator(cfg)


def test_points_mode_draws_the_scan_as_points_and_the_osm_part_as_boxes(tmp_path):
    """OSM boxes have one map_ds point each (a 20 m ground tile = one point): in points mode they are drawn as boxes,
    and their stand-in points are left out of the cloud; the scan stays points."""
    from pcd2sim.pcd_io import write_pcd
    from tests.test_osm_overlay import BASE_DATUM, _base, _osm_at_base_xy
    base, _ = _base(tmp_path)
    scan = np.random.default_rng(0).uniform([-5, -5, -0.4], [5, 5, 3.0], (500, 3)).astype(np.float32)
    write_pcd(f"{base}/map_ds.pcd", scan, np.full(len(scan), 50.0, np.float32))
    out = str(tmp_path / "scene")
    build(BASE_DATUM.lat, BASE_DATUM.lon, radius=40.0, out=out, base=base,
          osm_path=_osm_at_base_xy(tmp_path, (25.0, 0.0, 5.0, 2)))
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=1000", "gnss.datum=null", "web.port=8933",
                             "web.map=points", "viewer.map_color=osm"])
    cfg["scene"] = out
    sim = Simulator(cfg)
    web = WebViewer(cfg, sim)
    try:
        t0 = time.time()
        while web.map_info == "loading" and time.time() - t0 < 20:
            time.sleep(0.05)
        cloud, boxes = web.map_layers
        assert len(np.asarray(cloud.points)) == len(scan)                         # the scan only
        assert len(np.asarray(boxes.batched_positions)) == int((sim.colors[:, 3] > 0).sum())
        _choose(sim, web, "Plain")
        assert all(_layer_rgbs(h) <= {tuple(PLAIN_RGB), tuple((PLAIN_RGB * 0.85 + 0.5).astype(int))}
                   for h in web.map_layers)
    finally:
        web.server.stop()


def test_the_map_build_is_logged_with_what_was_drawn(tmp_path):
    lines = []
    osm = tmp_path / "overpass.json"
    osm.write_text(json.dumps({"elements": [square(15.0, 15.0, 5.0)]}))
    out = str(tmp_path / "scene")
    build(ENU0.lat, ENU0.lon, radius=30.0, out=out, osm_path=str(osm))
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=1000", "gnss.datum=null", "spawn=null",
                             "web.port=8934"])
    cfg["scene"] = out
    web = WebViewer(cfg, Simulator(cfg), log=lines.append)
    try:
        web._map_thread.join(20)
        assert any(line.startswith("web viewer: map ") and "OSM boxes" in line for line in lines)
    finally:
        web.server.stop()
