"""The MuJoCo viewer's map: boxes grouped by (scene colour, height band) into mesh geoms, so the colour mode can be
switched on a compiled model by writing geom_rgba (no recompile, the passive viewer just redraws)."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from quadsim.map_colors import PLAIN_RGB  # noqa: E402
from quadsim.model import add_map_mesh  # noqa: E402

RED, BLUE, NONE = [200, 20, 20, 255], [20, 20, 200, 255], [0, 0, 0, 0]
BOXES = np.array([[0, 0, 0.1, .5, .5, .1], [3, 0, 10.1, .5, .5, .1], [6, 0, 0.1, .5, .5, .1], [9, 0, 5.1, .5, .5, .1]])
RGBA = np.array([RED, RED, BLUE, NONE], np.uint8)


def _model():
    spec = mujoco.MjSpec()
    paint = add_map_mesh(spec, BOXES, rgba=RGBA)
    return spec.compile(), paint


def _rgb(m, name):
    return np.round(m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)][:3] * 255).astype(int).tolist()


def test_boxes_become_one_geom_per_colour_and_height_band_with_all_their_faces():
    m, paint = _model()
    assert len(paint.geoms) == 4                         # red low, red high, blue low, uncoloured middle
    assert int(m.mesh_facenum.sum()) == 12 * len(BOXES)


def test_the_mode_switches_on_the_compiled_model():
    m, paint = _model()
    paint.apply(m, "plain")
    assert all(_rgb(m, g) == PLAIN_RGB.tolist() for g in paint.geoms)
    paint.apply(m, "osm")
    rgbs = [_rgb(m, g) for g in paint.geoms]
    assert rgbs.count(RED[:3]) == 2 and rgbs.count(BLUE[:3]) == 1
    paint.apply(m, "height")
    heights = [_rgb(m, g) for g in paint.geoms]
    assert RED[:3] not in heights and BLUE[:3] not in heights
    assert len({tuple(c) for c in heights}) == 3          # red low and blue low share their band's colour

def test_a_scene_without_colours_draws_by_height_in_osm_mode():
    spec = mujoco.MjSpec()
    paint = add_map_mesh(spec, BOXES)
    m = spec.compile()
    paint.apply(m, "osm")
    osm = [_rgb(m, g) for g in paint.geoms]
    paint.apply(m, "height")
    assert osm == [_rgb(m, g) for g in paint.geoms]


def test_the_simulator_colours_its_viewer_map_from_the_scene_and_the_config_mode(tmp_path):
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    d = tmp_path / "scene"
    d.mkdir()
    floor = np.array([[0, 0, -0.1, 20, 20, 0.1], [5, 5, 2.0, 1, 1, 2.0]], np.float32)       # ground, a red block
    np.save(d / "boxes.npy", floor)
    np.save(d / "contact.npy", np.ones(2, bool))
    np.save(d / "colors.npy", np.array([NONE, RED], np.uint8))
    np.savez_compressed(d / "voxels.npz", keys=np.zeros((0, 3), np.int32), intensity=np.zeros(0, np.float32),
                        zmean=np.zeros(0, np.float32), origin=np.zeros(3), voxel=0.2, dims=np.ones(3, np.int64),
                        contact=np.zeros(0, bool))
    (d / "meta.json").write_text('{"format_version": 3, "voxel": 0.2, "gravity": [0, 0, -9.81], "spawn": [0, 0, 0]}')
    for mode, red in (("osm", True), ("plain", False)):
        cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=1000", "gnss.datum=null",
                                 f"viewer.map_color={mode}", "spawn=[0.0,0.0,0.0,0.0]"])
        cfg["scene"] = str(d)
        s = Simulator(cfg, visual_map=True)
        assert (RED[:3] in [_rgb(s.m, g) for g in s.map_paint.geoms]) is red
