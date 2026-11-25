"""scripts/make_slope.py: a smooth inclined plane (oriented collision box + sloped ray slab) with flat run-in and
landing, and a legged robot that rests on it at the plane's height and tilt."""
import json
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("open3d")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_slope  # noqa: E402

from tests.test_g23 import _cfg  # noqa: E402


def _box_corners(c, h, quat):
    import mujoco
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return np.asarray(c) + (signs * h) @ R.reshape(3, 3).T


def test_slope_geometry_is_one_smooth_plane():
    s = make_slope.slope_scene(angle_deg=10.0)
    t = math.tan(math.radians(10.0))
    x0, x1 = s["x0"], s["x1"]
    assert x1 - x0 >= 8.0 and s["width"] >= 6.0                      # at least 8 m x 6 m of incline
    # the ray slab's top face is the plane z = (x - x0) tan(angle)
    slab = s["slabs"][0]
    top = slab[[1, 3, 5, 7]]
    assert np.allclose(top[:, 2], (top[:, 0] - x0) * t, atol=1e-5)
    # the collision solid's top face lies on the same plane, from the foot (x0, 0) to the head (x1, h)
    (solid,) = s["solids"]
    corners = _box_corners(solid[:3], solid[3:6], solid[6:10])
    up = corners[[1, 3, 5, 7]]
    assert np.allclose(up[:, 2], (up[:, 0] - x0) * t, atol=1e-6)
    assert up[:, 0].max() == pytest.approx(x1, abs=1e-6)            # no lip above the landing
    assert up[:, 0].min() < x0                                        # tucked under the run-in floor
    # run-in floor top at 0 before x0, landing top at the plane's head height after x1, both touchable
    floor, landing = s["boxes"][0], s["boxes"][1]
    assert floor[2] + floor[5] == pytest.approx(0.0) and floor[0] + floor[3] == pytest.approx(x0)
    assert landing[2] + landing[5] == pytest.approx((x1 - x0) * t, abs=1e-6)
    assert landing[0] - landing[3] == pytest.approx(x1)
    assert s["contact"][:2].all()


def test_figure8_fits_on_the_incline():
    """Two 1.5 m lobes around the crossing point the scene advertises, plus 0.5 m of margin, stay on the incline."""
    s = make_slope.slope_scene(angle_deg=10.0)
    cx, cy = s["fig8_center"]
    assert s["x0"] + 0.5 <= cx - 1.5 and cx + 1.5 <= s["x1"] - 0.5
    assert abs(cy) + 3.0 + 0.5 <= s["width"] / 2


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("slope") / "slope_10")
    make_slope.write_scene(out, angle_deg=10.0)
    return out


def test_written_scene_is_format_3(scene):
    from pcd2sim import SCENE_FORMAT
    with open(os.path.join(scene, "meta.json")) as f:
        meta = json.load(f)
    assert meta["format_version"] == SCENE_FORMAT
    for name in ("boxes.npy", "contact.npy", "voxels.npz", "ground.npz", "solids.npy", "map_ds.pcd"):
        assert os.path.exists(os.path.join(scene, name)), name


def test_robot_rests_on_the_slope(scene):
    """A limp G23 dropped on the incline settles on it: the terrain rays see the plane's height there and the
    feet touch a surface tilted by the slope angle."""
    from quadsim.sim import Simulator
    t = math.tan(math.radians(10.0))
    meta = make_slope.slope_scene(angle_deg=10.0)
    x = meta["fig8_center"][0]
    zf = (x - meta["x0"]) * t
    sim = Simulator(_cfg(**{"scene": scene, "robot.controller.gt_addr": '""',
                            "robot.controller.ros_ascii_telemetry": "false", "robot.auto_stand": "false",
                            "spawn": f"[{x}, 0.0, 1.5708, {zf}]"}))
    try:
        assert sim.base.ground.z0 == pytest.approx(zf, abs=0.03)
        for _ in range(50):                                 # 1 s
            sim.control_tick()
        normals = []
        for i in range(sim.d.ncon):
            c = sim.d.contact[i]
            normals.append(np.array(c.frame[:3]))
        assert normals, "robot is not touching anything"
        tilt = np.degrees(np.arccos(np.clip(np.abs([n[2] for n in normals]), 0, 1)))
        assert np.allclose(tilt, 10.0, atol=0.5)
        assert sim.d.qpos[2] - zf < 0.4 and sim.d.qpos[2] - zf > 0.0     # lying on it, not fallen through
    finally:
        sim.close()
