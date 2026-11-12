"""IdealBase attitude = plane through the four foot heights (a synthetic step, no scene dir)."""
import math

import numpy as np
import pytest

pytest.importorskip("open3d")


def _base(boxes):
    from quadsim.base import IdealBase
    from quadsim.config import load_config
    from quadsim.raycast import RayScene
    cfg = load_config(None, [])["robot"]
    return IdealBase(cfg, RayScene(np.asarray(boxes, dtype=np.float32)), None, blocked=None), cfg


def test_flat_floor_is_level():
    base, _ = _base([[0.0, 0.0, -0.1, 2.0, 2.0, 0.1]])           # top at z = 0
    assert base.reset(0.0, 0.0, 0.0, 0.0)
    g = base.ground
    assert np.allclose(g.feet_h, 0.0, atol=1e-6)
    assert abs(g.z0) < 1e-6 and abs(g.roll) < 1e-6 and abs(g.pitch) < 1e-6


def test_front_feet_one_tread_up_pitch_the_body_through_the_feet():
    # floor top 0 for x < 0, tread top 0.2 for x >= 0; feet at x = +-0.19 -> plane slope 0.2 / 0.38
    base, cfg = _base([[-1.0, 0.0, -0.1, 1.0, 2.0, 0.1], [1.0, 0.0, 0.0, 1.0, 2.0, 0.2]])
    assert base.reset(0.0, 0.0, 0.0, 0.0)
    g = base.ground
    assert np.allclose(g.feet_h, [0.2, 0.2, 0.0, 0.0], atol=1e-6)
    expected = math.degrees(math.atan(0.2 / (2 * cfg["foot_offsets"][0][0])))
    assert abs(math.degrees(-g.pitch) - expected) < 1.0          # nose up = negative pitch
    assert abs(g.roll) < 1e-6
    assert abs(g.z0 - 0.1) < 1e-6                                 # plane through the feet at the centre


def test_a_fast_tick_stops_at_a_thin_wall_instead_of_jumping_it():
    # 60 km/h at 50 Hz is 0.33 m a tick: from x = 0.9 one check at 1.23 would land past a wall at x 1.0-1.2
    from quadsim.base import IdealBase
    from quadsim.config import load_config
    from quadsim.raycast import RayScene
    cfg = load_config(None, [])["robot"]
    wall = lambda x, *_: 1.0 <= x <= 1.2
    base = IdealBase(cfg, RayScene(np.array([[0.0, 0.0, -0.1, 5.0, 5.0, 0.1]], np.float32)), None, blocked=wall)
    assert base.reset(0.9, 0.0, 0.0, 0.0)
    base.v[:] = [60 / 3.6, 0.0, 0.0]
    assert base.update(0.02, np.array([60 / 3.6, 0.0, 0.0]), 0.0) == "collision"
    assert 0.9 < base.x < 1.0 and base.v[0] == 0.0
