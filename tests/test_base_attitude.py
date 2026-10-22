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
