import os

import numpy as np
import pytest

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp_1012")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "meta.json")),
                                reason="scenes/dp_1012 not built")


def test_low_obstacle_just_above_body_bottom_blocks():
    from quadsim.config import load_config
    from quadsim.proxies import BoxIndex
    from quadsim.sim import Simulator
    cfg = load_config(None, [])
    cfg["scene"] = SCENE
    s = Simulator(cfg)
    x, y, z_floor = 0.0, 0.0, 0.0
    z_center = z_floor + cfg["robot"]["stand_height"]
    bottom = z_center - s.half[2]
    top = bottom + 0.015                                  # 1.5 cm into the body's z band
    s.index = BoxIndex(np.array([[x + s.half[0] + 0.04, y, top - 0.05, 0.05, 0.05, 0.05]]))
    assert s._blocked(x, y, 0.0, z_center)
    s.index = BoxIndex(np.array([[x + s.half[0] + 0.04, y, bottom - 0.06, 0.05, 0.05, 0.05]]))
    assert not s._blocked(x, y, 0.0, z_center)            # below the body: not an obstacle


def test_set_pose_into_an_obstacle_moves_to_a_free_pose():
    from quadsim.config import load_config
    from quadsim.proxies import BoxIndex
    from quadsim.sim import Simulator
    cfg = load_config(None, [])
    cfg["scene"] = SCENE
    s = Simulator(cfg)
    x, y, z_floor = s.base.x, s.base.y, s.base.ground.z0
    z_mid = z_floor + cfg["robot"]["stand_height"]
    s.index = BoxIndex(np.array([[x, y, z_mid, 0.2, 0.2, 0.05]]))   # a block right in the body
    with pytest.warns(UserWarning, match="moved to"):
        s.set_pose(x, y, 0.0, z_floor)
    assert np.hypot(s.base.x - x, s.base.y - y) > 0.05
    assert not s._blocked(s.base.x, s.base.y, s.base.yaw, s.base.ground.z0 + cfg["robot"]["stand_height"])
