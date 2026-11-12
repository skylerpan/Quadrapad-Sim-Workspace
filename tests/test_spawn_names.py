"""Named spawns: config `spawns` holds [x, y, yaw, z_floor] by name and `spawn` is either such a list or a name."""
import math

import pytest

from quadsim.config import load_config
from quadsim.sim import spawn_pose

META = [1.0, 2.0, -0.4]


def test_a_name_picks_its_spawn_from_spawns():
    assert spawn_pose("gate", {"gate": [34.5, 41.0, 1.5708, -0.76]}, META) == (34.5, 41.0, 1.5708, -0.76)


def test_a_list_is_used_as_it_is_and_fills_yaw_and_floor():
    assert spawn_pose([3.0, 4.0], {}, META) == (3.0, 4.0, 0.0, -0.4)


def test_null_takes_the_scene_spawn():
    assert spawn_pose(None, {}, META) == (1.0, 2.0, 0.0, -0.4)


def test_an_unknown_name_is_refused_with_the_names_there_are():
    with pytest.raises(ValueError, match="gate"):
        spawn_pose("gat", {"gate": [0, 0, 0, 0]}, META)


def test_the_config_has_the_gate_spawn_facing_out_of_the_dp_gate():
    cfg = load_config(None, ["spawn=gate"])
    x, y, yaw, z = spawn_pose(cfg["spawn"], cfg.get("spawns") or {}, META)
    assert 29.1 <= x <= 39.9 and 38.0 <= y <= 50.0                      # between the reference points, cleared
    assert yaw == pytest.approx(math.pi / 2, abs=1e-3) and -1.0 < z < -0.5
    assert spawn_pose(load_config(None, [])["spawn"], {}, META) == (0.0, 0.0, 0.0, -0.4)   # the default stays
