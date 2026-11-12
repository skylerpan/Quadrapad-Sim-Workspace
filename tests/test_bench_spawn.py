"""scripts/bench_g23.py --spawn: x,y,z_floor as before, or a name in config spawns (gate); default the config spawn."""
import os
import sys

import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from bench_g23 import bench_spawn  # noqa: E402

from quadsim.config import load_config  # noqa: E402

CFG = load_config(None, [])


def test_numbers_are_x_y_and_floor():
    assert bench_spawn("0,0,-0.43", CFG) == [0.0, 0.0, -0.43]


def test_a_name_takes_that_spawn_and_its_floor():
    x, y, yaw, z = CFG["spawns"]["gate"]
    assert bench_spawn("gate", CFG) == [x, y, z + 0.05]


def test_the_default_is_the_config_spawn():
    assert bench_spawn(None, CFG) == [0.0, 0.0, -0.4 + 0.05]
    assert bench_spawn(None, load_config(None, ["spawn=gate"])) == bench_spawn("gate", CFG)


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="gate"):
        bench_spawn("gat", CFG)
