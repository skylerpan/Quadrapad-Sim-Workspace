"""quadsim/truth_log.py: per-physics-step ground truth of the feet and of every robot/world contact."""
import os

import numpy as np
import pytest

from quadsim.config import load_config
from tests.test_g23 import _cfg

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), load_config(None, [])["scene"])
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "contact.npy")),
                                 reason=f"{os.path.relpath(SCENE)} (config scene) not built as format 3")


@needs_scene
def test_truth_log_records_every_physics_step(tmp_path):
    from quadsim.sim import Simulator
    path = tmp_path / "truth.npz"
    sim = Simulator(_cfg(**{"robot.controller.gt_addr": '""', "robot.controller.ros_ascii_telemetry": "false",
                            "robot.controller.truth_log": str(path)}))
    try:
        steps0 = sim.step_count
        for _ in range(25):                      # 0.5 s: the limp robot settles on the floor
            sim.control_tick()
        n = sim.step_count - steps0
        t_end = sim.d.time
        mass = float(sim.m.body_subtreemass[sim.g23.root])
        gravity = float(np.linalg.norm(sim.m.opt.gravity))
    finally:
        sim.close()
    log = np.load(path)
    assert len(log["t"]) == n
    assert np.allclose(np.diff(log["t"]), sim.m.opt.timestep) and abs(log["t"][-1] - t_end) < 1e-9
    assert log["foot_v"].shape == (n, 4, 3) and log["foot_w"].shape == (n, 4, 3)
    assert log["foot_force"].shape == (n, 4) and log["foot_normal"].shape == (n, 4, 3)
    assert log["shank_force"].shape == (n, 4) and log["other_force"].shape == (n,)
    assert list(log["feet"]) == ["FL", "FR", "HL", "HR"]
    # at rest everything the robot weighs goes through the logged contacts, and nothing moves
    total = log["foot_force"][-50:].sum(1) + log["shank_force"][-50:].sum(1) + log["other_force"][-50:]
    assert abs(total.mean() - mass * gravity) < 0.1 * mass * gravity
    loaded = log["foot_force"][-1] > 1.0
    assert np.abs(log["foot_v"][-1][loaded]).max(initial=0.0) < 0.05
    # a loaded foot has a unit normal pointing from the ground into the foot; an unloaded one has none
    normals = np.linalg.norm(log["foot_normal"][-1], axis=1)
    assert np.allclose(normals[loaded], 1.0, atol=1e-6) and np.all(normals[~loaded] == 0.0)
    assert np.all(log["foot_normal"][-1][loaded][:, 2] > 0.5)


def test_truth_log_is_off_by_default():
    from quadsim.g23 import RefSimOptions
    assert RefSimOptions.from_env({}).truth_log is None
    assert RefSimOptions.from_env({"controller": {"truth_log": "x.npz"}}).truth_log == "x.npz"
