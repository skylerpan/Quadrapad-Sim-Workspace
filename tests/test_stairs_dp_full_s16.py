"""Headless climbs of the synthetic staircases in scenes/dp_full_003_s16 (scripts/make_stairs.py,
config/stairs_dp_full.yaml): east flight E1 (1F -0.35 -> landing 2.20 at x 6.2, rising north) up and down, and
the first west flight W1 (1F -0.30 -> half-landing 1.10 at x -10.7, rising north)."""
import json
import math
import os

import pytest

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp_full_003_s16")


def _has_synthetic_stairs():
    try:
        with open(os.path.join(SCENE, "meta.json")) as f:
            return "synthetic_stairs" in json.load(f)
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _has_synthetic_stairs(), reason="scenes/dp_full_003_s16 not built")
BAD = ("contact", "collision", "step_up", "step_down", "slope", "no_ground")


def _drive(sim, x, y, yaw, z_floor, vx, seconds):
    sim.set_pose(x, y, yaw, z_floor)
    events = []
    for _ in range(int(seconds * sim.cfg["physics"]["control_hz"])):
        sim.set_cmd(vx, 0.0, 0.0)
        out = sim.control_tick()
        events += [(e["reason"], round(e["pose"][0], 2), round(e["pose"][1], 2)) for e in out.events]
    return events


@pytest.fixture(scope="module")
def sim():
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    cfg = load_config(None, ["robot.collision_pool=1024", "robot.collision_margin=0.4"])
    cfg["scene"] = SCENE
    cfg["spawn"] = [6.2, -13.0, math.pi / 2, -0.4]
    return Simulator(cfg)


def test_climbs_east_flight_to_the_landing(sim):
    events = _drive(sim, 6.2, -13.0, math.pi / 2, -0.4, 0.5, 13.0)
    assert sim.base.y > -7.5 and abs(sim.base.ground.z0 - 2.2) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events


def test_descends_east_flight_to_the_floor(sim):
    events = _drive(sim, 6.2, -7.4, -math.pi / 2, 2.2, 0.5, 10.0)      # stops on the approach pad (z -0.35)
    assert -12.7 < sim.base.y < -12.0 and abs(sim.base.ground.z0 + 0.35) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events


def test_climbs_west_flight_to_the_half_landing(sim):
    events = _drive(sim, -10.7, -12.9, math.pi / 2, -0.3, 0.5, 7.0)
    assert sim.base.y > -10.3 and abs(sim.base.ground.z0 - 1.1) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events
