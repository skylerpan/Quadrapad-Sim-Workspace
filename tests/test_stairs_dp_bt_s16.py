"""Headless climbs of the synthetic staircases in scenes/dp_bt_s16 (scripts/make_stairs.py exact-box mode on the
1:1 build of Maps/archive/dp.bt, config/stairs_dp_bt.yaml): east flight E1 (1F -0.2 -> landing 2.3 at x 6.35,
rising north) up and down, west flight W1 (1F -0.2 -> half-landing 1.2 at x -10.75, rising north) and W2 (1.2 -> 2.5
at x -9.1, descending south in y)."""
import json
import math
import os

import pytest

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp_bt_s16")


def _mode():
    try:
        with open(os.path.join(SCENE, "meta.json")) as f:
            return json.load(f).get("synthetic_stairs", {}).get("mode")
    except OSError:
        return None


pytestmark = pytest.mark.skipif(_mode() != "exact_boxes", reason="scenes/dp_bt_s16 not built")
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
    cfg = load_config(None, ["robot.collision_pool=512", "robot.collision_margin=0.4"])
    cfg["scene"] = SCENE
    cfg["spawn"] = [6.35, -12.8, math.pi / 2, -0.2]
    return Simulator(cfg)


def test_climbs_east_flight_to_the_landing(sim):
    events = _drive(sim, 6.35, -12.8, math.pi / 2, -0.2, 0.5, 13.0)
    assert sim.base.y > -7.5 and abs(sim.base.ground.z0 - 2.3) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events


def test_descends_east_flight_to_the_floor(sim):
    events = _drive(sim, 6.35, -7.3, -math.pi / 2, 2.3, 0.5, 11.0)      # stops on the approach pad (z -0.2)
    assert -12.9 < sim.base.y < -11.9 and abs(sim.base.ground.z0 + 0.2) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events


def test_climbs_west_flight_to_the_half_landing(sim):
    events = _drive(sim, -10.75, -12.8, math.pi / 2, -0.2, 0.5, 5.5)   # stops on L1, short of the north wall
    assert sim.base.y > -10.2 and abs(sim.base.ground.z0 - 1.2) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events


def test_climbs_second_west_flight(sim):
    events = _drive(sim, -9.1, -9.5, -math.pi / 2, 1.2, 0.5, 6.5)     # stops on L2, short of the south wall
    assert sim.base.y < -12.6 and abs(sim.base.ground.z0 - 2.5) < 1e-6, (sim.base.y, sim.base.ground.z0, events)
    assert not [e for e in events if e[0] in BAD], events
