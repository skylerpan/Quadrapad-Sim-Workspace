"""Headless climb of the hand-drawn staircase D in the 1:1 dp_1012 scene (Maps/dp_1012_handmade.bt).
Risers 0.2 m, treads 0.2-0.6 m, x -5.5..-2.1 at y -17.0, from the 1F floor (z 0.0) to the landing (1.4)."""
import json
import math
import os

import pytest

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp_1012")


def _is_handmade_1to1():
    try:
        with open(os.path.join(SCENE, "meta.json")) as f:
            m = json.load(f)
    except OSError:
        return False
    return m.get("source_pcd", "").endswith("dp_1012_handmade.bt") and not m["build_args"]["flatten"]


pytestmark = pytest.mark.skipif(not _is_handmade_1to1(),
                                reason="scenes/dp_1012 is not the 1:1 build of Maps/dp_1012_handmade.bt")


def _drive(sim, x, y, yaw, z_floor, vx, seconds):
    sim.set_pose(x, y, yaw, z_floor)
    events = []
    for _ in range(int(seconds * sim.cfg["physics"]["control_hz"])):
        sim.set_cmd(vx, 0.0, 0.0)
        out = sim.control_tick()
        events += [(e["reason"], e["pose"][0]) for e in out.events]
    return events


@pytest.fixture(scope="module")
def sim():
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    cfg = load_config(None, [])
    cfg["scene"] = SCENE
    return Simulator(cfg)


def test_climbs_stair_d_without_refusals_or_contacts(sim):
    events = _drive(sim, -2.0, -17.0, math.pi, 0.0, 0.5, 10.0)
    assert abs(sim.base.ground.z0 - 1.4) < 1e-6, (sim.base.x, sim.base.ground.z0)
    assert not [e for e in events if e[0] == "contact"], events
    assert not [e for e in events if e[1] > -5.3], events          # only the landing wall may stop it


def test_descends_stair_d_without_events(sim):
    events = _drive(sim, -5.3, -17.0, 0.0, 1.4, 0.5, 10.0)
    assert abs(sim.base.ground.z0 - 0.0) < 1e-6, (sim.base.x, sim.base.ground.z0)
    assert events == []
