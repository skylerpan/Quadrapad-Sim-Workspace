"""quadsim as the MuJoCo side of SyncAI-Device-MotorControl: its packets, its operator words, the stand sequence,
rejected switches, stops, teleports, against the fake controller of tests/fake_motorcontrol.py."""
import os
import time

import numpy as np
import pytest

from quadsim.config import load_config
from quadsim.g23 import DEFAULT_Q
from tests.fake_motorcontrol import FakeMotorControl

SCENE = os.path.join(os.path.dirname(__file__), "..", load_config(None, [])["scene"])
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "contact.npy")),
                                 reason=f"{os.path.relpath(SCENE)} (config scene) not built as format 3")
# private ports: a real controller or another test on the machine never sees these packets
PORTS = dict(cmd_port=40101, joint_addr="127.0.0.1:40102", base_addr="127.0.0.1:40103", operator_addr="127.0.0.1:40120")


def _overrides(**over):
    o = ["robot.type=g23", "robot.controller.kind=motorcontrol", "robot.start_delay=0.0", "robot.stand_delay=0.0",
         "lidar.points_per_sec=2000"] + [f"robot.controller.motorcontrol.{k}={v}" for k, v in PORTS.items()]
    return o + [f"{k}={v}" for k, v in over.items()]


def _fake(**kw) -> FakeMotorControl:
    return FakeMotorControl(cmd_to=("127.0.0.1", PORTS["cmd_port"]), joint_port=40102, base_port=40103,
                            operator_port=40120, **kw)


def _backend(**over):
    """A MotorControlBackend on the bare G23 model (no scene): enough for the link and the sequence."""
    import mujoco
    from quadsim.g23 import RefSimOptions, build_g23_spec
    from quadsim.motorcontrol import MotorControlBackend
    cfg = load_config(None, _overrides(**over))
    o = RefSimOptions.from_env(cfg["robot"])
    m = build_g23_spec(cfg, [0, 0, -9.81], [0, 0, 0.5], 0, o).compile()
    d = mujoco.MjData(m)
    g = MotorControlBackend(m, d, cfg["robot"], mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "TORSO"), o)
    g.place([0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(m, d)
    return cfg, g


def _exchange(g, t: float, cmd=(0.0, 0.0, 0.0), fresh=False, settle: float = 0.03):
    """One physics step's worth of packets each way, then a control tick: telemetry out, the fake's reply and any
    operator lines back in."""
    g.send_telemetry(t)
    time.sleep(settle)
    return g.tick(t, np.asarray(cmd, dtype=float), cmd_fresh=fresh)


def _wait(pred, timeout=2.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.01)
    assert pred()


def test_packets_go_both_ways_and_the_sequence_stands_then_starts_the_behavior():
    fake = _fake()
    cfg, g = _backend()
    try:
        assert g.behavior == "rl_fcn" and g.cmd_port == 40101 and "40120" in g.describe_link()
        events = _exchange(g, 0.0)                                         # telemetry -> fake -> torque command back
        _wait(lambda: fake.n_joint >= 1 and fake.n_base >= 1 and g.n_cmd >= 1)
        assert np.allclose(fake.last_q, DEFAULT_Q) and len(fake.last_base) == 9
        events += _exchange(g, 0.001)
        states = [e["state"] for e in events if e["type"] == "controller"]
        assert states[:2] == ["connected", "stand"] or states[:3] == ["connected", "stand", "rl_fcn"], states
        for k in range(2, 6):
            events += _exchange(g, 0.001 * k)
        assert g.sent[:2] == ["stand", "rl_fcn"] and fake.behavior == "rl_fcn"
        _wait(lambda: g.state == "rl_fcn")
        assert not g.safety_locked and g.stage == "gait"
        states = [e["state"] for e in events if e["type"] == "controller"]
        assert "switched" in states and states.index("stand") < states.index("rl_fcn")
        # the fake holds the pose: kp / q_des arrived in the controller's interleaved layout
        assert np.allclose(g.q_des, DEFAULT_Q) and np.all(g.kp == 60.0) and np.all(g.kd == 2.0)
    finally:
        g.close(), fake.close()


def test_velocity_lines_follow_cmd_vel_and_stop_once():
    fake = _fake()
    cfg, g = _backend()
    try:
        for k in range(4):
            _exchange(g, 0.001 * k)
        assert g.stage == "gait"
        _exchange(g, 1.0, (0.5, 0.0, -0.2), fresh=True)
        assert g.sent[-1] == "vel 0.500 0.000 -0.200"
        n = len(g.sent)
        for k in range(5):
            _exchange(g, 1.1 + 0.001 * k, fresh=False)
        vel = [s for s in g.sent[n:] if s.startswith("vel")]
        assert vel == ["vel 0.000 0.000 0.000"]
        _wait(lambda: fake.vel == (0.0, 0.0, 0.0) and any(l == "vel 0.500 0.000 -0.200" for l in fake.received))
    finally:
        g.close(), fake.close()


def test_rl_him_velocity_is_predivided_by_the_controllers_command_scale():
    # the controller multiplies rl_him's command by (2, 2, 0.25) before the policy; the reference stack feeds it
    # raw, so quadsim divides first. rl_fcn is scaled by 1 on both sides and goes through unchanged.
    for behavior, expected in (("rl_him", "vel 0.200 0.100 2.400"), ("rl_fcn", "vel 0.400 0.200 0.600")):
        fake = _fake()
        cfg, g = _backend(**{"robot.behavior": behavior})
        try:
            for k in range(4):
                _exchange(g, 0.001 * k)
            _wait(lambda: fake.behavior == behavior)
            _exchange(g, 1.0)                                              # the status line naming it is drained
            assert g.state == behavior
            _exchange(g, 1.1, (0.4, 0.2, 0.6), fresh=True)
            assert g.sent[-1] == expected, behavior
        finally:
            g.close(), fake.close()


def test_a_rejected_behavior_is_resent_after_a_second_then_given_up():
    # the controller repeats an unchanged request line as a heartbeat: here faster than the ticks, so every tick
    # sees the same rejection again. A repeat is not a new verdict and must not postpone the retry.
    fake = _fake(reject_once=("rl_fcn",), heartbeat_s=0.01)
    cfg, g = _backend()
    try:
        for k in range(4):
            _exchange(g, 0.001 * k)
        assert g.sent.count("rl_fcn") == 1
        _wait(lambda: g.last_request is not None and g.last_request.state == "rejected")
        _exchange(g, 0.5)
        assert g.sent.count("rl_fcn") == 1                                 # not before a second has passed
        _exchange(g, 1.1)
        assert g.sent.count("rl_fcn") == 2
        _wait(lambda: fake.behavior == "rl_fcn")
        _exchange(g, 1.2)                                                  # the ok is drained by a tick
        assert g.last_request.state == "ok"
        for k in range(5):
            _exchange(g, 3.0 + k)
        assert g.sent.count("rl_fcn") == 2                                 # accepted: no more resends
    finally:
        g.close(), fake.close()
    fake = _fake(reject_always=("rl_him",), heartbeat_s=0.01)
    cfg, g = _backend(**{"robot.behavior": "rl_him"})
    try:
        events = []
        for k in range(4):
            events += _exchange(g, 0.001 * k)
        for k in range(1, 15):                                             # a rejection and a retry take two ticks
            events += _exchange(g, 1.0 * k + 0.5)
        assert g.sent.count("rl_him") == 6                                 # the first send and five retries
        assert any(e["state"] == "rejected" and e.get("target") == "rl_him" for e in events)
        assert g.stage == "gait" and fake.behavior == "stand"
    finally:
        g.close(), fake.close()


def test_policy_mode_maps_to_behaviors_and_unknown_slots_are_refused():
    fake = _fake(handover_s=0.1)                                            # a second request would supersede the stand
    cfg, g = _backend()
    try:
        _exchange(g, 0.0)                                                   # connected + stand (start_delay 0)
        _wait(lambda: fake.behavior == "stand")                             # stand_delay 0 here: let it land first
        for k in range(1, 4):
            _exchange(g, 0.001 * k)
        _wait(lambda: fake.behavior == "rl_fcn")
        e = g.set_policy_mode(1, 2.0)
        assert e["live"] and e["behavior"] == "rl_him" and g.behavior == "rl_him"
        assert g.sent[-1] == "stand"                                        # one edge at a time: stand first ...
        _wait(lambda: fake.behavior == "stand")
        for k in range(3):
            _exchange(g, 2.1 + 0.001 * k)
        assert g.sent[-1] == "rl_him"                                       # ... the behavior once stand is reported
        _wait(lambda: fake.behavior == "rl_him")
        assert g.sent.count("rl_him") == 1
        n = len(g.sent)
        e = g.set_policy_mode(4, 3.0)
        assert e["rejected"] and not e["live"] and len(g.sent) == n and g.behavior == "rl_him"
    finally:
        g.close(), fake.close()


def test_teleport_rests_first_then_resets_and_stands():
    fake = _fake()
    cfg, g = _backend()
    try:
        for k in range(4):
            _exchange(g, 0.001 * k)
        assert g.stage == "gait"
        n0 = len(g.sent)
        g.restart(5.0, teleport=True)
        assert g.sent[n0] == "rest" and g.stage == "idle"
        _exchange(g, 5.0)
        assert g.sent[n0 + 1:n0 + 3] == ["reset", "stand"]
        _wait(lambda: fake.behavior == "stand")
    finally:
        g.close(), fake.close()


def test_a_controller_stop_is_reported_not_cleared():
    fake = _fake()
    cfg, g = _backend()
    try:
        for k in range(4):
            _exchange(g, 0.001 * k)
        fake.inject_stop("hardware")
        events = []
        _wait(lambda: fake.stop == "hardware")
        for k in range(5):
            events += _exchange(g, 1.0 + 0.001 * k)
        stops = [e for e in events if e["type"] == "controller" and e["state"] == "stopped"]
        assert len(stops) == 1 and stops[0]["stop"] == "hardware"
        assert g.safety_locked and g.state == "rest" and "reset" not in g.sent
        # the operator's stand button is the way back: reset + stand
        e = g.operator("stand", 2.0)
        assert g.sent[-2:] == ["reset", "stand"] and e["source"] == "operator"
        events = []
        for k in range(5):
            events += _exchange(g, 2.0 + 0.001 * k)
        assert any(e["state"] == "released" for e in events) and not g.safety_locked
        _wait(lambda: fake.behavior == "stand" and fake.stop is None)
        assert g.operator("lie", 3.0)["state"] == "lie" and g.sent[-1] == "rest"
        assert g.operator("damp", 3.1)["state"] == "damp" and g.sent[-1] == "rest"
        assert g.operator("walk", 3.2)["state"] == "rl_fcn" and g.sent[-1] == "rl_fcn"
    finally:
        g.close(), fake.close()


@needs_scene
def test_model_stands_on_the_map_with_the_fake_motorcontrol():
    from quadsim.sim import Simulator
    fake = _fake()
    cfg = load_config(None, _overrides(**{"robot.start_delay": 0.1, "robot.stand_delay": 0.2}))
    sim = Simulator(cfg)
    try:
        assert sim.legged and type(sim.g23).__name__ == "MotorControlBackend"
        z_floor = sim.base.ground.z0
        events = []
        for _ in range(60):                                                 # 1.2 s of sim time
            time.sleep(0.002)
            events += sim.control_tick().events
        states = [e["state"] for e in events if e["type"] == "controller" and e["state"] != "switched"]
        assert states[:3] == ["connected", "stand", "rl_fcn"] or states[:3] == ["stand", "connected", "rl_fcn"], states
        assert sum(1 for e in events if e["state"] == "switched") == 2                 # rest -> stand -> rl_fcn
        assert sim.d.qpos[2] - z_floor > 0.25 and not sim.fallen             # standing on the floor proxies
        assert sim.d.xmat[sim.base_body].reshape(3, 3)[2, 2] > 0.95
        assert abs(fake.last_base[8] - 9.81) < 3.0                           # gravity in the body z of the base packet
        assert np.all(np.abs(fake.last_q - DEFAULT_Q) < 0.3)
        assert sim.g23.sent[:2] == ["stand", "rl_fcn"]
        sim.set_cmd(0.5, 0.0, -0.2)
        sim.control_tick()
        assert sim.g23.sent[-1] == "vel 0.500 0.000 -0.200"
        n1 = len(sim.g23.sent)
        e = sim.set_policy_mode(1)
        assert e["live"] and sim.g23.sent[n1] == "stand"
        for _ in range(10):
            time.sleep(0.002)
            sim.control_tick()
        switch = [s for s in sim.g23.sent[n1:] if not s.startswith("vel")]
        assert switch == ["stand", "rl_him"]                                 # the behavior once the controller reported stand
        n0 = len(sim.g23.sent)
        sim.set_pose(sim.base.x + 0.3, sim.base.y, 0.0, None)
        assert sim.g23.sent[n0] == "rest" and sim.g23.stage == "idle"
        for _ in range(10):
            time.sleep(0.002)
            sim.control_tick()
        seq = sim.g23.sent[n0:]
        assert "reset" in seq and seq.index("reset") < seq.index("stand")
    finally:
        sim.close(), fake.close()


def test_bad_kind_or_behavior_is_refused_at_construction():
    from quadsim.sim import Simulator
    with pytest.raises(ValueError):
        Simulator(load_config(None, _overrides(**{"robot.controller.kind": "other"})))
    with pytest.raises(ValueError):
        Simulator(load_config(None, _overrides(**{"robot.behavior": "fly"})))
