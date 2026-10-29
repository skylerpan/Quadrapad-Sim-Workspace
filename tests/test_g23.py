"""G23 legged mode: model build, the controller's UDP protocol, the stand sequence, a fake controller loop."""
import os
import socket
import struct
import time

import numpy as np
import pytest

from quadsim.config import load_config
from quadsim.g23 import CMD_BYTES, CMD_FMT, DEFAULT_Q, J60_PEAK, TELEMETRY_FMT, j60_limits, quat_to_rpy

SCENE = os.path.join(os.path.dirname(__file__), "..", load_config(None, [])["scene"])
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "contact.npy")),
                                 reason=f"{os.path.relpath(SCENE)} (config scene) not built as format 3")


def _cfg(**over):
    # private ports so a controller or another test on the machine never sees these packets
    o = ["robot.type=g23", "robot.controller.cmd_port=20101", "robot.controller.telemetry=[127.0.0.1:30110]",
         "robot.controller.command_addr=127.0.0.1:50151", "robot.start_delay=0.1", "robot.stand_delay=0.2",
         "lidar.points_per_sec=2000"]
    o += [f"{k}={v}" for k, v in over.items()]
    return load_config(None, o)


def test_j60_envelope_and_rpy():
    lo, hi = j60_limits(np.zeros(12))
    assert np.allclose(hi, J60_PEAK) and np.allclose(lo, -J60_PEAK)
    lo, hi = j60_limits(np.full(12, 100.0))                       # far past no-load speed
    assert np.allclose(hi, 0.0) and np.allclose(lo, -J60_PEAK)    # no drive torque, full braking
    assert np.allclose(quat_to_rpy(np.array([1.0, 0, 0, 0])), 0.0)
    yaw = quat_to_rpy(np.array([np.cos(0.3), 0, 0, np.sin(0.3)]))[2]
    assert abs(yaw - 0.6) < 1e-9


@needs_scene
def test_model_stands_on_the_map_with_a_fake_controller():
    from quadsim.sim import Simulator
    cfg = _cfg()
    sim = Simulator(cfg)
    try:
        assert sim.legged and sim.m.nu == 12 and sim.base_geom < 0
        # the controller's J60 joint model and vertical gravity (its own sim's conditions) by default
        import mujoco
        knee = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, "FL_Knee_joint")
        dof = sim.m.jnt_dofadr[knee]
        assert abs(sim.m.dof_armature[dof] - 1.1e-4 * 81) < 1e-9 and abs(sim.m.dof_frictionloss[dof] - 0.45) < 1e-9
        assert abs(sim.m.dof_damping[dof] - (0.92 - 0.45) / 15.5) < 1e-9
        assert np.allclose(sim.m.opt.gravity, [0, 0, -9.81])
        assert sim.proxies.index is sim.contact_index and sim.contact_index is not sim.index
        z_floor = sim.base.ground.z0
        assert abs(sim.d.qpos[2] - (z_floor + 0.40)) < 1e-6 and np.allclose(sim.d.qpos[sim.g23.jq], DEFAULT_Q)
        # a fake controller: listen for telemetry, answer with a PD hold of the default pose
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx.bind(("127.0.0.1", 30110))
        rx.settimeout(2.0)
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        txt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        txt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        txt.bind(("127.0.0.1", 50151))
        txt.settimeout(2.0)
        pd = struct.pack(CMD_FMT, *np.full(12, 60.0), *DEFAULT_Q, *np.full(12, 2.0), *np.zeros(12), *np.zeros(12))
        assert len(pd) == CMD_BYTES == 240
        events = []
        for _ in range(60):                                     # 1.2 s of sim time
            cmd_sock.sendto(pd, ("127.0.0.1", 20101))
            time.sleep(0.002)
            out = sim.control_tick()
            events += out.events
        pkt = rx.recv(4096)
        rx.settimeout(0.05)
        try:
            while True:                                         # the last packet: standing, not the first (free fall)
                pkt = rx.recv(4096)
        except socket.timeout:
            pass
        assert len(pkt) == struct.calcsize(TELEMETRY_FMT) == 228
        v = struct.unpack(TELEMETRY_FMT, pkt)
        t, rpy, acc, gyro, q, dq, tau, pos, vel, contacts = (v[0], v[1:4], v[4:7], v[7:10], v[10:22], v[22:34],
                                                             v[34:46], v[46:49], v[49:52], v[52:56])
        assert 0 < t <= sim.d.time and abs(acc[2] - 9.81) < 3.0       # gravity in the body z while standing
        assert sum(contacts) >= 3                                       # feet on the floor proxies
        assert all(abs(a - b) < 0.3 for a, b in zip(q, DEFAULT_Q))
        # standing: torso ~0.3-0.4 m over the floor, upright, 4-5 contacts
        assert sim.d.qpos[2] - z_floor > 0.25 and not sim.fallen
        assert sim.d.xmat[sim.base_body].reshape(3, 3)[2, 2] > 0.95
        states = [e["state"] for e in events if e["type"] == "controller"]
        assert states[:3] == ["connected", "stand", "rl"] or states[:3] == ["stand", "connected", "rl"], states
        lines = []
        txt.settimeout(0.5)
        try:
            while True:
                lines.append(txt.recv(256).decode().strip())
        except socket.timeout:
            pass
        assert "MODE STAND" in lines and "MODE RL" in lines
        assert not any(l.startswith("AXES ") for l in lines)             # no /cmd_vel yet: nothing forwarded
        # cmd_vel is forwarded as AXES once the gait is on; when it goes stale one zero is sent, then nothing
        sim.set_cmd(0.5, 0.0, -0.2)
        sim.control_tick()
        assert sim.g23.sent[-1] == "AXES 0.500 0.000 -0.200"
        n_sent = len(sim.g23.sent)
        for _ in range(40):                                             # 0.8 s > cmd_timeout 0.5 s
            sim.control_tick()
        axes = [l for l in sim.g23.sent[n_sent:] if l.startswith("AXES")]
        assert axes[-1] == "AXES 0.000 0.000 0.000" and len(axes) < 40
        # the start sequence used the configured policy (0 = PPO) before MODE RL; a switch is sent live
        assert sim.g23.sent[sim.g23.sent.index("MODE RL") - 1] == "MODE 0"
        e = sim.set_policy_mode(1)
        assert e["live"] and sim.g23.sent[-1] == "MODE 1" and e["policy_mode"] == 1
        # fall detection: lay the torso on its side
        sim.d.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        out = sim.control_tick()
        assert any(e["type"] == "fall" for e in out.events) and sim.fallen
        rx.close(), cmd_sock.close(), txt.close()
    finally:
        sim.close()


@needs_scene
def test_ideal_base_still_runs_at_1khz():
    from quadsim.sim import Simulator
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=2000"])
    sim = Simulator(cfg)
    assert not sim.legged and sim.dt == 0.001 and sim.ctrl_steps == 20 and sim.imu_steps == 5
    sim.set_cmd(0.5, 0.0, 0.0)
    for _ in range(25):
        sim.control_tick()
    assert sim.base.x > 0.15
    sim.close()


@needs_scene
def test_motor_model_and_gravity_can_be_switched_off():
    import mujoco
    from quadsim.sim import Simulator
    sim = Simulator(_cfg(**{"robot.motor_model": "ideal", "robot.level_gravity": "false"}))
    try:
        dof = sim.m.jnt_dofadr[mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, "FL_Knee_joint")]
        assert sim.m.dof_frictionloss[dof] == 0.0 and sim.m.dof_armature[dof] == 0.0
        assert np.allclose(sim.m.opt.gravity, sim.meta["gravity"])          # the scene's tilted gravity
    finally:
        sim.close()
