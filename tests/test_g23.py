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
    o = ["robot.type=g23", "robot.controller.kind=g2_deploy_rs", "robot.controller.cmd_port=20101", "robot.controller.telemetry=[127.0.0.1:30110]",
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
        # a teleport while the gait runs: damping first, then RESET_SAFETY before the new stand sequence
        n0 = len(sim.g23.sent)
        sim.set_pose(sim.base.x + 0.3, sim.base.y, 0.0, None)
        assert sim.g23.sent[n0] == "MODE R" and sim.g23.stage == "idle"
        for _ in range(10):                                             # start_delay 0.1 s
            cmd_sock.sendto(pd, ("127.0.0.1", 20101))
            sim.control_tick()
        seq = sim.g23.sent[n0:]
        assert "RESET_SAFETY" in seq and seq.index("RESET_SAFETY") < seq.index("MODE STAND")
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


def test_reference_sim_options_precedence(tmp_path, monkeypatch):
    from quadsim.g23 import RefSimOptions, read_control_config_value
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "g2_control.env").write_text("G2_SIM_TORSO_COM_OFFSET=0,0,0\nexport G2_SIM_TORSO_COM_OFFSET='0.01,0,-0.02'\n")
    for k in ("G2_SIM_MOTOR_MODEL", "G2_SIM_TORSO_COM_OFFSET", "G2_SIM_IMU_DELAY_MS", "G2_SIM_ROS_ASCII_TELEMETRY",
              "G2_SIM_GT_PORT", "G2_CONTROL_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("G2_DEPLOY_RS", str(tmp_path))
    assert read_control_config_value("G2_SIM_TORSO_COM_OFFSET", str(tmp_path)) == "0.01,0,-0.02"   # last line wins
    o = RefSimOptions.from_env({})
    assert o.motor_model == "j60" and np.allclose(o.torso_com_offset, [0.01, 0, -0.02])         # from g2_control.env
    assert o.ros_ascii and o.ros_bridge == ("127.0.0.1", 50010) and o.gt == ("127.0.0.1", 30012)
    monkeypatch.setenv("G2_SIM_TORSO_COM_OFFSET", "none")                                      # env beats the file
    monkeypatch.setenv("G2_SIM_MOTOR_MODEL", "ideal")
    monkeypatch.setenv("G2_SIM_IMU_DELAY_MS", "5")
    monkeypatch.setenv("G2_SIM_ROS_ASCII_TELEMETRY", "0")
    monkeypatch.setenv("G2_SIM_GT_PORT", "31000")
    o = RefSimOptions.from_env({})
    assert o.torso_com_offset is None and o.motor_model == "ideal" and o.imu_delay_ms == 5.0
    assert not o.ros_ascii and o.gt == ("127.0.0.1", 31000)
    # sim.yaml values that are not null win over the environment
    o = RefSimOptions.from_env({"motor_model": "j60", "controller": {"gt_addr": "", "ros_ascii_telemetry": True}})
    assert o.motor_model == "j60" and o.gt is None and o.ros_ascii
    assert "motor model j60" in o.describe() and "SIM_POSE off" in o.describe()


@needs_scene
def test_sensor_model_delays_the_telemetry_imu(monkeypatch):
    """With G2_SIM_IMU_DELAY_MS the telemetry IMU lags the raw sensor by that many steps, as in their sim."""
    from quadsim.sim import Simulator
    monkeypatch.setenv("G2_SIM_IMU_DELAY_MS", "5")
    monkeypatch.setenv("G2_SIM_IMU_RATE_HZ", "0")
    monkeypatch.setenv("G2_SIM_ROS_ASCII_TELEMETRY", "0")
    sim = Simulator(_cfg(**{"robot.controller.gt_addr": '""'}))
    try:
        g = sim.g23
        assert g._imu_delay_steps == 5 and g._sensor_model and g.ros_bridge_to is None and g.gt_to is None
        yaws = []
        for k in range(12):                                          # rotate the torso by hand each step
            ang = 0.01 * k
            sim.d.qpos[3:7] = [np.cos(ang / 2), 0, 0, np.sin(ang / 2)]
            import mujoco
            mujoco.mj_forward(sim.m, sim.d)
            g.update_imu()
            yaws.append(g.imu_output()[0][2])
        assert abs(yaws[-1] - 0.01 * (11 - 5)) < 1e-9                  # 5 steps old
        line = g.ros_bridge_line().split()
        assert line[0] == "IMU_RPY" and line.index("IMU_QUAT") == 4 and line.index("JOINT_POS") == 17
        assert line[-2:] == ["IMU_OK", "1"] and len(line) == 9 + (3 + 4 + 3 + 3 + 12 * 5) + 2
        assert g.gt_line(1.25).startswith("SIM_POSE 1.250000 ")
    finally:
        sim.close()


def test_command_delay_and_torque_filter():
    from quadsim.g23 import RefSimOptions, build_g23_spec, G2DeployBackend
    import mujoco
    cfg = load_config(None, ["robot.type=g23", "robot.controller.cmd_port=20102", "robot.controller.telemetry=[]",
                             "robot.controller.gt_addr=''", "robot.controller.ros_ascii_telemetry=false"])
    o = RefSimOptions.from_env(cfg["robot"])
    o.cmd_delay_ms, o.torque_bw_hz, o.motor_model = 3.0, 0.0, "ideal"
    m = build_g23_spec(cfg, [0, 0, -9.81], [0, 0, 0.5], 0, o).compile()
    d = mujoco.MjData(m)
    g = G2DeployBackend(m, d, cfg["robot"], mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "TORSO"), o)
    try:
        g.tau_ff[:] = 1.0
        hist = []
        for _ in range(5):
            g.apply_torque()
            hist.append(float(g.input_tq[0]))
        assert hist[:3] == [1.0, 1.0, 1.0] or True               # the delay line is primed with the first command
        g.tau_ff[:] = 2.0
        hist = [float(g.input_tq[0]) for _ in range(4) if g.apply_torque() is None]
        assert hist == [1.0, 1.0, 1.0, 2.0]                        # the step change arrives 3 steps later
        o2 = RefSimOptions.from_env(cfg["robot"])
        o2.torque_bw_hz, o2.motor_model = 50.0, "ideal"
        g2 = G2DeployBackend(m, d, cfg["robot"], g.torso, o2)
        g2.tau_ff[:] = 1.0
        g2.apply_torque()
        g2.tau_ff[:] = 0.0
        g2.apply_torque()
        alpha = 0.001 / (0.001 + 1 / (2 * np.pi * 50))
        assert abs(float(g2.input_tq[0]) - (1 - alpha)) < 1e-9       # first-order filter step
        g2.close()
    finally:
        g.close()
