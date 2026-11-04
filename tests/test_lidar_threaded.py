"""RollingLidar on a worker thread delivers the same frames as the synchronous path, one tick late."""
import os
import platform
import stat
import time

import pytest

import numpy as np

from quadsim.lidar import RollingLidar
from quadsim.raycast import RayScene


class _Refl:
    def reflectivity(self, pts):
        return np.full(len(pts), 100, np.uint8)


def _lidar(threaded):
    # a 10 x 10 m room: floor, ceiling and four walls as boxes
    boxes = np.array([[0, 0, -0.05, 5, 5, 0.05], [0, 0, 3.05, 5, 5, 0.05],
                      [5.05, 0, 1.5, 0.05, 5, 1.5], [-5.05, 0, 1.5, 0.05, 5, 1.5],
                      [0, 5.05, 1.5, 5, 0.05, 1.5], [0, -5.05, 1.5, 5, 0.05, 1.5]], dtype=np.float32)
    cfg = {"rate_hz": 10, "points_per_sec": 20000, "fov_deg": [-7.0, 52.0], "lines": 4, "min_range": 0.1,
           "max_range": 40.0, "range_noise": 0.0, "dropout": 0.0, "grazing_cos": 0.0, "seed": 0}
    return RollingLidar(cfg, RayScene(boxes, None), _Refl(), 0.1, threaded=threaded)


def _drive(lidar, ticks=30, steps=20, dt=0.001):
    lidar.reset(0.0)
    frames = []
    t = 0.0
    for _ in range(ticks):
        for _ in range(steps):
            lidar.record_pose(t, np.array([0.0, 0.0, 0.5]), np.eye(3))
            t += dt
        frames += lidar.process()
    return frames


def test_threaded_frames_match_synchronous():
    sync = _drive(_lidar(False))
    th = _lidar(True)
    frames = _drive(th)
    deadline = time.time() + 5.0
    while len(frames) < len(sync) and time.time() < deadline:      # the worker finishes the last tick's rays
        time.sleep(0.02)
        frames += th.process()
    assert len(sync) == 5 and len(frames) == len(sync)
    for a, b in zip(sync, frames):
        assert a.stamp == b.stamp and len(a.xyz) == len(b.xyz)
        assert np.allclose(a.xyz, b.xyz, atol=1e-5) and np.array_equal(a.offset, b.offset)


def test_threaded_reset_drops_old_frames():
    th = _lidar(True)
    _drive(th, ticks=8)                                             # ~1.6 s of rays queued / cast
    th.reset(0.0)
    time.sleep(0.3)
    assert th.process() == []                                       # nothing from before the reset comes back
    frames = _drive(th, ticks=15)
    deadline = time.time() + 5.0
    while len(frames) < 2 and time.time() < deadline:
        time.sleep(0.02)
        frames += th.process()
    assert [f.stamp for f in frames[:2]] == [0.0, 0.1]


def test_controller_process_runs_and_reports_exit(tmp_path):
    from quadsim.controller_proc import ControllerProcess
    exe = tmp_path / "fake_controller.sh"
    exe.write_text("#!/bin/sh\necho hello from $0 $1 $2\necho \"ORT=$ORT_DYLIB_PATH STATE=$G2_LMA_STATE_SOURCE\"\nexit 3\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    lines = []
    p = ControllerProcess({"kind": "g2_deploy_rs", "exec": str(exe), "args": ["sim", "ros"]}, lines.append, repo=str(tmp_path))
    assert p.available
    p.start()
    deadline = time.time() + 5.0
    while p.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.1)
    assert p.poll() == 3
    assert any("hello from" in l and "sim ros" in l for l in lines)
    assert any("STATE=ground_truth" in l for l in lines)
    missing = ControllerProcess({"kind": "g2_deploy_rs"}, lines.append, repo=str(tmp_path))
    assert not missing.available and missing.exe.endswith("target/release/g2_deploy_rs")
    assert p.spec.log_prefix == "[g2]" and any(l.startswith("[g2] ") for l in lines)


def test_motorcontrol_controller_process_defaults(tmp_path, monkeypatch):
    from quadsim.controller_proc import ControllerProcess
    monkeypatch.delenv("ROBOTDOG_BACKEND", raising=False)
    monkeypatch.setenv("ROBOTDOG_ROBOT", "from-env")                      # the environment wins over our defaults
    lib = tmp_path / "third_party" / platform.machine() / "lib" / "libonnxruntime.so.1.19.2"
    lib.parent.mkdir(parents=True)
    lib.write_text("")
    exe = tmp_path / "fake_robotdog.sh"
    exe.write_text("#!/bin/sh\necho \"argc=$# BACKEND=$ROBOTDOG_BACKEND ROBOT=$ROBOTDOG_ROBOT TARGET=$ROBOTDOG_SIM_TARGET\"\n"
                   "echo \"[STATE] rest ----> stand\"\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    cfg = {"kind": "motorcontrol", "exec": str(exe), "motorcontrol": {"cmd_port": 40101, "joint_addr": "127.0.0.1:40102",
           "base_addr": "127.0.0.1:40103", "operator_addr": "127.0.0.1:40120"}, "env": {"ROBOTDOG_TELEMETRY": "on"}}
    lines = []
    p = ControllerProcess(cfg, lines.append, repo=str(tmp_path))
    assert p.args == [] and p.spec.log_prefix == "[mc]"
    env = p.env
    assert env["ROBOTDOG_BACKEND"] == "sim" and env["ROBOTDOG_ROBOT"] == "from-env"
    assert env["ROBOTDOG_COMMAND_SOURCE"] == "udp" and env["ROBOTDOG_STATE_PUBLISHER"] == "udp"
    assert env["ROBOTDOG_TELEMETRY"] == "on"                                # robot.controller.env wins over our defaults
    assert env["ROBOTDOG_SHUTDOWN_SETTLE_SECS"] == "0"
    assert env["ROBOTDOG_SIM_TARGET"] == "127.0.0.1:40101" and env["ROBOTDOG_SIM_JOINT_PORT"] == "40102"
    assert env["ROBOTDOG_SIM_BASE_PORT"] == "40103" and env["ROBOTDOG_OPERATOR_UDP_PORT"] == "40120"
    assert env["ROBOTDOG_ONNXRUNTIME_LIB"] == str(lib)
    p.start()
    deadline = time.time() + 5.0
    while p.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.1)
    assert p.poll() == 0
    assert any("argc=0 BACKEND=sim ROBOT=from-env TARGET=127.0.0.1:40101" in l for l in lines)
    assert p.state is None and not p.safety_locked                          # stdout is not parsed for this controller
    missing = ControllerProcess({"kind": "motorcontrol"}, lines.append, repo=str(tmp_path))
    assert not missing.available and missing.exe.endswith("target/release/robotdog_control")
    default = ControllerProcess({}, lines.append, repo=str(tmp_path))
    assert default.spec.kind == "motorcontrol"                              # the default controller
    with pytest.raises(ValueError):
        ControllerProcess({"kind": "unknown"}, lines.append, repo=str(tmp_path))
