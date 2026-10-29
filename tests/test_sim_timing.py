import os

import numpy as np
import pytest

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp_1012")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "meta.json")),
                                reason="scenes/dp_1012 not built")


def _sim():
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    cfg = load_config(None, ["lidar.range_noise=0.0", "imu.acc_noise=0.0", "imu.gyro_noise=0.0",
                             "imu.acc_bias_rw=0.0", "imu.gyro_bias_rw=0.0"])
    cfg["scene"] = SCENE
    return Simulator(cfg)


def test_imu_stamped_with_the_state_it_measures():
    s = _sim()
    dt = s.dt * s.imu_steps
    out = s.control_tick()
    t = np.array([m.t for m in out.imu])
    # sample k describes the state at the start of the physics step that produced it
    assert np.allclose(t, out.t - s.dt - dt * np.arange(len(t))[::-1], atol=1e-9)


def test_lidar_pose_matches_ground_truth_while_turning():
    import mujoco
    s = _sim()
    for _ in range(100):                                  # spin in place at 1.5 rad/s
        s.set_cmd(0.0, 0.0, 1.5)
        s.control_tick()
    s.set_cmd(0.0, 0.0, 1.5)
    t0 = s.d.time
    mujoco.mj_step(s.m, s.d)                              # one step by hand, recorded like control_tick
    R_rec = s.d.site_xmat[s.lidar_site].reshape(3, 3).copy()
    _, quat, _, _ = s.ground_truth()                      # pose after the step (at s.d.time)
    yaw_after = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    yaw_rec = np.arctan2(R_rec[1, 0], R_rec[0, 0])
    w = s.d.qvel[5]
    # the recorded pose is the pose at t0 (one step earlier), not at s.d.time
    assert abs(np.angle(np.exp(1j * (yaw_after - yaw_rec))) - w * (s.d.time - t0)) < 2e-4


def test_control_tick_records_pose_with_its_own_time():
    s = _sim()
    for _ in range(50):
        s.set_cmd(0.0, 0.0, 1.5)
        s.control_tick()
    t_rec, R_rec = s.lidar._pose_t[-1], s.lidar._pose_R[-1]
    assert abs(t_rec - (s.d.time - s.dt)) < 1e-9
    _, quat, _, _ = s.ground_truth()
    yaw_now = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    yaw_rec = np.arctan2(R_rec[1, 0], R_rec[0, 0])
    assert abs(np.angle(np.exp(1j * (yaw_now - yaw_rec))) - s.d.qvel[5] * s.dt) < 2e-4
