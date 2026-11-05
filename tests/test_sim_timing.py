import os

import numpy as np
import pytest

from quadsim.config import load_config

SCENE = os.path.join(os.path.dirname(__file__), "..", load_config(None, [])["scene"])   # the config default scene
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "meta.json")),
                                reason=f"{os.path.relpath(SCENE)} (config/sim.yaml scene) not built")


def _site_yaw(s, R_site):
    """Yaw of the base from the recorded lidar site rotation: the mount (pitched Mid-360) is taken off first."""
    from quadsim.base import rot_zyx
    roll, pitch, yaw = s.cfg["lidar"].get("mount_rpy") or [0.0, 0.0, 0.0]
    R = R_site @ rot_zyx(yaw, pitch, roll).T
    return np.arctan2(R[1, 0], R[0, 0])


def _sim():
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    cfg = load_config(None, ["robot.type=ideal", "lidar.range_noise=0.0", "imu.acc_noise=0.0", "imu.gyro_noise=0.0",
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


def test_lidar_site_follows_the_urdf_mount():
    """lidar.mount_xyz / mount_rpy place the site like the URDF joint base_link -> lidar_top (pitched Mid-360)."""
    from quadsim.base import lidar_mount, rot_zyx
    from quadsim.sim import Simulator
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=2000", "lidar.mount_xyz=[0.195,0,0.155]",
                             "lidar.mount_rpy=[0,0.25,0]"])
    pos, quat = lidar_mount(cfg["lidar"])
    assert pos == [0.195, 0.0, 0.155] and abs(quat[0] - np.cos(0.125)) < 1e-9 and abs(quat[2] - np.sin(0.125)) < 1e-9
    s = Simulator(cfg)
    try:
        R_base = s.d.xmat[s.base_body].reshape(3, 3)
        R_site = s.d.site_xmat[s.lidar_site].reshape(3, 3)
        assert np.allclose(R_base.T @ R_site, rot_zyx(0.0, 0.25, 0.0), atol=1e-9)      # pitched on the base
        offset = R_base.T @ (s.d.site_xpos[s.lidar_site] - s.d.xpos[s.base_body])
        assert np.allclose(offset, [0.195, 0.0, 0.155], atol=1e-9)
        assert lidar_mount({"mount_xyz": [1, 2, 3]})[1] == [1.0, 0.0, 0.0, 0.0]           # no rpy: level
    finally:
        s.close()


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
    yaw_rec = _site_yaw(s, R_rec)
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
    yaw_rec = _site_yaw(s, R_rec)
    assert abs(np.angle(np.exp(1j * (yaw_now - yaw_rec))) - s.d.qvel[5] * s.dt) < 2e-4
