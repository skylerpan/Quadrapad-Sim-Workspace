"""The noise model of the IMU the g23 controller reads (quadsim/imu_noise.py), and its wiring into G23Physics."""
import os

import numpy as np
import pytest

from quadsim.config import load_config
from quadsim.imu_noise import ImuNoise, ImuNoiseParams

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), load_config(None, [])["scene"])
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "contact.npy")),
                                 reason=f"{os.path.relpath(SCENE)} (config scene) not built as format 3")

DT = 0.001
RPY = np.array([0.01, -0.02, 1.0])
GYRO = np.array([0.1, 0.2, 0.3])
ACC = np.array([0.0, 0.0, 9.81])


def _run(params, steps, seed=0):
    n = ImuNoise(params, DT, seed)
    out = [n.apply(RPY, GYRO, ACC) for _ in range(steps)]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]))


def test_all_zero_is_disabled_and_bit_exact():
    n = ImuNoise(ImuNoiseParams(), DT, 0)
    assert not n.enabled
    rpy, gyro, acc = n.apply(RPY, GYRO, ACC)
    assert np.array_equal(rpy, RPY) and np.array_equal(gyro, GYRO) and np.array_equal(acc, ACC)


def test_params_come_from_the_config_block_and_reject_unknown_keys():
    p = ImuNoiseParams.from_cfg({"gyro": 0.002, "acc": 0.02})
    assert p.gyro == 0.002 and p.acc == 0.02 and p.rp == 0.0 and ImuNoise(p, DT, 0).enabled
    assert not ImuNoise(ImuNoiseParams.from_cfg(None), DT, 0).enabled
    with pytest.raises(ValueError, match="gyros"):
        ImuNoiseParams.from_cfg({"gyros": 0.1})
    with pytest.raises(ValueError, match="negative"):
        ImuNoiseParams.from_cfg({"gyro": -0.1})


def test_same_seed_repeats_and_another_seed_differs():
    p = ImuNoiseParams(acc=0.02, gyro=0.002, acc_bias_rw=0.0005, gyro_bias_rw=0.00005, gyro_bias=0.002, rp=0.001)
    a, b, c = _run(p, 200, seed=3), _run(p, 200, seed=3), _run(p, 200, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])


def test_white_noise_has_the_configured_sigma_and_no_bias():
    rpy, gyro, acc = _run(ImuNoiseParams(acc=0.02, gyro=0.002, rp=0.001), 20000)
    assert np.allclose(gyro.std(axis=0), 0.002, rtol=0.05) and np.allclose(gyro.mean(axis=0), GYRO, atol=1e-4)
    assert np.allclose(acc.std(axis=0), 0.02, rtol=0.05) and np.allclose(acc.mean(axis=0), ACC, atol=1e-3)
    assert np.allclose(rpy[:, :2].std(axis=0), 0.001, rtol=0.05)
    assert np.array_equal(rpy[:, 2], np.full(20000, RPY[2]))         # yaw only moves through the gyro z bias


def test_yaw_drifts_with_the_integral_of_the_gyro_z_bias():
    """A constant gyro bias b shows up on the gyro and, integrated, on the yaw the AHRS would report."""
    steps = 5000
    rpy, gyro, _ = _run(ImuNoiseParams(gyro_bias=0.01), steps, seed=1)
    bias = gyro[0] - GYRO
    assert np.all(np.abs(bias) > 0) and np.allclose(gyro - GYRO, bias)                 # constant, per axis
    drift = rpy[:, 2] - RPY[2]
    assert np.allclose(drift, bias[2] * DT * np.arange(1, steps + 1), atol=1e-9)
    assert np.array_equal(rpy[:, :2], np.tile(RPY[:2], (steps, 1)))


def test_bias_random_walk_grows_like_sqrt_time():
    ends = np.array([_run(ImuNoiseParams(gyro_bias_rw=0.001), 1000, seed=s)[1][-1] - GYRO for s in range(60)])
    assert np.isclose(ends.std(), 0.001 * np.sqrt(1000 * DT), rtol=0.2)               # sigma * sqrt(t), t = 1 s


def test_yaw_stays_wrapped():
    n = ImuNoise(ImuNoiseParams(gyro_bias=0.5), DT, 0)
    for _ in range(20000):
        rpy, _, _ = n.apply(np.array([0.0, 0.0, 3.1]), GYRO, ACC)
        assert -np.pi <= rpy[2] <= np.pi


@needs_scene
def test_the_controller_telemetry_carries_the_noisy_imu_once_per_step():
    from quadsim.g23 import RefSimOptions
    from quadsim.sim import Simulator
    from tests.test_g23 import _cfg
    o = RefSimOptions.from_env({"controller": {"imu_noise": {"gyro": 0.002, "seed": 5}}})
    assert o.imu_noise.gyro == 0.002 and "IMU noise" in o.describe()
    assert "IMU noise" not in RefSimOptions.from_env({}).describe()
    sim = Simulator(_cfg(**{"robot.controller.gt_addr": '""', "robot.controller.ros_ascii_telemetry": "false",
                            "robot.controller.imu_noise": "{gyro: 0.002, seed: 5}"}))
    try:
        g = sim.g23
        g.update_imu()
        first = g.imu_output()
        assert not np.array_equal(first[1], g._raw_imu()[1])                           # noisy
        again = g.imu_output()
        assert all(np.array_equal(a, b) for a, b in zip(first, again))                 # one draw per physics step
        g.update_imu()
        assert not np.array_equal(g.imu_output()[1], first[1])
    finally:
        sim.close()
