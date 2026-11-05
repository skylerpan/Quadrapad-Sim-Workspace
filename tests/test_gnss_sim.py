"""The GNSS antenna in the simulator: the site follows the mount, the sky comes from the scene (indoors at
the dp origin there is none), the fixes land where the ground truth says, stamps follow the IMU rule."""
import os

import numpy as np
import pytest

from quadsim.config import load_config
from quadsim.gnss import enu_to_lla, lla_to_enu, map_to_enu

SCENE = os.path.join(os.path.dirname(__file__), "..", "scenes", "dp")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "meta.json")), reason="scenes/dp not built")


def _sim(*over):
    from quadsim.sim import Simulator
    cfg = load_config(None, ["robot.type=ideal", "lidar.points_per_sec=2000", *over])
    cfg["scene"] = SCENE
    return Simulator(cfg)


def _antenna_truth(s):
    from quadsim.base import rot_zyx
    pos, quat, _, _ = s.ground_truth()
    w, x, y, z = quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pos + rot_zyx(yaw, 0.0, 0.0) @ np.asarray(s.cfg["gnss"]["mount_xyz"], float)


def test_antenna_site_sits_at_the_mount_on_the_base():
    s = _sim()
    s.control_tick()
    assert np.allclose(s.d.site_xpos[s.gnss_site], _antenna_truth(s), atol=0.02)      # level base at spawn


def test_indoors_the_scene_gives_no_sky_and_no_fix():
    s = _sim()
    fixes = []
    for _ in range(15):                                              # 0.3 s: 3 fixes at 10 Hz
        fixes += s.control_tick().gnss
    assert len(fixes) == 3 and all(f.fix == "none" and f.sky < s.gnss.sky_cfg["poor"] for f in fixes)


def test_with_sky_forced_the_fix_lands_on_the_antenna_within_three_sigma():
    s = _sim("gnss.sky.mode=always", "gnss.rtk=true")
    fixes = []
    for _ in range(25):                                              # 0.5 s: 5 fixes
        fixes += s.control_tick().gnss
    assert len(fixes) == 5 and {f.fix for f in fixes} == {"rtk_fixed"} and all(f.sky == 1.0 for f in fixes)
    truth = map_to_enu(_antenna_truth(s), s.gnss.datum)
    for f in fixes[-3:]:
        err = lla_to_enu(f.lat, f.lon, f.alt, s.gnss.datum) - truth
        assert np.hypot(err[0], err[1]) < 3 * (f.sigma_h + s.gnss.noise["rtk_fixed"]["bias_sigma"]) + 0.02
        assert abs(err[2]) < 3 * (f.sigma_v + s.gnss.noise["rtk_fixed"]["bias_sigma"]) + 0.02
    lat, lon, _ = enu_to_lla(truth, s.gnss.datum)
    assert abs(fixes[-1].lat - lat) < 1e-5 and abs(fixes[-1].lon - lon) < 1e-5


def test_fixes_come_at_the_configured_rate_stamped_like_the_imu():
    s = _sim("gnss.sky.mode=always")
    ticks = round(s.cfg["physics"]["control_hz"])                      # one second
    fixes = []
    for _ in range(ticks):
        fixes += s.control_tick().gnss
    assert len(fixes) == s.cfg["gnss"]["rate_hz"] and s.gnss_steps * s.dt == pytest.approx(1.0 / s.cfg["gnss"]["rate_hz"])
    # like the IMU: a fix describes the state at the start of the physics step that produced it, so its stamp is
    # one step before a multiple of the GNSS period
    period = s.gnss_steps * s.dt
    for k, f in enumerate(fixes, start=1):
        assert f.t == pytest.approx(k * period - s.dt, abs=1e-9)


def test_without_a_datum_there_is_no_receiver():
    s = _sim("gnss.datum=null")
    assert s.gnss is None and s.control_tick().gnss == []
