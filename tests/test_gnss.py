"""The GNSS receiver model as pure functions: WGS84 geodesy around the map datum, the sky -> fix-type policy,
the noise per fix type, and the F9R holdover. No scene, no MuJoCo, no ROS."""
import math

import numpy as np
import pytest

from quadsim.gnss import (Datum, GnssModel, enu_to_lla, enu_to_map, fix_for, lla_to_enu, map_to_enu,
                          sky_directions, sky_fraction)

DATUM = Datum(lat=24.960860, lon=121.421968, alt=130.0, yaw_deg=31.0)
M_PER_DEG_LON = 100981.7                     # at lat 24.96 on the WGS84 ellipsoid
M_PER_DEG_LAT = 110772.3


def _cfg(**over):
    cfg = {"model": "f9p", "rate_hz": 10, "rtk": False,
           "datum": {"lat": DATUM.lat, "lon": DATUM.lon, "alt": DATUM.alt, "yaw_deg": DATUM.yaw_deg},
           "sky": {"mode": "map", "rays": 9, "half_angle_deg": 60, "range": 50.0, "good": 0.9, "degraded": 0.5, "poor": 0.2},
           "noise": {"rtk_fixed": {"sigma_h": 0.01, "sigma_v": 0.02, "bias_sigma": 0.01, "bias_tau": 50, "vel_sigma": 0.02},
                     "rtk_float": {"sigma_h": 0.3, "sigma_v": 0.5, "bias_sigma": 0.2, "bias_tau": 50, "vel_sigma": 0.05},
                     "single": {"sigma_h": 1.5, "sigma_v": 3.0, "bias_sigma": 1.0, "bias_tau": 100, "vel_sigma": 0.1}},
           "holdover": {"secs": 20.0, "drift": 0.2}}
    for k, v in over.items():
        cfg[k] = v
    return cfg


# -- geodesy -----------------------------------------------------------------------------------------------
def test_enu_lla_round_trip_is_exact_around_the_datum():
    for enu in ([0.0, 0.0, 0.0], [100.0, -50.0, 3.0], [-2000.0, 1500.0, -20.0]):
        lat, lon, alt = enu_to_lla(np.array(enu), DATUM)
        assert np.allclose(lla_to_enu(lat, lon, alt, DATUM), enu, atol=1e-6)
    assert enu_to_lla(np.zeros(3), DATUM) == pytest.approx((DATUM.lat, DATUM.lon, DATUM.alt), abs=1e-9)


def test_one_metre_east_and_north_are_the_ellipsoid_degrees():
    lat, lon, _ = enu_to_lla(np.array([1.0, 0.0, 0.0]), DATUM)
    assert (lon - DATUM.lon) * M_PER_DEG_LON == pytest.approx(1.0, abs=1e-3) and lat == pytest.approx(DATUM.lat, abs=1e-9)
    lat, lon, _ = enu_to_lla(np.array([0.0, 1.0, 0.0]), DATUM)
    assert (lat - DATUM.lat) * M_PER_DEG_LAT == pytest.approx(1.0, abs=1e-3) and lon == pytest.approx(DATUM.lon, abs=1e-9)


def test_map_x_axis_points_along_the_datum_heading():
    # the map's +x is 31 deg counter-clockwise from east: 10 m along it is 8.57 m east, 5.15 m north
    e, n, u = map_to_enu(np.array([10.0, 0.0, 2.0]), DATUM)
    assert (e, n, u) == pytest.approx((10 * math.cos(math.radians(31)), 10 * math.sin(math.radians(31)), 2.0), abs=1e-9)
    # the user's reference point A (map 29.10, 39.50) lands within the 1.5 m the two given points disagree by
    lat, lon, _ = enu_to_lla(map_to_enu(np.array([29.10, 39.50, -1.1]), DATUM), DATUM)
    assert abs(lat - 24.961306) * M_PER_DEG_LAT < 1.5 and abs(lon - 121.422022) * M_PER_DEG_LON < 1.5


# -- sky ---------------------------------------------------------------------------------------------------
def test_enu_to_map_undoes_map_to_enu():
    for xyz in ([10.0, 0.0, 1.0], [-3.0, 7.5, 0.0]):
        assert np.allclose(enu_to_map(map_to_enu(np.array(xyz), DATUM), DATUM), xyz, atol=1e-12)
    east = enu_to_map(np.array([1.0, 0.0, 0.0]), DATUM)       # east is 31 deg clockwise of map x
    assert np.allclose(east, [math.cos(math.radians(31.0)), -math.sin(math.radians(31.0)), 0.0])


def test_sky_rays_are_a_cone_around_zenith_and_the_fraction_counts_misses():
    d = sky_directions(9, 60.0)
    assert d.shape == (9, 3) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[0], [0, 0, 1]) and np.allclose(d[1:, 2], math.cos(math.radians(60)))
    t_hit = np.array([np.inf, 3.0, np.inf, np.inf, 80.0, 1.0, np.inf, np.inf, np.inf])   # 80 m is beyond the range
    assert sky_fraction(t_hit, 50.0) == pytest.approx(7 / 9)


def test_fix_type_follows_sky_rtk_and_model():
    sky = _cfg()["sky"]
    assert fix_for(1.0, sky, rtk=True, model="f9p", outage_s=0.0, holdover_s=20) == "rtk_fixed"
    assert fix_for(1.0, sky, rtk=False, model="f9p", outage_s=0.0, holdover_s=20) == "single"
    assert fix_for(0.7, sky, rtk=True, model="f9p", outage_s=0.0, holdover_s=20) == "rtk_float"
    assert fix_for(0.7, sky, rtk=False, model="f9p", outage_s=0.0, holdover_s=20) == "single"
    assert fix_for(0.3, sky, rtk=True, model="f9p", outage_s=0.0, holdover_s=20) == "single"
    assert fix_for(0.0, sky, rtk=True, model="f9p", outage_s=0.1, holdover_s=20) == "none"       # F9P: no sky, no fix
    assert fix_for(0.0, sky, rtk=True, model="f9r", outage_s=5.0, holdover_s=20) == "dead_reckoning"
    assert fix_for(0.0, sky, rtk=True, model="f9r", outage_s=25.0, holdover_s=20) == "none"


# -- the receiver ------------------------------------------------------------------------------------------
def _run(model, n, pos, sky, t0=0.0, vel=(0.0, 0.0, 0.0)):
    dt = 1.0 / model.rate_hz
    return [model.sample(t0 + i * dt, np.array(pos, dtype=float), np.array(vel, dtype=float), sky) for i in range(n)]


def test_fixes_scatter_with_the_sigma_of_their_fix_type():
    for rtk, fix, sig in ((True, "rtk_fixed", 0.01), (False, "single", 1.5)):
        m = GnssModel(_cfg(rtk=rtk), np.random.default_rng(1))
        fixes = _run(m, 2000, [5.0, -3.0, 0.5], sky=1.0)
        assert {f.fix for f in fixes} == {fix}
        enu = np.array([lla_to_enu(f.lat, f.lon, f.alt, m.datum) for f in fixes])
        err = enu - map_to_enu(np.array([5.0, -3.0, 0.5]), m.datum)
        # white noise plus a slow bias of the same order: the spread is between sigma and 1.5 sigma
        assert sig * 0.85 < err[:, 0].std() < sig * 1.6 and sig * 0.85 < err[:, 1].std() < sig * 1.6
        assert all(f.sigma_h == sig and f.sky == 1.0 for f in fixes)
        assert all(np.isfinite([f.lat, f.lon, f.alt]).all() for f in fixes)


def test_velocity_is_reported_in_enu():
    m = GnssModel(_cfg(rtk=True), np.random.default_rng(2))
    fixes = _run(m, 500, [0.0, 0.0, 0.0], sky=1.0, vel=(1.0, 0.0, 0.0))   # 1 m/s along map +x
    v = np.array([f.vel_enu for f in fixes]).mean(axis=0)
    assert v == pytest.approx((math.cos(math.radians(31)), math.sin(math.radians(31)), 0.0), abs=0.01)


def test_f9p_loses_the_fix_at_once_and_f9r_holds_it_over_with_growing_sigma():
    p = GnssModel(_cfg(model="f9p"), np.random.default_rng(3))
    _run(p, 10, [0.0, 0.0, 0.0], sky=1.0)
    lost = _run(p, 5, [0.0, 0.0, 0.0], sky=0.0, t0=1.0)
    assert all(f.fix == "none" and math.isnan(f.lat) and math.isinf(f.sigma_h) for f in lost)

    r = GnssModel(_cfg(model="f9r"), np.random.default_rng(3))
    _run(r, 10, [0.0, 0.0, 0.0], sky=1.0)
    held = _run(r, 250, [0.0, 0.0, 0.0], sky=0.0, t0=1.0)                 # 25 s without sky, holdover is 20 s
    kinds = [f.fix for f in held]
    assert kinds[0] == "dead_reckoning" and kinds[-1] == "none" and kinds.index("none") == pytest.approx(200, abs=1)
    sig = [f.sigma_h for f in held if f.fix == "dead_reckoning"]
    assert sig[0] < sig[-1] and all(np.isfinite([f.lat, f.lon]).all() for f in held if f.fix == "dead_reckoning")
    back = _run(r, 3, [0.0, 0.0, 0.0], sky=1.0, t0=30.0)                  # sky again: a plain fix, small sigma
    assert back[-1].fix == "single" and back[-1].sigma_h == 1.5


def test_datum_and_config_are_checked_at_the_boundary():
    with pytest.raises(ValueError):
        GnssModel(_cfg(model="f9x"), np.random.default_rng(0))
    with pytest.raises(ValueError):
        GnssModel(_cfg(datum={"lat": 95.0, "lon": 0.0, "alt": 0.0, "yaw_deg": 0.0}), np.random.default_rng(0))
    assert Datum.from_cfg({"lat": 1, "lon": 2, "alt": 3, "yaw_deg": 4}) == Datum(1.0, 2.0, 3.0, 4.0)
