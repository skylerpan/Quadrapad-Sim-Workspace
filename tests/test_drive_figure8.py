"""scripts/drive_figure8.py without ROS: the figure-8 geometry and the pure-pursuit tracker on a unicycle."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import drive_figure8 as f8  # noqa: E402

R = 1.5


@pytest.mark.parametrize("heading", [0.0, 1.0, -2.5])
def test_figure8_is_two_tangent_circles(heading):
    c = np.array([2.0, -1.0])
    p = f8.figure8_path(c, heading, R, laps=3, ds=0.01)
    left = np.array([-math.sin(heading), math.cos(heading)])
    lobes = (c + R * left, c - R * left)
    d = np.minimum(np.linalg.norm(p - lobes[0], axis=1), np.linalg.norm(p - lobes[1], axis=1))
    assert np.allclose(d, R, atol=1e-9)                             # every point on one of the two circles
    assert np.allclose(p[0], c) and np.allclose(p[-1], c)           # starts and ends at the crossing
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert seg.max() < 0.011
    assert seg.sum() == pytest.approx(3 * 2 * 2 * math.pi * R, rel=1e-3)
    d0 = p[1] - p[0]
    assert math.atan2(d0[1], d0[0]) == pytest.approx(heading, abs=1e-2)   # leaves the crossing along heading
    # first lobe on the left (counter-clockwise), then the right one (clockwise); 2 per lap
    q = p[len(p) // 12]
    assert np.linalg.norm(q - lobes[0]) == pytest.approx(R) and np.linalg.norm(q - lobes[1]) > R
    near_c = np.linalg.norm(p - c, axis=1) < 0.005
    passes = np.count_nonzero(np.diff(near_c.astype(int)) == 1) + int(near_c[0])
    assert passes == 2 * 3 + 0 or passes == 2 * 3 + 1


def test_lead_in_is_a_straight_line_to_the_crossing():
    c = np.array([8.0, 0.0])
    p, n_lead = f8.build_path(np.array([0.0, 0.0]), c, 0.0, R, laps=1, ds=0.02)
    assert np.allclose(p[:n_lead, 1], 0.0) and np.all(np.diff(p[:n_lead, 0]) > 0)
    assert np.allclose(p[n_lead], c)
    p2, n2 = f8.build_path(c.copy(), c, 0.0, R, laps=1, ds=0.02)
    assert n2 == 0 and np.allclose(p2[0], c)


def _unicycle(path, start, yaw, v=0.4, dt=0.02, seconds=200.0, wz_max=0.8, **kw):
    pp = f8.PurePursuit(path, **kw)
    x, y = start
    travelled, worst, n = 0.0, 0.0, 0
    while n * dt < seconds:
        wz, done, err = pp.step(x, y, yaw)
        if done:
            return True, travelled, worst, pp
        wz = max(-wz_max, min(wz_max, v * wz))
        worst = max(worst, err)
        yaw += wz * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        travelled += v * dt
        n += 1
    return False, travelled, worst, pp


def test_pure_pursuit_flies_the_whole_figure8_without_cutting_across():
    c = np.array([8.0, 0.0])
    path, n_lead = f8.build_path(np.array([0.0, 0.0]), c, 0.0, R, laps=3, ds=0.02)
    done, travelled, worst, pp = _unicycle(path, (0.0, 0.0), 0.0)
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert done
    assert travelled == pytest.approx(length, rel=0.03)              # no lobe skipped at the crossing
    assert worst < 0.08                                              # cross-track error, m
    assert pp.fraction_done(n_lead) == pytest.approx(1.0, abs=0.01)   # of the figure-8 after the lead-in


def test_pure_pursuit_starts_from_off_the_path():
    c = np.array([2.0, 0.0])
    path, n_lead = f8.build_path(np.array([0.0, 0.3]), c, 0.0, R, laps=1, ds=0.02)
    done, travelled, worst, _ = _unicycle(path, (0.0, 0.3), 0.3)
    assert done and worst < 0.3


def test_curvature_on_a_lobe_is_one_over_radius():
    c = np.array([0.0, 0.0])
    path = f8.figure8_path(c, 0.0, R, laps=1, ds=0.01)
    pp = f8.PurePursuit(path, lookahead=0.5)
    i = len(path) // 8                                               # on the left lobe, heading along it
    d = path[i + 1] - path[i]
    k, done, err = pp.step(path[i][0], path[i][1], math.atan2(d[1], d[0]))
    assert not done and err < 1e-3
    assert k == pytest.approx(1.0 / R, rel=0.03)
