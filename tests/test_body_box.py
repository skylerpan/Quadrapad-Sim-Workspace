"""Simulator._blocked: oriented body box vs map boxes (no scene needed)."""
import math

import numpy as np

from quadsim.base import body_lift


def _sim(half=(0.35, 0.155, 0.08)):
    from quadsim.proxies import BoxIndex
    from quadsim.sim import Simulator
    s = Simulator.__new__(Simulator)              # only .half and .index are used by _blocked
    s.half = np.asarray(half)
    s.index = BoxIndex(np.zeros((1, 6)))
    return s


def _set(s, boxes):
    from quadsim.proxies import BoxIndex
    s.index = BoxIndex(np.asarray(boxes, dtype=np.float64))


def test_level_box_matches_the_old_z_band_test():
    s = _sim()
    z_center = 0.30
    bottom = z_center - s.half[2]
    _set(s, [[s.half[0] + 0.04, 0.0, bottom + 0.015 - 0.05, 0.05, 0.05, 0.05]])   # 1.5 cm into the band
    assert s._blocked(0.0, 0.0, 0.0, z_center)
    _set(s, [[s.half[0] + 0.04, 0.0, bottom - 0.06, 0.05, 0.05, 0.05]])           # below the body
    assert not s._blocked(0.0, 0.0, 0.0, z_center)
    _set(s, [[0.0, s.half[1] + 0.06 + 0.05, z_center, 0.05, 0.05, 0.05]])         # 6 cm beside the body
    assert not s._blocked(0.0, 0.0, 0.0, z_center)
    assert s._blocked(0.0, 0.0, math.pi / 2, z_center)                            # yawed 90 deg: body x spans it


def test_pitched_body_clears_the_next_riser_a_level_body_hits():
    cfg = {"stand_height": 0.30}
    s = _sim()
    # Next riser: 0.35 m ahead, top 0.40 above the feet-plane origin (two 0.2 m treads up) -> a level body
    # (bottom at 0.22) hits it; on a 28 deg flight the body is pitched nose-up and lifted along the normal
    pitch = -math.atan(0.2 / 0.38)               # negative = nose up (base.py convention)
    _set(s, [[0.44, 0.0, 0.20, 0.10, 0.6, 0.20]])  # x 0.34..0.54 (1 cm into the body), z 0..0.4
    assert s._blocked(0.0, 0.0, 0.0, 0.0 + cfg["stand_height"])
    z = 0.0 + body_lift(cfg, 0.0, pitch)
    assert not s._blocked(0.0, 0.0, 0.0, z, 0.0, pitch)
    # the reverse: a lintel at the raised nose of the pitched body (top-front corner at x 0.27, z 0.57),
    # well above the level body's top (0.38)
    _set(s, [[0.27, 0.0, 0.57, 0.05, 0.6, 0.02]])  # x 0.22..0.32, z 0.55..0.59
    assert not s._blocked(0.0, 0.0, 0.0, 0.0 + cfg["stand_height"])
    assert s._blocked(0.0, 0.0, 0.0, z, 0.0, pitch)


def test_body_lift_keeps_the_bottom_clearance_on_a_slope():
    cfg = {"stand_height": 0.30}
    assert body_lift(cfg, 0.0, 0.0) == 0.30
    pitch = math.radians(30)
    # perpendicular distance from the plane z = tan(30) * x to the centre (0, lift) is stand_height
    lift = body_lift(cfg, 0.0, -pitch)
    assert abs(lift * math.cos(pitch) - 0.30) < 1e-9
