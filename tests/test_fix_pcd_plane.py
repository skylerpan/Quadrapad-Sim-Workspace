import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "fix_pcd_plane", os.path.join(os.path.dirname(__file__), "..", "scripts", "fix_pcd_plane.py"))
fix_pcd_plane = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix_pcd_plane)

FLOOR_Z, PLANE_Z = -0.5, -0.95
HOLE = (2.0, 2.0)          # 1 x 1 m unscanned patch of the floor


def _map(seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, 6.0, 0.04)
    x, y = [a.ravel() for a in np.meshgrid(g, g)]
    hole = (np.abs(x - HOLE[0]) < 0.5) & (np.abs(y - HOLE[1]) < 0.5)
    floor = np.column_stack([x, y, FLOOR_Z + 0.02 * (x - 3) + rng.normal(0, 0.03, len(x))])[~hole]
    zs = np.arange(FLOOR_Z, 2.5, 0.05)
    xs, zz = [a.ravel() for a in np.meshgrid(g, zs)]
    wall = np.column_stack([xs, np.full(len(xs), 6.02), zz])
    # synthetic plane: dense, flat, covering the floor, the hole and 5-10 m beyond the floor (no floor there)
    gp = np.arange(0.0, 11.0, 0.03)
    px, py = [a.ravel() for a in np.meshgrid(gp, gp)]
    plane = np.column_stack([px, py, PLANE_Z + rng.normal(0, 0.005, len(px))])
    return np.vstack([floor, wall, plane])


def test_plane_detected_lifted_in_hole_and_dropped_elsewhere():
    xyz = _map()
    pz = fix_pcd_plane.detect_plane_z(xyz[:, 2])
    assert abs(pz - PLANE_Z) < 0.01
    keep, new_z, st = fix_pcd_plane.fix_plane(xyz, pz)
    plane = np.abs(xyz[:, 2] - PLANE_Z) < 0.1
    kept_plane = plane & keep
    assert st["plane_points"] == plane.sum()
    # under the scanned floor the plane is gone
    under = plane & (np.abs(xyz[:, 0] - 4.0) < 0.5) & (np.abs(xyz[:, 1] - 4.0) < 0.5)
    assert not keep[under].any()
    # in the hole it is kept and lifted to the local floor height
    in_hole = plane & (np.abs(xyz[:, 0] - HOLE[0]) < 0.4) & (np.abs(xyz[:, 1] - HOLE[1]) < 0.4)
    assert keep[in_hole].all()
    expect = FLOOR_Z + 0.02 * (xyz[in_hole, 0] - 3)
    assert np.abs(new_z[in_hole] - expect).max() < 0.03
    # far outside the floor (no floor cell within 3 m) it is dropped
    far = plane & (xyz[:, 0] > 9.5) & (xyz[:, 1] > 9.5)
    assert not keep[far].any()
    # everything that is not plane is untouched
    assert keep[~plane].all() and np.array_equal(new_z[~plane], xyz[~plane, 2])
    assert kept_plane.sum() == st["lifted"]


def test_no_plane_means_nothing_to_do():
    xyz = _map()
    xyz = xyz[np.abs(xyz[:, 2] - PLANE_Z) > 0.1]
    assert fix_pcd_plane.detect_plane_z(xyz[:, 2]) is None
