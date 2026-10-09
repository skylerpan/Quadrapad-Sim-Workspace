import numpy as np

from pcd2sim.voxel import (boxes_to_world, fit_floor_normal, merge_boxes, remove_isolated, smooth_surface_z,
                           voxelize)


def _expand(boxes):
    """All voxel keys covered by inclusive box ranges, with multiplicity."""
    out = []
    for x0, x1, y0, y1, z0, z1 in boxes:
        g = np.mgrid[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1].reshape(3, -1).T
        out.append(g)
    return np.concatenate(out)


def test_merge_boxes_tiles_exactly():
    rng = np.random.default_rng(1)
    keys = np.unique(rng.integers(0, 12, size=(800, 3)), axis=0)
    # add a solid slab and a wall so merging has real work to do
    slab = np.mgrid[0:20, 0:15, 0:1].reshape(3, -1).T
    wall = np.mgrid[0:20, 20:21, 0:8].reshape(3, -1).T
    keys = np.unique(np.concatenate([keys, slab, wall]), axis=0)
    boxes = merge_boxes(keys)
    covered = _expand(boxes)
    assert len(covered) == len(keys)                      # no overlaps
    assert len(np.unique(covered, axis=0)) == len(keys)   # same set
    assert set(map(tuple, covered)) == set(map(tuple, keys))
    assert len(boxes) < len(keys) / 3                     # actually merged


def test_slab_merges_to_one_box():
    slab = np.mgrid[0:30, 0:10, 0:2].reshape(3, -1).T
    boxes = merge_boxes(slab)
    assert len(boxes) == 1
    assert boxes[0].tolist() == [0, 29, 0, 9, 0, 1]


def test_voxelize_boxes_world_and_isolated():
    rng = np.random.default_rng(2)
    floor = np.c_[rng.uniform(0, 4, 20000), rng.uniform(0, 3, 20000), rng.normal(0, 0.005, 20000)]
    lonely = np.array([[2.0, 1.5, 1.5]])
    g = voxelize(np.vstack([floor, lonely]), np.ones(20001, np.float32), 0.1)
    g2 = remove_isolated(g, 2)
    assert len(g2.keys) == len(g.keys) - 1                # the floating point is removed
    w = boxes_to_world(merge_boxes(g2.keys), g2)
    lo, hi = w[:, :3] - w[:, 3:], w[:, :3] + w[:, 3:]
    assert lo[:, 0].min() >= g2.origin[0] - 1e-9 and hi[:, 0].max() <= 4.1


def test_floor_normal_and_surface_smoothing():
    rng = np.random.default_rng(3)
    n = 30000
    tilt = np.deg2rad(2.0)
    x, y = rng.uniform(0, 5, n), rng.uniform(0, 5, n)
    z = np.tan(tilt) * x + rng.normal(0, 0.03, n)
    pts = np.c_[x, y, z]
    normal, _, _ = fit_floor_normal(pts, thresh=0.1)
    assert abs(np.degrees(np.arccos(normal[2])) - 2.0) < 0.3
    g = voxelize(pts, None, 0.1)
    zs = smooth_surface_z(g)
    resid_raw = g.zmean - np.tan(tilt) * g.centers()[:, 0]
    resid_s = zs - np.tan(tilt) * g.centers()[:, 0]
    assert resid_s.std() < 0.7 * resid_raw.std()          # noise reduced
