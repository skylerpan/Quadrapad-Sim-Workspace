import numpy as np

from pcd2sim.voxel import (boxes_to_world, components_26, remove_small_clusters, fit_floor_normal, merge_boxes, remove_isolated, smooth_surface_z,
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


def test_remove_small_clusters():
    from pcd2sim.voxel import voxelize
    block = np.array([(i, j, k) for i in range(3) for j in range(3) for k in range(3)], dtype=float)   # 27 voxels
    chain = np.array([(10, 0, 0), (11, 1, 0), (12, 2, 1)], dtype=float)                                  # 3, diagonal
    lone = np.array([(20, 20, 20)], dtype=float)
    g = voxelize(np.vstack([block, chain, lone]) + 0.5, None, 1.0)
    assert len(g.keys) == 31
    n, lab = components_26(g)
    assert n == 3 and sorted(np.bincount(lab).tolist()) == [1, 3, 27]
    assert len(remove_small_clusters(g, 0).keys) == 31 and len(remove_small_clusters(g, 1).keys) == 31
    assert len(remove_small_clusters(g, 2).keys) == 30
    assert len(remove_small_clusters(g, 4).keys) == 27
    assert len(remove_small_clusters(g, 28).keys) == 0


def _grid_from_keys(keys, voxel=0.03, intensity=None):
    from pcd2sim.voxel import VoxelGrid, encode
    keys = np.asarray(keys, dtype=np.int32)
    dims = keys.max(axis=0) + 2
    order = np.argsort(encode(keys, dims))
    keys = keys[order]
    inten = np.full(len(keys), 10.0, np.float32) if intensity is None else np.asarray(intensity, np.float32)[order]
    zmean = (keys[:, 2] + 0.5) * voxel
    return VoxelGrid(keys, np.ones(len(keys), np.int32), inten, zmean.astype(np.float32),
                     np.zeros(3), voxel, dims)


def test_close_holes_fills_sheet_hole_without_thickening(tmp_path):
    from pcd2sim.voxel import close_holes, encode
    sheet = np.mgrid[0:12, 0:12, 2:3].reshape(3, -1).T
    hole = (sheet[:, 0] == 5) & (sheet[:, 1] == 6)
    g = _grid_from_keys(sheet[~hole])
    g2, n = close_holes(g, rounds=1, workers=1, tmp_dir=str(tmp_path / "close"))
    assert n == 1
    assert len(g2.keys) == len(sheet)                            # hole filled, nothing else added
    assert set(map(tuple, g2.keys.tolist())) == set(map(tuple, sheet.tolist()))
    assert np.all(np.diff(encode(g2.keys, g2.dims)) > 0)        # still code-sorted
    assert not (tmp_path / "close").exists()                     # it created the dir, so it removed it


def test_close_holes_inside_corner_only_and_no_sheet_thickening(tmp_path):
    from pcd2sim.voxel import close_holes
    floor = np.mgrid[0:10, 0:10, 0:1].reshape(3, -1).T
    wall = np.mgrid[0:10, 9:10, 1:6].reshape(3, -1).T          # wall standing on the floor's far edge
    g = _grid_from_keys(np.vstack([floor, wall]))
    g2, n = close_holes(g, rounds=1, workers=1, tmp_dir=str(tmp_path))
    added = set(map(tuple, g2.keys.tolist())) - set(map(tuple, np.vstack([floor, wall]).tolist()))
    assert all(k[2] == 1 and k[1] == 8 for k in added)         # only the inside corner line (sees 9 + 6)
    assert n == len(added) <= 10


def test_close_holes_wall_gap_and_speck(tmp_path):
    from pcd2sim.voxel import close_holes
    wall = np.mgrid[0:10, 4:5, 0:8].reshape(3, -1).T
    gap = (wall[:, 0] == 4) & (wall[:, 2] == 3)
    speck = np.array([[20, 20, 20]])
    keys = np.vstack([wall[~gap], speck])
    inten = np.r_[np.full(len(wall[~gap]), 30.0), 200.0]
    g = _grid_from_keys(keys, intensity=inten)
    g2, n = close_holes(g, rounds=2, workers=1, tmp_dir=str(tmp_path))
    assert n == 1
    k = g2.keys.tolist()
    assert [4, 4, 3] in k and [20, 20, 20] in k                   # gap filled, speck untouched and not grown
    assert len(g2.keys) == len(keys) + 1
    added = np.flatnonzero((g2.keys == [4, 4, 3]).all(axis=1))[0]
    assert g2.counts[added] == 1 and abs(g2.intensity[added] - 30.0) < 1e-5
    assert abs(g2.zmean[added] - 3.5 * 0.03) < 1e-6


def test_close_holes_rounds_zero_and_chunking_invisible(tmp_path):
    from pcd2sim.voxel import close_holes
    rng = np.random.default_rng(3)
    floor = np.mgrid[0:40, 0:40, 1:2].reshape(3, -1).T
    keep = rng.random(len(floor)) > 0.15                         # 15 % holes
    g = _grid_from_keys(floor[keep])
    g0, n0 = close_holes(g, rounds=0)
    assert n0 == 0 and g0 is g
    a, na = close_holes(g, rounds=2, workers=1, tmp_dir=str(tmp_path / "a"))
    b, nb = close_holes(g, rounds=2, workers=3, tmp_dir=str(tmp_path / "b"), chunk=97)
    assert na == nb > 0
    assert np.array_equal(a.keys, b.keys) and np.allclose(a.intensity, b.intensity)
