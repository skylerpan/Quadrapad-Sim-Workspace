import numpy as np

from pcd2sim.ground import extract_ground
from pcd2sim.voxel import standable_mask, voxelize

SLOPE = np.tan(np.radians(0.5))      # SLAM-like floor tilt along x
STEP_X, STEP_H = 4.0, 0.17           # platform beyond x = 4 m
WALL_Y = 4.0
HOLE = (1.02, 1.02)
SPIKE = (2.02, 2.52)
LOW = (3.02, 1.52)                   # 1 ghost point 0.18 m up: becomes its column's top (ground cell) voxel


def floor_z(x):
    return SLOPE * x + np.where(x >= STEP_X, STEP_H, 0.0)


def _room(seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, 6.0, 0.05)
    x, y = [a.ravel() for a in np.meshgrid(g, np.arange(0.0, WALL_Y, 0.05))]
    hole = (np.abs(x - HOLE[0]) < 0.06) & (np.abs(y - HOLE[1]) < 0.06)      # unscanned floor patch
    floor = np.column_stack([x, y, floor_z(x) + rng.normal(0, 0.046, len(x))])[~hole]
    spike = np.array([[SPIKE[0], SPIKE[1], floor_z(SPIKE[0]) + 0.35]] * 2)  # 2 ghost points above the floor
    spike = np.vstack([spike, [LOW[0], LOW[1], floor_z(LOW[0]) + 0.18]])
    xs, zs = [a.ravel() for a in np.meshgrid(g, np.arange(0.0, 2.6, 0.05))]
    wall = np.column_stack([xs, np.full(len(xs), WALL_Y + 0.02), zs])
    ceil = np.column_stack([x, y, np.full(len(x), 2.6)])
    return np.vstack([floor, wall, ceil, spike])


def _ground():
    grid = voxelize(_room(), None, 0.1)
    return grid, extract_ground(grid, standable_mask(grid), min_cells=20)


def _covered(slabs, x, y):
    lo, hi = slabs[:, 0, :2], slabs[:, 7, :2]
    return bool(np.any((lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1])))


def test_noise_spike_removed_and_hole_filled():
    grid, g = _ground()
    k = np.floor((np.array([SPIKE[0], SPIKE[1], floor_z(SPIKE[0]) + 0.35]) - grid.origin) / grid.voxel)
    spike = np.all(grid.keys == k, axis=1)
    assert spike.any() and g.remove[spike].all()
    assert _covered(g.slabs, *SPIKE) and _covered(g.slabs, *HOLE)


def test_floor_is_flattened_onto_planes():
    grid, g = _ground()
    xy = grid.origin[:2] + (g.cell_keys[:, :2] + 0.5) * grid.voxel
    away = np.abs(xy[:, 0] - STEP_X) > 0.25           # cells straddling the step edge are ambiguous
    err = g.cell_h[away] - floor_z(xy[away, 0])
    assert g.stats["planar_fraction"] > 0.9
    assert np.abs(err - np.median(err)).max() < 0.005   # flat: no residual graininess
    assert abs(np.median(err)) < 0.01                   # and at the right height


def test_step_is_kept_as_two_regions():
    grid, g = _ground()
    xy = grid.origin[:2] + (g.cell_keys[:, :2] + 0.5) * grid.voxel
    low = g.cell_h[(xy[:, 0] > 3.0) & (xy[:, 0] < 3.7)]
    high = g.cell_h[(xy[:, 0] > 4.3) & (xy[:, 0] < 5.0)]
    assert abs((high.mean() - low.mean()) - (STEP_H + 0.9 * SLOPE)) < 0.02
    assert g.stats["planes"] >= 2


def test_wall_kept_and_slab_tops_on_plane():
    grid, g = _ground()
    zc = grid.origin[2] + (grid.keys[:, 2] + 0.5) * grid.voxel
    wall = grid.origin[1] + (grid.keys[:, 1] + 0.5) * grid.voxel > WALL_Y
    assert not g.remove[wall & (zc > 0.3)].any()
    assert g.remove[~wall & (zc < 0.15)].mean() > 0.95  # floor voxels replaced
    tops = g.slabs[:, 1::2].reshape(-1, 3)              # odd corners = top face
    inner = (tops[:, 0] < STEP_X - 0.05) | (tops[:, 0] > STEP_X + 0.05)
    err = tops[inner, 2] - floor_z(tops[inner, 0])
    assert np.abs(err - np.median(err)).max() < 0.01
    assert np.allclose(g.slabs[:, 1::2, 2] - g.slabs[:, 0::2, 2], 0.2, atol=1e-5)


def test_noise_voxel_that_is_the_ground_cell_itself_is_removed():
    grid, g = _ground()
    k = np.floor((np.array([LOW[0], LOW[1], floor_z(LOW[0]) + 0.18]) - grid.origin) / grid.voxel)
    low = np.all(grid.keys == k, axis=1)
    assert low.any() and g.remove[low].all()
    assert _covered(g.slabs, *LOW)
