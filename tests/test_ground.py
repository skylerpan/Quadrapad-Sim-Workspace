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


# -- stairs ---------------------------------------------------------------------------------------
RISE, DEPTH, WIDTH, N_TREADS = 0.17, 0.30, 1.2, 6
FLIGHT_X0, FLIGHT_Y = 2.0, (0.4, 1.6)        # lower landing x < 2, flight along +x, upper landing after


def stair_z(x, y):
    """True surface height of the stair room (landing 0, treads, upper landing)."""
    i = np.clip(np.floor((x - FLIGHT_X0) / DEPTH) + 1, 0, N_TREADS)
    on_flight = (x >= FLIGHT_X0) & (x < FLIGHT_X0 + N_TREADS * DEPTH)
    side = on_flight & ((y < FLIGHT_Y[0]) | (y > FLIGHT_Y[1]))
    return np.where(side, 0.0, i * RISE)


def _stair_room(seed=1):
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, 6.0, 0.04)
    x, y = [a.ravel() for a in np.meshgrid(g, np.arange(0.0, 2.0, 0.04))]
    floor = np.column_stack([x, y, stair_z(x, y) + rng.normal(0, 0.03, len(x))])
    ys = np.arange(FLIGHT_Y[0], FLIGHT_Y[1], 0.04)
    risers = []
    for i in range(N_TREADS):                     # vertical riser faces at the front of every tread
        zs = np.arange(i * RISE, (i + 1) * RISE, 0.04)
        yy, zz = [a.ravel() for a in np.meshgrid(ys, zs)]
        risers.append(np.column_stack([np.full(len(yy), FLIGHT_X0 + i * DEPTH), yy, zz]))
    xs, zz = [a.ravel() for a in np.meshgrid(g, np.arange(0.0, 3.0, 0.04))]
    wall = np.column_stack([xs, np.full(len(xs), 2.02), zz])
    ceil = np.column_stack([x, y, np.full(len(x), 3.0)])
    return np.vstack([floor, *risers, wall, ceil])


STAIR_KW = dict(min_cells=12, fill_guard=0.3, surface="median", peel=3, small_flat=60)   # the stair recipe


def _tread_levels(grid, g):
    """Median flattened height of the cells inside every tread footprint (interior 0.1 m from the risers)."""
    xy = grid.origin[:2] + (g.cell_keys[:, :2] + 0.5) * grid.voxel
    out = []
    for i in range(1, N_TREADS + 1):
        x0 = FLIGHT_X0 + (i - 1) * DEPTH
        m = (xy[:, 0] > x0 + 0.1) & (xy[:, 0] < x0 + DEPTH - 0.1) & (xy[:, 1] > 0.6) & (xy[:, 1] < 1.4)
        out.append(g.cell_h[m])
    return out


def test_stair_treads_are_separate_levels():
    grid = voxelize(_stair_room(), None, 0.1)
    g = extract_ground(grid, standable_mask(grid), **STAIR_KW)
    levels = _tread_levels(grid, g)
    for i, h in enumerate(levels, start=1):
        assert len(h) >= 6, f"tread {i} has only {len(h)} cells"
        assert abs(np.median(h) - i * RISE) < 0.03, f"tread {i} at {np.median(h):.3f}, expected {i * RISE:.2f}"
    rises = np.diff([np.median(h) for h in levels])
    assert np.all(np.abs(rises - RISE) < 0.04)


def test_stair_tread_not_swallowed_by_fill():
    grid = voxelize(_stair_room(), None, 0.1)
    g = extract_ground(grid, standable_mask(grid), **STAIR_KW)
    xy = grid.origin[:2] + (g.cell_keys[:, :2] + 0.5) * grid.voxel
    on = (xy[:, 0] >= FLIGHT_X0 + 0.05) & (xy[:, 0] < FLIGHT_X0 + N_TREADS * DEPTH) & \
         (xy[:, 1] > 0.6) & (xy[:, 1] < 1.4)
    truth = stair_z(xy[on, 0], xy[on, 1])
    assert (g.cell_h[on] - truth).min() > -0.08      # nothing flattened below its own tread (riser cells excepted)


def test_stairs_merge_into_the_landing_with_defaults():
    """Documents why the stair recipe exists: with the defaults tread 1 chains to the landing."""
    grid = voxelize(_stair_room(), None, 0.1)
    g = extract_ground(grid, standable_mask(grid))
    assert abs(np.median(_tread_levels(grid, g)[0])) < 0.03


def test_fill_guard_blocks_fill_under_solid_voxel():
    pts = _room()
    solid = np.array([[HOLE[0] + dx, HOLE[1] + dy, floor_z(HOLE[0]) + 0.17]
                      for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)] * 2)   # 18-point voxel
    grid = voxelize(np.vstack([pts, solid]), None, 0.1)
    stand = standable_mask(grid)
    g0 = extract_ground(grid, stand, min_cells=20)
    g1 = extract_ground(grid, stand, min_cells=20, fill_guard=0.3)
    assert _covered(g0.slabs, *HOLE) and not _covered(g1.slabs, *HOLE)
    assert g1.stats["cells_guarded"] >= 1 and g0.stats["cells_guarded"] == 0


def test_fill_guard_off_matches_previous_behaviour():
    grid, g = _ground()
    g_explicit = extract_ground(grid, standable_mask(grid), min_cells=20, fill_guard=0.0)
    assert g.stats == g_explicit.stats
    assert np.array_equal(g.cell_h, g_explicit.cell_h)


def test_small_flat_makes_treads_horizontal():
    grid = voxelize(_stair_room(), None, 0.1)
    g = extract_ground(grid, standable_mask(grid), **STAIR_KW)
    for i, h in enumerate(_tread_levels(grid, g), start=1):
        # a tread may be split into a few small regions, each snapped flat: no ramp, few distinct heights
        assert len(np.unique(np.round(h, 3))) <= 4, f"tread {i} is not piecewise flat"
        assert h.max() - h.min() < 0.07, f"tread {i} spread {h.max() - h.min():.3f}"
