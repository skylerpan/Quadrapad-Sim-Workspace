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


def test_spike_pts_zero_keeps_one_point_voxel_above_grown_floor():
    """At 2-3 cm voxels nearly every voxel holds one point, so the spike/sheet count tests must be
    switchable off: spike_pts=0 / sheet_pts=0 keep a 1-point voxel standing just above the floor."""
    from pcd2sim.voxel import VoxelGrid, encode, standable_mask
    floor = np.mgrid[0:30, 0:30, 0:1].reshape(3, -1).T
    post = np.array([[10, 10, 1], [10, 10, 2], [10, 10, 3]])        # 1-point voxels on the floor
    keys = np.vstack([floor, post]).astype(np.int32)
    dims = keys.max(axis=0) + 10
    order = np.argsort(encode(keys, dims))
    keys = keys[order]
    counts = np.where(keys[:, 2] == 0, 10, 1).astype(np.int32)
    zmean = ((keys[:, 2] + 0.5) * 0.1).astype(np.float32)
    g = VoxelGrid(keys, counts, np.zeros(len(keys), np.float32), zmean, np.zeros(3), 0.1, dims)
    stand = standable_mask(g, clearance=0.8, roof_min=0.0)
    # z index 1 (centre 0.15 m) lies inside the floor band (h + band_above) and goes either way; test 2 and 3
    post_idx = np.flatnonzero((keys[:, 0] == 10) & (keys[:, 1] == 10) & (keys[:, 2] >= 2))
    gd_default = extract_ground(g, stand, min_cells=10)
    gd_fine = extract_ground(g, stand, min_cells=10, spike_pts=0, sheet_pts=0, float_max_voxels=0)
    assert gd_default.remove[post_idx].any()                   # defaults call the 1-point post noise
    assert not gd_fine.remove[post_idx].any()                  # fine-voxel settings keep it


def _fine_grid(keys, voxel=0.03):
    """VoxelGrid from integer keys at a fine resolution: one point per voxel, zmean at the voxel centre."""
    from pcd2sim.voxel import VoxelGrid, encode
    keys = np.asarray(keys, dtype=np.int32)
    dims = keys.max(axis=0) + 40
    keys = keys[np.argsort(encode(keys, dims))]
    return VoxelGrid(keys, np.ones(len(keys), np.int32), np.zeros(len(keys), np.float32),
                     ((keys[:, 2] + 0.5) * voxel).astype(np.float32), np.zeros(3), voxel, dims)


FINE = dict(min_cells=90, surface="median", layers=7, grow=10, fill=30, max_above=7,
            spike_pts=0, sheet_pts=0, float_max_voxels=40)


def test_column_gap_fills_1f_hole_under_a_2f_floor():
    """Grow/fill used to skip any column that already had a ground cell on another storey."""
    from pcd2sim.voxel import standable_mask
    floor = np.mgrid[0:60, 0:60, 0:1].reshape(3, -1).T
    hole = (floor[:, 0] >= 25) & (floor[:, 0] < 35) & (floor[:, 1] >= 25) & (floor[:, 1] < 35)
    upper = np.mgrid[0:60, 0:60, 100:101].reshape(3, -1).T          # 2F floor 3 m up
    roof = np.mgrid[0:60, 0:60, 200:201].reshape(3, -1).T           # roof over 2F
    g = _fine_grid(np.vstack([floor[~hole], upper, roof]))
    stand = standable_mask(g, roof_cell=0.15)
    gd = extract_ground(g, stand, **FINE)
    low = gd.cell_keys[gd.cell_h < 1.0]
    in_hole = (low[:, 0] >= 25) & (low[:, 0] < 35) & (low[:, 1] >= 25) & (low[:, 1] < 35)
    assert in_hole.sum() == 0                                      # old rule: 2F cells block the column
    gd2 = extract_ground(g, stand, column_gap=0.3, **FINE)
    low2 = gd2.cell_keys[gd2.cell_h < 1.0]
    in_hole2 = (low2[:, 0] >= 25) & (low2[:, 0] < 35) & (low2[:, 1] >= 25) & (low2[:, 1] < 35)
    assert in_hole2.sum() == 100                                   # filled
    up2 = gd2.cell_keys[gd2.cell_h > 1.0]
    inside = (up2[:, 0] < 60) & (up2[:, 1] < 60)                   # (fill also grows the 2F edge outward)
    assert inside.sum() == 3600                                    # 2F untouched, no duplicate cells


def test_roof_cell_rescues_floor_under_a_sparse_ceiling():
    from pcd2sim.voxel import standable_mask
    rng = np.random.default_rng(5)
    floor = np.mgrid[0:60, 0:60, 0:1].reshape(3, -1).T
    ceil_all = np.mgrid[0:60, 0:60, 90:91].reshape(3, -1).T
    ceil = ceil_all[rng.random(len(ceil_all)) < 0.3]              # 30 % of the 3 cm columns have a ceiling point
    g = _fine_grid(np.vstack([floor, ceil]))
    s_default = standable_mask(g)
    s_coarse = standable_mask(g, roof_cell=0.15)
    n_floor_default = int((g.keys[s_default][:, 2] == 0).sum())
    n_floor_coarse = int((g.keys[s_coarse][:, 2] == 0).sum())
    assert n_floor_default < 0.4 * 3600 and n_floor_coarse > 0.95 * 3600
    assert not s_coarse[g.keys[:, 2] == 90].any()                  # ceiling top stays non-standable


def test_fine_floor_noise_links_into_one_region():
    """At 3 cm the floor scatter puts neighbouring top voxels 2-3 layers apart; they must still link."""
    from pcd2sim.voxel import standable_mask
    rng = np.random.default_rng(6)
    floor = np.mgrid[0:60, 0:60, 0:1].reshape(3, -1).T
    floor[:, 2] = rng.integers(0, 3, len(floor))                   # top voxel in layer 0, 1 or 2
    roof = np.mgrid[0:60, 0:60, 90:91].reshape(3, -1).T
    g = _fine_grid(np.vstack([floor, roof]))
    gd = extract_ground(g, standable_mask(g), **FINE)
    inside = (gd.cell_keys[:, 0] < 60) & (gd.cell_keys[:, 1] < 60)      # fill grows the edge outward too
    assert gd.stats["regions"] == 1 and inside.sum() == 3600 and gd.stats["cells_filled"] < 3000
