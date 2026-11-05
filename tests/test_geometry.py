"""pcd2sim.geometry on a synthetic room: a floor with a hole and a step, a ceiling, a wall, a trajectory."""
import numpy as np

from pcd2sim.geometry import FloorCells, floor_cells, holes, reachable, resample, trajectory_floor
from pcd2sim.voxel import VoxelGrid, encode

V = 0.1


def _grid(keys, dims=(80, 60, 40)):
    keys = np.asarray(keys, dtype=np.int32)
    dims = np.array(dims)
    keys = np.unique(keys, axis=0)
    keys = keys[np.argsort(encode(keys, dims), kind="stable")]
    n = len(keys)
    zmean = (keys[:, 2] + 1) * V - 1.0                              # top face, origin z = -1
    return VoxelGrid(keys, np.ones(n, np.int32), np.zeros(n, np.float32), zmean.astype(np.float32),
                     np.array([0.0, 0.0, -1.0]), V, dims)


def _room():
    floor = np.mgrid[0:60, 0:40, 10:11].reshape(3, -1).T              # 6 x 4 m at z top = 0.1
    hole = (floor[:, 0] >= 20) & (floor[:, 0] < 23) & (floor[:, 1] >= 10) & (floor[:, 1] < 12)   # 3x2 cells
    floor = floor[~hole]
    step = np.mgrid[40:60, 0:40, 11:12].reshape(3, -1).T              # +0.1 m step on the east part
    ceil = np.mgrid[0:60, 0:40, 36:37].reshape(3, -1).T               # 2.6 m up
    wall = np.mgrid[0:60, 40:41, 10:37].reshape(3, -1).T
    return _grid(np.concatenate([floor, step, ceil, wall]))


def test_floor_reachable_and_holes():
    g = _room()
    cells = floor_cells(g)
    assert cells.n == 60 * 40 - 6                                     # floor + step tops, minus the hole
    start = cells.nearest(1.0, 1.0, 0.1)
    reach = reachable(cells, start, dz_max=0.2)
    assert reach.all()                                                # the 0.1 m step is walkable
    h = holes(g, cells, reach, max_radius=1.5)
    assert h["components"] == 1 and h["cells"] == 6 and h["by_size"]["5-20"]["holes"] == 1
    assert h["largest"][0]["cells"] == 6 and abs(h["largest"][0]["x"] - 2.15) < 0.1
    # the floor's outer edge (nothing beyond the 6 x 4 m) is an open edge, not a hole
    assert h["open_edge_components"] >= 1 and h["open_edge_cells"] > 100


def test_reachable_stops_at_a_tall_step():
    floor = np.mgrid[0:40, 0:20, 10:11].reshape(3, -1).T
    ledge = np.mgrid[20:40, 0:20, 14:15].reshape(3, -1).T             # +0.4 m: not reachable with dz 0.2
    ceil = np.mgrid[0:40, 0:20, 36:37].reshape(3, -1).T
    g = _grid(np.concatenate([floor, ledge, ceil]))
    cells = floor_cells(g)
    reach = reachable(cells, cells.nearest(0.5, 0.5, 0.1), dz_max=0.2)
    assert reach.sum() == 20 * 20 and reach[cells.z > 0.3].sum() == 0


def test_trajectory_floor_reports_missing_floor_and_residual():
    g = _room()
    cells = floor_cells(g)
    # straight line along y = 1.05 from x = 0.5 to 5.5: crosses the hole (x 2.0-2.3) and the step (x >= 4)
    pts = np.array([[0.5, 1.05, 0.6], [5.5, 1.05, 0.6]])
    s = resample(pts, 0.1)
    out = trajectory_floor(g, cells, s, s[:, 2] - 0.5, search=0.0)
    assert out["samples"] == 51 and 2 <= out["no_floor"] <= 4          # the 0.3 m hole at 0.1 m spacing
    assert out["obstacle"] == 0
    # reference = the same floor shifted up 2 cm: residual -0.02 everywhere
    ref = cells.xy()
    out = trajectory_floor(g, cells, s, s[:, 2] - 0.5, reference=(ref[:, 0], ref[:, 1], cells.z + 0.02), search=0.0)
    assert abs(out["residual_median"] + 0.02) < 1e-6 and out["residual_max_abs"] < 0.021


def test_floor_cells_handles_two_storeys_in_one_column():
    g = _grid(np.concatenate([np.mgrid[0:10, 0:10, 5:6].reshape(3, -1).T,       # 1F
                              np.mgrid[0:10, 0:10, 30:31].reshape(3, -1).T,     # 2F slab (roof of 1F)
                              np.mgrid[0:10, 0:10, 55:56].reshape(3, -1).T]),   # roof of 2F, 2.5 m up
              dims=(80, 60, 60))
    cells = floor_cells(g)
    assert cells.n == 200
    lo, hi = cells.column_range(np.array([3 * 60 + 3]))
    assert hi[0] - lo[0] == 2                                           # both storeys in the column
    assert isinstance(cells, FloorCells)
