"""pcd2sim.stairs: a synthetic staircase on a floor is found and described as one flight plus a landing."""
import math

import numpy as np
import yaml

from pcd2sim.geometry import FloorCells
from pcd2sim.stairs import draft, to_yaml
from pcd2sim.voxel import VoxelGrid

V = 0.1


def _cells(keys, z, voxel=V, dims=(100, 100, 60)):
    grid = VoxelGrid(np.zeros((0, 3), np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32),
                     np.zeros(0, np.float32), np.array([-5.0, -5.0, -1.0]), voxel, np.array(dims))
    return FloorCells(grid, np.asarray(keys), np.asarray(z, dtype=np.float64))


def _staircase(voxel=V, rise=0.16, n=5, tread=0.4, width=1.0):
    """Floor at z=0 (8 x 8 m), a flight along +y from y=-1 (x -0.5..0.5), a 1.5 m landing at the top."""
    o = np.array([-5.0, -5.0, -1.0])
    dims = (int(round(10 / voxel)), int(round(10 / voxel)), 60)
    def cell_keys(x0, x1, y0, y1, z):
        ix = np.arange(int(round((x0 - o[0]) / voxel)), int(round((x1 - o[0]) / voxel)))
        iy = np.arange(int(round((y0 - o[1]) / voxel)), int(round((y1 - o[1]) / voxel)))
        gx, gy = np.meshgrid(ix, iy, indexing="ij")
        kz = int(np.floor((z - o[2]) / voxel - 1e-6)) - 1
        return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, kz)]), np.full(gx.size, z)
    keys, zs = [], []
    k, z = cell_keys(-4, 4, -4, -1, 0.0)                       # floor south of the flight
    keys.append(k); zs.append(z)
    k, z = cell_keys(-4, -0.5, -1, 4, 0.0); keys.append(k); zs.append(z)     # floor west of it
    k, z = cell_keys(0.5, 4, -1, 4, 0.0); keys.append(k); zs.append(z)       # floor east of it
    for s in range(1, n + 1):
        k, z = cell_keys(-width / 2, width / 2, -1 + (s - 1) * tread, -1 + s * tread, s * rise)
        keys.append(k); zs.append(z)
    k, z = cell_keys(-width / 2, width / 2, -1 + n * tread, -1 + n * tread + 1.5, n * rise)   # landing
    keys.append(k); zs.append(z)
    return _cells(np.concatenate(keys), np.concatenate(zs), voxel, dims)


def test_draft_finds_the_flight_and_landing():
    cells = _staircase()
    d = draft(cells)
    assert len(d) == 1
    st = d[0]
    assert len(st["flights"]) == 1
    f = st["flights"][0]
    assert f["steps"] == 5 and abs(f["rise"] - 0.16) < 1e-6
    assert f["yaw_deg"] == 90.0 and abs(f["width"] - 1.0) <= 0.11
    assert abs(f["start"][0]) < 0.06 and abs(f["start"][1] + 1.0) <= 0.15       # foot at y = -1
    assert abs(f["end"][1] - 1.0) <= 0.15                                        # head at y = -1 + 5 * 0.4
    assert abs(f["z0"]) < 1e-6 and abs(f["z1"] - 0.8) < 1e-6
    assert f["notes"] == [], f["notes"]
    assert len(st["landings"]) == 1                                              # the floor at the foot is open
    l = st["landings"][0]
    # the landing level includes the top tread (same height), so its box starts at the top tread (y 0.6)
    assert abs(l["z"] - 0.8) < 1e-6 and 0.5 <= l["rect"][2] <= 1.05 and abs(l["rect"][3] - 2.5) <= 0.11
    text = to_yaml(d, "scene")
    spec = yaml.safe_load(text)
    assert spec["stairs"][0]["flights"][0]["z1"] == f["z1"] and "# CHECK:" not in text


def test_draft_on_a_fine_grid_matches_the_coarse_one():
    fine = _staircase(voxel=0.03)
    d = draft(fine)
    f = d[0]["flights"][0]
    assert f["steps"] == 5 and abs(f["z1"] - 0.8) < 1e-6 and abs(f["width"] - 1.0) <= 0.11


def test_draft_handles_a_ramp_and_a_switchback():
    o = np.array([-5.0, -5.0, -1.0])
    # a 30 deg ramp along +x, 1.2 m wide, from z 0 at x 0 to z 0.8 at x 1.386, floor before and landing after
    ix, iy = np.meshgrid(np.arange(0, 70), np.arange(44, 56), indexing="ij")
    x = o[0] + (ix.ravel() + 0.5) * V
    z = np.clip((x - 0.0) * math.tan(math.radians(30)), 0.0, 0.8)
    keys = np.column_stack([ix.ravel(), iy.ravel(), np.floor((z - o[2]) / V - 1e-6).astype(int) - 1])
    d = draft(_cells(keys, z))
    assert len(d) == 1 and len(d[0]["flights"]) == 1
    f = d[0]["flights"][0]
    assert f["steps"] == 5 and abs(f["z1"] - 0.8) < 0.03 and f["yaw_deg"] == 0.0 and abs(f["start"][0]) < 0.2
    # switchback: flight A along +y (x in [-1.5, -0.5]), half landing, flight B along -y (x in [0, 1])
    def flight(x0, x1, y_from, y_to, z0, n=5, rise=0.16):
        ks, zs = [], []
        tread = (y_to - y_from) / n
        for k in range(1, n + 1):
            ya, yb = sorted([y_from + (k - 1) * tread, y_from + k * tread])
            iy0, iy1 = int(round((ya - o[1]) / V)), int(round((yb - o[1]) / V))
            gx, gy = np.meshgrid(np.arange(int(round((x0 - o[0]) / V)), int(round((x1 - o[0]) / V))),
                                 np.arange(iy0, iy1), indexing="ij")
            zk = z0 + k * rise
            ks.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, int(np.floor((zk - o[2]) / V - 1e-6)) - 1)]))
            zs.append(np.full(gx.size, zk))
        return np.concatenate(ks), np.concatenate(zs)
    ka, za = flight(-1.5, -0.5, -1.0, 1.0, 0.0)
    kb, zb = flight(0.0, 1.0, 1.0, -1.0, 0.8)
    gx, gy = np.meshgrid(np.arange(int(round((-1.5 - o[0]) / V)), int(round((1.0 - o[0]) / V))),
                         np.arange(int(round((1.0 - o[1]) / V)), int(round((2.2 - o[1]) / V))), indexing="ij")
    kl = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, int(np.floor((0.8 - o[2]) / V - 1e-6)) - 1)])
    zl = np.full(gx.size, 0.8)
    gx, gy = np.meshgrid(np.arange(0, 80), np.arange(0, 40), indexing="ij")             # floor y < -1
    kf = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 9)])
    d = draft(_cells(np.concatenate([ka, kb, kl, kf]), np.concatenate([za, zb, zl, np.zeros(gx.size)])))
    assert len(d) == 1, [s["name"] for s in d]
    st = d[0]
    assert len(st["flights"]) == 2
    a, b = sorted(st["flights"], key=lambda f: f["z0"])
    assert a["yaw_deg"] == 90.0 and abs(a["z0"]) < 1e-6 and abs(a["z1"] - 0.8) < 1e-6 and a["steps"] == 5
    assert b["yaw_deg"] == -90.0 and abs(b["z0"] - 0.8) < 1e-6 and abs(b["z1"] - 1.6) < 1e-6 and b["steps"] == 5
    assert any(abs(l["z"] - 0.8) < 1e-6 and l["m2"] > 2.0 for l in st["landings"])      # the half landing


def test_draft_ignores_a_single_step_and_flat_floor():
    o = np.array([-5.0, -5.0, -1.0])
    ix, iy = np.meshgrid(np.arange(0, 60), np.arange(0, 60), indexing="ij")
    keys = np.column_stack([ix.ravel(), iy.ravel(), np.full(ix.size, 9)])
    z = np.where(ix.ravel() >= 30, 0.15, 0.0)                                    # one 15 cm step across the room
    assert draft(_cells(keys, z)) == []
