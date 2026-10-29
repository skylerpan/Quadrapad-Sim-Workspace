"""build_scene stage files: created under <out>/.tmp, removed on success, kept with keep_tmp, reused with resume."""
import os

import numpy as np

from pcd2sim.build_scene import build
from pcd2sim.pcd_io import write_pcd


def _room_pcd(path):
    rng = np.random.default_rng(0)
    n = 60000
    floor = np.c_[rng.uniform(0, 6, n), rng.uniform(0, 5, n), rng.normal(0, 0.01, n)]
    wall = np.c_[rng.uniform(0, 6, n // 3), np.full(n // 3, 5.0) + rng.normal(0, 0.01, n // 3),
                 rng.uniform(0, 2.5, n // 3)]
    ceil = np.c_[rng.uniform(0, 6, n // 2), rng.uniform(0, 5, n // 2), 2.6 + rng.normal(0, 0.01, n // 2)]
    xyz = np.vstack([floor, wall, ceil])
    write_pcd(path, xyz.astype(np.float32), np.full(len(xyz), 20, np.float32))


def test_tmp_dir_removed_on_success(tmp_path):
    pcd = str(tmp_path / "room.pcd")
    _room_pcd(pcd)
    out = str(tmp_path / "scene")
    meta = build(pcd, out, voxel=0.1, close_rounds=1, workers=1)
    assert not os.path.exists(os.path.join(out, ".tmp"))
    assert meta["noise_filter"]["closed_added"] >= 0
    assert os.path.exists(os.path.join(out, "boxes.npy"))


def test_keep_tmp_and_resume(tmp_path, capsys):
    pcd = str(tmp_path / "room.pcd")
    _room_pcd(pcd)
    out = str(tmp_path / "scene")
    meta = build(pcd, out, voxel=0.1, close_rounds=1, workers=1, keep_tmp=True)
    tmp = os.path.join(out, ".tmp")
    assert sorted(os.listdir(tmp)) == ["stage_closed.npz", "stage_ground.npz"]
    boxes1 = np.load(os.path.join(out, "boxes.npy"))
    capsys.readouterr()
    meta2 = build(pcd, out, voxel=0.1, close_rounds=1, workers=1, resume=True)
    log = capsys.readouterr().out
    assert "resume: loaded closed voxels" in log and "resume: loaded ground" in log
    assert "closed:" not in log                                   # closing was skipped
    assert meta2["noise_filter"] == meta["noise_filter"] and meta2["boxes"] == meta["boxes"]
    assert np.array_equal(np.load(os.path.join(out, "boxes.npy")), boxes1)
    assert not os.path.exists(tmp)                                # resume without keep_tmp cleans up
