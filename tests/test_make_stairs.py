"""scripts/make_stairs.py on a tiny synthetic scene: original geometry in the band goes, clean steps come in,
and the ideal base climbs them with exact 0.16 m rises."""
import json
import math
import os
import sys

import numpy as np
import pytest
import yaml

pytest.importorskip("open3d")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import make_stairs  # noqa: E402

VOXEL = 0.05


def _scene(path):
    """Floor top at z = 0 over [-3, 3]^2, a noisy block inside the stair footprint, a ceiling box above it."""
    from pcd2sim.pcd_io import write_pcd
    origin = np.array([-3.0, -3.0, -1.0])
    n = int(6.0 / VOXEL)
    gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    kz = int(round((0.0 - origin[2]) / VOXEL)) - 1                   # top floor voxel: top face at z = 0
    keys = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, kz)]).astype(np.int32)
    junk = np.array([[0.0, 0.3, 0.75, 0.1, 0.1, 0.05],                   # noisy lump on the stairs (band)
                     [0.0, 0.0, 3.0, 2.0, 2.0, 0.05]], dtype=np.float32)   # ceiling (above the band)
    boxes = np.concatenate([np.array([[0.0, 0.0, -0.1, 3.0, 3.0, 0.1]], np.float32), junk])
    os.makedirs(path)
    np.save(os.path.join(path, "boxes.npy"), boxes)
    np.savez(os.path.join(path, "voxels.npz"), keys=keys, intensity=np.full(len(keys), 50, np.float32),
             zmean=np.zeros(len(keys), np.float32), origin=origin, voxel=VOXEL,
             dims=np.array([n, n, int(5.0 / VOXEL)]))
    slab = np.zeros((1, 8, 3), np.float32)
    for sx, sy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ci = 4 * sx + 2 * sy
        slab[0, ci] = [3.0 if sx else -3.0, 3.0 if sy else -3.0, -0.2]
        slab[0, ci + 1] = [3.0 if sx else -3.0, 3.0 if sy else -3.0, 0.0]
    np.savez(os.path.join(path, "ground.npz"), slabs=slab)
    pts = origin + (keys + 0.5) * VOXEL
    pts = np.concatenate([pts, [[0.0, 0.3, 0.3]]])                      # one map point in the band
    write_pcd(os.path.join(path, "map_ds.pcd"), pts.astype(np.float32), np.full(len(pts), 50, np.float32))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"format_version": 2, "voxel": VOXEL, "gravity": [0, 0, -9.81], "spawn": [0, -2, 0],
                   "boxes": len(boxes)}, f)


SPEC = {"riser": 0.16, "thickness": 0.30, "clear": [0.35, 0.60],
        "stairs": [{"name": "t",
                    "flights": [{"name": "F", "start": [0.0, -1.0], "end": [0.0, 1.0], "width": 1.0, "z0": 0.0, "z1": 0.8}],
                    "landings": [{"name": "L", "rect": [-0.5, 0.5, 1.0, 2.5], "z": 0.8}]}]}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("stairs")
    src, out, spec_path = os.path.join(root, "src"), os.path.join(root, "out"), os.path.join(root, "spec.yaml")
    _scene(src)
    with open(spec_path, "w") as f:
        yaml.safe_dump(SPEC, f)
    spec = make_stairs.load_spec(spec_path)
    meta = make_stairs.apply(src, out, spec, log=lambda *_: None)
    return src, out, spec, meta


def test_steps_and_clearing(built):
    src, out, spec, meta = built
    fl = spec["elements"][0]
    assert fl.n == 5 and abs(fl.rise - 0.16) < 1e-9 and abs(fl.tread - 0.4) < 1e-9
    boxes = np.load(os.path.join(out, "boxes.npy"))
    # the noisy lump is gone, the ceiling and the floor stay
    assert not np.any(np.all(np.isclose(boxes, [0.0, 0.3, 0.75, 0.1, 0.1, 0.05]), axis=1))
    assert np.any(np.all(np.isclose(boxes, [0.0, 0.0, 3.0, 2.0, 2.0, 0.05]), axis=1))
    assert np.any(np.all(np.isclose(boxes, [0.0, 0.0, -0.1, 3.0, 3.0, 0.1]), axis=1))
    # step tops are on the voxel grid within half a voxel of the exact height; zmean carries the exact height
    d = np.load(os.path.join(out, "voxels.npz"))
    c = d["origin"] + (d["keys"] + 0.5) * VOXEL
    for k in range(1, 6):
        s0, s1 = -1.0 + (k - 1) * 0.4, -1.0 + k * 0.4
        on_step = (np.abs(c[:, 0]) <= 0.5) & (c[:, 1] >= s0) & (c[:, 1] < s1) & (c[:, 2] > 0.0)
        assert on_step.any()
        top = c[on_step, 2].max() + VOXEL / 2
        assert abs(top - 0.16 * k) <= VOXEL / 2 + 1e-9
        tops = d["zmean"][on_step][np.isclose(c[on_step, 2] + VOXEL / 2, top)]
        assert np.allclose(tops, 0.16 * k)
    slabs = np.load(os.path.join(out, "ground.npz"))["slabs"]
    assert len(slabs) == 1 + 5 + 1                                       # floor + treads + landing
    assert np.isclose(sorted(slabs[:, 1, 2])[-1], 0.8)
    assert meta["boxes"] == len(boxes) and meta["synthetic_stairs"]["elements"][0]["steps"] == 5
    from pcd2sim.pcd_io import read_pcd
    pts, _ = read_pcd(os.path.join(out, "map_ds.pcd"))
    assert not np.any(np.all(np.isclose(pts, [0.0, 0.3, 0.3]), axis=1))


def test_base_climbs_with_exact_rises(built):
    from quadsim.config import load_config
    from quadsim.sim import Simulator
    _, out, _, _ = built
    cfg = load_config(None, [])
    cfg["scene"] = out
    cfg["spawn"] = [0.0, -2.0, math.pi / 2, 0.0]
    sim = Simulator(cfg)
    assert sim.spawn_ok and not sim.spawn_moved
    events = []
    for _ in range(int(8.0 * cfg["physics"]["control_hz"])):
        sim.set_cmd(0.5, 0.0, 0.0)
        events += sim.control_tick().events
    assert events == []
    assert sim.base.y > 1.4 and abs(sim.base.ground.z0 - 0.8) < 1e-6, (sim.base.y, sim.base.ground.z0)


def test_refuses_in_place(built):
    src, _, spec, _ = built
    with pytest.raises(SystemExit):
        make_stairs.apply(src, src, spec, log=lambda *_: None)
