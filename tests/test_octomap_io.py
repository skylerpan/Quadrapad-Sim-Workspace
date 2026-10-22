import json
import os
import struct

import numpy as np
import pytest

from pcd2sim.octomap_io import (BT_HEADER, OT_HEADER, TREE_DEPTH, expand_leaves, read_map_points,
                                read_octomap, voxel_grid_from_cells)

RES = 0.25          # power of two: every centre is exact in float64
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# -- tiny encoders (mirror octomap's writeBinaryNode / writeNodesRecurs) ------------------------------
def _child_center(c, depth, k):
    h = RES * 2.0 ** (TREE_DEPTH - depth - 2)
    return (c[0] + (h if k & 1 else -h), c[1] + (h if k & 2 else -h), c[2] + (h if k & 4 else -h))


def _inside(leaf_c, cube_c, cube_depth):
    half = RES * 2.0 ** (TREE_DEPTH - cube_depth - 1)
    return all(abs(a - b) <= half + 1e-9 for a, b in zip(leaf_c, cube_c))


def _encode_bt(leaves, c=(0.0, 0.0, 0.0), depth=0):
    """leaves: list of (center, depth, occupied: bool). Returns the depth-first byte stream."""
    assert depth < TREE_DEPTH, "duplicate or misaligned leaf"
    bits = 0
    sub = []
    for k in range(8):
        ck = _child_center(c, depth, k)
        inside = [lf for lf in leaves if _inside(lf[0], ck, depth + 1)]
        if not inside:
            continue
        if len(inside) == 1 and inside[0][1] == depth + 1:
            bits |= (0b10 if inside[0][2] else 0b01) << (2 * k)        # b1 occupied, b0 free
        else:
            bits |= 0b11 << (2 * k)
            sub.append(_encode_bt(inside, ck, depth + 1))
    return bytes([bits & 0xFF, bits >> 8]) + b"".join(sub)


def _encode_ot(leaves, c=(0.0, 0.0, 0.0), depth=0, extra=0):
    """Full tree: float32 log-odds (+extra bytes) and a child byte per node."""
    children = 0
    sub = []
    for k in range(8):
        ck = _child_center(c, depth, k)
        inside = [lf for lf in leaves if _inside(lf[0], ck, depth + 1)]
        if not inside:
            continue
        children |= 1 << k
        if len(inside) == 1 and inside[0][1] == depth + 1:
            sub.append(struct.pack("<f", inside[0][3]) + bytes(extra) + b"\x00")
        else:
            sub.append(_encode_ot(inside, ck, depth + 1, extra))
    return struct.pack("<f", 3.5) + bytes(extra) + bytes([children]) + b"".join(sub)


def _write(path, header, tree_id, payload, size=None):
    txt = f"{header}\n# comment\nid {tree_id}\nsize {size if size is not None else 0}\nres {RES}\ndata\n"
    path.write_bytes(txt.encode() + payload)


def _cells(xyz_idx):
    """Finest cell centres from integer indices (cell i covers [i*RES, (i+1)*RES))."""
    return [((i + 0.5) * RES, (j + 0.5) * RES, (k + 0.5) * RES) for i, j, k in xyz_idx]


def test_bt_roundtrip_with_pruned_and_free_leaves(tmp_path):
    occ = _cells([(0, 0, 0), (3, -2, 1), (-7, 5, -1), (10, 10, 10)])
    pruned_c = (2 * RES + RES, 2 * RES + RES, -2 * RES + RES)          # depth-15 cube: 2x2x2 cells, centre on a 2*RES grid
    leaves = [(c, TREE_DEPTH, True) for c in occ] + [(pruned_c, TREE_DEPTH - 1, True)]
    leaves.append((_cells([(5, 5, 5)])[0], TREE_DEPTH, False))          # free leaf: must be ignored
    p = tmp_path / "t.bt"
    _write(p, BT_HEADER, "OcTree", _encode_bt(leaves))
    L = read_octomap(str(p))
    assert L.res == RES and L.tree_id == "OcTree" and L.logodds is None
    assert L.n_free == 1 and len(L.centers) == 5
    got = {tuple(np.round(c, 6)) for c in L.centers}
    assert got == {tuple(np.round(c, 6)) for c in occ} | {tuple(np.round(pruned_c, 6))}
    assert sorted(L.depth.tolist()) == [15, 16, 16, 16, 16]
    x = expand_leaves(L)
    assert len(x) == 4 + 8
    tile = x[~np.isin(np.round(x, 6).view([("", x.dtype)] * 3).ravel(),
                      np.round(np.array(occ), 6).view([("", x.dtype)] * 3).ravel())]
    assert len(tile) == 8
    np.testing.assert_allclose(tile.mean(axis=0), pruned_c)
    np.testing.assert_allclose(np.abs(tile - pruned_c).max(axis=0), RES / 2)


def test_bt_trailing_bytes_and_size_mismatch(tmp_path):
    leaves = [(c, TREE_DEPTH, True) for c in _cells([(1, 1, 1)])]
    payload = _encode_bt(leaves)
    p = tmp_path / "bad.bt"
    _write(p, BT_HEADER, "OcTree", payload + b"\x00\x00")
    with pytest.raises(ValueError, match="trailing"):
        read_octomap(str(p))
    _write(p, BT_HEADER, "OcTree", payload[:-1])
    with pytest.raises(ValueError, match="truncated"):
        read_octomap(str(p))
    _write(p, BT_HEADER, "OcTree", payload, size=999)
    with pytest.warns(UserWarning, match="header says 999"):
        read_octomap(str(p))
    _write(p, "# not an octomap", "OcTree", payload)
    with pytest.raises(ValueError, match="not an OctoMap"):
        read_octomap(str(p))


@pytest.mark.parametrize("tree_id,extra", [("OcTree", 0), ("ColorOcTree", 3)])
def test_ot_logodds_threshold(tmp_path, tree_id, extra):
    occ = _cells([(0, 0, 0), (2, 3, 4)])
    weak = _cells([(6, 6, 6)])[0]
    free = _cells([(-3, 2, 1)])[0]
    pruned_c = (-4 * RES + RES, 4 * RES + RES, 2 * RES + RES)
    leaves = [(c, TREE_DEPTH, True, 2.0) for c in occ] + [(weak, TREE_DEPTH, True, 0.3), (free, TREE_DEPTH, False, -2.0),
                                                           (pruned_c, TREE_DEPTH - 1, True, 3.0)]
    p = tmp_path / "t.ot"
    _write(p, OT_HEADER, tree_id, _encode_ot(leaves, extra=extra))
    L = read_octomap(str(p))                                 # default thres 0.5 -> log-odds >= 0
    assert L.tree_id == tree_id and len(L.centers) == 4 and L.n_free == 1
    np.testing.assert_allclose(sorted(L.logodds.tolist()), [0.3, 2.0, 2.0, 3.0], atol=1e-6)
    L2 = read_octomap(str(p), occ_thres=0.7)                 # logit(0.7) = 0.847 drops the weak leaf
    assert len(L2.centers) == 3 and L2.n_free == 2
    assert len(expand_leaves(L)) == 3 + 8


def test_read_map_points_dispatch(tmp_path):
    leaves = [(c, TREE_DEPTH, True) for c in _cells([(0, 0, 0), (1, 0, 0)])]
    p = tmp_path / "m.bt"
    _write(p, BT_HEADER, "OcTree", _encode_bt(leaves))
    xyz, inten, res = read_map_points(str(p))
    assert inten is None and res == RES and xyz.shape == (2, 3)
    from pcd2sim.pcd_io import write_pcd
    q = tmp_path / "m.pcd"
    write_pcd(str(q), np.zeros((3, 3), np.float32), np.ones(3, np.float32))
    xyz, inten, res = read_map_points(str(q))
    assert res is None and xyz.shape == (3, 3) and inten is not None


def test_voxel_grid_from_cells_is_one_to_one():
    idx = np.array([(0, 0, 0), (3, -2, 1), (-7, 5, -1), (3, -2, 1)])           # one duplicate
    centers = (idx + 0.5) * RES
    g = voxel_grid_from_cells(centers, RES)
    assert len(g.keys) == 3 and g.voxel == RES
    np.testing.assert_allclose(np.sort(g.centers(), axis=0), np.sort(np.unique(centers, axis=0), axis=0))
    np.testing.assert_allclose(g.zmean, g.centers()[:, 2] + RES / 2)              # surface = cell top
    assert (g.counts == 1).all() and (g.intensity == 0).all()


def test_build_from_bt_room(tmp_path):
    """A 4 m x 4 m room, 2.5 m walls and a ceiling, a 2x2 hole in the floor, plus noise (a mid-air speck
    and a floating 3-voxel chain): the octree path maps every cell 1:1 to a voxel, applies the noise
    filter, then flattens the floor (hole filled); --no-flatten keeps the raw 1:1 voxels."""
    from pcd2sim.build_scene import build
    hole = {(-2, 1, -1), (-1, 1, -1), (-2, 2, -1), (-1, 2, -1)}
    room = [(i, j, -1) for i in range(-8, 8) for j in range(-8, 8) if (i, j, -1) not in hole]   # floor top z = 0
    room += [(i, j, 10) for i in range(-8, 8) for j in range(-8, 8)]                 # ceiling at 2.5 m
    for k in range(0, 10):
        room += [(i, j, k) for i in range(-8, 8) for j in (-8, 7)]
        room += [(i, j, k) for i in (-8, 7) for j in range(-8, 8)]
    room = sorted(set(room))                                                          # corners were listed twice
    noise = [(0, 0, 5), (3, 3, 4), (4, 3, 4), (5, 3, 4)]                              # speck + chain (ends have 1 nb)
    cells = room + noise
    leaves = [(c, TREE_DEPTH, True) for c in _cells(cells)]
    p = tmp_path / "room.bt"
    _write(p, BT_HEADER, "OcTree", _encode_bt(leaves), size=len(leaves))

    with pytest.warns(UserWarning):                        # size counts leaves only in this toy header
        mf = build(str(p), str(tmp_path / "flat"))        # defaults: noise filter + flatten
    g = mf["ground"]
    assert mf["build_args"]["flatten"] is True and os.path.exists(tmp_path / "flat" / "ground.npz")
    assert g["cells"] >= 150 and g["planar_fraction"] == 1.0 and g["cells_filled"] >= 4
    assert mf["noise_filter"]["isolated_removed"] == 3
    slabs = np.load(tmp_path / "flat" / "ground.npz")["slabs"].astype(np.float64)     # [K,8,3] box corners
    hx, hy = (-1.0 + 0.0) * RES, (1.5 + 0.5) * RES                                     # centre of the 2x2 hole
    lo, hi, top = slabs.min(axis=1), slabs.max(axis=1), slabs[:, :, 2].max(axis=1)
    over = (lo[:, 0] <= hx) & (hi[:, 0] >= hx) & (lo[:, 1] <= hy) & (hi[:, 1] >= hy)
    assert over.any() and abs(top[over].max() - 0.0) < 0.01                           # hole closed at floor level

    with pytest.warns(UserWarning):
        meta = build(str(p), str(tmp_path / "scene"), voxel=RES / 2, crop_pct=0.05, min_neighbors=0, flatten=False)
    assert meta["source_format"] == "octomap" and meta["octomap_res"] == RES and meta["voxel"] == RES
    assert meta["build_args"]["voxel"] == RES and meta["build_args"]["source"].endswith("room.bt")
    assert meta["build_args"]["flatten"] is False and meta["ground"] is None
    assert meta["points_raw"] == len(cells) and meta["voxels"] == len(cells)       # filter off: pure 1:1
    assert meta["noise_filter"] == {"sparse_removed": 0, "isolated_removed": 0, "small_clusters_removed": 0}
    assert os.path.exists(tmp_path / "scene" / "boxes.npy") and not os.path.exists(tmp_path / "scene" / "ground.npz")
    vox = np.load(tmp_path / "scene" / "voxels.npz")
    assert vox["voxel"] == RES and len(vox["keys"]) == len(cells)
    got = {tuple(c) for c in np.round(vox["origin"] + (vox["keys"] + 0.5) * RES, 6)}
    assert got == {tuple(np.round(c, 6)) for c in _cells(cells)}
    boxes = np.load(tmp_path / "scene" / "boxes.npy")
    assert boxes[:, 2].max() > 2.0 and abs(float((boxes[:, 3:] * 2).prod(axis=1).sum()) - len(cells) * RES ** 3) < 1e-6

    with pytest.warns(UserWarning):                        # default filter: speck + chain ends (<= 1 neighbour)
        m1 = build(str(p), str(tmp_path / "scene1"), flatten=False)
    assert m1["voxels"] == len(room) + 1 and m1["noise_filter"]["isolated_removed"] == 3
    assert m1["noise_filter"]["small_clusters_removed"] == 0
    with pytest.warns(UserWarning):                        # + cluster filter: the chain's lone middle goes too
        m4 = build(str(p), str(tmp_path / "scene4"), min_cluster=4, flatten=False)
    assert m4["voxels"] == len(room) and m4["noise_filter"]["isolated_removed"] == 3
    assert m4["noise_filter"]["small_clusters_removed"] == 1
    v4 = np.load(tmp_path / "scene4" / "voxels.npz")
    got4 = {tuple(c) for c in np.round(v4["origin"] + (v4["keys"] + 0.5) * RES, 6)}
    assert got4 == {tuple(np.round(c, 6)) for c in _cells(room)}
    with pytest.warns(UserWarning):
        m2 = build(str(p), str(tmp_path / "scene2"), z_min=0.0, min_neighbors=0, flatten=False)  # floor layer only
    assert m2["voxels"] == len(cells) - (256 - len(hole))
    with open(tmp_path / "scene" / "meta.json") as f:
        assert json.load(f)["source_format"] == "octomap"


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "Maps", "dp_1012.bt")), reason="Maps/dp_1012.bt not present")
def test_real_dp_1012_bt():
    L = read_octomap(os.path.join(REPO, "Maps", "dp_1012.bt"))
    assert L.res == 0.2 and len(L.centers) == 395012 and L.n_free == 0
    np.testing.assert_allclose(L.centers.min(axis=0), [-59.3, -25.5, -1.9], atol=0.3)
    np.testing.assert_allclose(L.centers.max(axis=0), [42.1, 61.9, 11.7], atol=0.3)
    assert len(expand_leaves(L)) == 436809
