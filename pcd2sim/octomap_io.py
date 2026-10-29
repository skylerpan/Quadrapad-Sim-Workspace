"""OctoMap (.bt / .ot) reader: occupied leaf centres from an OcTree, pure numpy.

Formats (octomap 1.9, OcTreeBaseImpl.hxx):
  .bt  header '# Octomap OcTree binary file', then 2 bytes per inner node, depth-first in child
       order 0..7. Child k uses bits 2k (b0) and 2k+1 (b1): b0=0,b1=1 occupied leaf; b0=1,b1=0 free
       leaf; b0=b1=1 inner node (its 2 bytes follow, recursively); 00 no child.
  .ot  header '# Octomap OcTree file', then per node: node data (OcTree: float32 log-odds;
       ColorOcTree: float32 + 3 bytes rgb), one child-bitmask byte, then each set child recursively.
       A node with no children is a leaf; occupied when log-odds >= logit(occ_thres).
Both: tree depth 16, root centred on (0, 0, 0); a child's centre is the parent's +- res * 2^(14-depth)
(x from bit 0, y from bit 1, z from bit 2). Pruned leaves above the finest depth cover
2^(16-depth) cells per axis; expand_leaves() tiles them into finest-resolution centres.
"""
from __future__ import annotations

import math
import os
import struct
import warnings
from dataclasses import dataclass

import numpy as np

from .pcd_io import read_pcd
from .voxel import VoxelGrid, encode

TREE_DEPTH = 16
BT_HEADER = "# Octomap OcTree binary file"
OT_HEADER = "# Octomap OcTree file"
_NODE_EXTRA = {"OcTree": 0, "ColorOcTree": 3}      # bytes after the float32 log-odds in .ot nodes
OCTOMAP_EXTS = (".bt", ".ot")

_bt_table: list | None = None


@dataclass
class OcTreeLeaves:
    res: float
    tree_id: str
    centers: np.ndarray             # [N,3] float64 occupied-leaf centres (any depth)
    depth: np.ndarray               # [N] int16, TREE_DEPTH = finest cell
    logodds: np.ndarray | None      # [N] float32 (.ot only)
    n_inner: int = 0
    n_free: int = 0

    def expand(self) -> np.ndarray:
        return expand_leaves(self)


def _parse_header(buf: bytes) -> tuple[str, dict, int]:
    """Return (first line, key->value, payload offset). Comments start with '#'; 'data' ends the header."""
    pos, meta, first = 0, {}, None
    while True:
        nl = buf.find(b"\n", pos)
        if nl < 0:
            raise ValueError("octomap header has no 'data' line")
        line = buf[pos:nl].decode("ascii", "replace").strip()
        pos = nl + 1
        if first is None:
            first = line
        if line == "data":
            return first, meta, pos
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition(" ")
        meta[key] = val.strip()


def _child_offsets(k: int) -> tuple[int, int, int]:
    return (1 if k & 1 else -1), (1 if k & 2 else -1), (1 if k & 4 else -1)


_SIGNS = [_child_offsets(k) for k in range(8)]


def _get_bt_table() -> list:
    """For every 16-bit child word: (occupied child indices, free child indices, inner child indices)."""
    global _bt_table
    if _bt_table is None:
        table = []
        for bits in range(65536):
            occ, free, inner = [], [], []
            for k in range(8):
                b0, b1 = (bits >> (2 * k)) & 1, (bits >> (2 * k + 1)) & 1
                if b1 and not b0:
                    occ.append(k)
                elif b0 and not b1:
                    free.append(k)
                elif b0 and b1:
                    inner.append(k)
            table.append((tuple(occ), tuple(free), tuple(inner)))
        _bt_table = table
    return _bt_table


def _decode_bt(payload: bytes, res: float):
    table = _get_bt_table()
    n = len(payload)
    pos = n_inner = n_free = 0
    occ: list[tuple] = []
    stack = [(0.0, 0.0, 0.0, 0)]                      # root centre, depth 0
    while stack:
        cx, cy, cz, d = stack.pop()
        if pos + 2 > n:
            raise ValueError("octomap .bt payload is truncated")
        occ_k, free_k, inner_k = table[payload[pos] | (payload[pos + 1] << 8)]
        pos += 2
        n_inner += 1
        n_free += len(free_k)
        h = res * 2.0 ** (TREE_DEPTH - d - 2)         # child centre offset = parent size / 4
        for k in occ_k:
            sx, sy, sz = _SIGNS[k]
            occ.append((cx + sx * h, cy + sy * h, cz + sz * h, d + 1))
        for k in reversed(inner_k):                   # pop order == stream order
            sx, sy, sz = _SIGNS[k]
            stack.append((cx + sx * h, cy + sy * h, cz + sz * h, d + 1))
    if pos != n:
        raise ValueError(f"octomap .bt payload has {n - pos} trailing bytes")
    return occ, None, n_inner, n_free


def _decode_ot(payload: bytes, res: float, extra: int, thres_lo: float):
    n = len(payload)
    stride = 4 + extra
    pos = n_inner = n_free = 0
    occ: list[tuple] = []
    lo_list: list[float] = []
    unpack = struct.Struct("<f").unpack_from
    stack = [(0.0, 0.0, 0.0, 0)]
    while stack:
        cx, cy, cz, d = stack.pop()
        if pos + stride + 1 > n:
            raise ValueError("octomap .ot payload is truncated")
        lo = unpack(payload, pos)[0]
        children = payload[pos + stride]
        pos += stride + 1
        if children == 0:
            if lo >= thres_lo:
                occ.append((cx, cy, cz, d))
                lo_list.append(lo)
            else:
                n_free += 1
            continue
        n_inner += 1
        h = res * 2.0 ** (TREE_DEPTH - d - 2)
        for k in range(7, -1, -1):
            if (children >> k) & 1:
                sx, sy, sz = _SIGNS[k]
                stack.append((cx + sx * h, cy + sy * h, cz + sz * h, d + 1))
    if pos != n:
        raise ValueError(f"octomap .ot payload has {n - pos} trailing bytes")
    return occ, np.asarray(lo_list, dtype=np.float32), n_inner, n_free


def read_octomap(path: str, occ_thres: float = 0.5) -> OcTreeLeaves:
    """Occupied leaves of a .bt / .ot OcTree. occ_thres (probability) only matters for .ot files."""
    with open(path, "rb") as f:
        buf = f.read()
    first, meta, off = _parse_header(buf)
    tree_id = meta.get("id", "OcTree")
    if "res" not in meta:
        raise ValueError(f"{path}: octomap header has no 'res'")
    res = float(meta["res"])
    payload = buf[off:]
    if first.startswith(BT_HEADER):
        if tree_id != "OcTree":
            raise ValueError(f"{path}: unsupported binary tree type {tree_id!r}")
        occ, lo, n_inner, n_free = _decode_bt(payload, res)
    elif first.startswith(OT_HEADER):
        if tree_id not in _NODE_EXTRA:
            raise ValueError(f"{path}: unsupported tree type {tree_id!r} (supported: {sorted(_NODE_EXTRA)})")
        if not 0.0 < occ_thres < 1.0:
            raise ValueError("occ_thres must be in (0, 1)")
        thres_lo = math.log(occ_thres / (1.0 - occ_thres))
        occ, lo, n_inner, n_free = _decode_ot(payload, res, _NODE_EXTRA[tree_id], thres_lo)
    else:
        raise ValueError(f"{path}: not an OctoMap file (first line {first!r})")
    if "size" in meta:
        total = n_inner + n_free + len(occ)
        if int(meta["size"]) != total:
            warnings.warn(f"{path}: header says {meta['size']} nodes, decoded {total}", stacklevel=2)
    arr = np.asarray(occ, dtype=np.float64).reshape(-1, 4)
    return OcTreeLeaves(res, tree_id, arr[:, :3], arr[:, 3].astype(np.int16), lo, n_inner, n_free)


def expand_leaves(leaves: OcTreeLeaves) -> np.ndarray:
    """Finest-resolution cell centres [M,3]: a leaf at depth d covers 2^(16-d) cells per axis."""
    out = [leaves.centers[leaves.depth == TREE_DEPTH]]
    for d in np.unique(leaves.depth[leaves.depth < TREE_DEPTH]):
        m = 2 ** (TREE_DEPTH - int(d))
        offs = (np.arange(m) - (m - 1) / 2.0) * leaves.res
        gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
        tile = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])          # [m^3,3]
        c = leaves.centers[leaves.depth == d]
        out.append((c[:, None, :] + tile[None, :, :]).reshape(-1, 3))
    return np.vstack(out) if out else np.zeros((0, 3))


def is_octomap_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in OCTOMAP_EXTS


def read_map_points(path: str, occ_thres: float = 0.5) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """(xyz float64 [N,3], intensity or None, octree resolution or None) from a .pcd, .bt or .ot file.
    For an octree the points are the finest-resolution occupied cell centres and intensity is None."""
    if is_octomap_path(path):
        leaves = read_octomap(path, occ_thres)
        return leaves.expand(), None, leaves.res
    xyz, inten = read_pcd(path)
    return xyz, inten, None


def voxel_grid_from_cells(centers: np.ndarray, res: float) -> VoxelGrid:
    """1:1 VoxelGrid from octree cell centres at the tree resolution: one voxel per cell, no binning.
    counts are 1 and intensity 0 (an octree has neither); zmean, the surface height the ground stage and the
    terrain rays use, is the cell's TOP face (an occupied cell is a solid block, so its surface is its top)."""
    origin = np.floor(centers.min(axis=0) / res) * res
    keys = np.rint((centers - origin) / res - 0.5).astype(np.int64)
    dims = keys.max(axis=0) + 1
    codes = encode(keys, dims)
    _, first = np.unique(codes, return_index=True)             # sorted by code, duplicates dropped
    keys = keys[first]
    return VoxelGrid(keys.astype(np.int32), np.ones(len(keys), dtype=np.int32), np.zeros(len(keys), dtype=np.float32),
                     (centers[first, 2] + res / 2).astype(np.float32), origin, float(res), dims)
