"""Embree ray casting (open3d) against the scene boxes + per-voxel attribute lookup."""
from __future__ import annotations

import numpy as np
import open3d as o3d

_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
# 12 triangles over the 8 corners above (outward winding is irrelevant for ray hits)
_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                   [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint32)


class RayScene:
    """Static scene: axis-aligned boxes [M,6] = (cx, cy, cz, hx, hy, hz) plus optional floor slabs
    [K,8,3] (8 corners in the same order as _CORNERS, sloped tops allowed; see pcd2sim/ground.py)."""

    def __init__(self, boxes: np.ndarray, slabs: np.ndarray | None = None):
        c, h = boxes[:, :3].astype(np.float32), boxes[:, 3:].astype(np.float32)
        corners = c[:, None, :] + _CORNERS[None] * h[:, None, :]
        if slabs is not None and len(slabs):
            corners = np.concatenate([corners, slabs.astype(np.float32)])
        verts = corners.reshape(-1, 3)
        tris = (_FACES[None] + 8 * np.arange(len(corners), dtype=np.uint32)[:, None, None]).reshape(-1, 3)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(tris))
        self.n_triangles = len(tris)

    def cast(self, origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (t_hit [N] with inf on miss, hit normals [N,3]). dirs must be unit length."""
        rays = np.empty((len(origins), 6), dtype=np.float32)
        rays[:, :3] = origins
        rays[:, 3:] = dirs
        out = self.scene.cast_rays(o3d.core.Tensor(rays))
        return out["t_hit"].numpy(), out["primitive_normals"].numpy()


class VoxelLookup:
    """Per-voxel attributes from voxels.npz: mean PCD intensity and mean point z."""

    def __init__(self, voxels_npz: str):
        d = np.load(voxels_npz)
        self.origin = d["origin"].astype(np.float64)
        self.voxel = float(d["voxel"])
        self.dims = d["dims"].astype(np.int64)
        keys = d["keys"].astype(np.int64)
        codes = (keys[:, 0] * self.dims[1] + keys[:, 1]) * self.dims[2] + keys[:, 2]
        order = np.argsort(codes)
        self.codes = codes[order]
        self.refl = np.clip(d["intensity"][order], 0, 255).astype(np.uint8)
        self.zmean = d["zmean"][order].astype(np.float64)     # every scene since format 2 carries zmean

    def _find(self, points: np.ndarray):
        """(index into attribute arrays, found mask) for the voxel containing each point."""
        k = np.floor((points - self.origin) / self.voxel).astype(np.int64)
        found = np.all((k >= 0) & (k < self.dims), axis=1)
        idx = np.zeros(len(points), dtype=np.int64)
        if found.any():
            c = (k[found, 0] * self.dims[1] + k[found, 1]) * self.dims[2] + k[found, 2]
            i = np.clip(np.searchsorted(self.codes, c), 0, len(self.codes) - 1)
            hit = self.codes[i] == c
            idx[found] = i
            found[found] = hit
        return idx, found

    def reflectivity(self, points: np.ndarray) -> np.ndarray:
        idx, found = self._find(points)
        out = np.where(found, self.refl[idx], 0).astype(np.uint8)
        miss = ~found
        if miss.any():          # a flattened floor plane can sit just above its points' voxel
            below = points[miss] - np.array([0.0, 0.0, self.voxel])
            idx2, found2 = self._find(below)
            out[miss] = np.where(found2, self.refl[idx2], 0)
        return out

    def surface_z(self, points: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Mean point z of the voxel containing each point (fallback where unknown or flattened)."""
        idx, found = self._find(points)
        z = np.where(found, self.zmean[idx], fallback)
        return np.where(np.isfinite(z), z, fallback)
