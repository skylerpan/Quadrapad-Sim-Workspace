"""PCD map -> MuJoCo-ready scene directory.

Usage:
  python3 -m pcd2sim.build_scene --pcd Maps/dp2f.pcd --out scenes/dp2f [--voxel 0.1]

Outputs (in --out):
  boxes.npy     float32 [M,6] (cx, cy, cz, hx, hy, hz) collision/LiDAR boxes, world = map frame
  ground.npz    float32 [K,8,3] flattened floor slabs (LiDAR + terrain rays, viewer); see ground.py
  voxels.npz    voxel keys + mean intensity (LiDAR reflectivity) + mean z (terrain height, NaN under slabs)
  map_ds.pcd    voxel centres + intensity (RViz / localization tests)
  meta.json     stats, floor tilt, gravity vector, spawn pose
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from .ground import extract_ground
from .pcd_io import read_pcd, write_pcd
from .voxel import (boxes_to_world, crop_percentile, find_spawn, fit_floor_normal, merge_boxes,
                    remove_isolated, smooth_surface_z, standable_mask, voxelize)


SCENE_FORMAT = 2          # 2: ground.npz + NaN zmean under slabs; keep in sync with quadsim.sim.SCENE_FORMAT


def build(pcd: str, out: str, voxel: float, min_pts: int, min_neighbors: int, crop_pct: float,
          z_min: float | None, z_max: float | None, flatten: bool = True, ground_tol: float = 0.06,
          ground_max_rms: float = 0.035, ground_thickness: float = 0.2, ground_fill: int = 12,
          ground_fill_nb: int = 3, ground_max_above: int = 2, ground_min_cells: int = 30,
          ground_layers: int = 2, ground_grow: int = 3, ground_floor_tol: float = 0.15,
          ground_fill_guard: float = 0.0, ground_surface: str = "mean", ground_peel: int = 0,
          ground_small_flat: int = 0) -> dict:
    build_args = {k: v for k, v in locals().items()}       # recorded in meta.json (scenes are a cache)
    os.makedirs(out, exist_ok=True)
    t = {"start": time.time()}
    log = lambda msg: print(f"[{time.time() - t['start']:6.1f}s] {msg}", flush=True)

    xyz, inten = read_pcd(pcd)
    n_raw = len(xyz)
    log(f"read {n_raw} points from {pcd}")

    keep = crop_percentile(xyz, crop_pct)
    if z_min is not None:
        keep &= xyz[:, 2] >= z_min
    if z_max is not None:
        keep &= xyz[:, 2] <= z_max
    xyz = xyz[keep]
    inten = inten[keep] if inten is not None else None
    log(f"cropped to {len(xyz)} points")

    grid = voxelize(xyz, inten, voxel)
    n_vox_raw = len(grid.keys)
    del xyz, inten
    grid = grid.subset(grid.counts >= min_pts)
    grid = remove_isolated(grid, min_neighbors)
    log(f"voxels: {n_vox_raw} -> {len(grid.keys)} after min_pts={min_pts}, min_neighbors={min_neighbors}")

    stand = standable_mask(grid)
    spawn = find_spawn(grid, stand)
    log(f"standable voxels: {int(stand.sum())}; spawn={None if spawn is None else np.round(spawn, 3).tolist()}")
    if flatten:                              # uses the raw per-voxel z, so run before smoothing
        ground = extract_ground(grid, stand, tol=ground_tol, max_rms=ground_max_rms, thickness=ground_thickness,
                                fill=ground_fill, fill_nb=ground_fill_nb, max_above=ground_max_above,
                                min_cells=ground_min_cells, layers=ground_layers, grow=ground_grow,
                                floor_tol=ground_floor_tol, fill_guard=ground_fill_guard,
                                surface=ground_surface, peel=ground_peel, small_flat=ground_small_flat)
        log(f"ground: {ground.stats}")
    grid.zmean = smooth_surface_z(grid)
    centers = grid.centers()

    # Fit the floor on standable (walkable, roofed) voxels so sloped ceilings don't bias gravity.
    floor_pts = centers[stand] if stand.sum() >= 500 else centers
    normal, d, n_inl = fit_floor_normal(floor_pts, thresh=1.5 * voxel)
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    gravity = (-9.81 * normal).tolist()
    log(f"floor plane: normal={np.round(normal, 4).tolist()} tilt={tilt_deg:.2f} deg ({n_inl} inliers)")

    solid = ~ground.remove if flatten else np.ones(len(grid.keys), dtype=bool)
    boxes_idx = merge_boxes(grid.keys[solid])
    boxes = boxes_to_world(boxes_idx, grid).astype(np.float32)
    log(f"merged {int(solid.sum())} voxels into {len(boxes)} boxes")

    np.save(os.path.join(out, "boxes.npy"), boxes)
    ground_path = os.path.join(out, "ground.npz")
    zmean = grid.zmean.copy()
    map_xyz, map_i = centers[solid], grid.intensity[solid]
    if flatten:
        np.savez_compressed(ground_path, slabs=ground.slabs)
        zmean[ground.remove] = np.nan        # terrain rays use the exact slab hit height there
        # filled/grown cells have no voxel of their own: they take intensity 0
        codes = grid.codes
        cell_codes = grid.encode(ground.cell_keys)
        top = np.clip(np.searchsorted(codes, cell_codes), 0, len(codes) - 1)
        cell_i = np.where(codes[top] == cell_codes, grid.intensity[top], 0).astype(grid.intensity.dtype)
        cell_xyz = np.column_stack([grid.origin[:2] + (ground.cell_keys[:, :2] + 0.5) * voxel, ground.cell_h])
        map_xyz = np.vstack([map_xyz, cell_xyz])
        map_i = np.concatenate([map_i, cell_i])
    elif os.path.exists(ground_path):
        os.remove(ground_path)
    np.savez_compressed(os.path.join(out, "voxels.npz"), keys=grid.keys, intensity=grid.intensity, zmean=zmean,
                        origin=grid.origin, voxel=grid.voxel, dims=grid.dims)
    write_pcd(os.path.join(out, "map_ds.pcd"), map_xyz.astype(np.float32), map_i)

    lo, hi = centers.min(axis=0), centers.max(axis=0)
    meta = {
        "format_version": SCENE_FORMAT,
        "source_pcd": os.path.abspath(pcd),
        "source_pcd_size": os.stat(pcd).st_size,
        "source_pcd_mtime": os.stat(pcd).st_mtime,
        "voxel": voxel,
        "min_pts": min_pts,
        "min_neighbors": min_neighbors,
        "crop_pct": crop_pct,
        "build_args": build_args,
        "points_raw": n_raw,
        "voxels_raw": n_vox_raw,
        "voxels": int(len(grid.keys)),
        "boxes": int(len(boxes)),
        "ground": ground.stats if flatten else None,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "floor_normal": normal.tolist(),
        "floor_tilt_deg": tilt_deg,
        "gravity": gravity,
        "spawn": None if spawn is None else spawn.tolist(),
        "build_seconds": round(time.time() - t["start"], 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote scene to {out}: {meta['boxes']} boxes, peak RSS {meta['peak_rss_mb']} MB")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.1, help="voxel size in metres")
    ap.add_argument("--min-pts", type=int, default=1, help="min points per voxel")
    ap.add_argument("--min-neighbors", type=int, default=2, help="min occupied 26-neighbours per voxel")
    ap.add_argument("--crop-pct", type=float, default=0.05, help="percentile crop per axis (0 = off)")
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--no-flatten", action="store_true", help="keep the raw voxel floor (no ground slabs)")
    ap.add_argument("--ground-tol", type=float, default=0.06, help="max height jump inside one floor region (m)")
    ap.add_argument("--ground-max-rms", type=float, default=0.035, help="max plane-fit RMS to snap a region (m)")
    ap.add_argument("--ground-thickness", type=float, default=0.2, help="floor slab thickness (m)")
    ap.add_argument("--ground-fill", type=int, default=12,
                    help="rounds of filling floor columns that have no points (holes in sparse SLAM floors)")
    ap.add_argument("--ground-fill-nb", type=int, default=3,
                    help="a hole column is filled when at least this many of its 8 neighbours are floor")
    ap.add_argument("--ground-max-above", type=int, default=2,
                    help="max voxels above the floor plane (within 0.8 m) for a column to be grown/filled as floor")
    ap.add_argument("--ground-min-cells", type=int, default=30,
                    help="smallest floor region kept (cells); stair treads need 8-12")
    ap.add_argument("--ground-layers", type=int, default=2,
                    help="voxel layers below the top voxel averaged into a cell's surface height")
    ap.add_argument("--ground-grow", type=int, default=3,
                    help="rings of neighbouring columns grown into the floor past noise voxels")
    ap.add_argument("--ground-floor-tol", type=float, default=0.15,
                    help="a grown column needs a voxel within this height of the floor plane (m)")
    ap.add_argument("--ground-fill-guard", type=float, default=0.0,
                    help="do not grow/fill under a column with a solid voxel up to this height above the "
                         "proposed floor (m; 0 = off). Keeps stair treads from being buried by the landing")
    ap.add_argument("--ground-surface", choices=["mean", "median"], default="mean",
                    help="per-cell surface height: count-weighted mean, or median (robust to a sparse noise "
                         "voxel on top and to riser points under a tread)")
    ap.add_argument("--ground-peel", type=int, default=0,
                    help="split a non-planar floor region into its dominant plane + the rest, this many "
                         "levels deep (0 = off). Separates landings from the treads chained to them")
    ap.add_argument("--ground-small-flat", type=int, default=0,
                    help="regions of at most this many cells get a horizontal plane at their median height "
                         "(0 = off; 60 for stair treads, which a free plane fit turns into ramps)")
    a = ap.parse_args(argv)
    build(a.pcd, a.out, a.voxel, a.min_pts, a.min_neighbors, a.crop_pct, a.z_min, a.z_max,
          not a.no_flatten, a.ground_tol, a.ground_max_rms, a.ground_thickness, a.ground_fill, a.ground_fill_nb,
          a.ground_max_above, a.ground_min_cells, a.ground_layers, a.ground_grow, a.ground_floor_tol,
          a.ground_fill_guard, a.ground_surface, a.ground_peel, a.ground_small_flat)


if __name__ == "__main__":
    main()
