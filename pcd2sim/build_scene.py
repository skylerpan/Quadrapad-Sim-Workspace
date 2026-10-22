"""Map (PCD point cloud or OctoMap .bt/.ot octree) -> MuJoCo-ready scene directory.

Usage:
  python3 -m pcd2sim.build_scene --map Maps/dp2f.pcd --out scenes/dp2f [--voxel 0.1]
  python3 -m pcd2sim.build_scene --map Maps/dp_1012.bt --out scenes/dp_1012_bt [--z-min -0.8]

The format follows the extension (.pcd / .bt / .ot); --pcd is an alias of --map. An octree is converted
1:1: every occupied cell (pruned leaves expanded to the tree resolution) becomes one voxel at the tree
resolution. Noise filter: voxels with fewer than --min-neighbors occupied 26-neighbours go (default 2;
0 = off), then --min-cluster drops 26-connected clusters smaller than N voxels (default off). Ground
flattening then runs exactly as for PCD (same --ground-* flags): floor holes are filled, every floor
region is snapped to a plane (slabs in ground.npz), floating specks go; --no-flatten keeps the raw 1:1
voxel floor instead. No percentile crop and no surface smoothing on octree input; --voxel is ignored.
OctoMap geometry is coarse (10-20 cm): fine for quick or navigation-only scenes, not for stairs (see
docs/pcd-to-sim-architecture.md section 11).

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
from .octomap_io import read_map_points, voxel_grid_from_cells
from .pcd_io import write_pcd
from .voxel import (boxes_to_world, crop_percentile, find_spawn, fit_floor_normal, merge_boxes,
                    remove_isolated, remove_small_clusters, smooth_surface_z, standable_mask, voxelize)


SCENE_FORMAT = 2          # 2: ground.npz + NaN zmean under slabs; keep in sync with quadsim.sim.SCENE_FORMAT
DEFAULT_VOXEL = 0.1       # PCD input; an octree defaults to its own resolution


def build(source: str, out: str, voxel: float | None = None, min_pts: int = 1, min_neighbors: int = 2,
          crop_pct: float = 0.05, z_min: float | None = None, z_max: float | None = None,
          flatten: bool = True, ground_tol: float = 0.06,
          ground_max_rms: float = 0.035, ground_thickness: float = 0.2, ground_fill: int = 12,
          ground_fill_nb: int = 3, ground_max_above: int = 2, ground_min_cells: int = 30,
          ground_layers: int = 2, ground_grow: int = 3, ground_floor_tol: float = 0.15,
          ground_fill_guard: float = 0.0, ground_surface: str = "mean", ground_peel: int = 0,
          ground_small_flat: int = 0, octomap_occ_thres: float = 0.5, min_cluster: int = 0) -> dict:
    build_args = {k: v for k, v in locals().items()}       # recorded in meta.json (scenes are a cache)
    os.makedirs(out, exist_ok=True)
    t = {"start": time.time()}
    log = lambda msg: print(f"[{time.time() - t['start']:6.1f}s] {msg}", flush=True)

    xyz, inten, tree_res = read_map_points(source, occ_thres=octomap_occ_thres)
    is_octomap = tree_res is not None
    if is_octomap:
        # Direct 1:1 conversion: one voxel per occupied cell at the tree resolution, nothing else.
        if voxel is not None and abs(voxel - tree_res) > 1e-9:
            log(f"warning: --voxel {voxel} ignored; octree cells map 1:1 at the tree resolution {tree_res}")
        voxel = tree_res
    elif voxel is None:
        voxel = DEFAULT_VOXEL
    build_args["voxel"] = voxel
    n_raw = len(xyz)
    log(f"read {n_raw} occupied octree cells (res {tree_res}) from {source}" if is_octomap
        else f"read {n_raw} points from {source}")

    keep = crop_percentile(xyz, 0.0 if is_octomap else crop_pct)
    if z_min is not None:
        keep &= xyz[:, 2] >= z_min
    if z_max is not None:
        keep &= xyz[:, 2] <= z_max
    xyz = xyz[keep]
    inten = inten[keep] if inten is not None else None
    log(f"cropped to {len(xyz)} {'cells' if is_octomap else 'points'}")

    if is_octomap:
        grid = voxel_grid_from_cells(xyz, voxel)             # 1:1 cells, then the noise filter below
        n_vox_raw = len(grid.keys)
    else:
        grid = voxelize(xyz, inten, voxel)
        n_vox_raw = len(grid.keys)
        grid = grid.subset(grid.counts >= min_pts)
    del xyz, inten
    n_sparse = n_vox_raw - len(grid.keys)
    grid = remove_isolated(grid, min_neighbors)
    n_iso = n_vox_raw - n_sparse - len(grid.keys)
    grid = remove_small_clusters(grid, min_cluster)
    n_small = n_vox_raw - n_sparse - n_iso - len(grid.keys)
    noise_filter = {"sparse_removed": n_sparse, "isolated_removed": n_iso, "small_clusters_removed": n_small}
    log(f"voxels: {n_vox_raw} -> {len(grid.keys)} after min_pts={min_pts}, min_neighbors={min_neighbors}, "
        f"min_cluster={min_cluster} ({'1:1 octree cells; ' if is_octomap else ''}sparse {n_sparse}, "
        f"isolated {n_iso}, small clusters {n_small})")
    if is_octomap and flatten:
        # Octree cells carry no point counts, and ground.py reads counts <= 3 as sparse noise (spike / sheet
        # tests) and > 3 as "solid" for the fill guard. 4 keeps the count tests quiet (the 26-neighbour filter
        # above already handled noise) so the floor gets filled and flattened exactly like a PCD floor.
        grid.counts = np.full(len(grid.keys), 4, dtype=np.int32)

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
    if not is_octomap:
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
        "source_pcd": os.path.abspath(source),          # the source map (.pcd or octree); key kept for old readers
        "source_pcd_size": os.stat(source).st_size,
        "source_pcd_mtime": os.stat(source).st_mtime,
        "source_format": "octomap" if is_octomap else "pcd",
        "octomap_res": tree_res,
        "voxel": voxel,
        "min_pts": min_pts,
        "min_neighbors": min_neighbors,
        "crop_pct": crop_pct,
        "build_args": build_args,
        "points_raw": n_raw,
        "voxels_raw": n_vox_raw,
        "voxels": int(len(grid.keys)),
        "noise_filter": noise_filter,
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
    ap.add_argument("--map", "--pcd", dest="source", required=True,
                    help="source map: .pcd point cloud or OctoMap .bt / .ot octree (format from the extension)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=None,
                    help="voxel size in metres (default 0.1; an octree always uses its own resolution)")
    ap.add_argument("--min-pts", type=int, default=1, help="min points per voxel")
    ap.add_argument("--min-neighbors", type=int, default=2,
                    help="min occupied 26-neighbours per voxel (noise filter; also applied to octree input, 0 = off)")
    ap.add_argument("--min-cluster", type=int, default=0,
                    help="drop 26-connected clusters with fewer than N voxels (0 = off). Octree noise is mostly "
                         "1-4 voxel specks; 4 is a good value there, PCD builds normally leave this off")
    ap.add_argument("--crop-pct", type=float, default=0.05, help="percentile crop per axis (0 = off)")
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--no-flatten", action="store_true",
                    help="keep the raw voxel floor (no ground slabs, no hole filling; octree input: pure 1:1 boxes)")
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
    ap.add_argument("--octomap-occ-thres", type=float, default=0.5,
                    help=".ot input only: occupancy probability at or above which a leaf counts as occupied")
    a = ap.parse_args(argv)
    build(a.source, a.out, a.voxel, a.min_pts, a.min_neighbors, a.crop_pct, a.z_min, a.z_max,
          not a.no_flatten, a.ground_tol, a.ground_max_rms, a.ground_thickness, a.ground_fill, a.ground_fill_nb,
          a.ground_max_above, a.ground_min_cells, a.ground_layers, a.ground_grow, a.ground_floor_tol,
          a.ground_fill_guard, a.ground_surface, a.ground_peel, a.ground_small_flat,
          octomap_occ_thres=a.octomap_occ_thres, min_cluster=a.min_cluster)


if __name__ == "__main__":
    main()
