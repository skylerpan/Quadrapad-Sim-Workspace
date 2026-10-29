"""Map (PCD point cloud or OctoMap .bt/.ot octree) -> MuJoCo-ready scene directory.

Usage:
  python3 -m pcd2sim.build_scene --map Maps/dp_1008.pcd --out scenes/dp_1008 [--voxel 0.1]
  python3 -m pcd2sim.build_scene --map Maps/dp_1012_handmade.bt --out scenes/dp_1012 --no-flatten --min-neighbors 0

The format follows the extension (.pcd / .bt / .ot); --pcd is an alias of --map. An octree is converted
1:1: every occupied cell (pruned leaves expanded to the tree resolution) becomes one voxel at the tree
resolution. Noise filter: voxels with fewer than --min-neighbors occupied 26-neighbours go (default 2;
0 = off), then --min-cluster drops 26-connected clusters smaller than N voxels (default off). Ground
flattening then runs exactly as for PCD (same --ground-* flags): floor holes are filled, every floor
region is snapped to a plane (slabs in ground.npz), floating specks go; --no-flatten keeps the raw 1:1
voxel floor instead. No percentile crop and no surface smoothing on octree input; --voxel is ignored.
OctoMap geometry is coarse (10-20 cm): fine for quick or navigation-only scenes, not for stairs (see
docs/pcd-to-sim-architecture.md section 11).

Fine PCD builds (2-3 cm): most voxels hold a single point, so surfaces are sieves. --close-rounds N fills
empty cells inside a surface (>= 6 of the 8 neighbours in one axis plane, or >= 12 of all 26; sheets close,
they do not thicken and edges do not grow; see voxel.close_holes), chunked over --workers forked processes.
The ground options that count points per voxel (--ground-spike-pts, --ground-sheet-pts) and voxels per
cluster (--ground-float-max) assume 10 cm voxels and must be lowered / raised with the resolution; the 0.03 m
recipe is in docs/usage.md section 2. Stage results (voxels, closed voxels, ground) are checkpointed in
<out>/.tmp (removed when the build succeeds; --keep-tmp keeps it, --resume reuses the stages found there).

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
import shutil
import time

import numpy as np

from .ground import Ground, extract_ground
from .octomap_io import read_map_points, voxel_grid_from_cells
from .pcd_io import write_pcd
from .voxel import (VoxelGrid, boxes_to_world, close_holes, crop_percentile, find_spawn, fit_floor_normal,
                    merge_boxes, remove_isolated, remove_small_clusters, smooth_surface_z, standable_mask, voxelize)


SCENE_FORMAT = 2          # 2: ground.npz + NaN zmean under slabs; keep in sync with quadsim.sim.SCENE_FORMAT
DEFAULT_VOXEL = 0.1       # PCD input; an octree defaults to its own resolution
CROP_PCT = 0.05           # per-axis percentile crop of PCD input (octree input is not cropped)


def build(source: str, out: str, voxel: float | None = None, min_neighbors: int = 2,
          z_min: float | None = None, flatten: bool = True, ground_tol: float = 0.06, ground_fill: int = 12,
          ground_max_above: int = 2, ground_min_cells: int = 30, ground_layers: int = 2, ground_grow: int = 3,
          ground_fill_guard: float = 0.0, ground_surface: str = "mean", ground_peel: int = 0,
          ground_small_flat: int = 0, min_cluster: int = 0, close_rounds: int = 0, ground_spike_pts: int = 3,
          ground_sheet_pts: int = 2, ground_float_max: int = 4, ground_column_gap: float = 0.0,
          ground_roof_cell: float | None = None, workers: int | None = None,
          keep_tmp: bool = False, resume: bool = False) -> dict:
    build_args = {k: v for k, v in locals().items()}       # recorded in meta.json (scenes are a cache)
    os.makedirs(out, exist_ok=True)
    tmp_dir = os.path.join(out, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    stage_voxels = os.path.join(tmp_dir, "stage_voxels.npz")
    stage_closed = os.path.join(tmp_dir, "stage_closed.npz")
    stage_ground = os.path.join(tmp_dir, "stage_ground.npz")
    t = {"start": time.time()}
    log = lambda msg: print(f"[{time.time() - t['start']:6.1f}s] {msg}", flush=True)
    try:
        meta = _build(**locals())
    except BaseException:
        log(f"build failed; stage files kept in {tmp_dir} (rerun with --resume to reuse them)")
        raise
    if keep_tmp:
        log(f"stage files kept in {tmp_dir} (--keep-tmp)")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return meta


def _save_grid(path: str, grid: VoxelGrid, **extra) -> None:
    np.savez(path, keys=grid.keys, counts=grid.counts, intensity=grid.intensity, zmean=grid.zmean,
             origin=grid.origin, voxel=grid.voxel, dims=grid.dims, extra=json.dumps(extra))


def _load_grid(path: str) -> tuple[VoxelGrid, dict]:
    d = np.load(path)
    grid = VoxelGrid(d["keys"], d["counts"], d["intensity"], d["zmean"], d["origin"], float(d["voxel"]), d["dims"])
    return grid, json.loads(str(d["extra"]))


def _build(source, out, voxel, min_neighbors, z_min, flatten, ground_tol, ground_fill, ground_max_above,
           ground_min_cells, ground_layers, ground_grow, ground_fill_guard, ground_surface, ground_peel,
           ground_small_flat, min_cluster, close_rounds, ground_spike_pts, ground_sheet_pts, ground_float_max,
           ground_column_gap, ground_roof_cell, workers, keep_tmp, resume, build_args, tmp_dir, stage_voxels,
           stage_closed, stage_ground, t, log) -> dict:
    # Stage order: voxels (read + voxelize + noise filters) -> closed (hole closing) -> ground. With --resume
    # the latest stage file present is loaded and the earlier stages are skipped; every stage file carries
    # what meta.json needs from the stages before it.
    grid = None
    is_octomap = tree_res = None
    if resume and os.path.exists(stage_closed):
        grid, st = _load_grid(stage_closed)
        log(f"resume: loaded closed voxels from {stage_closed} ({len(grid.keys)} voxels)")
    elif resume and os.path.exists(stage_voxels):
        grid, st = _load_grid(stage_voxels)
        log(f"resume: loaded voxels from {stage_voxels} ({len(grid.keys)} voxels)")
    if grid is not None:
        voxel, n_raw, n_vox_raw = st["voxel"], st["points_raw"], st["voxels_raw"]
        noise_filter, is_octomap, tree_res = st["noise_filter"], st["is_octomap"], st["tree_res"]
        build_args["voxel"] = voxel
    else:
        xyz, inten, tree_res = read_map_points(source)
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

        keep = crop_percentile(xyz, 0.0 if is_octomap else CROP_PCT)
        if z_min is not None:
            keep &= xyz[:, 2] >= z_min
        xyz = xyz[keep]
        inten = inten[keep] if inten is not None else None
        del keep
        log(f"cropped to {len(xyz)} {'cells' if is_octomap else 'points'}")

        if is_octomap:
            grid = voxel_grid_from_cells(xyz, voxel)             # 1:1 cells, then the noise filter below
            n_vox_raw = len(grid.keys)
        else:
            grid = voxelize(xyz, inten, voxel)
            n_vox_raw = len(grid.keys)
        del xyz, inten                                           # the raw points are the biggest arrays
        grid = remove_isolated(grid, min_neighbors)
        n_iso = n_vox_raw - len(grid.keys)
        grid = remove_small_clusters(grid, min_cluster)
        n_small = n_vox_raw - n_iso - len(grid.keys)
        noise_filter = {"isolated_removed": n_iso, "small_clusters_removed": n_small, "closed_added": 0}
        log(f"voxels: {n_vox_raw} -> {len(grid.keys)} after min_neighbors={min_neighbors}, "
            f"min_cluster={min_cluster} ({'1:1 octree cells; ' if is_octomap else ''}"
            f"isolated {n_iso}, small clusters {n_small})")
        stage = dict(voxel=voxel, points_raw=n_raw, voxels_raw=n_vox_raw, noise_filter=noise_filter,
                     is_octomap=is_octomap, tree_res=tree_res)
        if close_rounds > 0:
            _save_grid(stage_voxels, grid, **stage)
            grid, n_closed = close_holes(grid, rounds=close_rounds, workers=workers,
                                         tmp_dir=os.path.join(tmp_dir, "close"))
            noise_filter["closed_added"] = n_closed
            log(f"closed: +{n_closed} voxels ({close_rounds} rounds) -> {len(grid.keys)} voxels")
            _save_grid(stage_closed, grid, **stage)
            if os.path.exists(stage_voxels):
                os.remove(stage_voxels)
    if is_octomap and flatten:
        # Octree cells carry no point counts, and ground.py reads counts <= 3 as sparse noise (spike / sheet
        # tests) and > 3 as "solid" for the fill guard. 4 keeps the count tests quiet (the 26-neighbour filter
        # above already handled noise) so the floor gets filled and flattened exactly like a PCD floor.
        grid.counts = np.full(len(grid.keys), 4, dtype=np.int32)

    stand = standable_mask(grid, roof_cell=ground_roof_cell)
    spawn = find_spawn(grid, stand)
    log(f"standable voxels: {int(stand.sum())}; spawn={None if spawn is None else np.round(spawn, 3).tolist()}")
    if flatten:                              # uses the raw per-voxel z, so run before smoothing
        if resume and os.path.exists(stage_ground):
            d = np.load(stage_ground)
            ground = Ground(d["remove"], d["cell_keys"], d["cell_h"], d["slabs"], json.loads(str(d["stats"])))
            log(f"resume: loaded ground from {stage_ground}")
        else:
            ground = extract_ground(grid, stand, tol=ground_tol, fill=ground_fill, max_above=ground_max_above,
                                    min_cells=ground_min_cells, layers=ground_layers, grow=ground_grow,
                                    fill_guard=ground_fill_guard,
                                    surface=ground_surface, peel=ground_peel, small_flat=ground_small_flat,
                                    spike_pts=ground_spike_pts, sheet_pts=ground_sheet_pts,
                                    float_max_voxels=ground_float_max, column_gap=ground_column_gap)
            np.savez(stage_ground, remove=ground.remove, cell_keys=ground.cell_keys, cell_h=ground.cell_h,
                     slabs=ground.slabs, stats=json.dumps(ground.stats))
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
        "min_neighbors": min_neighbors,
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
    ap.add_argument("--min-neighbors", type=int, default=2,
                    help="min occupied 26-neighbours per voxel (noise filter; also applied to octree input, 0 = off)")
    ap.add_argument("--min-cluster", type=int, default=0,
                    help="drop 26-connected clusters with fewer than N voxels (0 = off). Octree noise is mostly "
                         "1-4 voxel specks; 4 is a good value there, PCD builds normally leave this off")
    ap.add_argument("--z-min", type=float, default=None, help="drop everything below this height (m)")
    ap.add_argument("--no-flatten", action="store_true",
                    help="keep the raw voxel floor (no ground slabs, no hole filling; octree input: pure 1:1 boxes)")
    ap.add_argument("--ground-tol", type=float, default=0.06, help="max height jump inside one floor region (m)")
    ap.add_argument("--ground-fill", type=int, default=12,
                    help="rounds of filling floor columns that have no points (holes in sparse SLAM floors; "
                         "a column is filled when >= 3 of its 8 neighbours are floor)")
    ap.add_argument("--ground-max-above", type=int, default=2,
                    help="max voxels above the floor plane (within 0.8 m) for a column to be grown/filled as floor")
    ap.add_argument("--ground-min-cells", type=int, default=30,
                    help="smallest floor region kept (cells); stair treads need 8-12")
    ap.add_argument("--ground-layers", type=int, default=2,
                    help="voxel layers below the top voxel averaged into a cell's surface height")
    ap.add_argument("--ground-grow", type=int, default=3,
                    help="rings of neighbouring columns grown into the floor past noise voxels")
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
    ap.add_argument("--ground-spike-pts", type=int, default=3,
                    help="a voxel with at most this many points above a grown/filled floor column is SLAM noise, "
                         "and the fill guard needs more points than this to call a voxel solid. Sized for 10 cm "
                         "voxels; 0 at 2-3 cm (every voxel is real)")
    ap.add_argument("--ground-sheet-pts", type=int, default=2,
                    help="voxels with at most this many points one layer above the floor band with nothing on "
                         "top are the upper tail of the floor noise and go. 0 at 2-3 cm voxels")
    ap.add_argument("--ground-float-max", type=int, default=4,
                    help="clusters of at most this many voxels floating within 0.5 m above the floor are noise "
                         "(scale with 1/voxel^2: 4 at 10 cm, ~40 at 3 cm)")
    ap.add_argument("--ground-roof-cell", type=float, default=None,
                    help="the roof test of a floor voxel (structure >= 1.5 m above) looks at this wide an xy "
                         "column (m; default: one voxel). Sparse ceilings leave most 3 cm columns roofless: 0.15")
    ap.add_argument("--ground-column-gap", type=float, default=0.0,
                    help="multi-storey fill: a column is only skipped when it already has a ground cell within "
                         "this height (m) of the proposed floor (0 = any cell in the column blocks, the old "
                         "behaviour; 0.3 lets the 1F floor under the 2F slabs be filled)")
    ap.add_argument("--close-rounds", type=int, default=0,
                    help="hole-closing rounds (0 = off): fill empty cells inside a surface so 2-3 cm surfaces "
                         "become contiguous (2 rounds close gaps up to 2 voxels wide)")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for the chunked hole closing (default: all CPUs)")
    ap.add_argument("--keep-tmp", action="store_true", help="keep the stage files (<out>/.tmp) after a successful build")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the stage files found in <out>/.tmp (closed voxels, ground) instead of recomputing")
    a = ap.parse_args(argv)
    build(a.source, a.out, voxel=a.voxel, min_neighbors=a.min_neighbors, z_min=a.z_min, flatten=not a.no_flatten,
          ground_tol=a.ground_tol, ground_fill=a.ground_fill, ground_max_above=a.ground_max_above,
          ground_min_cells=a.ground_min_cells, ground_layers=a.ground_layers, ground_grow=a.ground_grow,
          ground_fill_guard=a.ground_fill_guard, ground_surface=a.ground_surface, ground_peel=a.ground_peel,
          ground_small_flat=a.ground_small_flat, min_cluster=a.min_cluster, close_rounds=a.close_rounds,
          ground_spike_pts=a.ground_spike_pts, ground_sheet_pts=a.ground_sheet_pts,
          ground_float_max=a.ground_float_max, ground_column_gap=a.ground_column_gap,
          ground_roof_cell=a.ground_roof_cell, workers=a.workers, keep_tmp=a.keep_tmp, resume=a.resume)


if __name__ == "__main__":
    main()
