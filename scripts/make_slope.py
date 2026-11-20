"""Build a synthetic scene with one smooth inclined plane: flat run-in, incline, flat landing.

Usage (inside the container or the pixi env):
  python3 scripts/make_slope.py --out scenes/slope_10 --angle 10

Layout (x up the incline, y across, the run-in floor top at z = 0):
  run-in   x in [-4, 3]    z = 0             spawn at (0, 0), facing +x
  incline  x in [3, 13]    z = (x - 3) tan(angle)   12 m wide (y in [-6, 6])
  landing  x in [13, 15]   z = 10 tan(angle)
A figure-8 of two 1.5 m lobes with its crossing at meta.json `fig8_center` (8, 0), heading +x, lies on the incline
with >= 0.5 m to spare (lobes x in [6.5, 9.5], y in [-3, 3]).

Scene boxes are axis-aligned, so the incline can not be a box. It is:
  ground.npz  one sloped slab (8 corners): what the LiDAR, the terrain rays and the viewers see;
  solids.npy  one oriented box [cx, cy, cz, hx, hy, hz, qw, qx, qy, qz] whose top face is the plane: a static
              collision geom the simulator adds to the MuJoCo model (quadsim.sim, scene file `solids.npy`).
The run-in floor and the landing are ordinary touchable boxes (contact.npy); a ceiling (not touchable) gives the
LiDAR something to see above, as in scenes/flat_test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

X_RUNIN, X0, X1, X_END = -4.0, 3.0, 13.0, 15.0
HALF_Y = 6.0
THICK = 0.30            # incline solid thickness, normal to the plane
FLOOR_THICK = 0.10
CEILING_Z = 4.0
VOXEL = 0.1


def slope_scene(angle_deg: float = 10.0) -> dict:
    """The scene geometry as arrays (no files): boxes, contact flags, ray slab, collision solid, figure-8 centre."""
    th = math.radians(angle_deg)
    t = math.tan(th)
    h = (X1 - X0) * t
    floor = [(X_RUNIN + X0) / 2, 0.0, -FLOOR_THICK / 2, (X0 - X_RUNIN) / 2, HALF_Y, FLOOR_THICK / 2]
    # the landing is a solid block from the floor up to the head of the incline (no overhang for the rays)
    landing = [(X1 + X_END) / 2, 0.0, (h - FLOOR_THICK) / 2, (X_END - X1) / 2, HALF_Y, (h + FLOOR_THICK) / 2]
    ceiling = [(X_RUNIN + X_END) / 2, 0.0, CEILING_Z + 0.05, (X_END - X_RUNIN) / 2, HALF_Y, 0.05]
    boxes = np.array([floor, landing, ceiling], dtype=np.float32)
    contact = np.array([True, True, False])

    slab = np.zeros((1, 8, 3), np.float32)                  # corner index = 4*ix + 2*iy + iz
    for ix, x in enumerate((X0, X1)):
        for iy, y in enumerate((-HALF_Y, HALF_Y)):
            top = (x - X0) * t
            slab[0, 4 * ix + 2 * iy] = [x, y, top - THICK]
            slab[0, 4 * ix + 2 * iy + 1] = [x, y, top]

    # Oriented box: its top face is the plane, ending exactly at the head (x1, h) and reaching 0.1 m (along the
    # plane) past the foot, where it is under the run-in floor's top.
    ext = 0.10
    length = (X1 - X0) / math.cos(th) + ext                 # along the plane
    u = np.array([math.cos(th), 0.0, math.sin(th)])        # up the plane
    n = np.array([-math.sin(th), 0.0, math.cos(th)])       # plane normal
    head = np.array([X1, 0.0, h])
    centre = head - u * length / 2 - n * THICK / 2
    quat = [math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0]   # rotation about -y by th: +x -> u
    solid = np.array([*centre, length / 2, HALF_Y, THICK / 2, *quat], dtype=np.float64)
    return {"boxes": boxes, "contact": contact, "slabs": slab, "solids": solid[None], "x0": X0, "x1": X1,
            "width": 2 * HALF_Y, "height": h, "angle_deg": angle_deg, "fig8_center": [8.0, 0.0]}


def surface_z(x, angle_deg: float):
    t = math.tan(math.radians(angle_deg))
    return np.clip(np.asarray(x, float) - X0, 0.0, X1 - X0) * t


def write_scene(out: str, angle_deg: float = 10.0) -> dict:
    from pcd2sim import SCENE_FORMAT
    from pcd2sim.pcd_io import write_pcd

    s = slope_scene(angle_deg)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "boxes.npy"), s["boxes"])
    np.save(os.path.join(out, "contact.npy"), s["contact"])
    np.save(os.path.join(out, "solids.npy"), s["solids"])
    np.savez_compressed(os.path.join(out, "ground.npz"), slabs=s["slabs"])

    # one surface voxel per 0.1 m column (zmean NaN on the incline: the terrain rays give its exact height);
    # the ceiling as one layer of voxels
    origin = np.array([X_RUNIN, -HALF_Y, -0.2])
    nx, ny = int(round((X_END - X_RUNIN) / VOXEL)), int(round(2 * HALF_Y / VOXEL))
    nz = int(round((CEILING_Z + 0.4 - origin[2]) / VOXEL))
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx = origin[0] + (gx.ravel() + 0.5) * VOXEL
    cy = origin[1] + (gy.ravel() + 0.5) * VOXEL
    z = surface_z(cx, angle_deg)
    kz = np.floor((z - 1e-6 - origin[2]) / VOXEL).astype(np.int64)       # the voxel just under the surface
    flat = (cx < X0) | (cx > X1)
    kz_ceil = int(np.floor((CEILING_Z + 0.05 - origin[2]) / VOXEL))
    keys = np.concatenate([np.column_stack([gx.ravel(), gy.ravel(), kz]),
                           np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, kz_ceil)])]).astype(np.int32)
    zmean = np.concatenate([np.where(flat, z, np.nan), np.full(gx.size, CEILING_Z + 0.05)]).astype(np.float32)
    contact_v = np.concatenate([flat, np.zeros(gx.size, bool)])
    np.savez_compressed(os.path.join(out, "voxels.npz"), keys=keys, intensity=np.full(len(keys), 50, np.float32),
                        zmean=zmean, origin=origin, voxel=VOXEL, dims=np.array([nx, ny, nz]), contact=contact_v)
    pts = np.column_stack([cx, cy, z])
    pts = np.concatenate([pts, np.column_stack([cx, cy, np.full(len(cx), CEILING_Z)])])
    write_pcd(os.path.join(out, "map_ds.pcd"), pts.astype(np.float32), np.full(len(pts), 50, np.float32))

    meta = {"format_version": SCENE_FORMAT, "voxel": VOXEL, "gravity": [0, 0, -9.81], "spawn": [0.0, 0.0, 0.0],
            "boxes": int(len(s["boxes"])), "contact": {"boxes": int(s["contact"].sum()), "voxels": int(contact_v.sum())},
            "ground": {"slabs": 1}, "solids": int(len(s["solids"])), "source_format": "synthetic",
            "slope": {"angle_deg": angle_deg, "x0": X0, "x1": X1, "width": s["width"], "height": round(s["height"], 4),
                      "fig8_center": s["fig8_center"], "fig8_heading": 0.0},
            "build_args": {"script": "scripts/make_slope.py", "angle": angle_deg}}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--angle", type=float, default=10.0, help="incline angle (deg)")
    a = ap.parse_args()
    meta = write_scene(a.out, a.angle)
    print(f"{a.out}: incline {a.angle} deg, x {X0}..{X1} m, rises {meta['slope']['height']} m")


if __name__ == "__main__":
    main()
