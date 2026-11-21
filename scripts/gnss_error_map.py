#!/usr/bin/env python3
"""GNSS error map of a built scene: on every walkable floor cell, the sim's sky cone (quadsim.gnss, the
config's gnss.sky) cast from the antenna height, the fix type that sky gives, and the expected horizontal
error. Writes a self-contained HTML report (scripts/gnss_error_map_template.html) next to a JSON of the grid.

Runs on the host: the rays march through the scene's own occupancy grid (voxels.npz, step 0.1 m), not
Embree, so no open3d / MuJoCo is needed. A hit within gnss.sky.range is "no sky", a ray that leaves the
scanned volume is sky, the same as the sim's Embree cast.

  python3 scripts/gnss_error_map.py                      # config/sim.yaml scene -> results/gnss_errmap/<scene>.html
  python3 scripts/gnss_error_map.py --scene scenes/dp_1012 --cell 2

Floor cells: an occupied voxel with >= 1.0 m free above it and >= 6 of its 8 neighbours alike (within one
voxel up / down), so wall tops and furniture edges drop out. A map cell (--cell voxels square) takes the
lowest such voxel inside each height band (LEVELS). The antenna sits stand_height + mount_xyz z above it.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from quadsim.gnss import sky_directions  # noqa: E402

LEVELS = [("1F 地面層", -1.6, 1.6), ("2F", 1.6, 6.0), ("3F 以上 / 屋頂", 6.0, 30.0)]
BLOCK = 4                                       # 3D view: building cubes of BLOCK voxels (0.8 m at 0.2 m)
HEADROOM = 5                                    # voxels of free space above a floor voxel (1.0 m at 0.2 m)


def floor_voxels(occ: np.ndarray) -> np.ndarray:
    free = ~occ
    head = np.ones(occ.shape, bool)
    for j in range(1, HEADROOM + 1):
        sh = np.ones(occ.shape, bool)
        sh[:, :, :-j] = free[:, :, j:]
        head &= sh
    surf = occ & head
    nb = np.zeros(occ.shape, np.int8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            s = np.zeros(occ.shape, bool)
            for dz in (-1, 0, 1):
                s |= np.roll(np.roll(np.roll(surf, dx, 0), dy, 1), dz, 2)
            nb += s
    return surf & (nb >= 6)


def march(occ, org, v, origins, dirs, rng, step=0.1) -> np.ndarray:
    """Distance to the first occupied voxel within rng along each ray, inf on a miss."""
    dims = np.array(occ.shape)
    hit = np.zeros(len(origins), bool)
    dist = np.full(len(origins), np.inf)
    alive = np.ones(len(origins), bool)
    for s in np.arange(0.3, rng, step):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        p = origins[idx] + dirs[idx] * s
        kk = np.floor((p - org) / v).astype(np.int64)
        out = np.any(kk < 0, 1) | np.any(kk >= dims, 1)
        h = np.zeros(len(idx), bool)
        inn = ~out
        h[inn] = occ[kk[inn, 0], kk[inn, 1], kk[inn, 2]]
        hit[idx[h]] = True
        dist[idx[h]] = s
        alive[idx[h | out]] = False             # out of the scanned volume (above or sideways) = sky
    return dist


def dense_directions() -> np.ndarray:
    """Zenith + rings at 20 / 40 / 60 deg (8 / 16 / 24 rays): the physical reference next to the sim's 9."""
    d = [[0.0, 0.0, 1.0]]
    for a, n in ((20, 8), (40, 16), (60, 24)):
        ar = math.radians(a)
        for az in np.linspace(0, 2 * np.pi, n, endpoint=False):
            d.append([math.sin(ar) * math.cos(az), math.sin(ar) * math.sin(az), math.cos(ar)])
    return np.array(d)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config/sim.yaml"))
    ap.add_argument("--scene", default=None, help="scene directory (default: the config's scene)")
    ap.add_argument("--cell", type=int, default=2, help="map cell size in voxels (2 = 0.4 m at 0.2 m)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/gnss_errmap"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gn = cfg["gnss"]
    sky_cfg = gn["sky"]
    scene = args.scene or cfg["scene"]
    scene_dir = scene if os.path.isabs(scene) else os.path.join(ROOT, scene)
    meta = json.load(open(os.path.join(scene_dir, "meta.json")))
    datum = meta.get("geo") or gn["datum"]
    d = np.load(os.path.join(scene_dir, "voxels.npz"))
    keys, dims, org, v = d["keys"], d["dims"], d["origin"].astype(float), float(d["voxel"])
    occ = np.zeros(dims, bool)
    occ[keys[:, 0], keys[:, 1], keys[:, 2]] = True
    floor = floor_voxels(occ)

    ant = float(cfg["robot"]["stand_height"]) + float(gn["mount_xyz"][2])
    zc = org[2] + (np.arange(dims[2]) + 1.0) * v            # top face of voxel layer k
    C = args.cell
    nx, ny = int(dims[0] // C), int(dims[1] // C)
    rng = float(sky_cfg["range"])
    dirs9 = sky_directions(int(sky_cfg["rays"]), float(sky_cfg["half_angle_deg"]))
    dense = dense_directions()

    t0 = time.time()
    levels = []
    for name, z0, z1 in LEVELS:
        kmask = (zc >= z0) & (zc < z1)
        blk = floor[:nx * C, :ny * C][:, :, kmask].reshape(nx, C, ny, C, -1).any((1, 3))     # [nx, ny, k]
        has = blk.any(2)
        kz = np.argmax(blk, 2)
        fz = np.where(has, zc[kmask][kz], np.nan)
        ii, jj = np.nonzero(has)
        pts = np.column_stack([org[0] + (ii * C + C / 2) * v, org[1] + (jj * C + C / 2) * v, fz[ii, jj] + ant + 0.05])
        res = {}
        for tag, D in (("sky9", dirs9), ("skyd", dense)):
            o = np.repeat(pts, len(D), 0)
            dd = np.tile(D, (len(pts), 1))
            t = np.concatenate([march(occ, org, v, o[c:c + 400000], dd[c:c + 400000], rng)
                                for c in range(0, len(o), 400000)]) if len(o) else np.zeros(0)
            sky = np.full((nx, ny), np.nan)
            sky[ii, jj] = 1.0 - np.isfinite(t).reshape(len(pts), len(D)).mean(1)
            res[tag] = sky
            if tag == "sky9":                   # per-ray hit distance for the 3D view: 0.2 m steps, 255 = sky
                res["r9"] = np.where(np.isfinite(t), np.clip(np.round(t / 0.2), 0, 254), 255)
        print(f"{name}: {len(pts)} cells, {time.time() - t0:.0f} s", flush=True)
        n9 = len(dirs9)
        levels.append({
            "name": name, "z": [z0, z1], "n": int(len(pts)),
            "sky9": b64(np.where(has, np.round(np.nan_to_num(res["sky9"]) * n9), 255)),
            "skyd": b64(np.where(has, np.round(np.nan_to_num(res["skyd"]) * 200), 255)),
            "r9": b64(res["r9"]),
            "fz": b64(np.where(has, np.clip(np.round((np.nan_to_num(fz) + 3.0) / 0.1), 0, 254), 255)),
        })

    anyocc = occ.any(2)
    top = (dims[2] - 1 - np.argmax(occ[:, :, ::-1], 2)) * v + org[2] + v
    top = np.where(anyocc, top, np.nan)[:nx * C, :ny * C].reshape(nx, C, ny, C)
    top = np.where(np.isfinite(top), top, -np.inf).max((1, 3))
    top[np.isinf(top)] = np.nan
    # coarse scene for the 3D view: BLOCK^3 voxel cubes, only those with an open face
    B = BLOCK
    bd = -(-dims // B)
    blk = np.zeros(bd, bool)
    blk[keys[:, 0] // B, keys[:, 1] // B, keys[:, 2] // B] = True
    pad = np.pad(blk, 1)
    inner = np.ones(bd, bool)
    for ax in range(3):
        for sh in (-1, 1):
            inner &= np.roll(pad, sh, ax)[1:-1, 1:-1, 1:-1]
    bk = np.argwhere(blk & ~inner).astype(np.uint8)
    data = {
        "scene": os.path.relpath(scene_dir, ROOT), "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nx": nx, "ny": ny, "cell": C * v, "origin": [float(org[0]), float(org[1])],
        "antenna": round(ant, 2), "datum": datum, "datum_src": "meta.json geo" if meta.get("geo") else "config gnss.datum",
        "sky_cfg": sky_cfg, "noise": gn["noise"], "holdover": gn["holdover"], "model": gn["model"], "rtk": gn["rtk"],
        "rate_hz": gn["rate_hz"], "spawn": cfg.get("spawn"), "rays9": len(dirs9), "raysd": len(dense),
        "top": b64(np.where(np.isfinite(top), np.clip(np.round((np.nan_to_num(top) + 3.0) / 0.2), 0, 254), 255)),
        "levels": levels, "dirs9": np.round(dirs9, 5).tolist(),
        "blocks": b64(bk.reshape(-1)), "block": B * v, "zorg": float(org[2]),
    }
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, os.path.basename(scene_dir.rstrip("/")))
    json.dump(data, open(stem + ".json", "w"))
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnss_error_map_template.html")).read()
    open(stem + ".html", "w").write(tpl.replace("__DATA__", json.dumps(data)))
    print(f"wrote {stem}.html, {stem}.json")


if __name__ == "__main__":
    main()
