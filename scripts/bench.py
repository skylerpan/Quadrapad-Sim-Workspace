"""Headless benchmark + sanity checks of the simulator core (no ROS needed).

Usage: python3 scripts/bench.py [--scene scenes/dp_1012] [--seconds 30] [key=value overrides...]

Checks:
  - real-time factor and time split (physics / lidar / control)
  - points per frame, offset_time monotonic within [0, frame period)
  - IMU at rest ~ 1 g along the floor normal
  - static LiDAR frame vs map_ds.pcd: nearest-neighbour distance P95 (< 2 voxels expected)
  - driving: stuck/collision events
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pcd2sim.pcd_io import read_pcd  # noqa: E402
from quadsim.config import load_config  # noqa: E402
from quadsim.sim import Simulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    if a.scene:
        cfg["scene"] = a.scene
    t_load = time.time()
    sim = Simulator(cfg)
    print(f"scene {cfg['scene']}: {sim.meta['boxes']} boxes, {sim.rays.n_triangles} triangles, "
          f"load {time.time() - t_load:.2f}s, model ngeom={sim.m.ngeom}")
    print(f"spawn pose {np.round(sim.d.qpos[:3], 3).tolist()}")

    frames, events, imu_rest = [], [], []
    static_check = None
    wall0 = time.perf_counter()
    while sim.d.time < a.seconds:
        t = sim.d.time
        if t < 1.5:
            sim.set_cmd(0.0, 0.0, 0.0)          # stand still: IMU + static LiDAR checks
        else:                                     # wander: forward with a slow weave
            sim.set_cmd(0.8, 0.0, 0.6 * np.sin(0.4 * t))
        out = sim.control_tick()
        events += out.events
        if 0.5 < t < 1.5:
            imu_rest += [s.acc for s in out.imu]
        for f in out.frames:
            frames.append(f)
            if static_check is None and 0.5 < f.stamp < 1.4 and len(f.xyz):
                R = sim.d.site_xmat[sim.lidar_site].reshape(3, 3).copy()
                p = sim.d.site_xpos[sim.lidar_site].copy()
                static_check = (f.xyz @ R.T) + p
    wall = time.perf_counter() - wall0

    sim_t = sim.d.time
    print(f"\nsim {sim_t:.1f}s in {wall:.2f}s wall -> RTF {sim_t / wall:.1f}x")
    tm = sim.timing
    print("time split: " + ", ".join(f"{k} {v:.2f}s ({100 * v / wall:.0f}%)" for k, v in tm.items()))
    n = np.array([len(f.xyz) for f in frames])
    print(f"frames {len(frames)}, points/frame mean {n.mean():.0f} min {n.min()} max {n.max()}")
    period = 1.0 / cfg["lidar"]["rate_hz"]
    mono = all(np.all(np.diff(f.offset) >= 0) for f in frames)
    rng_ok = all(len(f.offset) == 0 or (f.offset.min() >= 0 and f.offset.max() < period) for f in frames)
    print(f"offset_time monotonic: {mono}, within [0,{period}): {rng_ok}")
    if imu_rest:
        acc = np.mean(imu_rest, axis=0)
        print(f"IMU at rest mean acc {np.round(acc, 4).tolist()} (g), "
              f"|a|={np.linalg.norm(acc):.4f}")
    if static_check is not None:
        from scipy.spatial import cKDTree

        mp, _ = read_pcd(os.path.join(cfg["scene"], "map_ds.pcd"))
        dist, _ = cKDTree(mp).query(static_check, k=1)
        print(f"static frame vs map: {len(static_check)} pts, NN dist P50 {np.percentile(dist, 50):.3f} "
              f"P95 {np.percentile(dist, 95):.3f} m (voxel {sim.meta['voxel']})")
    print(f"proxy overflow ticks: {sim.proxies.overflow}")
    kinds = {}
    for e in events:
        key = e["type"] + (":" + e["reason"] if "reason" in e else "")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"events: {kinds}")
    print(f"final pose {np.round(sim.d.qpos[:3], 2).tolist()}")


if __name__ == "__main__":
    main()
