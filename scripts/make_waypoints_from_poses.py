#!/usr/bin/env python3
"""Sample navigation waypoints from a SLAM keyframe trajectory (Maps/<map>/poses.txt).

The keyframe trajectory is where the mapping robot actually drove, so points on it are
reachable and in the same `map` frame as the scene and the planner's octomap.

  python3 scripts/make_waypoints_from_poses.py Maps/dp_1008/poses.txt \
      --spacing 8 --z-min -1 --z-max 1 -o config/nav_waypoints_dp_1008.yaml

poses.txt lines: `<name>.pcd x y z qw qx qy qz`. Keyframes whose z lies outside
[--z-min, --z-max] are skipped (use this to pick one storey). The sampled z is the keyframe z
(LiDAR height, the same convention as /robot01/tracked_pose).
"""
from __future__ import annotations

import argparse
import math
import sys


def load_poses(path: str) -> list[tuple[float, float, float, float]]:
    """Return (x, y, z, yaw) per keyframe."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            x, y, z, qw, qx, qy, qz = (float(v) for v in parts[1:8])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            out.append((x, y, z, yaw))
    return out


def sample(poses, spacing: float, z_min: float, z_max: float, min_sep: float):
    """Walk the trajectory and keep a keyframe every `spacing` metres of arc length.

    Points closer than `min_sep` to an already kept point are skipped so a trajectory that
    revisits a corridor does not produce near-duplicates.
    """
    kept: list[tuple[float, float, float, float]] = []
    acc = spacing  # keep the first valid keyframe
    prev = None
    for p in poses:
        if prev is not None:
            acc += math.hypot(p[0] - prev[0], p[1] - prev[1])
        prev = p
        if not (z_min <= p[2] <= z_max):
            continue
        if acc < spacing:
            continue
        if any(math.hypot(p[0] - k[0], p[1] - k[1]) < min_sep for k in kept):
            continue
        kept.append(p)
        acc = 0.0
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", help="Maps/<map>/poses.txt")
    ap.add_argument("--spacing", type=float, default=8.0, help="arc-length spacing between waypoints (m)")
    ap.add_argument("--min-sep", type=float, default=4.0, help="drop a sample closer than this to a kept one (m)")
    ap.add_argument("--z-min", type=float, default=-1.0)
    ap.add_argument("--z-max", type=float, default=1.0)
    ap.add_argument("--prefix", default="wp")
    ap.add_argument("-o", "--output", help="write YAML here (default: stdout)")
    a = ap.parse_args()

    poses = load_poses(a.poses)
    if not poses:
        print(f"no keyframes in {a.poses}", file=sys.stderr)
        return 1
    pts = sample(poses, a.spacing, a.z_min, a.z_max, a.min_sep)

    lines = [
        f"# Waypoints sampled from {a.poses}: every ~{a.spacing:g} m of trajectory, keyframe z in",
        f"# [{a.z_min:g}, {a.z_max:g}] (one storey). {len(pts)} of {len(poses)} keyframes kept.",
        "# Coordinates are in the `map` frame; z is the keyframe (LiDAR) height; yaw in radians.",
        "# Same per-waypoint keys as syncai_waypoint_sequencer/config/waypoints.yaml.",
        "frame_id: map",
        "waypoints:",
    ]
    for i, (x, y, z, yaw) in enumerate(pts, 1):
        lines.append(f"  - {{name: {a.prefix}{i:02d}, x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, yaw: {yaw:.2f}}}")
    text = "\n".join(lines) + "\n"
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(pts)} waypoints to {a.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
