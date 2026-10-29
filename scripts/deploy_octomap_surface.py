#!/usr/bin/env python3
"""Deploy a hand-edited OctoMap (.bt) to the SyncAI planner workspace as a surface PCD.

One command does what was done by hand for dp_1012 / dp:
  1. sample the exposed faces of the .bt into Maps/<stem>_surface.pcd (same as bt_to_pcd.py)
  2. copy that PCD into <ws>/src/syncai_planner/octo_planner_3d/maps/<name>/
  3. make sure the planner's copy of the .bt is identical to the source (never overwrite an
     existing .bt: if the two differ the script stops and lets you decide)
  4. touch the .bt and the .layers cache so they are NEWER than the PCD. The planner
     (octo_planner_rviz_node.cpp) rebuilds and OVERWRITES the .bt from the PCD whenever the
     PCD is newer, and global_planner.cpp recomputes .layers for the same reason.
  5. point maps_config.yaml  maps.<name>.pcd_path  at the new surface PCD
  6. point config/instances/robot01.ini  [3Dmap] pcd  at it (only if it already points into
     maps/<name>/, otherwise it is left alone with a warning; --force-ini overrides)

Usage:
  python3 scripts/deploy_octomap_surface.py Maps/dp.bt
  python3 scripts/deploy_octomap_surface.py Maps/dp_1012_handmade.bt --name dp_1012
  python3 scripts/deploy_octomap_surface.py Maps/dp.bt --dry-run        # print the plan only

Config paths are rewritten by file name only: the directory prefix already in the config
(/mnt/ssd/sync_ai/src/SyncAI-Device-Workspace/...) is kept as is.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from pcd2sim import octomap_io  # noqa: E402
from pcd2sim.pcd_io import write_pcd  # noqa: E402
from bt_to_pcd import exposed_faces, report, sample_points  # noqa: E402

DEFAULT_WS = os.path.expanduser("~/Skyler_test/SyncAI-Device-Workspace")
MAPS_REL = "src/syncai_planner/octo_planner_3d/maps"
MAPS_CONFIG_REL = "src/syncai_planner/octo_planner_3d/config/maps_config.yaml"
INI_REL = "config/instances/robot01.ini"
DEFAULT_POINTS = 12_800_000


class Deploy:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.changes: list[str] = []

    def log(self, msg: str) -> None:
        print(("[dry-run] " if self.dry_run else "") + msg)

    def do(self, msg: str, fn) -> None:
        self.log(msg)
        self.changes.append(msg)
        if not self.dry_run:
            fn()


def fail(msg: str) -> None:
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def default_name(bt_path: str) -> str:
    stem = os.path.splitext(os.path.basename(bt_path))[0]
    return re.sub(r"_handmade$", "", stem)


# ---------------------------------------------------------------- step 1: sample
def sample_surface(bt_path: str, out_pcd: str, points: int, seed: int, dep: Deploy,
                   skip_down: bool = False) -> None:
    t0 = time.time()
    leaves = octomap_io.read_octomap(bt_path, 0.5)
    centers = octomap_io.expand_leaves(leaves)
    res = leaves.res
    faces, dirs, _ = exposed_faces(centers, res, skip_down=skip_down)
    print(f"{bt_path}: res {res} m, occupied voxels {len(centers):,}, exposed faces {len(faces):,}"
          + (" (down-facing faces skipped)" if skip_down else ""))
    if dep.dry_run:
        dep.log(f"would write {out_pcd} ({points:,} points)")
        return
    rng = np.random.default_rng(seed)
    xyz = sample_points(faces, dirs, res, points, 0.0, rng)
    report(xyz)
    write_pcd(out_pcd, xyz.astype(np.float32))
    print(f"wrote {out_pcd} ({os.path.getsize(out_pcd) / 1e6:.0f} MB) in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------- step 5: maps_config.yaml
def update_maps_config(path: str, name: str, pcd_name: str, bt_name: str, dep: Deploy) -> None:
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    # locate `maps:` then the `  <name>:` block under it
    maps_i = next((i for i, l in enumerate(lines) if l.rstrip() == "maps:"), None)
    if maps_i is None:
        fail(f"{path}: no top-level 'maps:' key")
    block_re = re.compile(rf"^  {re.escape(name)}:\s*$")
    entry_re = re.compile(r"^  \S+:\s*$")
    pcd_re = re.compile(r'^(\s+pcd_path:\s*")([^"]*)(".*)$')
    start = next((i for i in range(maps_i + 1, len(lines)) if block_re.match(lines[i])), None)

    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if entry_re.match(lines[i]) or
                    (lines[i] and not lines[i].startswith(" "))), len(lines))
        for i in range(start + 1, end):
            m = pcd_re.match(lines[i])
            if m:
                old = m.group(2)
                new = os.path.join(os.path.dirname(old), pcd_name)
                if old == new:
                    dep.log(f"{os.path.relpath(path)}: maps.{name}.pcd_path already {pcd_name}")
                    return
                lines[i] = m.group(1) + new + m.group(3)

                def write(lines=lines):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                dep.do(f"{os.path.relpath(path)}: maps.{name}.pcd_path {os.path.basename(old)} -> {pcd_name}", write)
                return
        fail(f"{path}: block maps.{name} has no pcd_path line")

    # no block: add one modelled on the first existing entry's prefix
    first_pcd = next((pcd_re.match(l) for l in lines[maps_i + 1:] if pcd_re.match(l)), None)
    if first_pcd is None:
        fail(f"{path}: cannot infer the maps directory prefix (no pcd_path under maps:)")
    prefix = os.path.dirname(os.path.dirname(first_pcd.group(2)))  # .../maps
    d = f"{prefix}/{name}"
    block = [f"  {name}:",
             f'    pcd_path: "{d}/{pcd_name}"',
             f'    bt_path: "{d}/{bt_name}"',
             f'    layers_path: "{d}/{name}.layers"']
    lines[maps_i + 1:maps_i + 1] = block

    def write_new(lines=lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    dep.do(f"{os.path.relpath(path)}: add maps.{name} block -> {d}/{pcd_name}", write_new)


# ---------------------------------------------------------------- step 6: robot01.ini
def update_ini(path: str, name: str, pcd_name: str, force: bool, dep: Deploy) -> None:
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    sec = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            sec = s[1:-1]
            continue
        if sec != "3Dmap":
            continue
        m = re.match(r"^(pcd\s*[:=]\s*)(\S+)\s*$", line)
        if not m:
            continue
        old = m.group(2)
        in_map_dir = os.path.basename(os.path.dirname(old)) == name
        if not in_map_dir and not force:
            dep.log(f"{os.path.relpath(path)}: [3Dmap] pcd points to {old}, not into maps/{name}/ -> left alone "
                    "(use --force-ini to override)")
            return
        new_dir = os.path.dirname(old) if in_map_dir else os.path.join(os.path.dirname(os.path.dirname(old)), name)
        new = os.path.join(new_dir, pcd_name)
        if old == new:
            dep.log(f"{os.path.relpath(path)}: [3Dmap] pcd already {pcd_name}")
            return
        lines[i] = m.group(1) + new

        def write(lines=lines):
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        dep.do(f"{os.path.relpath(path)}: [3Dmap] pcd {os.path.basename(old)} -> {pcd_name}", write)
        return
    fail(f"{path}: no 'pcd:' line in [3Dmap]")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bt", help="hand-edited OctoMap (.bt)")
    ap.add_argument("--name", help="map name in maps_config.yaml / maps/<name>/ (default: bt stem minus _handmade)")
    ap.add_argument("--ws", default=DEFAULT_WS, help=f"SyncAI-Device-Workspace checkout (default {DEFAULT_WS})")
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS, help=f"surface points (default {DEFAULT_POINTS:,})")
    ap.add_argument("--out", help="surface PCD to write (default Maps/<bt stem>_surface.pcd)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-down", action="store_true",
                    help="do not sample faces that point down (undersides of floor slabs and ceilings). "
                         "Without it a one-voxel floor contributes its bottom face too, and the local "
                         "elevation map ends up with 0.2 m holes next to a 0.2 m ledge (2026-10-22)")
    ap.add_argument("--no-ini", action="store_true", help="do not touch robot01.ini")
    ap.add_argument("--force-ini", action="store_true", help="rewrite [3Dmap] pcd even if it points to another map")
    ap.add_argument("--dry-run", action="store_true", help="print what would be done, change nothing")
    args = ap.parse_args(argv)

    dep = Deploy(args.dry_run)
    bt = os.path.abspath(args.bt)
    if not os.path.isfile(bt):
        fail(f"{args.bt} not found")
    name = args.name or default_name(bt)
    stem = os.path.splitext(os.path.basename(bt))[0]
    out_pcd = os.path.abspath(args.out or os.path.join(os.path.dirname(bt), f"{stem}_surface.pcd"))
    pcd_name = os.path.basename(out_pcd)
    bt_name = os.path.basename(bt)

    ws = os.path.abspath(os.path.expanduser(args.ws))
    maps_dir = os.path.join(ws, MAPS_REL, name)
    maps_config = os.path.join(ws, MAPS_CONFIG_REL)
    ini = os.path.join(ws, INI_REL)
    for p in (maps_config, ini):
        if not os.path.isfile(p):
            fail(f"{p} not found (is --ws right?)")
    dst_pcd = os.path.join(maps_dir, pcd_name)
    dst_bt = os.path.join(maps_dir, bt_name)
    dst_layers = os.path.join(maps_dir, f"{name}.layers")

    print(f"map '{name}': {bt} -> {dst_pcd}")

    # 1. sample
    sample_surface(bt, out_pcd, args.points, args.seed, dep, skip_down=args.skip_down)

    # 2. copy pcd
    if not os.path.isdir(maps_dir):
        dep.do(f"mkdir {maps_dir}", lambda: os.makedirs(maps_dir))
    if os.path.realpath(out_pcd) == os.path.realpath(dst_pcd):
        dep.log(f"surface PCD already written in place: {dst_pcd}")
    else:
        dep.do(f"copy {os.path.relpath(out_pcd)} -> {dst_pcd}", lambda: shutil.copyfile(out_pcd, dst_pcd))

    # 3. .bt protection
    if os.path.realpath(bt) == os.path.realpath(dst_bt):
        dep.log(f"source .bt is the planner's copy: {dst_bt}")
    elif not os.path.exists(dst_bt):
        dep.do(f"copy {bt_name} -> {dst_bt} (planner copy did not exist)", lambda: shutil.copyfile(bt, dst_bt))
    elif filecmp.cmp(bt, dst_bt, shallow=False):
        dep.log(f"planner .bt identical to source, left untouched: {dst_bt}")
    else:
        fail(f"planner .bt differs from source and will NOT be overwritten:\n"
             f"  source : {bt}\n  planner: {dst_bt}\n"
             f"Decide which one is the hand-edited version. To deploy the planner's copy, run:\n"
             f"  python3 scripts/deploy_octomap_surface.py {dst_bt} --name {name}")

    # 4. touch bt + layers so the planner keeps the .bt instead of rebuilding it from the PCD
    def touch_all():
        now = time.time() + 2.0  # strictly newer than the PCD even on coarse filesystems
        os.utime(dst_bt, (now, now))
        if os.path.exists(dst_layers):
            os.utime(dst_layers, (now, now))
        pcd_t = os.path.getmtime(dst_pcd)
        if not os.path.getmtime(dst_bt) > pcd_t:
            fail(f"{dst_bt} is not newer than {dst_pcd}; the planner would overwrite the .bt")
        if os.path.exists(dst_layers) and not os.path.getmtime(dst_layers) >= pcd_t:
            fail(f"{dst_layers} is older than {dst_pcd}; the planner would rebuild the layers")
    layers_note = "" if os.path.exists(dst_layers) else " (no .layers yet, planner will build it once)"
    dep.do(f"touch {bt_name} + {name}.layers so they are newer than {pcd_name}{layers_note}", touch_all)

    # 5. maps_config.yaml
    update_maps_config(maps_config, name, pcd_name, bt_name, dep)

    # 6. robot01.ini
    if args.no_ini:
        dep.log("robot01.ini skipped (--no-ini)")
    else:
        update_ini(ini, name, pcd_name, args.force_ini, dep)

    # 7. summary
    print("\nsummary:")
    for c in dep.changes:
        print("  - " + c)
    if not dep.changes:
        print("  (nothing to change)")
    if not dep.dry_run:
        for p in (dst_pcd, dst_bt, dst_layers):
            if os.path.exists(p):
                print(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p)))}  "
                      f"{os.path.getsize(p):>12,}  {p}")
    print("restart the octo planner: its log must say 'Found up-to-date OctoMap file. Loading directly'\n"
          "and must NOT say 'Source PCD is newer than cache'. If the planner runs in a container with a\n"
          "stale ini mount, relaunch with system_config:=config/instances/robot01.ini.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
