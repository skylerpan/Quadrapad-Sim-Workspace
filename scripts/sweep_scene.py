#!/usr/bin/env python3
"""Build a map with several build_scene parameter sets and evaluate each with scripts/eval_scene.py.

Runs inside the sim container (eval_scene needs Embree + MuJoCo):
  docker/run.sh "python3 scripts/sweep_scene.py --pcd Maps/dp_1012_fixed.pcd --prefix dp_1012 --stage 1 \\
                 --stairs-box=-46,-30,-3,11 --stairs-box=-5,-2,-18,-14 --csv results/sweep_dp_1012/stage1.csv"
  docker/run.sh "python3 scripts/sweep_scene.py ... --stage 2 --variant 'tol05:--ground-tol 0.05' --variant 'fill16:--ground-fill 16'"

Every variant is built into <out>/<prefix>_<tag>/ (skipped when meta.json already records the same
build_args), evaluated (eval.json), and one row per variant goes to the CSV: build flags, ground stats,
stair metrics per box, walkability and spawn. The CSV is rewritten from all evaluated variants, so
re-runs and later stages merge into it. --only tag,tag limits the run; --variant adds ad-hoc variants.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")

STAIR = "--ground-min-cells 12 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3"
STAGES = {
    1: [
        ("base", ""),
        ("mc12", "--ground-min-cells 12"),
        ("mc12_guard", "--ground-min-cells 12 --ground-fill-guard 0.3"),
        ("mc12_guard_med", "--ground-min-cells 12 --ground-fill-guard 0.3 --ground-surface median"),
        ("mc12_guard_peel", "--ground-min-cells 12 --ground-fill-guard 0.3 --ground-peel 3"),
        ("stair", STAIR),
        ("stair_l1", STAIR + " --ground-layers 1"),
        ("stair_mc8", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 8")),
        ("stair_ma1", STAIR + " --ground-max-above 1"),
        ("stair_tol05", STAIR + " --ground-tol 0.05"),
    ],
    2: [            # around the stage-1 findings: treads come out as small ramps -> snap small regions flat
        ("flat60", STAIR + " --ground-small-flat 60"),
        ("mc8_flat60", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 8") + " --ground-small-flat 60"),
        ("mc8_flat60_l1", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 8") + " --ground-small-flat 60 --ground-layers 1"),
        ("mc8_flat60_tol05", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 8") + " --ground-small-flat 60 --ground-tol 0.05"),
        ("mc8_flat100", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 8") + " --ground-small-flat 100"),
        ("mc6_flat60", STAIR.replace("--ground-min-cells 12", "--ground-min-cells 6") + " --ground-small-flat 60"),
    ],
}

# CLI flag -> build() kwarg name, to compare with meta.build_args
FLAG2ARG = {"--voxel": "voxel", "--min-pts": "min_pts", "--min-neighbors": "min_neighbors", "--crop-pct": "crop_pct",
            "--z-min": "z_min", "--z-max": "z_max", "--ground-tol": "ground_tol", "--ground-max-rms": "ground_max_rms",
            "--ground-thickness": "ground_thickness", "--ground-fill": "ground_fill", "--ground-fill-nb": "ground_fill_nb",
            "--ground-max-above": "ground_max_above", "--ground-min-cells": "ground_min_cells",
            "--ground-layers": "ground_layers", "--ground-grow": "ground_grow", "--ground-floor-tol": "ground_floor_tol",
            "--ground-fill-guard": "ground_fill_guard", "--ground-surface": "ground_surface", "--ground-peel": "ground_peel",
            "--ground-small-flat": "ground_small_flat"}
DEFAULTS = {"voxel": 0.1, "min_pts": 1, "min_neighbors": 2, "crop_pct": 0.05, "z_min": None, "z_max": None,
            "ground_tol": 0.06, "ground_max_rms": 0.035, "ground_thickness": 0.2, "ground_fill": 12, "ground_fill_nb": 3,
            "ground_max_above": 2, "ground_min_cells": 30, "ground_layers": 2, "ground_grow": 3, "ground_floor_tol": 0.15,
            "ground_fill_guard": 0.0, "ground_surface": "mean", "ground_peel": 0, "ground_small_flat": 0}


def flags_to_args(flags: str) -> dict:
    args = dict(DEFAULTS)
    toks = shlex.split(flags)
    i = 0
    while i < len(toks):
        name = toks[i]
        if name not in FLAG2ARG:
            raise SystemExit(f"unknown build flag {name}")
        raw = toks[i + 1]
        key = FLAG2ARG[name]
        cur = DEFAULTS[key]
        args[key] = raw if isinstance(cur, str) else (int(raw) if isinstance(cur, int) and not isinstance(cur, bool)
                                                       and "." not in raw else float(raw))
        i += 2
    return args


def same_build(meta_path: str, want: dict, pcd: str) -> bool:
    if not os.path.exists(meta_path):
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    have = meta.get("build_args") or {}
    if os.path.basename(have.get("pcd", "")) != os.path.basename(pcd):
        return False
    for k, v in want.items():
        hv = have.get(k)
        if isinstance(v, float) or isinstance(hv, float):
            if hv is None or abs(float(hv) - float(v)) > 1e-9:
                return False
        elif hv != v:
            return False
    try:
        st = os.stat(pcd)
        return st.st_size == meta.get("source_pcd_size") and abs(st.st_mtime - meta.get("source_pcd_mtime", 0)) < 1e-3
    except OSError:
        return False


def run(cmd: list[str], log: str) -> int:
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return p.returncode


def row_from_eval(tag: str, flags: str, ev: dict, build_s) -> dict:
    g = ev.get("ground") or {}
    r = {"tag": tag, "flags": flags, "build_s": build_s}
    for k in ("planar_fraction", "plane_rms_m", "regions", "planes", "cells", "cells_grown", "cells_filled",
              "cells_guarded", "slabs"):
        r[k] = g.get(k)
    r["boxes"] = ev.get("boxes")
    sp = ev.get("spawn") or {}
    r["spawn_ok"], r["spawn_moved"], r["spawn_floor_z"] = sp.get("ok"), sp.get("moved"), sp.get("floor_z")
    w = ev.get("walk") or {}
    r["reach_m2"] = w.get("reachable_m2")
    r["walk_samples"] = w.get("samples")
    r["walk_valid"] = w.get("valid_samples")
    r["walk_edge"] = w.get("edge_fraction")
    r["walk_fail"] = w.get("fail_fraction")
    r["walk_foot_gap"] = w.get("unsupported_foot_fraction")
    for k in ("no_ground", "step_up", "step_down", "slope", "collision"):
        r[f"walk_{k}"] = (w.get("events") or {}).get(k)
    for i, s in enumerate(ev.get("stairs") or []):
        p = f"s{i}_"
        for k in ("n_levels", "n_rises", "rise_mean", "rise_std", "rise_min", "rise_max", "off_level_fraction",
                  "ramp_fraction", "coverage", "residual_p95", "buried_fraction", "slab_cells"):
            r[p + k] = s.get(k)
    for i, s in enumerate(ev.get("stairs_walk") or []):
        p = f"s{i}_"
        for mode in ("config", "climb"):
            m = s.get(mode) or {}
            r[f"{p}{mode}_fail"] = m.get("fail_fraction")
            r[f"{p}{mode}_edge"] = m.get("edge_fraction")
            r[f"{p}{mode}_samples"] = m.get("valid_samples")
            r[f"{p}{mode}_events"] = json.dumps(m.get("events")) if m.get("events") else None
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--out", default="scenes/sweep")
    ap.add_argument("--prefix", default=None, help="scene name prefix (default: PCD basename)")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--variant", action="append", default=[], help="tag:build flags (adds to the stage)")
    ap.add_argument("--only", default=None, help="comma-separated tags to run")
    ap.add_argument("--stairs-box", action="append", default=[])
    ap.add_argument("--csv", default=None, help="summary CSV (default results/sweep_<prefix>/stage<N>.csv)")
    ap.add_argument("--eval-args", default="", help="extra eval_scene.py arguments (quoted)")
    ap.add_argument("--no-auto", action="store_true", help="skip the reachable-floor walk")
    ap.add_argument("--force", action="store_true", help="rebuild even if meta.build_args match")
    a = ap.parse_args(argv)

    prefix = a.prefix or os.path.splitext(os.path.basename(a.pcd))[0]
    variants = list(STAGES.get(a.stage, []))
    for v in a.variant:
        tag, _, flags = v.partition(":")
        variants.append((tag.strip(), flags.strip()))
    if a.only:
        want = {t.strip() for t in a.only.split(",")}
        variants = [v for v in variants if v[0] in want]
    if not variants:
        raise SystemExit("no variants to run")
    csv_path = a.csv or os.path.join("results", f"sweep_{prefix}", f"stage{a.stage}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(os.path.dirname(csv_path), f"stage{a.stage}.log")

    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows[r["tag"]] = r
    for tag, flags in variants:
        scene = os.path.join(a.out, f"{prefix}_{tag}")
        meta = os.path.join(scene, "meta.json")
        want = flags_to_args(flags)
        t0 = time.time()
        if a.force or not same_build(meta, want, a.pcd):
            print(f"[{tag}] build {flags or '(defaults)'}", flush=True)
            rc = run([sys.executable, "-m", "pcd2sim.build_scene", "--pcd", a.pcd, "--out", scene] + shlex.split(flags), log)
            if rc != 0:
                print(f"[{tag}] BUILD FAILED (see {log})", flush=True)
                rows[tag] = {"tag": tag, "flags": flags, "build_s": "FAILED"}
                continue
            build_s = round(time.time() - t0, 1)
        else:
            print(f"[{tag}] build up to date", flush=True)
            with open(meta) as f:
                build_s = json.load(f).get("build_seconds")
        cmd = [sys.executable, os.path.join(HERE, "eval_scene.py"), "--scene", scene, "--pcd", a.pcd,
               "--json", os.path.join(scene, "eval.json")]
        if not a.no_auto:
            cmd.append("--auto")
        for b in a.stairs_box:
            cmd.append(f"--stairs-box={b}")
        cmd += shlex.split(a.eval_args)
        t1 = time.time()
        rc = run(cmd, log)
        if rc != 0:
            print(f"[{tag}] EVAL FAILED (see {log})", flush=True)
            rows[tag] = {"tag": tag, "flags": flags, "build_s": build_s, "eval": "FAILED"}
            continue
        with open(os.path.join(scene, "eval.json")) as f:
            ev = json.load(f)
        rows[tag] = row_from_eval(tag, flags, ev, build_s)
        r = rows[tag]
        stairs = " ".join(f"s{i}:{r.get(f's{i}_n_levels')}lv/{r.get(f's{i}_n_rises')}r/sd{r.get(f's{i}_rise_std')}/"
                          f"ramp{r.get(f's{i}_ramp_fraction')}/cov{r.get(f's{i}_coverage')}/bur{r.get(f's{i}_buried_fraction')} "
                          f"cfg {r.get(f's{i}_config_fail')} climb {r.get(f's{i}_climb_fail')}"
                          for i in range(len(a.stairs_box)))
        print(f"[{tag}] planar {r['planar_fraction']} rms {r['plane_rms_m']} regions {r['regions']} | walk fail "
              f"{r['walk_fail']} edge {r['walk_edge']} gap {r['walk_foot_gap']} reach {r['reach_m2']} m2 | {stairs} | "
              f"eval {time.time() - t1:.0f}s",
              flush=True)
        keys = []
        for rr in rows.values():
            for k in rr:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for rr in rows.values():
                wr.writerow(rr)
    print(f"summary: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
