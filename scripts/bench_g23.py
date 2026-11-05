#!/usr/bin/env python3
"""Phase 2 go/no-go: can the Jetson step the G23 legged model at 1 kHz on a map scene with floor proxies?

Loads robot/g23/mjcf/g23_mjcf.xml with MjSpec, drops its flat floor, adds a pool of mocap proxy boxes that
ProxyPool moves onto the scene's contact boxes (contact.npy, floors included) around the robot, holds the
default standing pose with a joint PD, and steps the physics at 1 kHz with the proxies refreshed at 50 Hz.
No LiDAR, no ROS, no controller: this is the physics cost alone.

  docker/run.sh "python3 scripts/bench_g23.py --pool 128 256 512 --seconds 5"
  docker/run.sh "python3 scripts/bench_g23.py --scene scenes/dp_1008 --spawn 0,0,-0.43 --margin 0.6"

Prints, per pool size: wall time per simulated second (RTF), physics vs proxy-update split, mean / max
contacts, how often the window overflowed the pool, and whether the robot was still standing at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quadsim.config import load_config  # noqa: E402
from quadsim.proxies import BoxIndex, ProxyPool  # noqa: E402

G23_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot", "g23", "mjcf", "g23_mjcf.xml")
DEFAULT_Q = np.array([0.0, -0.65, 1.3] * 4)


def build(n_proxies: int, timestep: float, spawn, keep_floor: bool = False):
    spec = mujoco.MjSpec.from_file(G23_XML)
    spec.option.timestep = timestep
    if not keep_floor:
        spec.delete(spec.geom("floor"))
    torso = spec.body("TORSO")
    torso.pos = [spawn[0], spawn[1], spawn[2] + 0.42]
    for i in range(n_proxies):
        b = spec.worldbody.add_body(name=f"proxy_{i}", mocap=True, pos=[0, 0, -1000 - 10 * i])
        b.add_geom(name=f"proxy_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05],
                   contype=1, conaffinity=0, group=3, rgba=[0.9, 0.3, 0.3, 0.6])
    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


def run(scene: str, spawn, pool: int, seconds: float, margin: float, timestep: float, kp: float, kd: float,
        flat: bool = False) -> dict:
    boxes = np.load(os.path.join(scene, "boxes.npy"))
    cp = os.path.join(scene, "contact.npy")
    contact = np.load(cp) if os.path.exists(cp) else np.ones(len(boxes), dtype=bool)
    index = BoxIndex(boxes[contact])
    m, d = build(pool, timestep, spawn, keep_floor=flat)
    mujoco.mj_resetData(m, d)
    jq = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint", "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint", "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"]])
    jv = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint", "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint", "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"]])
    d.qpos[jq] = DEFAULT_Q
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
    proxies = ProxyPool(m, d, index, pool) if not flat else None
    ctrl_every = int(round(0.02 / timestep))          # 50 Hz proxy refresh
    n_steps = int(round(seconds / timestep))
    t_phys = t_prox = 0.0
    ncon = []
    n_win = []
    mujoco.mj_forward(m, d)
    for k in range(n_steps):
        if proxies is not None and k % ctrl_every == 0:
            t0 = time.perf_counter()
            c = d.xpos[torso]
            n_win.append(proxies.update(c, 0.35, c[2] - 0.7, c[2] + 0.4, margin))
            t_prox += time.perf_counter() - t0
        d.ctrl[:] = kp * (DEFAULT_Q - d.qpos[jq]) - kd * d.qvel[jv]
        t0 = time.perf_counter()
        mujoco.mj_step(m, d)
        t_phys += time.perf_counter() - t0
        if k % 10 == 0:
            ncon.append(d.ncon)
    z = float(d.xpos[torso][2])
    up = float(d.xmat[torso].reshape(3, 3)[2, 2])
    return {"pool": pool, "sim_s": seconds, "wall_s": round(t_phys + t_prox, 3),
            "rtf": round(seconds / (t_phys + t_prox), 2), "physics_ms_per_sim_s": round(1000 * t_phys / seconds, 1),
            "proxy_ms_per_sim_s": round(1000 * t_prox / seconds, 1), "ncon_mean": round(float(np.mean(ncon)), 1),
            "ncon_max": int(np.max(ncon)), "window_boxes_mean": round(float(np.mean(n_win)), 1) if n_win else None,
            "window_boxes_max": int(np.max(n_win)) if n_win else None,
            "overflow": proxies.overflow if proxies is not None else 0, "ngeom": m.ngeom,
            "torso_z": round(z, 3), "torso_up": round(up, 3), "standing": bool(up > 0.9 and z > spawn[2] + 0.2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="scene directory (default: config/sim.yaml scene)")
    ap.add_argument("--spawn", default=None, help="x,y,z_floor (default: config spawn x,y and its z_floor)")
    ap.add_argument("--pool", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.6, help="proxy window beyond the body footprint (m)")
    ap.add_argument("--timestep", type=float, default=0.001)
    ap.add_argument("--kp", type=float, default=60.0)
    ap.add_argument("--kd", type=float, default=2.0)
    ap.add_argument("--flat", action="store_true", help="baseline: the model's own flat floor, no map, no proxies")
    a = ap.parse_args(argv)
    cfg = load_config(None, [])
    scene = a.scene or cfg["scene"]
    if a.spawn:
        spawn = [float(t) for t in a.spawn.split(",")]
    else:
        s = cfg["spawn"]
        spawn = [s[0], s[1], s[3] + 0.05]
    print(f"scene {scene}, spawn {spawn}, timestep {a.timestep}, margin {a.margin}, {a.seconds} s each")
    if a.flat:
        r = run(scene, [0, 0, 0], 0, a.seconds, a.margin, a.timestep, a.kp, a.kd, flat=True)
        print("flat floor, no proxies:", r)
    for pool in a.pool:
        r = run(scene, spawn, pool, a.seconds, a.margin, a.timestep, a.kp, a.kd)
        print(f"pool {pool:4d}: RTF {r['rtf']:5.2f}  physics {r['physics_ms_per_sim_s']:6.1f} ms/s  proxies "
              f"{r['proxy_ms_per_sim_s']:5.1f} ms/s  ncon {r['ncon_mean']:.1f}/{r['ncon_max']}  window "
              f"{r['window_boxes_mean']}/{r['window_boxes_max']} boxes  overflow {r['overflow']}  "
              f"geoms {r['ngeom']}  torso z {r['torso_z']} up {r['torso_up']}  standing {r['standing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
