"""Simulator core (no ROS): MuJoCo physics + ideal base + rolling LiDAR + IMU."""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .base import IdealBase, body_lift, rot_zyx
from .config import load_scene_meta
from .lidar import LidarFrame, RollingLidar
from .model import add_map_mesh, build_mjcf
from .proxies import BoxIndex, ProxyPool
from .raycast import RayScene, VoxelLookup

G = 9.81
SCENE_FORMAT = 2          # bump with pcd2sim.build_scene.SCENE_FORMAT when scene outputs change


def _warn_stale_scene(scene_dir: str, meta: dict):
    """Scenes are a build cache: warn (don't fail) if built by older code or from a changed PCD."""
    why = []
    if meta.get("format_version", 1) < SCENE_FORMAT:
        why.append(f"format {meta.get('format_version', 1)} < {SCENE_FORMAT}")
    src = meta.get("source_pcd")
    if src and os.path.exists(src) and "source_pcd_mtime" in meta:
        st = os.stat(src)
        if st.st_size != meta.get("source_pcd_size") or abs(st.st_mtime - meta["source_pcd_mtime"]) > 1e-3:
            why.append(f"{src} changed since the build")
    if why:
        warnings.warn(f"scene {scene_dir} looks stale ({'; '.join(why)}); rebuild with scripts/build_scenes.sh",
                      stacklevel=3)


@dataclass
class ImuSample:
    t: float
    acc: np.ndarray    # sensor frame, unit per cfg (g or m/s^2)
    gyro: np.ndarray   # rad/s


@dataclass
class TickOutput:
    t: float
    imu: list[ImuSample] = field(default_factory=list)
    frames: list[LidarFrame] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    clock: list[float] = field(default_factory=list)


class Simulator:
    def __init__(self, cfg: dict, visual_map: bool = False):
        self.cfg = cfg
        scene_dir = cfg["scene"]
        self.meta = load_scene_meta(scene_dir)
        _warn_stale_scene(scene_dir, self.meta)
        t0 = time.time()
        boxes = np.load(os.path.join(scene_dir, "boxes.npy"))
        ground = os.path.join(scene_dir, "ground.npz")      # flattened floor (absent in older scenes)
        slabs = np.load(ground)["slabs"] if os.path.exists(ground) else None
        self.rays = RayScene(boxes, slabs)
        self.voxels = VoxelLookup(os.path.join(scene_dir, "voxels.npz"))
        self.index = BoxIndex(boxes)
        self.load_seconds = time.time() - t0

        r = cfg["robot"]
        self.half = np.asarray(r["body_size"]) / 2.0
        spawn = self._spawn_pose()
        self.base = IdealBase(r, self.rays, self.voxels, blocked=self._blocked)
        # Kept for the ROS front-end: the requested spawn can differ from where the base ends up
        # (no ground near z_floor, or the body overlapped the map and was moved to a free pose).
        self.spawn_requested = spawn
        self.spawn_ok, self.spawn_moved = self._reset_clear(*spawn)
        self.spawn_floor_z = self.base.ground.z0
        pos, quat = self.base.target_pose()

        xml = build_mjcf(cfg, self.meta["gravity"], pos, r["collision_pool"])
        if visual_map:                      # only for the viewer: headless runs keep the 130-geom model
            spec = mujoco.MjSpec.from_string(xml)
            add_map_mesh(spec, boxes, slabs)
            self.m = spec.compile()
        else:
            self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.proxies = ProxyPool(self.m, self.d, self.index, r["collision_pool"])
        self.base_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
        self.lidar_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.mocap_target = self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.acc_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")]
        self.gyro_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]

        self.lidar = RollingLidar(cfg["lidar"], self.rays, self.voxels, self.meta["voxel"])
        ph = cfg["physics"]
        self.dt = ph["timestep"]
        self.ctrl_steps = self._steps_per(ph["control_hz"], "physics.control_hz")
        self.imu_steps = self._steps_per(cfg["imu"]["rate_hz"], "imu.rate_hz")
        self.clock_steps = self._steps_per(cfg["ros"]["clock_hz"], "ros.clock_hz")
        self.rng = np.random.default_rng(cfg["lidar"].get("seed", 0) + 1)
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.cmd = np.zeros(3)
        self.cmd_time = -1e9
        self.in_collision = False
        self.last_stuck = (None, -1e9)
        self.step_count = 0
        self.timing = {"physics": 0.0, "lidar": 0.0, "control": 0.0}
        self._place(pos, quat)

    # -- setup -----------------------------------------------------------------
    def _steps_per(self, hz: float, name: str) -> int:
        steps = 1.0 / (hz * self.dt)
        if abs(steps - round(steps)) > 1e-6 or round(steps) < 1:
            raise ValueError(f"{name}={hz} Hz is not an integer number of physics steps "
                             f"(timestep {self.dt}s -> {steps:.3f} steps)")
        return int(round(steps))

    def _spawn_pose(self):
        s = self.cfg.get("spawn")
        meta_spawn = self.meta.get("spawn")
        if s is not None:
            z = meta_spawn[2] if meta_spawn else 0.0
            return float(s[0]), float(s[1]), float(s[2] if len(s) > 2 else 0.0), float(s[3] if len(s) > 3 else z)
        if meta_spawn is None:
            raise RuntimeError("scene has no spawn point; set `spawn: [x, y, yaw, z_floor]` in the config")
        return float(meta_spawn[0]), float(meta_spawn[1]), 0.0, float(meta_spawn[2])

    def _place(self, pos, quat):
        self.d.qpos[0:3] = pos
        self.d.qpos[3:7] = quat
        self.d.qvel[:] = 0.0
        self.d.mocap_pos[self.mocap_target] = pos
        self.d.mocap_quat[self.mocap_target] = quat
        self._refresh_proxies(pos)
        mujoco.mj_forward(self.m, self.d)
        self.lidar.reset(self.d.time)
        self.lidar.record_pose(self.d.time, self.d.site_xpos[self.lidar_site], self.d.site_xmat[self.lidar_site].reshape(3, 3))

    def set_pose(self, x: float, y: float, yaw: float, z_floor_hint: float | None = None) -> tuple[bool, bool]:
        """Teleport the base. Returns (ground found, moved to a free pose instead of (x, y, yaw))."""
        z = self.base.ground.z0 if z_floor_hint is None else z_floor_hint
        ok, moved = self._reset_clear(x, y, yaw, z)
        pos, quat = self.base.target_pose()
        self._place(pos, quat)
        return ok, moved

    def _reset_clear(self, x: float, y: float, yaw: float, z_floor_hint: float) -> tuple[bool, bool]:
        """base.reset, moved to the nearest free pose (spiral <= 1 m, 4 yaws) if the body would overlap
        the map there; warns when it moves the base or finds no free pose.
        Returns (ground found, base moved away from the requested pose)."""
        ok = self.base.reset(x, y, yaw, z_floor_hint)
        r_cfg = self.cfg["robot"]
        blocked = lambda px, py, pyaw: self._blocked(
            px, py, pyaw, self.base.ground.z0 + body_lift(r_cfg, self.base.ground.roll, self.base.ground.pitch),
            self.base.ground.roll, self.base.ground.pitch)
        if not blocked(x, y, yaw):
            return ok, False
        for r in np.arange(0.1, 1.01, 0.1):
            n = max(8, int(2 * np.pi * r / 0.1))
            for a in np.arange(n) * 2 * np.pi / n:
                cx, cy = x + r * np.cos(a), y + r * np.sin(a)
                for dyaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                    if self.base.reset(cx, cy, yaw + dyaw, z_floor_hint) and not blocked(cx, cy, yaw + dyaw):
                        warnings.warn(f"pose ({x:.2f}, {y:.2f}, yaw {yaw:.2f}) overlaps the map; moved to "
                                      f"({cx:.2f}, {cy:.2f}, yaw {yaw + dyaw:.2f})", stacklevel=3)
                        return True, True
        self.base.reset(x, y, yaw, z_floor_hint)
        warnings.warn(f"pose ({x:.2f}, {y:.2f}) overlaps the map and no free pose within 1 m", stacklevel=3)
        return ok, False

    def set_cmd(self, vx: float, vy: float, wz: float):
        self.cmd = np.array([vx, vy, wz], dtype=np.float64)
        self.cmd_time = self.d.time

    # -- stepping --------------------------------------------------------------
    def _refresh_proxies(self, pos):
        r = self.cfg["robot"]
        half_xy = float(np.hypot(self.half[0], self.half[1]))
        z_lo = pos[2] - self.half[2] - 0.05
        z_hi = pos[2] + self.half[2] + 0.3
        self.proxies.update(pos, half_xy, z_lo, z_hi, r["collision_margin"])

    def _blocked(self, x: float, y: float, yaw: float, z_center: float, roll: float = 0.0,
                 pitch: float = 0.0) -> bool:
        """Would the body box, oriented like the physical body (yaw, pitch, roll), overlap any map box?
        Oriented box vs axis-aligned boxes: AABB pre-query, then a vectorised separating-axis test.
        The box is the physical body minus 5 mm at the bottom (numeric slack): the bottom stands ~0.22 m
        above the feet plane, so the floor is never in it; a larger margin hides low obstacles MuJoCo
        still hits. Level (roll = pitch = 0) this is the yawed-rectangle / z-band test of before; on a
        flight of stairs the pitched box clears the next riser that a level box's front edge hit."""
        hx, hy, hz = self.half
        R = rot_zyx(yaw, pitch, roll)                                  # columns = body axes in world
        half = np.array([hx, hy, hz - 0.0025])
        centre = np.array([x, y, z_center]) + R[:, 2] * 0.0025
        ext = np.abs(R) @ half                                         # world AABB of the oriented box
        ids = self.index.query(centre - ext, centre + ext)
        if len(ids) == 0:
            return False
        b = self.index.boxes[ids]
        d = b[:, :3] - centre
        bh = b[:, 3:6]
        axes = [np.eye(3)[i] for i in range(3)] + [R[:, i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                L = np.cross(np.eye(3)[i], R[:, j])
                n = np.linalg.norm(L)
                if n > 1e-6:
                    axes.append(L / n)
        separated = np.zeros(len(ids), dtype=bool)
        for L in axes:
            r_body = float(np.sum(half * np.abs(R.T @ L)))
            r_map = bh @ np.abs(L)
            separated |= np.abs(d @ L) > r_body + r_map
        return bool((~separated).any())

    def control_tick(self) -> TickOutput:
        """Advance one control period (ctrl_steps physics steps)."""
        out = TickOutput(t=self.d.time)
        t0 = time.perf_counter()
        reason = self.base.update(self.ctrl_steps * self.dt, self.cmd, self.d.time - self.cmd_time)
        pos, quat = self.base.target_pose()
        self.d.mocap_pos[self.mocap_target] = pos
        self.d.mocap_quat[self.mocap_target] = quat
        self._refresh_proxies(self.d.qpos[0:3])
        if reason and (reason != self.last_stuck[0] or self.d.time - self.last_stuck[1] > 1.0):
            kind = "collision" if reason == "collision" else "stuck"
            out.events.append({"type": kind, "reason": reason, "t": self.d.time,
                               "pose": [self.base.x, self.base.y, self.base.yaw]})
            self.last_stuck = (reason, self.d.time)
        t1 = time.perf_counter()
        self.timing["control"] += t1 - t0

        imu_cfg = self.cfg["imu"]
        for _ in range(self.ctrl_steps):
            # mj_step evaluates kinematics/sensors at the current state, then integrates and advances
            # d.time: afterwards site_xpos/sensordata describe the state at t_state, not at d.time.
            t_state = self.d.time
            mujoco.mj_step(self.m, self.d)
            self.step_count += 1
            self.lidar.record_pose(t_state, self.d.site_xpos[self.lidar_site],
                                   self.d.site_xmat[self.lidar_site].reshape(3, 3))
            if self.step_count % self.imu_steps == 0:
                out.imu.append(self._imu_sample(imu_cfg, t_state))
            if self.step_count % self.clock_steps == 0:
                out.clock.append(self.d.time)
        t2 = time.perf_counter()
        self.timing["physics"] += t2 - t1

        out.frames = self.lidar.process()
        self.timing["lidar"] += time.perf_counter() - t2

        hit = self._base_contact()
        if hit is not None and not self.in_collision:
            out.events.append({"type": "collision", "reason": "contact", "t": self.d.time, "point": hit,
                               "pose": [self.base.x, self.base.y, self.base.yaw]})
        self.in_collision = hit is not None
        out.t = self.d.time
        return out

    def _imu_sample(self, c, t: float) -> ImuSample:
        dt = self.imu_steps * self.dt
        acc = self.d.sensordata[self.acc_adr:self.acc_adr + 3].copy()
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        self.acc_bias += self.rng.normal(0, c["acc_bias_rw"] * np.sqrt(dt), 3)
        self.gyro_bias += self.rng.normal(0, c["gyro_bias_rw"] * np.sqrt(dt), 3)
        acc += self.acc_bias + self.rng.normal(0, c["acc_noise"], 3)
        gyro += self.gyro_bias + self.rng.normal(0, c["gyro_noise"], 3)
        if c.get("acc_unit", "g") == "g":
            acc /= G
        return ImuSample(t, acc, gyro)

    def _base_contact(self):
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if con.geom1 == self.base_geom or con.geom2 == self.base_geom:
                return con.pos.tolist()
        return None

    # -- state -----------------------------------------------------------------
    def ground_truth(self):
        """(pos [3], quat wxyz [4], linear vel in body frame [3], angular vel in body frame [3])."""
        q = self.d.qpos
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q[3:7])          # from qpos: d.xmat still holds the pre-step pose
        v_body = R.reshape(3, 3).T @ self.d.qvel[0:3]
        return q[0:3].copy(), q[3:7].copy(), v_body, self.d.qvel[3:6].copy()

    def lidar_pose_in_base(self):
        return np.asarray(self.cfg["lidar"]["mount_xyz"], dtype=np.float64)
