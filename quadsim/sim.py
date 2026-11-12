"""Simulator core (no ROS): MuJoCo physics + ideal base + rolling LiDAR + IMU."""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from pcd2sim import SCENE_FORMAT

from .base import IdealBase, body_lift, rot_zyx
from .config import load_scene_meta
from .g23 import JOINTS, RefSimOptions, build_g23_spec
from .gnss import GnssFix, GnssModel, resolve_datum, sky_directions, sky_fraction
from .backends import make_backend
from .lidar import LidarFrame, RollingLidar
from .model import add_map_mesh, build_mjcf, nearest_boxes
from .proxies import BoxIndex, ProxyPool
from .raycast import RayScene, VoxelLookup

G = 9.81


def _check_scene_format(scene_dir: str, meta: dict) -> None:
    """meta.json format_version against pcd2sim.SCENE_FORMAT. Format 2 (no contact set) still loads for the
    ideal base, with a warning; anything else must be rebuilt."""
    v = meta.get("format_version")
    if v == SCENE_FORMAT:
        return
    if v == 2:
        warnings.warn(f"scene {scene_dir} is format 2 (no contact.npy): fine for the ideal base, a legged robot "
                      f"needs format {SCENE_FORMAT}; rebuild with pcd2sim.build_scene", stacklevel=3)
        return
    raise RuntimeError(f"scene {scene_dir} has format_version {v!r}; this quadsim reads format {SCENE_FORMAT} "
                       f"(and 2 for the ideal base). Rebuild it with pcd2sim.build_scene "
                       f"(flags in meta.json build_args) or make_stairs")


def _warn_stale_scene(scene_dir: str, meta: dict):
    """Scenes are a build cache: warn (don't fail) if the source map changed since the build."""
    why = []
    src = meta.get("source_pcd")
    if src and os.path.exists(src) and "source_pcd_mtime" in meta:
        st = os.stat(src)
        if st.st_size != meta.get("source_pcd_size") or abs(st.st_mtime - meta["source_pcd_mtime"]) > 1e-3:
            why.append(f"{src} changed since the build")
    if why:
        warnings.warn(f"scene {scene_dir} looks stale ({'; '.join(why)}); rebuild with pcd2sim.build_scene (flags in meta.json build_args)",
                      stacklevel=3)


@dataclass
class ImuSample:
    t: float
    acc: np.ndarray    # sensor frame, in g (like livox_ros_driver2)
    gyro: np.ndarray   # rad/s


@dataclass
class TickOutput:
    t: float
    imu: list[ImuSample] = field(default_factory=list)
    gnss: list[GnssFix] = field(default_factory=list)
    frames: list[LidarFrame] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    clock: list[float] = field(default_factory=list)


class Simulator:
    def __init__(self, cfg: dict, visual_map: bool = False):
        self.cfg = cfg
        scene_dir = cfg["scene"]
        self.meta = load_scene_meta(scene_dir)
        _check_scene_format(scene_dir, self.meta)
        _warn_stale_scene(scene_dir, self.meta)
        boxes = np.load(os.path.join(scene_dir, "boxes.npy"))
        contact_path = os.path.join(scene_dir, "contact.npy")   # format 3: boxes a legged robot can touch
        self.contact = np.load(contact_path) if os.path.exists(contact_path) else None
        ground = os.path.join(scene_dir, "ground.npz")      # flattened floor slabs (absent in --no-flatten scenes)
        slabs = np.load(ground)["slabs"] if os.path.exists(ground) else None
        self.boxes, self.slabs = boxes, slabs            # kept for the viewers (visual map)
        self.rays = RayScene(boxes, slabs)
        self.voxels = VoxelLookup(os.path.join(scene_dir, "voxels.npz"))
        self.index = BoxIndex(boxes)

        r = cfg["robot"]
        self.robot_type = str(r.get("type", "ideal"))
        if self.robot_type not in ("ideal", "g23"):
            raise ValueError(f"robot.type={self.robot_type!r}: expected 'g23' (legged) or 'ideal' (floating base)")
        self.legged = self.robot_type == "g23"
        self.half = np.asarray(r["body_size"]) / 2.0
        spawn = self._spawn_pose()
        # The ideal base is the terrain probe in both modes: it finds the floor under a pose and the free
        # pose nearest a blocked one (spawn, set_pose); in g23 mode it then just tracks the torso pose.
        self.base = IdealBase(r, self.rays, self.voxels, blocked=self._blocked)
        # Kept for the ROS front-end: the requested spawn can differ from where the base ends up
        # (no ground near z_floor, or the body overlapped the map and was moved to a free pose).
        self.spawn_requested = spawn
        self.spawn_ok, self.spawn_moved = self._reset_clear(*spawn)
        self.spawn_floor_z = self.base.ground.z0
        pos, quat = self.base.target_pose()

        if self.legged:
            if self.contact is None:
                raise RuntimeError(f"scene {scene_dir} has no contact.npy: a legged robot needs a format "
                                   f"{SCENE_FORMAT} scene (rebuild with pcd2sim.build_scene)")
            self.contact_index = BoxIndex(boxes[self.contact])       # floors included: the feet stand on them
            pos = np.array([pos[0], pos[1], self.base.ground.z0 + float(r.get("spawn_height", 0.40))])
            self.ref_opts = RefSimOptions.from_env(r)          # the reference sim's G2_SIM_* switches
            spec = build_g23_spec(cfg, self.meta["gravity"], pos, r["collision_pool"], self.ref_opts)
        else:
            self.contact_index = self.index
            spec = mujoco.MjSpec.from_string(build_mjcf(cfg, self.meta["gravity"], pos, r["collision_pool"]))
        if visual_map:                      # only for the viewer: headless runs keep the small model
            max_vis = int(cfg.get("viewer", {}).get("max_map_boxes", 400_000))
            # draw the boxes nearest the spawn (a 0.03 m map has millions)
            add_map_mesh(spec, nearest_boxes(boxes, pos, max_vis), slabs, max_boxes=max_vis)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.proxies = ProxyPool(self.m, self.d, self.contact_index, r["collision_pool"])
        self.lidar_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.gnss_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gnss")       # -1 without a gnss section
        if self.legged:
            self.base_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "TORSO")
            self.base_geom = -1
            self.mocap_target = None
            self.acc_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "lidar_acc")]
            self.gyro_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "lidar_gyro")]
            self.g23 = make_backend(self.m, self.d, r, self.base_body, self.ref_opts)
            self.fallen = False
            self.low_since = None
        else:
            self.base_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
            self.base_geom = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "base_geom")
            self.mocap_target = self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "target")]
            self.acc_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")]
            self.gyro_adr = self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
            self.g23 = None

        # legged mode: Embree runs on a worker thread so the 1 kHz loop never waits for a frame
        self.lidar = RollingLidar(cfg["lidar"], self.rays, self.voxels, self.meta["voxel"], threaded=self.legged)
        ph = cfg["physics"]
        self.dt = ph["timestep"]
        self.ctrl_steps = self._steps_per(ph["control_hz"], "physics.control_hz")
        self.imu_steps = self._steps_per(cfg["imu"]["rate_hz"], "imu.rate_hz")
        self.clock_steps = self._steps_per(cfg["ros"]["clock_hz"], "ros.clock_hz")
        self.rng = np.random.default_rng(cfg["lidar"].get("seed", 0) + 1)
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        # the GNSS receiver (quadsim/gnss.py): only with a datum; the antenna is the `gnss` site
        gn = cfg.get("gnss") or {}
        self.datum = resolve_datum(self.meta, gn.get("datum"))       # the scene's geo, else gnss.datum
        self.gnss = (GnssModel({**gn, "datum": asdict(self.datum)}, np.random.default_rng(cfg["lidar"].get("seed", 0) + 2))
                     if self.datum is not None else None)
        if self.gnss is not None:
            if self.gnss_site < 0:
                raise ValueError("gnss.datum is set but gnss.mount_xyz is not: the antenna has no place on the base")
            self.gnss_steps = self._steps_per(gn["rate_hz"], "gnss.rate_hz")
            self.sky_dirs = sky_directions(int(gn["sky"]["rays"]), float(gn["sky"]["half_angle_deg"]))
            self.sky_mode = str(gn["sky"].get("mode", "map"))
            if self.sky_mode not in ("map", "always", "none"):
                raise ValueError(f"gnss.sky.mode={self.sky_mode!r}: expected map, always or none")

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
        if self.legged:
            pos = np.array([pos[0], pos[1], self.base.ground.z0 + float(self.cfg["robot"].get("spawn_height", 0.40))])
            teleport = self.d.time > 0.0              # set_pose, not the spawn
            if teleport:
                self.g23.restart(self.d.time, teleport=True)   # damp the legs first: the gait must not flail mid-air
            self.g23.place(pos, quat)
            if not teleport:
                self.g23.restart(self.d.time)
            self.fallen = False
            self.low_since = None
        else:
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

    def set_policy_mode(self, mode: int) -> dict | None:
        """g23 RL policy switch (0 PPO, 1 HIMLoco ...); None in ideal mode."""
        return self.g23.set_policy_mode(mode, self.d.time) if self.legged else None

    def set_motion(self, mode: str) -> dict | None:
        """g23 operator command: stand | walk | lie | damp (quadsim.sequence.StandSequence.operator); None in ideal mode."""
        if not self.legged:
            return None
        if mode == "stand":
            self.fallen = False
            self.low_since = None
        return self.g23.operator(mode, self.d.time)

    def set_cmd(self, vx: float, vy: float, wz: float):
        self.cmd = np.array([vx, vy, wz], dtype=np.float64)
        self.cmd_time = self.d.time

    # -- stepping --------------------------------------------------------------
    def _refresh_proxies(self, pos):
        r = self.cfg["robot"]
        if self.legged:                     # torso centre; the window reaches the feet and a step below them
            self.proxies.update(pos, 0.35, pos[2] - 0.75, pos[2] + 0.4, r["collision_margin"])
            return
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
        if self.legged:
            return self._control_tick_g23()
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
            if self.gnss is not None and self.step_count % self.gnss_steps == 0:
                out.gnss.append(self._gnss_sample(t_state))
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

    def _control_tick_g23(self) -> TickOutput:
        """Legged robot: the controller drives the joints over UDP; quadsim steps, senses and reports."""
        out = TickOutput(t=self.d.time)
        r = self.cfg["robot"]
        t0 = time.perf_counter()
        fresh = self.d.time - self.cmd_time <= r["cmd_timeout"]
        out.events += self.g23.tick(self.d.time, self.cmd if fresh else np.zeros(3), cmd_fresh=fresh)
        self._refresh_proxies(self.d.qpos[0:3])
        t1 = time.perf_counter()
        self.timing["control"] += t1 - t0

        imu_cfg = self.cfg["imu"]
        for _ in range(self.ctrl_steps):
            t_state = self.d.time
            self.g23.apply_torque()
            mujoco.mj_step(self.m, self.d)
            self.step_count += 1
            self.g23.update_imu()
            self.g23.send_telemetry(self.d.time)
            self.lidar.record_pose(t_state, self.d.site_xpos[self.lidar_site],
                                   self.d.site_xmat[self.lidar_site].reshape(3, 3))
            if self.step_count % self.imu_steps == 0:
                out.imu.append(self._imu_sample(imu_cfg, t_state))
            if self.gnss is not None and self.step_count % self.gnss_steps == 0:
                out.gnss.append(self._gnss_sample(t_state))
            if self.step_count % self.clock_steps == 0:
                out.clock.append(self.d.time)
        t2 = time.perf_counter()
        self.timing["physics"] += t2 - t1

        out.frames = self.lidar.process()
        self.timing["lidar"] += time.perf_counter() - t2

        q = self.d.qpos
        self.base.x, self.base.y = float(q[0]), float(q[1])
        self.base.yaw = float(np.arctan2(2.0 * (q[3] * q[6] + q[4] * q[5]), 1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2)))
        # floor under the torso (terrain rays, 50 Hz) so events and height carry the local floor on stairs
        g = self.base._sample(self.base.x, self.base.y, self.base.yaw, np.full(4, float(q[2])), 1.5)
        if g.valid:
            self.base.ground = g
        up = float(self.d.xmat[self.base_body].reshape(3, 3)[2, 2])
        height = float(q[2]) - self.base.ground.z0
        # "collapsed" only counts once the gait has been on for 2 s: before that the robot lies, stands up or
        # settles on purpose
        walking = self.g23.stage == "gait" and self.g23.t_gait is not None and self.d.time - self.g23.t_gait > 2.0
        low = walking and height < float(r.get("fall_height", 0.18))
        self.low_since = (self.low_since if low and self.low_since is not None else self.d.time) if low else None
        collapsed = self.low_since is not None and self.d.time - self.low_since > 1.0
        if (up < 0.5 or collapsed) and not self.fallen:
            self.fallen = True
            out.events.append({"type": "fall", "reason": "tipped" if up < 0.5 else "collapsed", "t": self.d.time,
                               "pose": [self.base.x, self.base.y, self.base.yaw], "z": float(q[2]),
                               "height": round(height, 3), "up": round(up, 3)})
        out.t = self.d.time
        return out

    def close(self):
        if self.g23 is not None:
            self.g23.close()

    def _imu_sample(self, c, t: float) -> ImuSample:
        dt = self.imu_steps * self.dt
        acc = self.d.sensordata[self.acc_adr:self.acc_adr + 3].copy()
        gyro = self.d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        self.acc_bias += self.rng.normal(0, c["acc_bias_rw"] * np.sqrt(dt), 3)
        self.gyro_bias += self.rng.normal(0, c["gyro_bias_rw"] * np.sqrt(dt), 3)
        acc += self.acc_bias + self.rng.normal(0, c["acc_noise"], 3)
        gyro += self.gyro_bias + self.rng.normal(0, c["gyro_noise"], 3)
        acc /= G                                    # livox_ros_driver2 publishes acceleration in g
        return ImuSample(t, acc, gyro)

    def _gnss_sample(self, t: float) -> GnssFix:
        """The antenna's true position and velocity (world frame, at t like the IMU) and the sky it sees."""
        pos = self.d.site_xpos[self.gnss_site].copy()
        vel6 = np.empty(6)                                   # [angular, linear] in the world frame
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE, self.gnss_site, vel6, 0)
        if self.sky_mode == "always":
            sky = 1.0
        elif self.sky_mode == "none":
            sky = 0.0
        else:
            origins = np.repeat((pos + [0.0, 0.0, 0.05])[None], len(self.sky_dirs), axis=0)
            t_hit, _ = self.rays.cast(origins.astype(np.float32), self.sky_dirs.astype(np.float32))
            sky = sky_fraction(t_hit, float(self.cfg["gnss"]["sky"]["range"]))
        return self.gnss.sample(t, pos, vel6[3:6], sky)

    def _base_contact(self):
        if self.base_geom < 0:
            return None
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

    def joint_states(self):
        """(names, q, dq, tau) of the 12 leg joints, or None for the ideal base (no legs)."""
        if not self.legged:
            return None
        jq, jv = self.g23.jq, self.g23.jv
        return JOINTS, self.d.qpos[jq].copy(), self.d.qvel[jv].copy(), self.d.qfrc_actuator[jv].copy()
