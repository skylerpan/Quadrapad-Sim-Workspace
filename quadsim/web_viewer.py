"""Browser viewer for a headless sim (viser): the map, the robot, the LiDAR cloud and the planned path over
HTTP, no display or GL in the container.

Runs inside sim_node. The main loop calls update() after every control tick; the viewer rate-limits itself
(web.hz for poses, web.lidar_hz for the cloud) and does nothing while no browser is connected. viser's own
server thread only serves what update() handed over: it never reads the sim state.

Controls: a joystick (a draggable gizmo riding on the robot: its offset is the velocity command, re-sent every
tick so robot.cmd_timeout never trips, zero on release), stand / walk / lie / damping and the RL policy for the
G23 controller, and reset buttons (teleport to the spawn / re-stand here). The viser thread only sets flags;
sim_node polls them on the main thread and applies them through the same path as the ROS topics.

Static content is sent once per client connect: the map (map_ds.pcd as a point cloud, voxel-downsampled
to web.max_map_points, or with web.map=boxes the scene boxes as instanced cubes, one per box, coloured by
height with a checkerboard shade so every box edge shows) and the robot's visual geoms.
The map is built on a background thread so the sim does not wait for a 17 M point read.
"""
from __future__ import annotations

import math
import os
import socket
import threading
import time

import numpy as np

from .map_colors import height_colors
from .model import box_corners, hexahedra_arrays, nearest_boxes

JOY_LIFT = np.array([0.0, 0.0, 0.35])          # the joystick handle floats this much above the base
JOY_RANGE = 1.0                                  # m of handle offset for full speed
MAX_SPEED = 60.0 / 3.6                           # top of the speed slider: 60 km/h, for crossing an OSM area

DEFAULTS = {
    "enabled": True,
    "host": "0.0.0.0",
    "port": 8755,
    "hz": 30,                      # max rate (wall clock) of robot pose updates
    "lidar_hz": 10,                # max rate of LiDAR cloud updates (the sensor runs at lidar.rate_hz)
    "lidar_points": 8000,          # points per displayed LiDAR frame (uniformly decimated)
    "map": "points",               # points: map_ds.pcd as a point cloud | boxes: one instance per scene box, coloured by height | none
    "max_map_points": 1_500_000,   # voxel-downsample the map cloud beyond this
    "max_map_boxes": 400_000,      # boxes mode: nearest to the spawn first (same rule as the MuJoCo viewer)
    "map_precision": "float32",    # float16 halves the map download but quantises to ~3 cm at 50 m
    "path_topic": "/robot01/planned_path",   # nav_msgs/Path drawn as a line (empty = off)
}

def box_colors(boxes: np.ndarray, voxel: float) -> np.ndarray:
    """uint8 [N,3] colour per scene box (cx, cy, cz, hx, hy, hz): the height palette of the point-cloud map,
    darkened by 15 % on every other cell of a 3-D checkerboard (min-corner grid index), so two boxes that touch at
    the same height still differ and the edge between them is visible from any direction."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    if len(b) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rgb = height_colors(b[:, 2]).astype(np.float64)
    cell = np.floor((b[:, :3] - b[:, 3:6]) / max(float(voxel), 1e-6) + 1e-6).astype(np.int64)
    odd = (cell.sum(axis=1) % 2).astype(bool)
    rgb[odd] *= 0.85
    return (rgb + 0.5).astype(np.uint8)


def downsample_points(xyz: np.ndarray, max_points: int, voxel: float) -> tuple[np.ndarray, float]:
    """Keep one point per cell of a voxel grid coarse enough that at most max_points remain.
    Returns (points, cell size used); the input comes back untouched when it already fits."""
    xyz = np.asarray(xyz)
    if len(xyz) <= max_points or max_points <= 0:
        return xyz, voxel
    v = voxel
    for _ in range(12):
        # a scanned map is surface-like: the count falls ~1/v^2
        v *= max(1.15, math.sqrt(len(xyz) / max_points) * 1.05)
        q = np.floor(xyz / v).astype(np.int64)
        q -= q.min(axis=0)
        dims = q.max(axis=0) + 1
        key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        _, first = np.unique(key, return_index=True)
        xyz = xyz[np.sort(first)]
        if len(xyz) <= max_points:
            break
    return xyz, v


def decimate(xyz: np.ndarray, max_points: int) -> np.ndarray:
    """Every k-th point so that at most max_points remain."""
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz
    return xyz[::int(math.ceil(len(xyz) / max_points))]


class WebViewer:
    """viser server fed by the sim main thread. All sim reads happen in update() (main thread)."""

    def __init__(self, cfg: dict, sim, log=None):
        import viser                                   # only here: headless runs without it still work

        self.cfg = w = {**DEFAULTS, **(cfg.get("web") or {})}
        self.log = log or (lambda msg: None)
        self.period = 1.0 / max(1.0, float(w["hz"]))
        self.lidar_period = 1.0 / max(0.1, float(w["lidar_hz"]))
        self.lidar_points = int(w["lidar_points"])
        self.scene_dir = cfg["scene"]
        self.voxel = float(sim.meta.get("voxel", 0.1))

        self.server = viser.ViserServer(host=str(w["host"]), port=int(w["port"]), verbose=False)
        self.port = self.server.get_port()
        self.url = f"http://{socket.gethostname()}:{self.port}"
        sc = self.server.scene
        sc.set_up_direction("+z")
        sc.configure_default_lights(enabled=True, cast_shadow=False)
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible", dark_mode=True, show_logo=False, show_share_button=False)
        gui.set_panel_label("quadsim")
        with gui.add_folder("Sim"):
            self.g_time = gui.add_text("Sim time", "0.0 s", disabled=True)
            self.g_rtf = gui.add_text("Real-time factor", "-", disabled=True)
            self.g_pose = gui.add_text("Base x y z", "-", disabled=True)
            self.g_lidar = gui.add_text("LiDAR", "-", disabled=True)
            self.g_gnss = gui.add_text("GNSS", "-", disabled=True)
            self.g_map = gui.add_text("Map", "loading...", disabled=True)
        with gui.add_folder("Layers", expand_by_default=False):
            self.c_map = gui.add_checkbox("Map", True)
            self.c_lidar = gui.add_checkbox("LiDAR", True)
            self.c_path = gui.add_checkbox("Planned path", True)
            self.c_follow = gui.add_checkbox("Follow robot", True)
        self._teleop = None                            # (vx, vy, wz) while the joystick is held, or None = ROS /cmd_vel
        self._stop_pending = False                     # send one zero command, then hand back to ROS
        self._drag = False                             # joystick handle being dragged (viser thread sets, main thread reads)
        self._reset = None                             # "spawn" | "here", consumed by take_reset()
        self._reset_wall = -1e9                        # wall time of the last reset (debounce, "resetting..." text)
        self._motion = None                            # "stand" | "walk" | "lie" | "damp", consumed by take_motion()
        self._policy = None                            # RL policy slot, consumed by take_policy()
        with gui.add_folder("Teleop"):
            gui.add_markdown("Drag the handle on the robot: it walks toward the handle and stops when it gets there; "
                             "release = zero.")
            self.s_speed = gui.add_slider("Max speed m/s", min=0.1, max=round(MAX_SPEED, 1), step=0.1,
                                          initial_value=0.4, hint="16.7 m/s = 60 km/h; the ideal base clips "
                                                                  "at robot.vel_limits, a legged gait at its policy")
            self.s_turn = gui.add_slider("Max turn rad/s", min=0.2, max=1.5, step=0.05, initial_value=0.8)
            self.g_cmd = gui.add_text("Command", "ROS /cmd_vel", disabled=True)
            b_stop = gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP, color="red",
                                    hint="zero velocity now and hand the command back to /cmd_vel")
        self.legged = bool(getattr(sim, "legged", False))
        if self.legged:
            kind = (cfg["robot"].get("controller", {}) or {}).get("kind") or "motorcontrol"
            if kind == "motorcontrol":                 # the controller's behaviors, by their policy_mode slots
                policies = {"rl_fcn (PPO flat)": 0, "rl_him (HIMLoco stairs)": 1}
                cur = {"rl_fcn": 0, "rl_him": 1}.get(cfg["robot"].get("behavior", "rl_fcn"), 0)
                hints = ("reset + stand / the behavior / rest / rest to the controller",
                         "by way of stand while walking")
            else:
                policies = {"0 PPO flat": 0, "1 HIMLoco stairs": 1, "4 IsaacSim": 4}
                cur = int(cfg["robot"].get("policy_mode") or 0)
                hints = ("MODE STAND / MODE RL / MODE LIE / MODE R to the controller", "MODE <n>: switches live while walking")
            initial = next((k for k, v in policies.items() if v == cur), next(iter(policies)))
            with gui.add_folder("Motion"):
                self.bg_motion = gui.add_button_group("Motion", ["Stand", "Walk", "Lie down", "Damping"], hint=hints[0])
                self.dd_policy = gui.add_dropdown("RL policy", list(policies), initial_value=initial, hint=hints[1])
            motions = {"Stand": "stand", "Walk": "walk", "Lie down": "lie", "Damping": "damp"}
            self.bg_motion.on_click(lambda _: setattr(self, "_motion", motions[self.bg_motion.value]))
            self.dd_policy.on_update(lambda _: setattr(self, "_policy", policies[self.dd_policy.value]))
        with gui.add_folder("Reset"):
            b_spawn = gui.add_button("Reset to spawn", icon=viser.Icon.HOME,
                                     hint="teleport back to the configured spawn (a localization stack needs a re-seed)")
            b_here = gui.add_button("Re-stand here", icon=viser.Icon.REFRESH,
                                    hint="set_pose at the current x, y, yaw: a fallen robot gets back up")
        b_spawn.on_click(lambda _: setattr(self, "_reset", "spawn"))
        b_here.on_click(lambda _: setattr(self, "_reset", "here"))

        # joystick: a draggable gizmo riding on the robot; the offset from the robot is the velocity command
        self.joy = sc.add_transform_controls("/joystick", scale=0.5, disable_rotations=True, active_axes=(True, True, False),
                                             depth_test=False, opacity=0.85, position=sim.d.xpos[sim.base_body] + JOY_LIFT)

        def _release(_evt=None) -> None:
            self._drag = False
            self._teleop = (0.0, 0.0, 0.0)
            self._stop_pending = True

        def _joy_event(evt) -> None:                   # viser thread: flags only, the command is computed in update()
            if evt.phase == "start":
                self._drag = True
            elif evt.phase == "end":
                _release()

        self.joy.on_update(_joy_event)
        b_stop.on_click(_release)

        self.map_root = sc.add_frame("/map", show_axes=False)
        self.lidar_handle = None
        self.path_handle = None
        self.c_map.on_update(lambda _: setattr(self.map_root, "visible", self.c_map.value))
        self.c_lidar.on_update(lambda _: self.lidar_handle is not None
                               and setattr(self.lidar_handle, "visible", self.c_lidar.value))
        self.c_path.on_update(lambda _: self.path_handle is not None
                              and setattr(self.path_handle, "visible", self.c_path.value))

        self.base_body = sim.base_body
        self.lidar_site = sim.lidar_site
        self._base = np.array(sim.d.xpos[self.base_body])
        self._last_base = self._base.copy()
        self.body_frames = self._add_robot(sim)

        self._pending_frame = None
        self._path = None
        self._next = 0.0
        self._next_lidar = 0.0
        self._next_stats = 0.0
        self._stats_t = (time.perf_counter(), float(sim.d.time))
        self.n_clients = 0
        self.map_info = "loading"
        self._lidar_info = "-"

        @self.server.on_client_connect
        def _(client) -> None:                          # viser thread: only reads self._base (replaced whole)
            b = self._base
            client.camera.position = b + np.array([-4.0, -3.0, 2.5])
            client.camera.look_at = b

        self._map_thread = threading.Thread(target=self._build_map, args=(sim,), daemon=True, name="web-map")
        self._map_thread.start()

    def describe(self) -> str:
        w = self.cfg
        return (f"map {w['map']} (<= {w['max_map_points']} points / {w['max_map_boxes']} boxes), poses <= {w['hz']} Hz, "
                f"lidar <= {w['lidar_hz']} Hz x {w['lidar_points']} points, path {w['path_topic'] or 'off'}; "
                f"--no-web or web.enabled=false to disable")

    # -- static content ---------------------------------------------------------
    def _add_robot(self, sim) -> dict:
        """Visual geoms (groups 0 and 2; collision group 1 and proxies group 3 are skipped) under one frame per
        body; the frames follow d.xpos / d.xquat in update()."""
        import mujoco

        m, d, sc = sim.m, sim.d, self.server.scene
        frames: dict[int, object] = {}
        for g in range(m.ngeom):
            bid = int(m.geom_bodyid[g])
            if bid == 0 or int(m.geom_group[g]) not in (0, 2):
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
            if bid not in frames:
                frames[bid] = sc.add_frame(f"/robot/{bname}", show_axes=False, position=d.xpos[bid], wxyz=d.xquat[bid])
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            color = tuple(int(round(255 * float(c))) for c in rgba[:3])
            opacity = float(rgba[3]) if rgba[3] < 0.999 else None
            path, pos, quat, size = f"/robot/{bname}/{gname}", m.geom_pos[g], m.geom_quat[g], m.geom_size[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                sc.add_mesh_simple(path, m.mesh_vert[va:va + vn].astype(np.float32), m.mesh_face[fa:fa + fn].astype(np.uint32),
                                   color=color, opacity=opacity, position=pos, wxyz=quat, cast_shadow=False)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                sc.add_box(path, color=color, dimensions=2.0 * size, opacity=opacity, position=pos, wxyz=quat, cast_shadow=False)
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                sc.add_icosphere(path, radius=float(size[0]), color=color, subdivisions=2, opacity=opacity,
                                 position=pos, wxyz=quat, cast_shadow=False)
            elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                import trimesh
                make = trimesh.creation.capsule if t == mujoco.mjtGeom.mjGEOM_CAPSULE else trimesh.creation.cylinder
                tm = make(radius=float(size[0]), height=2.0 * float(size[1]))
                sc.add_mesh_simple(path, np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32),
                                   color=color, opacity=opacity, position=pos, wxyz=quat, cast_shadow=False)
        return frames

    def _build_map(self, sim) -> None:
        """Background thread: read the scene and add it under /map. Reads only the scene files and the arrays the
        Simulator loaded at start (sim.boxes / sim.slabs, never written afterwards)."""
        w, sc = self.cfg, self.server.scene
        t0 = time.perf_counter()
        try:
            mode = str(w["map"])
            if mode == "points":
                from pcd2sim.pcd_io import read_pcd
                xyz, _ = read_pcd(os.path.join(self.scene_dir, "map_ds.pcd"))
                n_all = len(xyz)
                xyz, cell = downsample_points(xyz, int(w["max_map_points"]), self.voxel)
                xyz32 = xyz.astype(np.float32)
                sc.add_point_cloud("/map/points", xyz32, height_colors(xyz32[:, 2]), point_size=float(cell),
                                   point_shape="square", point_shading="flat", precision=str(w["map_precision"]))
                info = f"{len(xyz)} points" + (f" (of {n_all}, {cell:.2f} m cells)" if len(xyz) < n_all else "")
            elif mode == "boxes":
                boxes = np.asarray(nearest_boxes(sim.boxes, self._base[:2], int(w["max_map_boxes"])), dtype=np.float64)
                # one unit cube (same winding as the MuJoCo viewer mesh), instanced per box: position = centre,
                # per-axis scale = full extents, its own colour so neighbouring boxes do not merge into one face
                verts, faces = hexahedra_arrays(box_corners(np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])))
                n = len(boxes)
                sc.add_batched_meshes_simple(
                    "/map/boxes", verts.astype(np.float32), faces.astype(np.uint32),
                    batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
                    batched_positions=boxes[:, :3].astype(np.float32),
                    batched_scales=(2.0 * boxes[:, 3:6]).astype(np.float32),
                    batched_colors=box_colors(boxes, self.voxel),
                    lod="auto", flat_shading=True, cast_shadow=False, receive_shadow=False)
                info = f"{n} boxes, per-box colour" + (f" ({n} of {len(sim.boxes)}, nearest the spawn)" if n < len(sim.boxes) else "")
                if sim.slabs is not None and len(sim.slabs):
                    verts, faces = hexahedra_arrays(sim.slabs)
                    sc.add_mesh_simple("/map/ground", verts.astype(np.float32), faces.astype(np.uint32), color=(107, 120, 107),
                                       flat_shading=True, cast_shadow=False, receive_shadow=False)
                    info += f", {len(sim.slabs)} floor slabs"
            else:
                info = "off"
        except Exception as e:  # noqa: BLE001  (a viewer problem must not take the sim down)
            info = f"failed: {e!r}"
        self.map_info = info
        self.g_map.value = info
        self.log(f"web viewer: map {info} in {time.perf_counter() - t0:.1f} s")

    # -- controls (main thread polls, viser thread sets) -------------------------
    def teleop_cmd(self):
        """(vx, vy, wz) while the joystick is held: send it every tick. After release / Stop, one zero command is
        returned and then None, so ROS /cmd_vel drives again."""
        cmd = self._teleop
        if cmd is None:
            return None
        if self._stop_pending:
            self._stop_pending = False
            self._teleop = None
        return cmd

    def take_reset(self, sim):
        """A pending reset as a set_pose request (x, y, yaw, z_floor), or None. Clicks within 3 s of the last reset are
        dropped: a second set_pose restarts the stand sequence from scratch."""
        kind, self._reset = self._reset, None
        if kind is None:
            return None
        now = time.perf_counter()
        if now - self._reset_wall < 3.0:
            return None
        self._reset_wall = now
        if kind == "spawn":
            x, y, yaw, z_floor = sim.spawn_requested
            return (float(x), float(y), float(yaw), float(z_floor))
        b = sim.base
        return (float(b.x), float(b.y), float(b.yaw), float(b.ground.z0))

    def take_motion(self):
        """"stand" | "walk" | "lie" | "damp" once per click, or None."""
        m, self._motion = self._motion, None
        return m

    def take_policy(self):
        """RL policy slot once per dropdown change, or None."""
        p, self._policy = self._policy, None
        return p

    def _joystick(self, d) -> None:
        """Main thread: velocity from the handle's offset in the robot's yaw frame while dragging; otherwise the handle
        rides on the robot. The command shrinks as the robot reaches the handle, so it walks there and stops."""
        q = d.xquat[self.base_body]
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        if self._drag:
            off = np.asarray(self.joy.position)[:2] - self._base[:2]
            c, s = math.cos(yaw), math.sin(yaw)
            rx, ry = c * off[0] + s * off[1], -s * off[0] + c * off[1]
            dist = math.hypot(rx, ry)
            if dist < 0.05:
                self._teleop = (0.0, 0.0, 0.0)
                return
            theta = math.atan2(ry, rx)
            wz = self.s_turn.value * max(-1.0, min(1.0, theta / (math.pi / 2)))
            vx = self.s_speed.value * min(1.0, dist / JOY_RANGE) * math.cos(theta) if abs(theta) < math.pi / 2 else 0.0
            self._teleop = (vx, 0.0, wz)
        else:
            target = self._base + JOY_LIFT
            if float(np.abs(target - np.asarray(self.joy.position)).max()) > 1e-3:
                self.joy.position = target
            # the handle's arrows follow the robot's heading (yaw only: level on slopes and stairs)
            wxyz = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
            if float(np.abs(wxyz - np.asarray(self.joy.wxyz)).max()) > 4e-3:       # ~0.5 deg
                self.joy.wxyz = wxyz

    # -- per tick (main thread) -------------------------------------------------
    def set_path(self, points) -> None:
        """nav_msgs/Path positions [N,3] (map frame); drawn on the next update."""
        self._path = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    def update(self, sim, frames=()) -> float:
        """Push the current state; returns the wall time spent. Call after every control tick: the LiDAR frame
        is kept until the next slot, poses are sent at most web.hz times a second, nothing without a client."""
        if frames:
            self._pending_frame = frames[-1]
        now = time.perf_counter()
        if now < self._next:
            return 0.0
        self._next = now + self.period
        d = sim.d
        self._base = np.array(d.xpos[self.base_body])
        clients = self.server.get_clients()
        self.n_clients = len(clients)
        if not clients:
            self._pending_frame = None
            self._last_base = self._base
            if self._drag:                          # the browser went away mid-drag: let go
                self._drag = False
                self._teleop = (0.0, 0.0, 0.0)
                self._stop_pending = True
            return time.perf_counter() - now
        self._joystick(d)
        with self.server.atomic():
            for bid, fr in self.body_frames.items():
                fr.position = d.xpos[bid]
                fr.wxyz = d.xquat[bid]
            if self._pending_frame is not None and now >= self._next_lidar:
                self._push_lidar(d, self._pending_frame)
                self._pending_frame = None
                self._next_lidar = now + self.lidar_period
            if self._path is not None:
                self._push_path(self._path)
                self._path = None
            if self.c_follow.value:
                delta = self._base - self._last_base
                if float(np.dot(delta, delta)) > 1e-6:
                    for c in clients.values():
                        c.camera.position = c.camera.position + delta
                        c.camera.look_at = self._base
        self._last_base = self._base
        if now >= self._next_stats:
            self._next_stats = now + 0.5
            w0, t0 = self._stats_t
            self._stats_t = (now, float(d.time))
            self.g_time.value = f"{d.time:.1f} s"
            self.g_rtf.value = f"{(d.time - t0) / max(1e-6, now - w0):.2f}"
            self.g_pose.value = " ".join(f"{v:.2f}" for v in self._base)
            t = self._teleop
            if now - self._reset_wall < 5.0:
                cmd_txt = "resetting..."
            elif t is None:
                cmd_txt = "ROS /cmd_vel"
            else:
                cmd_txt = f"joystick vx {t[0]:.2f} wz {t[2]:.2f}"
            if self.g_cmd.value != cmd_txt:
                self.g_cmd.value = cmd_txt
            if self.g_lidar.value != self._lidar_info:
                self.g_lidar.value = self._lidar_info
            gnss_txt = gnss_text(sim.gnss.latest) if getattr(sim, "gnss", None) is not None else "off"
            if self.g_gnss.value != gnss_txt:
                self.g_gnss.value = gnss_txt
        return time.perf_counter() - now

    def _push_lidar(self, d, f) -> None:
        """The latest frame in the map frame. Points are in the sensor frame at their emission time: placing them
        all with the current site pose smears a moving scan slightly, which is fine for a display."""
        if not self.c_lidar.value and self.lidar_handle is not None:
            return
        xyz = decimate(f.xyz, self.lidar_points)
        R = d.site_xmat[self.lidar_site].reshape(3, 3)
        world = (xyz.astype(np.float64) @ R.T + d.site_xpos[self.lidar_site]).astype(np.float32)
        if self.lidar_handle is None:
            self.lidar_handle = self.server.scene.add_point_cloud(
                "/lidar", world, (255, 176, 64), point_size=0.5 * self.voxel, point_shape="square",
                point_shading="flat", precision="float32", visible=self.c_lidar.value)
        else:
            self.lidar_handle.points = world
        self._lidar_info = f"{len(f.xyz)} points/frame, {len(world)} shown"

    def _push_path(self, pts: np.ndarray) -> None:
        if len(pts) < 2:
            if self.path_handle is not None:
                self.path_handle.visible = False
            return
        seg = np.stack([pts[:-1], pts[1:]], axis=1) + np.array([0.0, 0.0, 0.03], dtype=np.float32)   # just above the floor
        if self.path_handle is None:
            self.path_handle = self.server.scene.add_line_segments("/path", seg, (64, 220, 120), thickness=0.04,
                                                                   visible=self.c_path.value)
        else:
            self.path_handle.points = seg
            self.path_handle.visible = self.c_path.value

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001
            pass


def gnss_text(fix) -> str:
    """The GNSS line of the Sim folder: fix type, position, sigma, sky."""
    if fix is None:
        return "waiting"
    if fix.fix == "none":
        return f"no fix (sky {fix.sky:.0%})"
    return f"{fix.fix} {fix.lat:.6f} {fix.lon:.6f} alt {fix.alt:.1f} ±{fix.sigma_h:.2f} m (sky {fix.sky:.0%})"
