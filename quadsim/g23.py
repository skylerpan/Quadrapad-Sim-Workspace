"""G23 legged robot in quadsim: the model, and the UDP side of the Rust controller's simulation mode.

The controller (g2_deploy_rs, `g2_deploy_rs sim ros`) expects the MuJoCo simulation process it ships
with (interface/robot/simulation/mujoco_simulation_G23.py). quadsim plays that process, and keeps to it:
same joint model, gravity, torque formula, sensor / actuator options, telemetry packets and targets. Where
quadsim differs it says so in docs/usage.md section 3.1; measure any doubt against their sim with
scripts/idle_drift_theirs.py rather than changing the robot on this side.

  controller -> sim   UDP :20001   240 bytes, 5 x 12 float32: kp | q_des | kd | dq_des | tau_ff
                                   torque = kp (q_des - q) + kd (dq_des - dq) + tau_ff, clipped to the J60
                                   speed-torque envelope, every physics step (1 kHz)
  sim -> controller   UDP :30010   228 bytes at 1 kHz: float64 t, then float32 rpy[3] acc[3] gyro[3] q[12]
                      (+ GUI :30011)  dq[12] tau[12] base_pos[3] base_vel[3] foot_contact[4]
  sim -> ROS bridge   UDP :50010   their ASCII state line (IMU_RPY ... IMU_OK 1), G2_SIM_ROS_ASCII_TELEMETRY
  sim -> GT           UDP :30012   "SIM_POSE t x y z qw qx qy qz vx vy vz wx wy wz"
  sim -> controller   UDP :50051   text lines the ROS bridge would send: "MODE STAND", "MODE <policy>",
                                   "MODE RL" / "MODE LMA", "AXES vx vy wz" (m/s, m/s, rad/s)

Their sim is configured by G2_SIM_* environment variables and config/g2_control.env (same precedence as the
controller: G2_CONTROL_CONFIG > <repo>/config/g2_control.env > /opt/g2/current/config/g2_control.env);
RefSimOptions reads the same ones so both sims run under one configuration.

The model is robot/g23/mjcf/g23_mjcf.xml (see robot/g23/provenance.json) with its flat floor removed, a
`lidar` site + `lidar_acc` / `lidar_gyro` sensors on the torso for the ROS side (the controller keeps its own
`imu_acc` / `imu_gyro` at imu_site), and the mocap proxy pool. Proxy geoms are contype=1 so the G23 collision
geoms (contype=0, conaffinity=1) hit them; the pool is fed from the scene's contact boxes, floors included.
"""
from __future__ import annotations

import collections
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

G23_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot", "g23", "mjcf", "g23_mjcf.xml")
JOINTS = ["FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint", "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
          "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint", "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"]
FEET = ["FL_FOOT_collision", "FR_FOOT_collision", "HL_FOOT_collision", "HR_FOOT_collision"]
DEFAULT_Q = np.array([0.0, -0.65, 1.3] * 4)
CMD_FMT = "<" + "12f" * 5                     # 240 bytes
CMD_BYTES = struct.calcsize(CMD_FMT)
TELEMETRY_FMT = "<d55f"                       # 228 bytes
# J60 motors: peak torque and no-load speed per (HipX, HipY, Knee); same-direction torque fades linearly
# with speed, braking torque stays at the peak (mujoco_simulation_G23.j60_torque_limits)
J60_PEAK = np.tile([19.94, 19.94, 30.50], 4)
J60_WMAX = np.tile([24.18, 24.18, 15.49], 4)
# J60 joint model their sim puts on the 12 leg DOFs (mujoco_simulation_G23._apply_motor_model): rotor inertia
# through the gear ratio (6 on the hips, 9 on the knee), dry friction and viscous damping per (HipX, HipY, Knee)
J60_ARMATURE = [1.1e-4 * 6 ** 2, 1.1e-4 * 6 ** 2, 1.1e-4 * 9 ** 2]
J60_FRICTION = [0.40, 0.40, 0.45]
J60_DAMPING = [(0.81 - 0.40) / 24.6, (0.81 - 0.40) / 24.6, (0.92 - 0.45) / 15.5]


def j60_limits(dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.clip(J60_PEAK * (1.0 - dq / J60_WMAX), 0.0, J60_PEAK)
    lower = -np.clip(J60_PEAK * (1.0 + dq / J60_WMAX), 0.0, J60_PEAK)
    return lower, upper


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def _addr(s: str) -> tuple[str, int]:
    host, _, port = str(s).rpartition(":")
    return host or "127.0.0.1", int(port)


# -- the reference simulation's configuration ------------------------------------------------------------
def read_control_config_value(key: str, repo: str | None = None) -> str:
    """Same lookup as the controller and their sim: G2_CONTROL_CONFIG > <repo>/config/g2_control.env >
    /opt/g2/current/config/g2_control.env, last `KEY=value` wins ("export " and quotes stripped); "" if absent."""
    explicit = os.environ.get("G2_CONTROL_CONFIG", "").strip()
    repo = repo or os.environ.get("G2_DEPLOY_RS", "/g2")
    candidates = [explicit] if explicit else [os.path.join(repo, "config", "g2_control.env"),
                                               "/opt/g2/current/config/g2_control.env"]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        value = ""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith(f"{key}="):
                    value = line.split("=", 1)[1].strip().strip("\"'")
        return value
    return ""


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def _vec3(raw: str, key: str) -> np.ndarray | None:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "none":
        return None
    v = np.array([float(x) for x in raw.split(",")], dtype=float)
    if v.shape != (3,):
        raise ValueError(f"{key} needs 3 values, got {raw!r}")
    return v


@dataclass
class RefSimOptions:
    """Every switch of mujoco_simulation_G23.py, read from the same environment variables (and
    g2_control.env for the COM offset), with their defaults. sim.yaml values that are not null win over the
    environment (see from_env)."""
    motor_model: str = "j60"                    # G2_SIM_MOTOR_MODEL j60 | ideal
    torso_com_offset: np.ndarray | None = None  # G2_SIM_TORSO_COM_OFFSET dx,dy,dz (torso frame) on body_ipos
    imu_pos: np.ndarray | None = None           # G2_SIM_IMU_POS x,y,z: moves imu_site
    ahrs_tau: float = 0.0                       # G2_SIM_IMU_AHRS_TAU s: complementary-filter roll / pitch
    imu_rate_hz: float = 0.0                    # G2_SIM_IMU_RATE_HZ: sample-and-hold of the telemetry IMU
    imu_delay_ms: float = 0.0                   # G2_SIM_IMU_DELAY_MS: telemetry IMU delay
    torque_bw_hz: float = 0.0                   # G2_SIM_TORQUE_BW_HZ: first-order torque filter
    cmd_delay_ms: float = 0.0                   # G2_SIM_CMD_DELAY_MS: torque command delay line
    motor_temp: bool = False                    # G2_SIM_MOTOR_TEMP=1 (their thermal model; quadsim reports 0 C)
    ros_ascii: bool = True                      # G2_SIM_ROS_ASCII_TELEMETRY != 0: ASCII state line to the ROS bridge
    ros_bridge: tuple[str, int] | None = ("127.0.0.1", 50010)   # G2_SIM_ROS_BRIDGE_IP / _PORT
    gt: tuple[str, int] | None = ("127.0.0.1", 30012)           # G2_SIM_GT_IP / _PORT: SIM_POSE lines
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, robot_cfg: dict | None = None, repo: str | None = None) -> "RefSimOptions":
        env = os.environ
        o = cls()
        o.motor_model = env.get("G2_SIM_MOTOR_MODEL", "j60").strip().lower() or "j60"
        raw = env.get("G2_SIM_TORSO_COM_OFFSET", "").strip() or read_control_config_value("G2_SIM_TORSO_COM_OFFSET", repo)
        o.torso_com_offset = _vec3(raw, "G2_SIM_TORSO_COM_OFFSET")
        o.imu_pos = _vec3(env.get("G2_SIM_IMU_POS", ""), "G2_SIM_IMU_POS")
        o.ahrs_tau = _env_float("G2_SIM_IMU_AHRS_TAU")
        o.imu_rate_hz = _env_float("G2_SIM_IMU_RATE_HZ")
        o.imu_delay_ms = _env_float("G2_SIM_IMU_DELAY_MS")
        o.torque_bw_hz = _env_float("G2_SIM_TORQUE_BW_HZ")
        o.cmd_delay_ms = _env_float("G2_SIM_CMD_DELAY_MS")
        o.motor_temp = env.get("G2_SIM_MOTOR_TEMP", "0") == "1"
        o.ros_ascii = env.get("G2_SIM_ROS_ASCII_TELEMETRY", "1") != "0"
        o.ros_bridge = (env.get("G2_SIM_ROS_BRIDGE_IP", "127.0.0.1"), int(env.get("G2_SIM_ROS_BRIDGE_PORT", "50010")))
        o.gt = (env.get("G2_SIM_GT_IP", "127.0.0.1"), int(env.get("G2_SIM_GT_PORT", "30012")))
        r = robot_cfg or {}
        ctl = r.get("controller", {}) or {}
        if r.get("motor_model") is not None:
            o.motor_model = str(r["motor_model"]).lower()
        if ctl.get("ros_ascii_telemetry") is not None:
            o.ros_ascii = bool(ctl["ros_ascii_telemetry"])
        if ctl.get("ros_bridge_addr") is not None:
            o.ros_bridge = _addr(ctl["ros_bridge_addr"]) if ctl["ros_bridge_addr"] else None
        if ctl.get("gt_addr") is not None:
            o.gt = _addr(ctl["gt_addr"]) if ctl["gt_addr"] else None
        if o.motor_model not in ("j60", "ideal"):
            raise ValueError(f"G2_SIM_MOTOR_MODEL / robot.motor_model must be j60 or ideal, got {o.motor_model!r}")
        return o

    def describe(self) -> str:
        parts = [f"motor model {self.motor_model}"]
        if self.torso_com_offset is not None:
            parts.append(f"torso COM offset {self.torso_com_offset.tolist()}")
        if self.imu_pos is not None:
            parts.append(f"IMU site {self.imu_pos.tolist()}")
        if self.ahrs_tau or self.imu_rate_hz or self.imu_delay_ms or self.torque_bw_hz or self.cmd_delay_ms:
            parts.append(f"sensor/actuator model: AHRS tau {self.ahrs_tau} s, IMU {self.imu_rate_hz or 1000:.0f} Hz "
                         f"delay {self.imu_delay_ms} ms, torque bandwidth {self.torque_bw_hz or 'inf'} Hz, "
                         f"cmd delay {self.cmd_delay_ms} ms")
        parts.append("ROS ASCII telemetry " + ("-> %s:%d" % self.ros_bridge if self.ros_ascii and self.ros_bridge else "off"))
        parts.append("SIM_POSE " + ("-> %s:%d" % self.gt if self.gt else "off"))
        if self.motor_temp:
            parts.append("motor temperature model requested (not simulated by quadsim, JOINT_TEMP stays 0)")
        return "; ".join(parts)


def build_g23_spec(cfg: dict, gravity, spawn_pos, n_proxies: int, opts: RefSimOptions | None = None) -> mujoco.MjSpec:
    """The G23 model on a map scene: floor removed, LiDAR site + sensors on the torso, proxy pool, the
    reference sim's joint model, IMU position and gravity."""
    r, li, ph = cfg["robot"], cfg["lidar"], cfg["physics"]
    opts = opts or RefSimOptions.from_env(r)
    spec = mujoco.MjSpec.from_file(G23_XML)
    spec.option.timestep = ph["timestep"]
    # Their sim has gravity straight down. The scene's gravity follows the fitted floor normal (the ideal
    # base's level map frame); fed to the policy it put the accelerometer 0.5-1 deg off the rpy.
    spec.option.gravity = [0.0, 0.0, -9.81] if r.get("level_gravity", True) else list(gravity)
    if opts.motor_model == "j60":
        for i, name in enumerate(JOINTS):
            j = spec.joint(name)
            j.armature = J60_ARMATURE[i % 3]
            j.frictionloss = J60_FRICTION[i % 3]
            j.damping = [J60_DAMPING[i % 3], 0.0, 0.0]        # MjsJoint.damping takes a 3-vector (hinge uses [0])
    if opts.imu_pos is not None:
        spec.site("imu_site").pos = opts.imu_pos.tolist()
    spec.delete(spec.geom("floor"))
    torso = spec.body("TORSO")
    torso.pos = [spawn_pos[0], spawn_pos[1], spawn_pos[2]]
    torso.add_site(name="lidar", pos=list(li["mount_xyz"]), size=[0.03, 0.03, 0.03], rgba=[0.1, 0.1, 0.9, 1])
    spec.add_sensor(name="lidar_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="lidar")
    spec.add_sensor(name="lidar_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="lidar")
    for i in range(n_proxies):
        b = spec.worldbody.add_body(name=f"proxy_{i}", mocap=True, pos=[0, 0, -1000 - 10 * i])
        b.add_geom(name=f"proxy_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05],
                   contype=1, conaffinity=0, group=3, rgba=[0.9, 0.3, 0.3, 0.6])
    spec.stat.center = [spawn_pos[0], spawn_pos[1], spawn_pos[2]]
    spec.stat.extent = 10.0
    return spec


class G23Backend:
    """Torque commands in, telemetry out, text commands to the controller, the stand / gait sequence, and the
    reference sim's sensor / actuator options."""

    def __init__(self, m, d, robot_cfg: dict, torso_body: int, opts: RefSimOptions | None = None):
        self.m, self.d, self.c = m, d, robot_cfg
        self.opts = opts or RefSimOptions.from_env(robot_cfg)
        self.dt = float(m.opt.timestep)
        self.torso = torso_body
        self.jq = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
        self.jv = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_ctrl") for n in JOINTS])
        self.feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n): i for i, n in enumerate(FEET)}
        self.root = m.body_rootid[torso_body]
        self.acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")]
        self.gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        if self.opts.torso_com_offset is not None:          # their _apply_torso_com_offset
            m.body_ipos[torso_body] += self.opts.torso_com_offset
        self.kp = np.zeros(12)
        self.q_des = np.zeros(12)
        self.kd = np.zeros(12)
        self.dq_des = np.zeros(12)
        self.tau_ff = np.zeros(12)
        self.lock = threading.Lock()
        self.last_cmd_wall = None
        self.n_cmd = 0
        self.connected = False
        self.t_connected = None                    # sim time of the first torque command
        self.lost = False
        # sensor / actuator model (their _init_sensor_model)
        o = self.opts
        self._cmd_delay_steps = int(round(o.cmd_delay_ms * 1e-3 / self.dt))
        self._cmd_history: collections.deque = collections.deque(maxlen=self._cmd_delay_steps + 1)
        self._tau_filtered = None
        self._imu_delay_steps = int(round(o.imu_delay_ms * 1e-3 / self.dt))
        self._imu_history: collections.deque = collections.deque(maxlen=self._imu_delay_steps + 1)
        self._imu_period_steps = max(1, int(round(1.0 / (o.imu_rate_hz * self.dt)))) if o.imu_rate_hz > 0 else 1
        self._imu_held = None
        self._ahrs_q = None
        self._sensor_model = any(v > 0 for v in (o.ahrs_tau, o.imu_rate_hz, o.imu_delay_ms))
        self.step = 0
        self.input_tq = np.zeros(12)
        ctl = robot_cfg.get("controller", {}) or {}
        self.cmd_port = int(ctl.get("cmd_port", 20001))
        self.telemetry_to = [_addr(a) for a in ctl.get("telemetry", ["127.0.0.1:30010", "127.0.0.1:30011"])]
        self.command_to = _addr(ctl.get("command_addr", "127.0.0.1:50051"))
        self.gt_to = o.gt
        self.ros_bridge_to = o.ros_bridge if o.ros_ascii else None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.2)
        self.sock.bind(("0.0.0.0", self.cmd_port))
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self.rx = threading.Thread(target=self._receiver, daemon=True)
        self.rx.start()
        # stand / gait sequence (sim time)
        self.gait = str(robot_cfg.get("gait", "rl")).upper()
        self.policy_mode = robot_cfg.get("policy_mode")      # g23 RL: integer policy slot sent before MODE RL
        self.auto = bool(robot_cfg.get("auto_stand", True))
        self.start_delay = float(robot_cfg.get("start_delay", 1.0))
        self.stand_delay = float(robot_cfg.get("stand_delay", 4.0))
        self.sent = []                             # text commands sent (for tests / logs)
        self.axes_active = False
        self.restart(0.0)

    def describe(self) -> str:
        return self.opts.describe()

    # -- controller link --------------------------------------------------------------------------
    def _receiver(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < CMD_BYTES:
                continue
            v = struct.unpack(CMD_FMT, data[:CMD_BYTES])
            with self.lock:
                self.kp[:] = v[0:12]
                self.q_des[:] = v[12:24]
                self.kd[:] = v[24:36]
                self.dq_des[:] = v[36:48]
                self.tau_ff[:] = v[48:60]
                self.last_cmd_wall = time.monotonic()
                self.n_cmd += 1

    def apply_torque(self):
        """Their _apply_joint_torque: kp (q_des - q) + kd (dq_des - dq) + tau_ff through the optional command
        delay line and torque filter, within the J60 envelope -> d.ctrl."""
        q = self.d.qpos[self.jq]
        dq = self.d.qvel[self.jv]
        with self.lock:
            cmd = (self.kp.copy(), self.q_des.copy(), self.kd.copy(), self.dq_des.copy(), self.tau_ff.copy())
        if self._cmd_delay_steps:
            self._cmd_history.append(cmd)
            cmd = self._cmd_history[0]
        kp, q_des, kd, dq_des, tau_ff = cmd
        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
        if self.opts.torque_bw_hz > 0:
            alpha = self.dt / (self.dt + 1.0 / (2.0 * np.pi * self.opts.torque_bw_hz))
            if self._tau_filtered is None:
                self._tau_filtered = tau.copy()
            self._tau_filtered = self._tau_filtered + alpha * (tau - self._tau_filtered)
            tau = self._tau_filtered.copy()
        if self.opts.motor_model == "j60":
            lo, hi = j60_limits(dq)
            tau = np.clip(tau, lo, hi)
        self.input_tq = tau
        self.d.ctrl[self.act] = tau

    # -- sensors --------------------------------------------------------------------------------------
    def _raw_imu(self):
        d = self.d
        return (quat_to_rpy(d.qpos[3:7]), d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy(),
                d.sensordata[self.acc_adr:self.acc_adr + 3].copy())

    def update_imu(self):
        """Their _update_imu, once per physics step after mj_step: AHRS roll / pitch estimate, sample-and-hold
        at the IMU rate, delay line. No-op unless one of those is configured."""
        self.step += 1
        if not self._sensor_model:
            return
        rpy, gyro, acc = self._raw_imu()
        o = self.opts
        if o.ahrs_tau > 0:
            qt = self.d.qpos[3:7].copy()
            if self._ahrs_q is None:
                self._ahrs_q = qt.copy()
            w, x, y, z = self._ahrs_q
            g_est = np.array([2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z])
            a = acc / max(np.linalg.norm(acc), 1e-6)
            omega = gyro + (1.0 / o.ahrs_tau) * np.cross(a, g_est)
            dq = 0.5 * np.array([-x * omega[0] - y * omega[1] - z * omega[2],
                                 w * omega[0] + y * omega[2] - z * omega[1],
                                 w * omega[1] - x * omega[2] + z * omega[0],
                                 w * omega[2] + x * omega[1] - y * omega[0]])
            self._ahrs_q = self._ahrs_q + dq * self.dt
            self._ahrs_q /= np.linalg.norm(self._ahrs_q)
            est = quat_to_rpy(self._ahrs_q)
            rpy = np.array([est[0], est[1], rpy[2]])
        if self._imu_held is None or self.step % self._imu_period_steps == 0:
            self._imu_held = (rpy, gyro, acc)
        self._imu_history.append(self._imu_held)

    def imu_output(self):
        if not self._sensor_model or not self._imu_history:
            return self._raw_imu()
        return self._imu_history[0]

    def foot_contacts(self) -> np.ndarray:
        """1.0 per foot touching something outside the robot (the floor proxies)."""
        c = np.zeros(4, dtype=np.float32)
        m, d = self.m, self.d
        for i in range(d.ncon):
            con = d.contact[i]
            for foot, other in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                leg = self.feet.get(int(foot))
                if leg is None or m.body_rootid[m.geom_bodyid[other]] == self.root:
                    continue
                c[leg] = 1.0
        return c

    # -- telemetry ------------------------------------------------------------------------------------
    def telemetry(self, t: float) -> bytes:
        d = self.d
        rpy, gyro, acc = self.imu_output()
        payload = np.concatenate([rpy, acc, gyro, d.qpos[self.jq], d.qvel[self.jv],
                                  d.qfrc_actuator[self.jv], d.qpos[0:3], d.qvel[0:3], self.foot_contacts()])
        return struct.pack(TELEMETRY_FMT, t, *payload.astype(np.float32))

    def ros_bridge_line(self) -> str:
        """Their _send_ros_bridge_ascii: the state line udp_ros_bridge_node parses."""
        d = self.d
        rpy, gyro, acc = self.imu_output()
        q = d.qpos[3:7]

        def vec(v):
            return " ".join(f"{float(x):.6f}" for x in np.asarray(v).ravel())

        return (f"IMU_RPY {vec(rpy)} IMU_QUAT {vec([q[1], q[2], q[3], q[0]])} ACC {vec(acc)} OMEGA {vec(gyro)} "
                f"JOINT_POS {vec(d.qpos[self.jq])} JOINT_VEL {vec(d.qvel[self.jv])} "
                f"JOINT_TAU {vec(d.qfrc_actuator[self.jv])} JOINT_TEMP {vec(np.zeros(12))} "
                f"JOINT_ERR {' '.join('0' for _ in range(12))} IMU_OK 1\n")

    def gt_line(self, t: float) -> str:
        d = self.d
        p, q, v, w = d.qpos[0:3], d.qpos[3:7], d.qvel[0:3], d.qvel[3:6]
        return (f"SIM_POSE {t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")

    def send_telemetry(self, t: float):
        pkt = self.telemetry(t)
        for a in self.telemetry_to:
            try:
                self.out.sendto(pkt, a)
            except OSError:
                pass
        if self.ros_bridge_to is not None:
            try:
                self.out.sendto(self.ros_bridge_line().encode(), self.ros_bridge_to)
            except OSError:
                pass
        if self.gt_to is not None:
            try:
                self.out.sendto(self.gt_line(t).encode(), self.gt_to)
            except OSError:
                pass

    def send_command(self, text: str):
        self.sent.append(text)
        try:
            self.out.sendto((text + "\n").encode(), self.command_to)
        except OSError:
            pass

    # -- sequence ---------------------------------------------------------------------------------
    def restart(self, t: float):
        """(Re)start the stand -> gait sequence from sim time t (spawn, set_pose)."""
        self.t0 = t
        self.t_gait = None
        self.stage = "idle" if self.auto else "manual"

    def tick(self, t: float, cmd: np.ndarray, cmd_fresh: bool = True) -> list[dict]:
        """Per control tick: advance the sequence, forward cmd_vel (cmd_fresh: a /cmd_vel arrived within the
        timeout), watch the controller link."""
        events = []
        with self.lock:
            last, n = self.last_cmd_wall, self.n_cmd
        if n > 0 and not self.connected:
            self.connected = True
            self.t_connected = t
            events.append({"type": "controller", "state": "connected", "t": t})
        # the sequence starts once the controller talks (its first LMA start can take minutes) and after
        # start_delay s past the spawn / set_pose
        ready = self.connected and t - max(self.t0, self.t_connected) >= self.start_delay
        if self.stage == "idle" and ready:
            self.send_command("MODE STAND")
            self.stage = "stand"
            self.t_stand = t
            events.append({"type": "controller", "state": "stand", "t": t})
        elif self.stage == "stand" and t - self.t_stand >= self.stand_delay:
            if self.policy_mode is not None and self.gait == "RL":
                self.send_command(f"MODE {int(self.policy_mode)}")     # RL policy slot (0 PPO flat, 1 HIMLoco, ...)
            self.send_command(f"MODE {self.gait}")
            self.stage = "gait"
            self.t_gait = t
            events.append({"type": "controller", "state": self.gait.lower(), "t": t})
        if self.stage in ("gait", "manual"):
            # forward a live /cmd_vel every tick; when it has gone stale send one zero and then nothing, so
            # the controller's own watchdog marks the command inactive (a stream of zero AXES keeps the RL
            # policy stepping in place)
            active = cmd_fresh
            if active or self.axes_active:
                self.send_command(f"AXES {cmd[0]:.3f} {cmd[1]:.3f} {cmd[2]:.3f}" if active else "AXES 0.000 0.000 0.000")
            self.axes_active = active
        if self.connected and last is not None:
            silent = time.monotonic() - last > 1.0
            if silent and not self.lost:
                self.lost = True
                events.append({"type": "controller", "state": "lost", "t": t})
            elif not silent and self.lost:
                self.lost = False
                events.append({"type": "controller", "state": "connected", "t": t})
        return events

    def set_policy_mode(self, mode: int, t: float) -> dict:
        """Choose the RL policy (0 PPO, 1 HIMLoco, 4 IsaacSim): sent now when the gait is on (the controller
        switches live), otherwise before MODE RL in the start sequence."""
        self.policy_mode = int(mode)
        if self.stage == "gait" and self.gait == "RL":
            self.send_command(f"MODE {self.policy_mode}")
        return {"type": "controller", "state": "policy", "policy_mode": self.policy_mode, "t": t,
                "live": self.stage == "gait" and self.gait == "RL"}

    def place(self, pos, quat):
        """Torso pose + the default joint angles, at rest."""
        d = self.d
        d.qpos[0:3] = pos
        d.qpos[3:7] = quat
        d.qpos[self.jq] = DEFAULT_Q
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
            self.out.close()
        except OSError:
            pass
