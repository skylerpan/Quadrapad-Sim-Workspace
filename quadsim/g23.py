"""G23 legged robot in quadsim: the model, and the UDP side of the Rust controller's simulation mode.

The controller (g2_deploy_rs, `g2_deploy_rs sim ros`) expects the MuJoCo simulation process it ships
with (interface/robot/simulation/mujoco_simulation_G23.py). quadsim plays that process:

  controller -> sim   UDP :20001   240 bytes, 5 x 12 float32: kp | q_des | kd | dq_des | tau_ff
                                   torque = kp (q_des - q) + kd (dq_des - dq) + tau_ff, clipped to the J60
                                   speed-torque envelope, every physics step (1 kHz)
  sim -> controller   UDP :30010   228 bytes at 1 kHz: float64 t, then float32 rpy[3] acc[3] gyro[3] q[12]
                      (+ GUI :30011)  dq[12] tau[12] base_pos[3] base_vel[3] foot_contact[4]
  sim -> controller   UDP :50051   text lines the ROS bridge would send: "MODE STAND", "MODE RL" / "MODE LMA",
                                   "AXES vx vy wz" (m/s, m/s, rad/s), "ESTOP", "RESET_SAFETY"

The model is robot/g23/mjcf/g23_mjcf.xml (see robot/g23/provenance.json) with its flat floor removed, a
`lidar` site + `lidar_acc` / `lidar_gyro` sensors on the torso for the ROS side (the controller keeps its own
`imu_acc` / `imu_gyro` at imu_site), and the mocap proxy pool. Proxy geoms are contype=1 so the G23 collision
geoms (contype=0, conaffinity=1) hit them; the pool is fed from the scene's contact boxes, floors included.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time

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


def build_g23_spec(cfg: dict, gravity, spawn_pos, n_proxies: int) -> mujoco.MjSpec:
    """The G23 model on a map scene: floor removed, LiDAR site + sensors on the torso, proxy pool."""
    r, li, ph = cfg["robot"], cfg["lidar"], cfg["physics"]
    spec = mujoco.MjSpec.from_file(G23_XML)
    spec.option.timestep = ph["timestep"]
    spec.option.gravity = list(gravity)
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


def _addr(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)


class G23Backend:
    """Torque commands in, telemetry out, text commands to the controller, the stand / gait sequence."""

    def __init__(self, m, d, robot_cfg: dict, torso_body: int):
        self.m, self.d, self.c = m, d, robot_cfg
        self.torso = torso_body
        self.jq = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
        self.jv = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_ctrl") for n in JOINTS])
        self.feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n): i for i, n in enumerate(FEET)}
        self.root = m.body_rootid[torso_body]
        self.acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")]
        self.gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
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
        ctl = robot_cfg.get("controller", {}) or {}
        self.cmd_port = int(ctl.get("cmd_port", 20001))
        self.telemetry_to = [_addr(a) for a in ctl.get("telemetry", ["127.0.0.1:30010", "127.0.0.1:30011"])]
        self.command_to = _addr(ctl.get("command_addr", "127.0.0.1:50051"))
        self.gt_to = _addr(ctl["gt_addr"]) if ctl.get("gt_addr") else None
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
        self.auto = bool(robot_cfg.get("auto_stand", True))
        self.start_delay = float(robot_cfg.get("start_delay", 1.0))
        self.stand_delay = float(robot_cfg.get("stand_delay", 4.0))
        self.sent = []                             # text commands sent (for tests / logs)
        self.restart(0.0)

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
        """kp (q_des - q) + kd (dq_des - dq) + tau_ff within the J60 envelope -> d.ctrl."""
        q = self.d.qpos[self.jq]
        dq = self.d.qvel[self.jv]
        with self.lock:
            tau = self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq) + self.tau_ff
        lo, hi = j60_limits(dq)
        self.d.ctrl[self.act] = np.clip(tau, lo, hi)

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

    def telemetry(self, t: float) -> bytes:
        d = self.d
        rpy = quat_to_rpy(d.qpos[3:7])
        acc = d.sensordata[self.acc_adr:self.acc_adr + 3]
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        payload = np.concatenate([rpy, acc, gyro, d.qpos[self.jq], d.qvel[self.jv],
                                  d.qfrc_actuator[self.jv], d.qpos[0:3], d.qvel[0:3], self.foot_contacts()])
        return struct.pack(TELEMETRY_FMT, t, *payload.astype(np.float32))

    def send_telemetry(self, t: float):
        pkt = self.telemetry(t)
        for a in self.telemetry_to:
            try:
                self.out.sendto(pkt, a)
            except OSError:
                pass
        if self.gt_to is not None:
            d = self.d
            p, q, v, w = d.qpos[0:3], d.qpos[3:7], d.qvel[0:3], d.qvel[3:6]
            line = (f"SIM_POSE {t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                    f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")
            try:
                self.out.sendto(line.encode(), self.gt_to)
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
        self.stage = "idle" if self.auto else "manual"

    def tick(self, t: float, cmd: np.ndarray) -> list[dict]:
        """Per control tick: advance the sequence, forward cmd_vel, watch the controller link."""
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
            self.send_command(f"MODE {self.gait}")
            self.stage = "gait"
            events.append({"type": "controller", "state": self.gait.lower(), "t": t})
        if self.stage in ("gait", "manual"):
            self.send_command(f"AXES {cmd[0]:.3f} {cmd[1]:.3f} {cmd[2]:.3f}")
        if self.connected and last is not None:
            silent = time.monotonic() - last > 1.0
            if silent and not self.lost:
                self.lost = True
                events.append({"type": "controller", "state": "lost", "t": t})
            elif not silent and self.lost:
                self.lost = False
                events.append({"type": "controller", "state": "connected", "t": t})
        return events

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
