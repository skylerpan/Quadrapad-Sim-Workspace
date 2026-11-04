"""The wire of SyncAI-Device-MotorControl's simulation backend, as pure functions: nothing here touches a socket.

Three UDP packets, little-endian float32, no header, the length is the identity (src/devices/common/sim_packets.rs
on their side):

  controller -> sim   [pos, vel, tau, kp, kd] x 12 motors   240 bytes   torque = kp (pos - q) + kd (vel - dq) + tau
  sim -> controller   [q, dq, tau] x 12 motors              144 bytes
  sim -> controller   [roll, pitch, yaw, wx, wy, wz, ax, ay, az]   36 bytes   radians, body frame, acc includes gravity

A packet of the wrong length, or holding a NaN / Inf anywhere, is dropped whole: the fields come from one simulation
step, a bad one means the step itself is bad.

Operator lines (src/operator/protocol/line.rs): `kind=<x> key=value ...`, an empty field is `-`, spaces inside a value
became `_`. Only status, request and event lines are parsed; the 20 Hz `kind=state` line is not needed here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np

MOTORS = 12
_CMD_FMT = "<" + "f" * (MOTORS * 5)
_JOINT_FMT = "<" + "f" * (MOTORS * 3)
_BASE_FMT = "<9f"
MC_CMD_BYTES = struct.calcsize(_CMD_FMT)
MC_JOINT_BYTES = struct.calcsize(_JOINT_FMT)
MC_BASE_BYTES = struct.calcsize(_BASE_FMT)

Behavior = Literal["rest", "stand", "rl_fcn", "rl_him", "gait_mpc"]
BEHAVIORS: tuple[str, ...] = ("rest", "stand", "rl_fcn", "rl_him", "gait_mpc")
# quadsim's policy_mode integers (ROS /policy_mode, the web viewer, walk_g23 --switch) to the controller's RL behaviors;
# other slots (4 = IsaacSim on g2_deploy_rs) have no counterpart
POLICY_TO_BEHAVIOR: dict[int, str] = {0: "rl_fcn", 1: "rl_him"}


def encode_joint(q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> bytes:
    return struct.pack(_JOINT_FMT, *np.stack([q, dq, tau], axis=1).astype(np.float32).ravel())


def encode_base(rpy: np.ndarray, omega: np.ndarray, acc: np.ndarray) -> bytes:
    return struct.pack(_BASE_FMT, *np.concatenate([rpy, omega, acc]).astype(np.float32))


def decode_command(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """(pos, vel, tau, kp, kd), each (12,), or None for a packet of the wrong length or with a non-finite value."""
    if len(data) != MC_CMD_BYTES:
        return None
    v = np.frombuffer(data, dtype="<f4").astype(float).reshape(MOTORS, 5)
    if not np.all(np.isfinite(v)):
        return None
    return v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]
