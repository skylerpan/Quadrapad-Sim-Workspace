"""Noise on the IMU the g23 controller reads (rpy, gyro, acc in the telemetry packet).

MuJoCo's sensors and the torso quaternion are exact, so without this the controller's yaw never drifts and
every estimator downstream of it looks better than it is on the robot. The model is the usual one:

  gyro, acc    white noise + a bias: a constant drawn once per axis (gyro only) plus a random walk
  roll, pitch  white noise (an AHRS observes them through gravity, so they do not drift)
  yaw          the true yaw plus the integral of the gyro z bias (an AHRS without a magnetometer cannot
               observe yaw: its error is the integrated gyro bias)

All sigmas default to 0, which disables the model: apply() then returns its arguments untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class ImuNoiseParams:
    acc: float = 0.0            # m/s^2, white noise sigma per sample
    gyro: float = 0.0           # rad/s, white noise sigma per sample
    acc_bias_rw: float = 0.0    # m/s^2/sqrt(s), bias random walk
    gyro_bias_rw: float = 0.0   # rad/s/sqrt(s), bias random walk
    gyro_bias: float = 0.0      # rad/s, sigma of the constant bias drawn once per axis
    rp: float = 0.0             # rad, white noise sigma on roll and pitch
    seed: int = 0

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "ImuNoiseParams":
        cfg = cfg or {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"robot.controller.imu_noise: unknown key(s) {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        values = {k: (int(v) if k == "seed" else float(v)) for k, v in cfg.items() if v is not None}
        negative = sorted(k for k, v in values.items() if k != "seed" and v < 0)
        if negative:
            raise ValueError(f"robot.controller.imu_noise: negative sigma for {', '.join(negative)}")
        return cls(**values)

    @property
    def enabled(self) -> bool:
        return any(getattr(self, f.name) > 0 for f in fields(self) if f.name != "seed")

    def describe(self) -> str:
        return (f"IMU noise: gyro {self.gyro} rad/s (bias {self.gyro_bias}, walk {self.gyro_bias_rw}), "
                f"acc {self.acc} m/s^2 (walk {self.acc_bias_rw}), roll/pitch {self.rp} rad, seed {self.seed}")


class ImuNoise:
    """Stateful: call apply() exactly once per physics step."""

    def __init__(self, params: ImuNoiseParams, dt: float, seed: int | None = None):
        self.p = params
        self.dt = float(dt)
        self.rng = np.random.default_rng(params.seed if seed is None else seed)
        self.gyro_bias = self.rng.normal(0.0, params.gyro_bias, 3) if params.gyro_bias > 0 else np.zeros(3)
        self.acc_bias = np.zeros(3)
        self.yaw_drift = 0.0

    @property
    def enabled(self) -> bool:
        return self.p.enabled

    def apply(self, rpy, gyro, acc):
        if not self.enabled:
            return rpy, gyro, acc
        p, rng, root_dt = self.p, self.rng, np.sqrt(self.dt)
        if p.gyro_bias_rw > 0:
            self.gyro_bias = self.gyro_bias + rng.normal(0.0, p.gyro_bias_rw * root_dt, 3)
        if p.acc_bias_rw > 0:
            self.acc_bias = self.acc_bias + rng.normal(0.0, p.acc_bias_rw * root_dt, 3)
        self.yaw_drift += self.gyro_bias[2] * self.dt
        gyro = np.asarray(gyro, dtype=float) + self.gyro_bias
        acc = np.asarray(acc, dtype=float) + self.acc_bias
        if p.gyro > 0:
            gyro = gyro + rng.normal(0.0, p.gyro, 3)
        if p.acc > 0:
            acc = acc + rng.normal(0.0, p.acc, 3)
        rpy = np.array(rpy, dtype=float)
        if p.rp > 0:
            rpy[:2] += rng.normal(0.0, p.rp, 2)
        if self.yaw_drift != 0.0:
            rpy[2] = (rpy[2] + self.yaw_drift + np.pi) % (2.0 * np.pi) - np.pi
        return rpy, gyro, acc
