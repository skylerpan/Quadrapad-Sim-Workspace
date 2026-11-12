"""A u-blox ZED-F9P / ZED-F9R receiver on the robot, as pure functions: nothing here touches MuJoCo or ROS.

Glossary
  datum        lat / lon / ellipsoid alt of the map origin plus the heading of the map x axis from east
               (degrees, counter-clockwise): the scene's meta.json `geo`, else config/sim.yaml gnss.datum
               (resolve_datum). Without a datum there is no GNSS.
  antenna      the `gnss` site on the base (gnss.mount_xyz); the sim hands its map position and velocity in.
  sky          the fraction of rays in a cone around zenith that leave the scene without a hit: 0 indoors,
               1 in the open. The sim casts them in the LiDAR's Embree scene; `sky.mode: always` skips that.
  fix type     none | single | rtk_float | rtk_fixed | dead_reckoning, decided by `fix_for` from the sky
               fraction, whether RTK corrections exist, the model, and how long the sky has been gone.
  holdover     F9R only: with the sky gone the receiver dead-reckons on its IMU for `holdover.secs`, the
               position walking away at `holdover.drift` m/sqrt(s) and the reported sigma growing; an F9P
               reports no fix at once.
Noise per fix type: white noise (sigma_h, sigma_v) plus a first-order Gauss-Markov bias (bias_sigma,
bias_tau) for multipath-like drift, and white noise on the velocity (vel_sigma).
Geodesy: WGS84 through ECEF, exact at any distance; `map_to_enu` only rotates by the datum heading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

FixType = Literal["none", "single", "rtk_float", "rtk_fixed", "dead_reckoning"]
MODELS = ("f9p", "f9r")
NOISE_TYPES = ("rtk_fixed", "rtk_float", "single")
NOISE_KEYS = ("sigma_h", "sigma_v", "bias_sigma", "bias_tau", "vel_sigma")


@dataclass(frozen=True)
class Datum:
    lat: float
    lon: float
    alt: float
    yaw_deg: float

    @classmethod
    def from_cfg(cls, d: dict) -> "Datum":
        try:
            datum = cls(float(d["lat"]), float(d["lon"]), float(d["alt"]), float(d["yaw_deg"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"gnss.datum needs lat, lon, alt, yaw_deg: {d!r}") from e
        if not (-90.0 <= datum.lat <= 90.0 and -180.0 <= datum.lon <= 180.0):
            raise ValueError(f"gnss.datum lat/lon out of range: {d!r}")
        return datum


def resolve_datum(scene_meta: dict, cfg_datum: dict | None) -> Datum | None:
    """The one place that picks the datum: a georeferenced scene carries its own (meta.json `geo`, written by
    pcd2sim.osm_scene); a scene without one (dp, built from a scan) falls back to config gnss.datum."""
    geo = scene_meta.get("geo")
    if geo:
        return Datum.from_cfg(geo)
    return Datum.from_cfg(cfg_datum) if cfg_datum else None


@dataclass
class GnssFix:
    t: float
    fix: FixType
    lat: float
    lon: float
    alt: float
    vel_enu: np.ndarray
    sigma_h: float
    sigma_v: float
    sky: float


# -- geodesy -----------------------------------------------------------------------------------------------
def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo), (n + alt) * math.cos(la) * math.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt) * math.sin(la)])


def ecef_to_lla(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in xyz)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    alt = 0.0
    for _ in range(10):                             # converges to well under a micrometre in a few rounds
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(math.atan2(y, x)), alt


def _enu_rotation(datum: Datum) -> np.ndarray:
    """Rows are the east, north, up unit vectors in ECEF at the datum."""
    la, lo = math.radians(datum.lat), math.radians(datum.lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def enu_to_lla(enu: np.ndarray, datum: Datum) -> tuple[float, float, float]:
    return ecef_to_lla(lla_to_ecef(datum.lat, datum.lon, datum.alt) + _enu_rotation(datum).T @ np.asarray(enu, float))


def lla_to_enu(lat: float, lon: float, alt: float, datum: Datum) -> np.ndarray:
    return _enu_rotation(datum) @ (lla_to_ecef(lat, lon, alt) - lla_to_ecef(datum.lat, datum.lon, datum.alt))


def map_to_enu(xyz: np.ndarray, datum: Datum) -> np.ndarray:
    """Map frame -> ENU: a rotation about the vertical by the datum heading (the map z axis is up)."""
    c, s = math.cos(math.radians(datum.yaw_deg)), math.sin(math.radians(datum.yaw_deg))
    x, y, z = (float(v) for v in xyz)
    return np.array([c * x - s * y, s * x + c * y, z])


def enu_to_map(enu: np.ndarray, datum: Datum) -> np.ndarray:
    """ENU -> map frame: the inverse of map_to_enu."""
    c, s = math.cos(math.radians(datum.yaw_deg)), math.sin(math.radians(datum.yaw_deg))
    e, n, u = (float(v) for v in enu)
    return np.array([c * e + s * n, -s * e + c * n, u])


enu_velocity = map_to_enu                         # a velocity rotates the same way


# -- sky ---------------------------------------------------------------------------------------------------
def sky_directions(n: int, half_angle_deg: float) -> np.ndarray:
    """Unit vectors: zenith, then n-1 directions evenly spaced in azimuth at half_angle from zenith."""
    a = math.radians(half_angle_deg)
    az = np.linspace(0.0, 2.0 * math.pi, max(n - 1, 0), endpoint=False)
    ring = np.stack([math.sin(a) * np.cos(az), math.sin(a) * np.sin(az), np.full_like(az, math.cos(a))], axis=1)
    return np.vstack([[0.0, 0.0, 1.0], ring])


def sky_fraction(t_hit: np.ndarray, range_m: float) -> float:
    """The share of rays that hit nothing within range_m (Embree gives inf on a miss)."""
    return float(np.mean(np.asarray(t_hit) > range_m))


def fix_for(sky: float, sky_cfg: dict, rtk: bool, model: str, outage_s: float, holdover_s: float) -> FixType:
    """The one place that decides the fix type."""
    if sky >= sky_cfg["good"]:
        return "rtk_fixed" if rtk else "single"
    if sky >= sky_cfg["degraded"]:
        return "rtk_float" if rtk else "single"
    if sky >= sky_cfg["poor"]:
        return "single"
    if model == "f9r" and outage_s < holdover_s:
        return "dead_reckoning"
    return "none"


# -- the receiver ------------------------------------------------------------------------------------------
class GnssModel:
    """Owns the receiver's noise state (bias, holdover) and turns the antenna's true motion into fixes."""

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.datum = Datum.from_cfg(cfg["datum"])
        self.model = str(cfg.get("model", "f9p"))
        if self.model not in MODELS:
            raise ValueError(f"gnss.model={self.model!r}: expected one of {MODELS}")
        self.rate_hz = float(cfg["rate_hz"])
        if self.rate_hz <= 0:
            raise ValueError(f"gnss.rate_hz={self.rate_hz}: must be positive")
        self.rtk = bool(cfg.get("rtk", False))
        self.sky_cfg = dict(cfg["sky"])
        if not self.sky_cfg["good"] >= self.sky_cfg["degraded"] >= self.sky_cfg["poor"] >= 0.0:
            raise ValueError(f"gnss.sky thresholds must be good >= degraded >= poor >= 0: {self.sky_cfg}")
        self.noise = {k: {n: float(cfg["noise"][k][n]) for n in NOISE_KEYS} for k in NOISE_TYPES}
        hold = cfg.get("holdover") or {}
        self.holdover_s = float(hold.get("secs", 0.0))
        self.drift = float(hold.get("drift", 0.0))
        self.rng = rng
        self.bias = np.zeros(3)
        self.outage_since: float | None = None
        self.dr_offset = np.zeros(3)
        self.last = self.noise["single"]           # the noise the dead-reckoning starts from
        self.latest: GnssFix | None = None         # the last fix, for the viewers

    def vel_sigma(self, fix: GnssFix) -> float:
        """The velocity noise the fix was made with (dead reckoning keeps the last good value)."""
        if fix.fix == "none":
            return math.inf
        return self.noise[fix.fix]["vel_sigma"] if fix.fix in self.noise else self.last["vel_sigma"]

    def sample(self, t: float, pos_map: np.ndarray, vel_map: np.ndarray, sky: float) -> GnssFix:
        self.latest = self._sample(t, pos_map, vel_map, sky)
        return self.latest

    def _sample(self, t: float, pos_map: np.ndarray, vel_map: np.ndarray, sky: float) -> GnssFix:
        dt = 1.0 / self.rate_hz
        if sky >= self.sky_cfg["poor"]:
            self.outage_since = None
        elif self.outage_since is None:
            self.outage_since = t
        outage = 0.0 if self.outage_since is None else t - self.outage_since
        fix = fix_for(sky, self.sky_cfg, self.rtk, self.model, outage, self.holdover_s)
        if fix == "none":
            self.dr_offset[:] = 0.0
            return GnssFix(t, fix, math.nan, math.nan, math.nan, np.full(3, math.nan), math.inf, math.inf, sky)
        enu = map_to_enu(pos_map, self.datum)
        if fix == "dead_reckoning":
            self.dr_offset += self.rng.normal(0.0, self.drift * math.sqrt(dt), 3)
            enu = enu + self.bias + self.dr_offset
            sigma_h, sigma_v = self.last["sigma_h"] + self.drift * outage, self.last["sigma_v"] + self.drift * outage
            vel_sigma = self.last["vel_sigma"]
        else:
            n = self.noise[fix]
            decay = math.exp(-dt / n["bias_tau"])
            self.bias = self.bias * decay + self.rng.normal(0.0, n["bias_sigma"] * math.sqrt(1.0 - decay * decay), 3)
            enu = enu + self.bias + self.rng.normal(0.0, [n["sigma_h"], n["sigma_h"], n["sigma_v"]])
            sigma_h, sigma_v, vel_sigma = n["sigma_h"], n["sigma_v"], n["vel_sigma"]
            self.last = n
            self.dr_offset[:] = 0.0
        lat, lon, alt = enu_to_lla(enu, self.datum)
        vel = enu_velocity(vel_map, self.datum) + self.rng.normal(0.0, vel_sigma, 3)
        return GnssFix(t, fix, lat, lon, alt, vel, sigma_h, sigma_v, sky)
