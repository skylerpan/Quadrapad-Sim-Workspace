"""Rolling Livox Mid-360-like LiDAR.

Every ray has its own emission time. Rays are cast from the recorded sensor pose
nearest to their emission time (<= half a physics step away), so motion distortion appears
naturally (what Point-LIO / FAST-LIO expect from real hardware). Points are
expressed in the sensor frame at their emission time, like the real driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GOLDEN = 0.6180339887498949


def _radical_inverse_base2(i: np.ndarray) -> np.ndarray:
    """Van der Corput sequence (base 2) for uint64 indices."""
    i = i.astype(np.uint64)
    i = ((i >> np.uint64(1)) & np.uint64(0x5555555555555555)) | ((i & np.uint64(0x5555555555555555)) << np.uint64(1))
    i = ((i >> np.uint64(2)) & np.uint64(0x3333333333333333)) | ((i & np.uint64(0x3333333333333333)) << np.uint64(2))
    i = ((i >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F)) | ((i & np.uint64(0x0F0F0F0F0F0F0F0F)) << np.uint64(4))
    i = ((i >> np.uint64(8)) & np.uint64(0x00FF00FF00FF00FF)) | ((i & np.uint64(0x00FF00FF00FF00FF)) << np.uint64(8))
    i = ((i >> np.uint64(16)) & np.uint64(0x0000FFFF0000FFFF)) | ((i & np.uint64(0x0000FFFF0000FFFF)) << np.uint64(16))
    i = (i >> np.uint64(32)) | (i << np.uint64(32))
    return i.astype(np.float64) / 2.0 ** 64


class ScanPattern:
    """Maps global ray indices to (time, unit direction in sensor frame, line).

    synthetic: one azimuth revolution per frame, elevation from a base-2
    low-discrepancy sequence over the whole run, so successive frames hit new
    elevations (non-repetitive coverage that densifies with integration time).
    csv: a recorded table (t, azimuth_deg, elevation_deg[, line]) replayed periodically.
    """

    def __init__(self, cfg: dict):
        self.rate = float(cfg["points_per_sec"])
        self.frame_hz = float(cfg["rate_hz"])
        self.el_lo, self.el_hi = np.deg2rad(cfg["fov_deg"])
        self.lines = int(cfg.get("lines", 4))
        self.table = None
        if cfg.get("pattern", "synthetic") != "synthetic":
            tab = np.loadtxt(cfg["pattern"], delimiter=",", ndmin=2, comments="#")
            order = np.argsort(tab[:, 0])
            tab = tab[order]
            self.table = {
                "t": tab[:, 0] - tab[0, 0],
                "dir": _sph_to_dir(np.deg2rad(tab[:, 1]), np.deg2rad(tab[:, 2])).astype(np.float32),
                "line": (tab[:, 3] if tab.shape[1] > 3 else np.arange(len(tab)) % self.lines).astype(np.uint8),
            }
            dt = np.median(np.diff(self.table["t"])) if len(tab) > 1 else 1.0 / self.rate
            self.period = float(self.table["t"][-1] + dt)
            self.rate = len(tab) / self.period

    def rays(self, t0: float, t1: float):
        """Rays emitted in [t0, t1): (times [N], dirs [N,3] float32, line [N] uint8)."""
        if self.table is None:
            g0, g1 = int(np.ceil(t0 * self.rate)), int(np.ceil(t1 * self.rate))
            g = np.arange(g0, g1, dtype=np.int64)
            t = g / self.rate
            frame = np.floor(t * self.frame_hz)
            phase = (frame * _GOLDEN) % 1.0                       # interleave revolutions
            az = 2 * np.pi * ((t * self.frame_hz) % 1.0 + phase / (self.rate / self.frame_hz))
            el = self.el_lo + (self.el_hi - self.el_lo) * _radical_inverse_base2(g)
            return t, _sph_to_dir(az, el).astype(np.float32), (g % self.lines).astype(np.uint8)
        # replay recorded table
        n0, n1 = int(np.floor(t0 / self.period)), int(np.floor(t1 / self.period))
        ts, ds, ls = [], [], []
        for n in range(n0, n1 + 1):
            base = n * self.period
            lo = np.searchsorted(self.table["t"], t0 - base, "left")
            hi = np.searchsorted(self.table["t"], t1 - base, "left")
            ts.append(self.table["t"][lo:hi] + base)
            ds.append(self.table["dir"][lo:hi])
            ls.append(self.table["line"][lo:hi])
        return np.concatenate(ts), np.concatenate(ds), np.concatenate(ls)


def _sph_to_dir(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1)


@dataclass
class LidarFrame:
    stamp: float            # frame start time (s) == livox timebase
    xyz: np.ndarray         # [N,3] float32, sensor frame at each point's emission time
    reflectivity: np.ndarray  # [N] uint8
    line: np.ndarray        # [N] uint8
    offset: np.ndarray      # [N] float64 seconds since stamp (monotonic)


class RollingLidar:
    def __init__(self, cfg: dict, ray_scene, intensity_lookup, voxel: float):
        self.cfg = cfg
        self.pattern = ScanPattern(cfg)
        self.scene = ray_scene
        self.intensity = intensity_lookup
        self.inside = 0.25 * voxel                      # nudge into the hit voxel for intensity lookup
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.frame_period = 1.0 / float(cfg["rate_hz"])
        self._pose_t: list[float] = []
        self._pose_p: list[np.ndarray] = []
        self._pose_R: list[np.ndarray] = []
        self._t_done = None                             # rays emitted before this time are processed
        self._frame_idx = None
        self._buf: list[tuple] = []

    def reset(self, t: float):
        self._pose_t.clear(); self._pose_p.clear(); self._pose_R.clear()
        self._t_done = t
        self._frame_idx = int(np.floor(t / self.frame_period))
        self._buf.clear()

    def record_pose(self, t: float, pos: np.ndarray, R: np.ndarray):
        """Sensor pose at sim time t (poses must be recorded in increasing t)."""
        self._pose_t.append(t)
        self._pose_p.append(pos.copy())
        self._pose_R.append(R.copy())

    def process(self) -> list[LidarFrame]:
        """Cast every ray emitted up to the last recorded pose; return completed frames."""
        if not self._pose_t:
            return []
        t_end = self._pose_t[-1]
        times, dirs, line = self.pattern.rays(self._t_done, t_end)
        frames: list[LidarFrame] = []
        if len(times):
            pt = np.asarray(self._pose_t)
            hi = np.clip(np.searchsorted(pt, times, "left"), 0, len(pt) - 1)
            lo = np.maximum(hi - 1, 0)
            idx = np.where(np.abs(pt[lo] - times) <= np.abs(pt[hi] - times), lo, hi)   # nearest pose
            P = np.asarray(self._pose_p)[idx]
            R = np.asarray(self._pose_R)[idx]
            d_world = np.einsum("nij,nj->ni", R, dirs)
            t_hit, normals = self.scene.cast(P, d_world)
            c = self.cfg
            ok = np.isfinite(t_hit) & (t_hit >= c["min_range"]) & (t_hit <= c["max_range"])
            ok &= np.abs(np.einsum("ni,ni->n", normals, d_world)) >= c.get("grazing_cos", 0.0)
            if c.get("dropout", 0.0) > 0:
                ok &= self.rng.random(len(t_hit)) >= c["dropout"]
            r = t_hit[ok] + self.rng.normal(0.0, c["range_noise"], ok.sum()).astype(np.float32)
            refl = self.intensity.reflectivity(P[ok] + d_world[ok] * (t_hit[ok] + self.inside)[:, None])
            self._buf.append((times[ok], dirs[ok] * r[:, None], refl, line[ok]))
        self._t_done = t_end
        # keep only the newest pose (needed for rays emitted right after t_end)
        self._pose_t = self._pose_t[-1:]; self._pose_p = self._pose_p[-1:]; self._pose_R = self._pose_R[-1:]
        frame_end = (self._frame_idx + 1) * self.frame_period
        while t_end >= frame_end:
            frames.append(self._pop_frame(frame_end))
            self._frame_idx += 1
            frame_end = (self._frame_idx + 1) * self.frame_period
        return frames

    def _pop_frame(self, frame_end: float) -> LidarFrame:
        start = frame_end - self.frame_period
        if self._buf:
            t = np.concatenate([b[0] for b in self._buf])
            xyz = np.concatenate([b[1] for b in self._buf]).astype(np.float32)
            refl = np.concatenate([b[2] for b in self._buf])
            line = np.concatenate([b[3] for b in self._buf])
        else:
            t = np.zeros(0); xyz = np.zeros((0, 3), np.float32)
            refl = np.zeros(0, np.uint8); line = np.zeros(0, np.uint8)
        cur = t < frame_end
        rest = ~cur
        self._buf = [(t[rest], xyz[rest], refl[rest], line[rest])] if rest.any() else []
        return LidarFrame(start, xyz[cur], refl[cur], line[cur], np.maximum(t[cur] - start, 0.0))
