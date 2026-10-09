"""Minimal PCD reader/writer (ascii + binary) built on numpy.

Handles arbitrary FIELDS/SIZE/TYPE/COUNT layouts, e.g. FAST-LIO maps
(x y z intensity as float32) and exported maps with float64 xyz plus extra
fields. binary_compressed falls back to open3d (xyz + intensity only).
"""
from __future__ import annotations

import numpy as np

_TYPE_MAP = {
    ("F", 4): "f4", ("F", 8): "f8",
    ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("U", 8): "u8",
    ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4", ("I", 8): "i8",
}


def _parse_header(f):
    header = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected EOF in PCD header")
        line = line.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
        if key.upper() == "DATA":
            return header


def _dtype(header) -> np.dtype:
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    descr, seen = [], set()
    for i, (name, size, typ, cnt) in enumerate(zip(fields, sizes, types, counts)):
        if name == "_" or name in seen:  # padding / duplicate names
            name = f"_pad{i}"
        seen.add(name)
        base = "<" + _TYPE_MAP[(typ.upper(), size)]
        descr.append((name, base) if cnt == 1 else (name, base, (cnt,)))
    return np.dtype(descr)


def read_pcd(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (xyz float64 [N,3], intensity float32 [N] or None)."""
    with open(path, "rb") as f:
        header = _parse_header(f)
        npts = int(header["POINTS"][0])
        fmt = header["DATA"][0].lower()
        dt = _dtype(header)
        if fmt == "binary":
            arr = np.frombuffer(f.read(npts * dt.itemsize), dtype=dt, count=npts)
        elif fmt == "ascii":
            arr = np.loadtxt(f, dtype=dt, max_rows=npts)
        elif fmt == "binary_compressed":
            return _read_with_open3d(path)
        else:
            raise ValueError(f"unsupported PCD DATA format: {fmt}")
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    inten = arr["intensity"].astype(np.float32) if "intensity" in dt.names else None
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], (inten[ok] if inten is not None else None)


def _read_with_open3d(path):
    import open3d as o3d

    pc = o3d.t.io.read_point_cloud(path)
    xyz = pc.point.positions.numpy().astype(np.float64)
    inten = pc.point.intensity.numpy().reshape(-1).astype(np.float32) if "intensity" in pc.point else None
    return xyz, inten


def write_pcd(path: str, xyz: np.ndarray, intensity: np.ndarray | None = None) -> None:
    """Write a binary PCD with float32 x y z [intensity]."""
    n = len(xyz)
    names = ["x", "y", "z"] + (["intensity"] if intensity is not None else [])
    arr = np.empty(n, dtype=[(k, "<f4") for k in names])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        arr["intensity"] = intensity
    k = len(names)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {' '.join(names)}\nSIZE {' '.join(['4'] * k)}\nTYPE {' '.join(['F'] * k)}\n"
        f"COUNT {' '.join(['1'] * k)}\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
