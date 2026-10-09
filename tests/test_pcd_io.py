import numpy as np

from pcd2sim.pcd_io import read_pcd, write_pcd


def test_roundtrip_float32(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(1000, 3)).astype(np.float32)
    inten = rng.uniform(0, 255, 1000).astype(np.float32)
    p = tmp_path / "a.pcd"
    write_pcd(str(p), xyz, inten)
    xyz2, inten2 = read_pcd(str(p))
    np.testing.assert_allclose(xyz2, xyz, atol=1e-6)
    np.testing.assert_allclose(inten2, inten)


def test_mixed_layout_float64_with_extra_fields(tmp_path):
    # same layout as dpmap.pcd: double xyz + float extras + uint32 rgb + uint8
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f4"), ("normal_x", "<f4"),
                   ("normal_y", "<f4"), ("normal_z", "<f4"), ("curvature", "<f4"), ("rgb", "<u4"),
                   ("synthetic", "u1")])
    arr = np.zeros(5, dtype=dt)
    arr["x"], arr["y"], arr["z"] = np.arange(5) + 0.123456789, -np.arange(5), 2.0
    arr["intensity"] = [1, 2, 3, 4, 5]
    header = ("VERSION 0.7\nFIELDS x y z intensity normal_x normal_y normal_z curvature rgb synthetic\n"
              "SIZE 8 8 8 4 4 4 4 4 4 1\nTYPE F F F F F F F F U U\nCOUNT 1 1 1 1 1 1 1 1 1 1\n"
              "WIDTH 00000000000000000005\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 00000000000000000005\n"
              "DATA binary\n")
    p = tmp_path / "b.pcd"
    p.write_bytes(b"# .PCD v0.7\n" + header.encode() + arr.tobytes())
    xyz, inten = read_pcd(str(p))
    assert xyz.dtype == np.float64
    np.testing.assert_allclose(xyz[:, 0], arr["x"])
    np.testing.assert_allclose(inten, arr["intensity"])


def test_ascii(tmp_path):
    p = tmp_path / "c.pcd"
    p.write_text("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\n"
                 "POINTS 2\nDATA ascii\n1 2 3\n4 5 6\n")
    xyz, inten = read_pcd(str(p))
    assert inten is None
    np.testing.assert_allclose(xyz, [[1, 2, 3], [4, 5, 6]])
