"""pcd2sim.osm_scene.build with a base scene: OSM is laid around a scanned scene in the scan's own map frame
(its datum and heading), the OSM buildings and ground are cut out where the scan has geometry, the scan is kept
as it was, and the fit between the scan's building and its OSM outline is measured."""
import json
import os

import numpy as np
import pytest

from pcd2sim.osm_scene import build
from quadsim.gnss import Datum, enu_to_lla, map_to_enu
from tests.osm_fixture import geom

BASE_DATUM = Datum(lat=24.95919, lon=121.41896, alt=10.0, yaw_deg=31.0)
FLOOR = -0.4


def _box(x0, x1, y0, y1, z0, z1):
    return [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2]


def _base(tmp_path, geo=True):
    """A scanned 12 x 12 m building at the base origin: floor, four walls 4 m high, a ceiling, and the scanned
    yard east of it (low: not part of the building)."""
    d = tmp_path / "base"
    d.mkdir()
    boxes = np.array([_box(-6, 6, -6, 6, FLOOR - 0.2, FLOOR),
                      _box(-6, 6, -6, -5.8, FLOOR, 3.6), _box(-6, 6, 5.8, 6, FLOOR, 3.6),
                      _box(-6, -5.8, -6, 6, FLOOR, 3.6), _box(5.8, 6, -6, 6, FLOOR, 3.6),
                      _box(-6, 6, -6, 6, 3.4, 3.6), _box(6, 16, -6, 6, FLOOR - 0.2, FLOOR)], np.float32)
    np.save(d / "boxes.npy", boxes)
    np.save(d / "contact.npy", np.array([True, True, True, False, False, False, True]))
    np.savez_compressed(d / "voxels.npz", keys=np.zeros((1, 3), np.int32), intensity=np.full(1, 7.0, np.float32),
                        zmean=np.zeros(1, np.float32), origin=np.zeros(3), voxel=0.2, dims=np.ones(3, np.int64),
                        contact=np.ones(1, bool))
    meta = {"format_version": 3, "voxel": 0.2, "gravity": [0, 0, -9.81], "spawn": [0.0, 0.0, FLOOR]}
    if geo:
        meta["geo"] = {"lat": BASE_DATUM.lat, "lon": BASE_DATUM.lon, "alt": BASE_DATUM.alt, "yaw_deg": BASE_DATUM.yaw_deg}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d), boxes


def _osm_at_base_xy(tmp_path, *buildings):
    """OSM squares given in the base map frame (axis-aligned there, so turned 31 deg in ENU): (x, y, half, osm_id)."""
    enu_frame = Datum(BASE_DATUM.lat, BASE_DATUM.lon, BASE_DATUM.alt, 0.0)
    els = []
    for x, y, h, osm_id in buildings:
        ring = [map_to_enu(np.array([x + dx, y + dy, 0.0]), BASE_DATUM)[:2] for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        els.append({"type": "way", "id": osm_id, "tags": {"building": "yes", "height": "10"},
                    "geometry": geom(ring, centre=enu_frame)})
    p = tmp_path / "overpass.json"
    p.write_text(json.dumps({"elements": els}))
    return str(p)


def _overlap(boxes, x0, x1, y0, y1):
    lo, hi = boxes[:, :2] - boxes[:, 3:5], boxes[:, :2] + boxes[:, 3:5]
    return (lo[:, 0] < x1) & (hi[:, 0] > x0) & (lo[:, 1] < y1) & (hi[:, 1] > y0)


def test_the_scan_is_kept_and_osm_is_cut_out_around_it(tmp_path):
    base, base_boxes = _base(tmp_path)
    osm = _osm_at_base_xy(tmp_path, (0.0, 0.0, 6.0, 1), (30.0, 0.0, 5.0, 2))     # the scanned building + a neighbour
    out = str(tmp_path / "scene")
    meta = build(BASE_DATUM.lat, BASE_DATUM.lon, radius=50.0, out=out, osm_path=osm, base=base, margin=2.0)
    boxes = np.load(os.path.join(out, "boxes.npy"))
    contact = np.load(os.path.join(out, "contact.npy"))
    n = len(base_boxes)
    assert np.array_equal(boxes[:n], base_boxes)                                     # the scan, untouched, first
    assert np.array_equal(contact[:n], [True, True, True, False, False, False, True]) and contact[n:].all()
    osm_part = boxes[n:]
    assert not _overlap(osm_part, -7.9, 17.9, -7.9, 7.9).any()                       # scan + 2 m margin left alone
    tall = osm_part[:, 2] + osm_part[:, 5] > FLOOR + 1.0
    assert _overlap(osm_part[tall], 25, 35, -5, 5).any()                             # the neighbour stands
    ground = osm_part[~tall]
    assert np.allclose(ground[:, 2] + ground[:, 5], FLOOR)                           # at the scan's floor height
    assert _overlap(ground, 18.5, 19.5, -0.5, 0.5).any() and _overlap(ground, -9.5, -8.5, -0.5, 0.5).any()
    g = meta["geo"]
    assert (g["lat"], g["lon"], g["yaw_deg"]) == (BASE_DATUM.lat, BASE_DATUM.lon, 31.0)
    assert meta["spawn"] == [0.0, 0.0, FLOOR] and meta["voxel"] == 0.2
    assert np.load(os.path.join(out, "voxels.npz"))["intensity"][0] == 7.0           # the scan's voxels
    fit = meta["overlay"]
    assert fit["osm_id"] == 1 and fit["iou"] > 0.9 and fit["offset_m"] < 0.5


def test_a_base_without_geo_takes_the_fallback_datum_and_a_shifted_scan_shows_in_the_fit(tmp_path):
    base, _ = _base(tmp_path, geo=False)
    osm = _osm_at_base_xy(tmp_path, (3.0, 0.0, 6.0, 1))                              # OSM outline 3 m off the scan
    fallback = {"lat": BASE_DATUM.lat, "lon": BASE_DATUM.lon, "alt": BASE_DATUM.alt, "yaw_deg": BASE_DATUM.yaw_deg}
    meta = build(BASE_DATUM.lat, BASE_DATUM.lon, radius=40.0, out=str(tmp_path / "scene"), osm_path=osm, base=base,
                 fallback_datum=fallback)
    assert meta["geo"]["yaw_deg"] == 31.0
    assert meta["overlay"]["offset_m"] == pytest.approx(3.0, abs=0.3)
    assert meta["overlay"]["iou"] == pytest.approx(9.0 / 15.0, abs=0.03)            # 9 x 12 shared of 15 x 12


def test_the_streets_gl_centre_sets_the_square_in_the_base_frame(tmp_path):
    base, _ = _base(tmp_path)
    centre_map = np.array([100.0, 0.0])
    e, n, _ = map_to_enu(np.array([*centre_map, 0.0]), BASE_DATUM)
    lat, lon, _ = enu_to_lla(np.array([e, n, 0.0]), BASE_DATUM)
    osm = _osm_at_base_xy(tmp_path)
    meta = build(lat, lon, radius=30.0, out=str(tmp_path / "scene"), osm_path=osm, base=base)
    assert meta["bbox_min"][0] == pytest.approx(-6.0, abs=0.01)                     # the scan is still there
    assert meta["bbox_max"][0] == pytest.approx(130.0, abs=0.01)                    # square edge: 100 + 30
    assert meta["spawn"] == [0.0, 0.0, FLOOR]                                       # the scan's, not the centre
