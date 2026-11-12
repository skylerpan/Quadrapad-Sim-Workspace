"""Overpass answers built from ENU metres around a centre, for the OSM tests."""
import numpy as np

from quadsim.gnss import Datum, enu_to_lla

ENU0 = Datum(lat=24.95919, lon=121.41896, alt=0.0, yaw_deg=0.0)       # the streets.gl view centre, heading 0


def geom(enu_xy, close=True, centre: Datum = ENU0):
    pts = [enu_to_lla(np.array([e, n, 0.0]), centre) for e, n in enu_xy]
    if close:
        pts.append(pts[0])
    return [{"lat": la, "lon": lo} for la, lo, _ in pts]


def square(cx, cy, half, osm_id=1, centre: Datum = ENU0, **tags):
    ring = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    return {"type": "way", "id": osm_id, "tags": {"building": "yes", **tags}, "geometry": geom(ring, centre=centre)}
