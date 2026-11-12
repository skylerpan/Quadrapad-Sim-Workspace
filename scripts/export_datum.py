#!/usr/bin/env python3
"""Write a scene's datum as the nav stack's map sidecar (<map>.geo.yaml), so the simulator's GNSS and the nav
stack's navsat_transform_node (syncai_gnss_localization) use the same georeference.

  python3 scripts/export_datum.py                         # print the config scene's datum as YAML
  python3 scripts/export_datum.py --scene scenes/osm_tucheng_dp -o /path/to/maps/dp/dp.geo.yaml

The datum is resolved the way the sim does it (quadsim.gnss.resolve_datum): the scene's meta.json `geo`, else
config/sim.yaml gnss.datum. The sidecar goes next to the nav stack's localization map: [map] pcd = .../dp.pcd
-> .../dp.geo.yaml. No ROS needed.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quadsim.config import load_config, load_scene_meta  # noqa: E402
from quadsim.gnss import Datum, resolve_datum  # noqa: E402


def sidecar_text(datum: Datum, scene: str) -> str:
    return (f"# Georeference of the map: origin lat / lon [deg], ellipsoid height [m], and the direction of the map +x\n"
            f"# axis counter-clockwise from east [deg]. Exported from the simulator scene {scene}\n"
            f"# (Quadrapad-Sim-Workspace scripts/export_datum.py); read by syncai_gnss_localization.\n"
            f"datum: {{lat: {datum.lat!r}, lon: {datum.lon!r}, alt: {datum.alt!r}, yaw_deg: {datum.yaw_deg!r}}}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene", default=None, help="scene directory; default: the config's scene")
    ap.add_argument("-o", "--output", default=None, help="file to write; default: print to stdout")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    scene = a.scene or cfg["scene"]
    datum = resolve_datum(load_scene_meta(scene), cfg["gnss"].get("datum"))
    if datum is None:
        print(f"no datum: scene {scene} has no geo and config gnss.datum is null", file=sys.stderr)
        return 1
    text = sidecar_text(datum, scene)
    if a.output is None:
        sys.stdout.write(text)
    else:
        with open(a.output, "w") as f:
            f.write(text)
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
