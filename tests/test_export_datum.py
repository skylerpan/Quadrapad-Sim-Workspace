"""scripts/export_datum.py: the sidecar it writes carries the datum the sim resolves, unrounded."""
import importlib.util
import os

import yaml

from quadsim.gnss import Datum

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("export_datum", os.path.join(HERE, "..", "scripts", "export_datum.py"))
export_datum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_datum)


def test_sidecar_round_trips_the_datum():
    datum = Datum(24.960726893908642, 121.42204957522816, 130.0, 31.0)
    data = yaml.safe_load(export_datum.sidecar_text(datum, "scenes/osm_tucheng_dp"))
    assert Datum.from_cfg(data["datum"]) == datum


def test_sidecar_has_only_the_datum_key():
    data = yaml.safe_load(export_datum.sidecar_text(Datum(1.0, 2.0, 3.0, -45.5), "scenes/x"))
    assert set(data) == {"datum"} and set(data["datum"]) == {"lat", "lon", "alt", "yaw_deg"}
