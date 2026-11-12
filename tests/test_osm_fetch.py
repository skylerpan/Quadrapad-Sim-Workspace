"""pcd2sim.osm_scene.fetch_overpass: download once, try the next server on an error page or a network error, write
the cache whole or not at all (urlopen faked: no network)."""
import io
import json
import os
import urllib.parse

import pytest

from pcd2sim import osm_scene


class _Server:
    def __init__(self, answers):
        self.answers, self.calls = dict(answers), []

    def __call__(self, req, timeout):
        self.calls.append((req.full_url, urllib.parse.parse_qs(req.data.decode())["data"][0]))
        a = self.answers[req.full_url]
        if isinstance(a, Exception):
            raise a
        return io.BytesIO(a)


def test_an_error_page_is_skipped_and_the_next_server_answers(tmp_path, monkeypatch):
    srv = _Server({"https://a": b"<html>rate limited</html>", "https://b": b'{"elements": []}'})
    monkeypatch.setattr(osm_scene.urllib.request, "urlopen", srv)
    path = str(tmp_path / "osm" / "ground.json")
    osm_scene.fetch_overpass(path, "QUERY", urls=("https://a", "https://b"))
    assert json.load(open(path)) == {"elements": []} and not os.path.exists(path + ".part")
    assert [u for u, _ in srv.calls] == ["https://a", "https://b"] and srv.calls[0][1] == "QUERY"
    osm_scene.fetch_overpass(path, "QUERY", urls=("https://a", "https://b"))       # cached: no more calls
    assert len(srv.calls) == 2


def test_when_every_server_fails_nothing_is_written(tmp_path, monkeypatch):
    srv = _Server({"https://a": OSError("down"), "https://b": b"not json"})
    monkeypatch.setattr(osm_scene.urllib.request, "urlopen", srv)
    path = str(tmp_path / "ground.json")
    with pytest.raises(RuntimeError, match="https://a: down"):
        osm_scene.fetch_overpass(path, "Q", urls=("https://a", "https://b"))
    assert os.listdir(tmp_path) == []
