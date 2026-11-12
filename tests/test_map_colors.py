"""quadsim.map_colors: the one place that decides a map colour, for the MuJoCo viewer and the browser alike."""
import numpy as np
import pytest

from quadsim.map_colors import PLAIN_RGB, height_colors, map_colors, parse_map_color

Z = np.array([0.0, 5.0, 10.0])
RGBA = np.array([[200, 10, 10, 255], [0, 0, 0, 0], [10, 200, 10, 255]], np.uint8)     # middle row: no OSM colour


def test_osm_mode_uses_the_scene_colour_and_height_where_there_is_none():
    c = map_colors(Z, RGBA, "osm")
    assert c.dtype == np.uint8 and c.shape == (3, 3)
    assert c[0].tolist() == [200, 10, 10] and c[2].tolist() == [10, 200, 10]
    assert c[1].tolist() == height_colors(Z)[1].tolist()


def test_height_mode_ignores_the_scene_colours():
    assert np.array_equal(map_colors(Z, RGBA, "height"), height_colors(Z))


def test_plain_mode_is_one_grey():
    assert (map_colors(Z, RGBA, "plain") == PLAIN_RGB).all()


def test_a_scene_without_colours_is_coloured_by_height_in_osm_mode():
    assert np.array_equal(map_colors(Z, None, "osm"), height_colors(Z))


def test_the_mode_is_parsed_once_and_anything_else_is_refused():
    assert parse_map_color("OSM") == "osm" and parse_map_color(" plain ") == "plain"
    with pytest.raises(ValueError, match="map_color"):
        parse_map_color("rainbow")
