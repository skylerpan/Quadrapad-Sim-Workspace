"""pcd2sim.osm_colors: OSM tags -> colours, parsed at the boundary (pure functions, no files)."""
import pytest

from pcd2sim.osm_colors import (BUILDING_PALETTE, GROUND_RGB, GroundClass, building_colour, ground_class,
                                parse_colour, road_width)


@pytest.mark.parametrize("raw, rgb", [("#996B1F", (0x99, 0x6B, 0x1F)), ("#fff", (255, 255, 255)),
                                      (" Grey ", (128, 128, 128)), ("gray", (128, 128, 128)), ("brown", (165, 42, 42))])
def test_colours_parse_from_hex_and_names(raw, rgb):
    assert parse_colour(raw) == rgb


@pytest.mark.parametrize("raw", ["", "#12", "#gggggg", "sort of red", None, "rgb(1,2,3)"])
def test_anything_else_is_no_colour(raw):
    assert parse_colour(raw) is None


def test_a_building_takes_its_tag_then_its_roof_then_its_type_then_the_default():
    assert building_colour({"building": "school", "building:colour": "#000000"}) == (0, 0, 0)
    assert building_colour({"building": "school", "roof:colour": "white"}) == (255, 255, 255)
    assert building_colour({"building": "school", "building:colour": "nonsense"}) == BUILDING_PALETTE["school"]
    assert building_colour({"building": "school"}) == BUILDING_PALETTE["school"]
    assert building_colour({"building": "spaceport"}) == BUILDING_PALETTE["default"]
    assert BUILDING_PALETTE["school"] != BUILDING_PALETTE["apartments"]


@pytest.mark.parametrize("tags, cls", [
    ({"highway": "residential"}, GroundClass.ROAD), ({"highway": "primary_link"}, GroundClass.ROAD),
    ({"highway": "footway"}, GroundClass.PATH), ({"amenity": "parking"}, GroundClass.PARKING),
    ({"leisure": "park"}, GroundClass.GRASS), ({"landuse": "grass"}, GroundClass.GRASS),
    ({"natural": "water"}, GroundClass.WATER), ({"waterway": "riverbank"}, GroundClass.WATER),
    ({"landuse": "forest"}, GroundClass.FOREST), ({"landuse": "farmland"}, GroundClass.FARM),
    ({"landuse": "residential"}, GroundClass.BUILT), ({"highway": "bus_stop"}, None), ({"name": "x"}, None)])
def test_ground_classes_from_tags(tags, cls):
    assert ground_class(tags) is cls


def test_roads_paint_over_water_over_parks_over_built_land():
    assert GroundClass.ROAD > GroundClass.WATER > GroundClass.GRASS > GroundClass.BUILT > GroundClass.BARE
    assert set(GROUND_RGB) == set(GroundClass)


def test_road_width_from_the_width_tag_then_lanes_then_the_class():
    assert road_width({"highway": "residential", "width": "7.5 m"}) == 7.5
    assert road_width({"highway": "residential", "lanes": "4"}) == 14.0
    assert road_width({"highway": "primary"}) == 10.0
    assert road_width({"highway": "footway"}) == 2.0
    assert road_width({"highway": "residential", "width": "wide"}) == 6.0
