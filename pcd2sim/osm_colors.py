"""OpenStreetMap tags -> scene colours, as pure functions: the boundary where colour tags are parsed.

Glossary
  colour        an sRGB (r, g, b) tuple of ints 0-255.
  building colour  `building:colour`, else `roof:colour` (OSM's own colour tags, hex or a CSS name), else the
                palette colour of its `building=*` type, else BUILDING_PALETTE["default"]. Near Tucheng 6 of 226
                buildings carry a colour tag, so the type palette is what tells the rest apart.
  ground class  what the ground is at a spot (GroundClass), from the tags of an Overpass way / relation:
                roads and paths (lines with a width), parking, parks and grass, water, forest, farmland and
                built-up landuse (areas). Its value is its paint priority: a higher class paints over a lower one
                (a road over a river is a bridge, a footpath through a park stays a path).
  road width    `width` (m), else `lanes` x LANE_WIDTH, else by `highway` class (ROAD_WIDTHS).
"""
from __future__ import annotations

import enum
import re

Colour = tuple[int, int, int]

CSS_COLOURS: dict[str, Colour] = {
    "white": (255, 255, 255), "black": (0, 0, 0), "grey": (128, 128, 128), "gray": (128, 128, 128),
    "silver": (192, 192, 192), "lightgrey": (211, 211, 211), "lightgray": (211, 211, 211),
    "darkgrey": (169, 169, 169), "darkgray": (169, 169, 169), "red": (255, 0, 0), "maroon": (128, 0, 0),
    "brown": (165, 42, 42), "beige": (245, 245, 220), "tan": (210, 180, 140), "orange": (255, 165, 0),
    "yellow": (255, 255, 0), "olive": (128, 128, 0), "green": (0, 128, 0), "teal": (0, 128, 128),
    "blue": (0, 0, 255), "navy": (0, 0, 128), "purple": (128, 0, 128), "pink": (255, 192, 203),
    "cream": (255, 253, 208), "ivory": (255, 255, 240), "sand": (194, 178, 128),
}

BUILDING_PALETTE: dict[str, Colour] = {
    "apartments": (214, 199, 178), "residential": (224, 204, 176), "house": (226, 196, 160),
    "detached": (226, 196, 160), "school": (206, 152, 120), "university": (206, 152, 120),
    "dormitory": (214, 170, 140), "industrial": (168, 172, 182), "warehouse": (168, 172, 182),
    "commercial": (196, 178, 206), "retail": (204, 168, 196), "office": (168, 190, 212),
    "temple": (190, 84, 62), "religious": (190, 84, 62), "parking": (150, 150, 154), "roof": (188, 190, 198),
    "public": (202, 190, 150), "government": (202, 190, 150), "service": (180, 180, 170),
    "default": (206, 196, 186),
}


class GroundClass(enum.IntEnum):
    """What the ground is; the value is the paint priority (higher paints over lower)."""
    BARE = 0
    BUILT = 1
    FARM = 2
    FOREST = 3
    GRASS = 4
    PARKING = 5
    WATER = 6
    PATH = 7
    ROAD = 8


GROUND_RGB: dict[GroundClass, Colour] = {
    GroundClass.BARE: (190, 190, 184), GroundClass.BUILT: (208, 204, 198), GroundClass.FARM: (196, 210, 150),
    GroundClass.FOREST: (96, 142, 88), GroundClass.GRASS: (152, 192, 120), GroundClass.PARKING: (150, 150, 156),
    GroundClass.WATER: (122, 172, 222), GroundClass.PATH: (196, 184, 166), GroundClass.ROAD: (96, 98, 104),
}

ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
         "living_street", "road", "busway"}
PATHS = {"footway", "path", "cycleway", "pedestrian", "steps", "track", "bridleway", "corridor"}
ROAD_WIDTHS = {"motorway": 14.0, "trunk": 12.0, "primary": 10.0, "secondary": 8.0, "tertiary": 7.0,
               "residential": 6.0, "unclassified": 6.0, "living_street": 5.0, "road": 6.0, "busway": 6.0,
               "service": 4.0}
PATH_WIDTH = 2.0
LANE_WIDTH = 3.5
_AREAS = (
    ("amenity", {"parking"}, GroundClass.PARKING),
    ("natural", {"water", "wetland"}, GroundClass.WATER),
    ("waterway", {"riverbank", "dock"}, GroundClass.WATER),
    ("landuse", {"reservoir", "basin"}, GroundClass.WATER),
    ("leisure", {"park", "garden", "pitch", "playground", "golf_course", "recreation_ground"}, GroundClass.GRASS),
    ("landuse", {"grass", "meadow", "recreation_ground", "village_green", "cemetery"}, GroundClass.GRASS),
    ("natural", {"grassland", "heath"}, GroundClass.GRASS),
    ("landuse", {"forest"}, GroundClass.FOREST),
    ("natural", {"wood", "scrub"}, GroundClass.FOREST),
    ("landuse", {"farmland", "orchard", "vineyard", "allotments", "plant_nursery"}, GroundClass.FARM),
    ("landuse", {"residential", "commercial", "industrial", "retail", "construction", "railway",
                 "religious", "education"}, GroundClass.BUILT),
)
_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def parse_colour(raw) -> Colour | None:
    """An OSM colour value (#rgb, #rrggbb or a CSS name) -> (r, g, b); None for anything else."""
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    m = _HEX.match(v)
    if m:
        h = m.group(1)
        h = "".join(c * 2 for c in h) if len(h) == 3 else h
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return CSS_COLOURS.get(v)


def building_colour(tags: dict) -> Colour:
    """building:colour, roof:colour, the palette colour of the building type, the default (first that parses)."""
    for key in ("building:colour", "roof:colour"):
        c = parse_colour(tags.get(key))
        if c is not None:
            return c
    return BUILDING_PALETTE.get(str(tags.get("building", "")).strip().lower(), BUILDING_PALETTE["default"])


def ground_class(tags: dict) -> GroundClass | None:
    """The ground class of a way / relation, or None when its tags say nothing about the ground."""
    hw = tags.get("highway")
    if hw in PATHS:
        return GroundClass.PATH
    if hw in ROADS or (isinstance(hw, str) and hw.endswith("_link") and hw[:-5] in ROADS):
        return GroundClass.ROAD
    for key, values, cls in _AREAS:
        if tags.get(key) in values:
            return cls
    return None


def _metres(raw) -> float | None:
    m = _NUMBER.match(str(raw)) if raw is not None else None
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def road_width(tags: dict) -> float:
    """Metres across a road or path: width, else lanes x LANE_WIDTH, else by its highway class."""
    w = _metres(tags.get("width"))
    if w is not None:
        return w
    lanes = _metres(tags.get("lanes"))
    if lanes is not None:
        return lanes * LANE_WIDTH
    hw = str(tags.get("highway", ""))
    if hw in PATHS:
        return PATH_WIDTH
    return ROAD_WIDTHS.get(hw.removesuffix("_link"), ROAD_WIDTHS["residential"])
