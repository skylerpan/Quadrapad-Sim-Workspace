"""Colours: 256-colour purple, an 8-colour magenta fallback, and plain reverse / bold without colours (or with
NO_COLOR set). To change the look, change the tables here."""
from __future__ import annotations

import curses
import os

# name: (foreground, background, bold) in xterm-256 numbers
PURPLE_256 = {
    "screen": (254, 54, False), "window": (234, 189, False), "border": (55, 189, False), "title": (55, 189, True),
    "focus": (231, 129, True), "button": (55, 189, False), "entry": (231, 98, False), "value": (55, 189, True),
    "warn": (161, 189, True), "dim": (60, 189, False), "shadow": (16, 16, False),
}
PURPLE_8 = {
    "screen": (curses.COLOR_WHITE, curses.COLOR_MAGENTA, True), "window": (curses.COLOR_BLACK, curses.COLOR_WHITE, False),
    "border": (curses.COLOR_MAGENTA, curses.COLOR_WHITE, False), "title": (curses.COLOR_MAGENTA, curses.COLOR_WHITE, True),
    "focus": (curses.COLOR_WHITE, curses.COLOR_MAGENTA, True), "button": (curses.COLOR_MAGENTA, curses.COLOR_WHITE, False),
    "entry": (curses.COLOR_BLACK, curses.COLOR_CYAN, False), "value": (curses.COLOR_MAGENTA, curses.COLOR_WHITE, True),
    "warn": (curses.COLOR_RED, curses.COLOR_WHITE, True), "dim": (curses.COLOR_BLACK, curses.COLOR_WHITE, False),
    "shadow": (curses.COLOR_BLACK, curses.COLOR_BLACK, False),
}
MONO = {"title": curses.A_BOLD, "focus": curses.A_REVERSE, "entry": curses.A_UNDERLINE, "value": curses.A_BOLD,
        "warn": curses.A_BOLD}


def palette(colors: int, has_colors: bool, env: dict[str, str]) -> dict | None:
    """The colour table for this terminal; None = no colours."""
    if not has_colors or env.get("NO_COLOR"):
        return None
    return PURPLE_256 if colors >= 256 else PURPLE_8


class Theme:
    """style name -> curses attribute. Needs curses to be started."""

    def __init__(self):
        table = palette(getattr(curses, "COLORS", 0), curses.has_colors(), dict(os.environ))
        self.mono = table is None
        self.attr = {name: MONO.get(name, 0) for name in PURPLE_256}
        if table:
            for n, (name, (fg, bg, bold)) in enumerate(table.items(), start=1):
                curses.init_pair(n, fg, bg)
                self.attr[name] = curses.color_pair(n) | (curses.A_BOLD if bold else 0)

    def __getitem__(self, name: str) -> int:
        return self.attr[name]
