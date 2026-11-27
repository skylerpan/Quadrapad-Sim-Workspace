"""The terminal: the coloured background with a header and a footer line, window frames, the stack of open
windows, and keyboard / mouse input."""
from __future__ import annotations

import curses

from .form import CLICK, WHEEL_DOWN, WHEEL_UP, window_rect
from .theme import Theme

MIN_W, MIN_H = 60, 16


class HitMap:
    """Where the clickable things were drawn: add(y, x, h, w, target), at(y, x) -> (target, dy, dx) or None."""

    def __init__(self):
        self.rects: list[tuple[int, int, int, int, object]] = []

    def clear(self) -> None:
        self.rects.clear()

    def add(self, y: int, x: int, h: int, w: int, target) -> None:
        self.rects.append((y, x, h, w, target))

    def at(self, y: int, x: int):
        for ry, rx, h, w, target in reversed(self.rects):       # the last drawn is on top
            if ry <= y < ry + h and rx <= x < rx + w:
                return target, y - ry, x - rx
        return None


class Screen:
    def __init__(self, scr, header: str = ""):
        self.scr, self.header = scr, header
        self.theme = Theme()
        self.stack: list = []                   # open windows (forms), the last one on top
        self.hits = HitMap()                    # of the top window
        self.cursor: tuple[int, int] | None = None
        self.mouse: tuple[int, int, int] | None = None      # (y, x, buttons) of the last KEY_MOUSE
        scr.keypad(True)
        self.resume()

    def resume(self) -> None:
        """Ask for mouse clicks and the wheel (again, after the terminal was lent to another program)."""
        try:
            curses.mousemask(CLICK | WHEEL_UP | WHEEL_DOWN)
            curses.mouseinterval(0)
        except curses.error:
            pass

    # ---- drawing

    def put(self, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and 0 <= x < w - 1 and text:
            try:
                self.scr.addstr(y, x, text[:w - 1 - x], attr)
            except curses.error:
                pass

    def size(self) -> tuple[int, int]:
        return self.scr.getmaxyx()

    def begin(self, hint: str) -> tuple[int, int]:
        """Paint the background with the header and footer; wait while the terminal is too small."""
        while True:
            self.scr.bkgd(" ", self.theme["screen"])
            self.scr.erase()
            h, w = self.size()
            if h >= MIN_H and w >= MIN_W:
                break
            self.put(0, 0, f"Enlarge the terminal to {MIN_W}x{MIN_H} or more", self.theme["screen"])
            self.scr.getch()
        self.put(0, 1, self.header, self.theme["screen"])
        self.put(h - 1, 1, hint, self.theme["screen"])
        return h, w

    def window(self, h: int, w: int, title: str) -> tuple[int, int, int, int]:
        """Draw an empty window of about h x w (clipped to the terminal) and return its (y, x, h, w)."""
        th, tw = self.size()
        y, x, h, w = window_rect(th, tw, h, w)
        t, scr = self.theme, self.scr
        if not t.mono:
            for i in range(1, h + 1):
                self.put(y + i, x + w, "  ", t["shadow"])
            self.put(y + h, x + 2, " " * w, t["shadow"])
        for i in range(h):
            self.put(y + i, x, " " * w, t["window"])
        b = t["border"]
        try:
            scr.hline(y, x + 1, curses.ACS_HLINE | b, w - 2)
            scr.hline(y + h - 1, x + 1, curses.ACS_HLINE | b, w - 2)
            scr.vline(y + 1, x, curses.ACS_VLINE | b, h - 2)
            scr.vline(y + 1, x + w - 1, curses.ACS_VLINE | b, h - 2)
            for cy, cx, ch in ((y, x, curses.ACS_ULCORNER), (y, x + w - 1, curses.ACS_URCORNER),
                               (y + h - 1, x, curses.ACS_LLCORNER), (y + h - 1, x + w - 1, curses.ACS_LRCORNER)):
                scr.addch(cy, cx, ch | b)
            text = f" {title} "[:w - 6]
            tx = x + (w - len(text) - 2) // 2
            scr.addch(y, tx, curses.ACS_RTEE | b)
            self.put(y, tx + 1, text, t["title"])
            scr.addch(y, tx + 1 + len(text), curses.ACS_LTEE | b)
        except curses.error:
            pass
        return y, x, h, w

    def arrows(self, y: int, x: int, h: int, w: int, above: bool, below: bool) -> None:
        """Scroll marks on the right edge of the window at (y, x, h, w)."""
        try:
            if above:
                self.scr.addch(y + 1, x + w - 1, curses.ACS_UARROW | self.theme["value"])
            if below:
                self.scr.addch(y + h - 2, x + w - 1, curses.ACS_DARROW | self.theme["value"])
        except curses.error:
            pass

    # ---- the window stack

    def push(self, form) -> None:
        self.stack.append(form)

    def pop(self) -> None:
        self.stack.pop()

    def paint(self) -> None:
        """Draw every open window, the top one last; the cursor and the click map are the top one's."""
        self.begin(self.stack[-1].hint if self.stack else "")
        for form in self.stack:
            self.cursor = None
            self.hits.clear()
            form.draw()

    # ---- input

    def key(self) -> int:
        """The next key. A mouse event comes as curses.KEY_MOUSE with its place and buttons in self.mouse."""
        try:
            curses.curs_set(1 if self.cursor else 0)
        except curses.error:
            pass
        if self.cursor:
            try:
                self.scr.move(*self.cursor)
            except curses.error:
                pass
        k = self.scr.getch()
        if k == curses.KEY_MOUSE:
            try:
                _, x, y, _, buttons = curses.getmouse()
                self.mouse = (y, x, buttons)
            except curses.error:
                self.mouse = None
        return k

    def choose(self, title: str, items: list[str], index: int = 0) -> int | None:
        from .windows import choice_window          # windows are built on forms, which draw on a screen
        return choice_window(self, title, items, index)
