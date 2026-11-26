"""A small form toolkit on curses, laid out like nmtui (newt): a coloured screen, centred windows with a border,
a title in the top edge and a shadow, <Buttons>, Tab to move. Nothing here knows about the sim (quadsim.tui does).

  Screen                    the terminal: colours (Theme), the header / footer lines, drawing a window
  Label, Spacer, Entry, Checkbox, Select, Button      form rows
  Form(ui, title, fields, buttons).run()               -> the name of the button pressed, None on Esc
  listbox(ui, title, items, index)                     -> the row chosen, None on Esc / <Back>
  message(ui, title, lines, buttons)                   -> the name of the button pressed, None on Esc

Keys: Tab / Down and Shift-Tab / Up move, Left / Right move along the buttons, Enter presses a button, opens a
<select> or goes to the next field, Space toggles a [ ] box, PgUp / PgDn scroll, Esc cancels.
Colours live in Theme only: 256-colour purple, an 8-colour magenta fallback, and plain reverse / bold without
colours (or with NO_COLOR set). Standard library only.
"""
from __future__ import annotations

import curses
import os
import textwrap

MIN_W, MIN_H = 60, 16
ENTER = (curses.KEY_ENTER, 10, 13)
BACKSPACE = (curses.KEY_BACKSPACE, 127, 8)
ESC, TAB = 27, 9
HINT_FORM = "Tab/Up/Down: move   Enter: select   Space: toggle   Esc: cancel"
HINT_LIST = "Up/Down: move   Enter: select   Esc: back"
CANCEL = object()

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


# ---- geometry (no curses) -------------------------------------------------------------------------------------------

def window_rect(term_h: int, term_w: int, h: int, w: int) -> tuple[int, int, int, int]:
    """(y, x, h, w) of a centred window that leaves the header / footer lines and room for the shadow."""
    h, w = max(5, min(h, term_h - 4)), max(20, min(w, term_w - 6))
    return (term_h - h) // 2, (term_w - w) // 2, h, w


def scroll_to(offset: int, focus: int | None, rows: int, total: int) -> int:
    """The first visible row so that row `focus` shows (None: keep the offset), within the content."""
    if focus is not None:
        if focus < offset:
            offset = focus
        elif focus >= offset + rows:
            offset = focus - rows + 1
    return max(0, min(offset, max(0, total - rows)))


def wrap(lines: list, width: int) -> list[tuple[str, str]]:
    """(text, style) rows: each line (a string or a (text, style) pair) wrapped to width."""
    out = []
    for line in lines:
        text, style = (line, "window") if isinstance(line, str) else line
        out += [(part, style) for part in textwrap.wrap(text, max(10, width), subsequent_indent="    ") or [""]]
    return out


# ---- the terminal ---------------------------------------------------------------------------------------------------

class Screen:
    def __init__(self, scr, header: str = ""):
        self.scr, self.header, self.theme = scr, header, Theme()
        self.cursor: tuple[int, int] | None = None
        scr.keypad(True)

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
            self.cursor = None
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

    def key(self) -> int:
        try:
            curses.curs_set(1 if self.cursor else 0)
        except curses.error:
            pass
        if self.cursor:
            try:
                self.scr.move(*self.cursor)
            except curses.error:
                pass
        return self.scr.getch()

    def choose(self, title: str, items: list[str], index: int = 0) -> int | None:
        return listbox(self, title, items, index)


# ---- form rows ------------------------------------------------------------------------------------------------------

class Field:
    focusable = False
    label = ""
    visible = True          # or a callable

    def shown(self) -> bool:
        return self.visible() if callable(self.visible) else self.visible

    def natural(self, label_w: int) -> int:
        return 0

    def draw(self, ui: Screen, y: int, x: int, w: int, label_w: int, focused: bool) -> None:
        pass


class Label(Field):
    def __init__(self, text: str = "", style: str = "window"):
        self.text, self.style = text, style

    def natural(self, label_w):
        return len(self.text)

    def draw(self, ui, y, x, w, label_w, focused):
        ui.put(y, x, self.text[:w], ui.theme[self.style])


def Spacer() -> Label:
    return Label("")


class Entry(Field):
    """One line of text. `value` is the stripped text."""
    focusable = True

    def __init__(self, label: str, text: str = "", width: int = 32):
        self.label, self.text, self.width, self.pos = label, text, width, len(text)

    @property
    def value(self) -> str:
        return self.text.strip()

    def set(self, text: str) -> None:
        self.text, self.pos = text, len(text)

    def handle(self, key: int) -> bool:
        """Edit with the key; False when it is not an editing key."""
        if 32 <= key < 127:
            self.text = self.text[:self.pos] + chr(key) + self.text[self.pos:]
            self.pos += 1
        elif key in BACKSPACE:
            if self.pos:
                self.text = self.text[:self.pos - 1] + self.text[self.pos:]
                self.pos -= 1
        elif key == curses.KEY_DC:
            self.text = self.text[:self.pos] + self.text[self.pos + 1:]
        elif key == curses.KEY_LEFT:
            self.pos = max(0, self.pos - 1)
        elif key == curses.KEY_RIGHT:
            self.pos = min(len(self.text), self.pos + 1)
        elif key == curses.KEY_HOME:
            self.pos = 0
        elif key == curses.KEY_END:
            self.pos = len(self.text)
        else:
            return False
        return True

    def view(self, fw: int) -> tuple[str, int]:
        """(the fw characters to show, the cursor column in them): the text scrolls to keep the cursor in."""
        start = max(0, self.pos - fw + 1)
        return self.text[start:start + fw].ljust(fw), self.pos - start

    def natural(self, label_w):
        return label_w + 1 + self.width

    def draw(self, ui, y, x, w, label_w, focused):
        ui.put(y, x, self.label.ljust(label_w), ui.theme["window"])
        fw = max(4, min(self.width, w - label_w - 1))
        shown, col = self.view(fw)
        ui.put(y, x + label_w + 1, shown, ui.theme["entry"] | (curses.A_BOLD if focused else 0))
        if focused:
            ui.cursor = (y, x + label_w + 1 + col)


class Checkbox(Field):
    focusable = True

    def __init__(self, text: str, checked: bool = False):
        self.text, self.checked = text, checked

    def toggle(self) -> None:
        self.checked = not self.checked

    def natural(self, label_w):
        return 4 + len(self.text)

    def draw(self, ui, y, x, w, label_w, focused):
        t = ui.theme
        ui.put(y, x, f"[{'X' if self.checked else ' '}]", t["focus"] if focused else t["value"])
        ui.put(y, x + 4, self.text[:w - 4], t["focus"] if focused else t["window"])


class Select(Field):
    """One of options [(key, text), ...], shown as <text>; Enter opens the list. on_change(select) after a change."""
    focusable = True

    def __init__(self, label: str, options: list[tuple[str, str]], key=None, on_change=None):
        self.label, self.options, self.on_change = label, list(options), on_change
        self.key = key if key in dict(self.options) else (self.options[0][0] if self.options else None)

    def text(self) -> str:
        return dict(self.options).get(self.key, "(none)")

    def index(self) -> int:
        return next((i for i, (k, _) in enumerate(self.options) if k == self.key), 0)

    def set(self, key) -> None:
        if key in dict(self.options) and key != self.key:
            self.key = key
            if self.on_change:
                self.on_change(self)

    def set_options(self, options: list[tuple[str, str]]) -> None:
        """Replace the choices; the current one is kept when it is still there, else the first is taken."""
        self.options = list(options)
        if self.key not in dict(self.options):
            self.key = None
            if self.options:
                self.set(self.options[0][0])

    def natural(self, label_w):
        return label_w + 3 + max([len(t) for _, t in self.options] or [6])

    def draw(self, ui, y, x, w, label_w, focused):
        t = ui.theme
        ui.put(y, x, self.label.ljust(label_w), t["window"])
        ui.put(y, x + label_w + 1, f"<{self.text()}>"[:max(3, w - label_w - 1)], t["focus"] if focused else t["value"])


class Button(Field):
    focusable = True

    def __init__(self, name: str, text: str | None = None):
        self.name, self.text = name, f"<{text or name}>"

    def natural(self, label_w):
        return len(self.text)

    def draw(self, ui, y, x, w, label_w, focused):
        ui.put(y, x, self.text, ui.theme["focus" if focused else "button"])


# ---- form -----------------------------------------------------------------------------------------------------------

class Form:
    """Rows of fields over a row of buttons. run() returns the name of the button pressed (a Button among the
    fields counts too), or None on Esc. The form keeps its state: run() again continues where it was."""

    def __init__(self, ui, title: str, fields: list[Field], buttons: list[Button], focus: Field | None = None,
                 hint: str = HINT_FORM):
        self.ui, self.title, self.fields, self.buttons, self.hint = ui, title, fields, buttons, hint
        self.offset = 0
        self.current = focus or next((f for f in fields if f.focusable and f.shown()), None) or buttons[-1]

    def rows(self) -> list[Field]:
        return [f for f in self.fields if f.shown()]

    def order(self) -> list[Field]:
        return [f for f in self.rows() if f.focusable] + self.buttons

    def move(self, step: int) -> None:
        order = self.order()
        i = order.index(self.current) if self.current in order else 0
        self.current = order[(i + step) % len(order)]

    def feed(self, key: int):
        """Apply one key. None = keep going, CANCEL, or the name of the button pressed."""
        if self.current not in self.order():        # its row was hidden by a change
            self.current = self.order()[0]
        cur = self.current
        if isinstance(cur, Entry) and cur.handle(key):
            return None
        if key == ESC:
            return CANCEL
        if key in (TAB, curses.KEY_DOWN):
            self.move(1)
        elif key in (curses.KEY_BTAB, curses.KEY_UP):
            self.move(-1)
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT) and cur in self.buttons:
            i = self.buttons.index(cur) + (1 if key == curses.KEY_RIGHT else -1)
            self.current = self.buttons[max(0, min(i, len(self.buttons) - 1))]
        elif key == curses.KEY_NPAGE:
            self.offset += 5
        elif key == curses.KEY_PPAGE:
            self.offset = max(0, self.offset - 5)
        elif key in ENTER or key == ord(" "):
            if isinstance(cur, Button):
                return cur.name
            if isinstance(cur, Checkbox):
                cur.toggle()
            elif isinstance(cur, Select):
                if cur.options:
                    i = self.ui.choose(cur.label, [t for _, t in cur.options], cur.index())
                    if i is not None:
                        cur.set(cur.options[i][0])
            elif key in ENTER:
                self.move(1)
        return None

    def draw(self) -> None:
        ui, rows = self.ui, self.rows()
        ui.begin(self.hint)
        label_w = max([len(f.label) for f in rows if f.label] or [0])
        buttons_w = sum(len(b.text) + 2 for b in self.buttons)
        inner = max([f.natural(label_w) for f in rows] + [buttons_w, len(self.title) + 6])
        y, x, h, w = ui.window(len(rows) + 5, inner + 4, self.title)
        visible = h - 5
        focus = rows.index(self.current) if self.current in rows else None
        self.offset = scroll_to(self.offset, focus, visible, len(rows))
        for i, f in enumerate(rows[self.offset:self.offset + visible]):
            f.draw(ui, y + 2 + i, x + 2, w - 4, label_w, f is self.current)
        ui.arrows(y, x, h, w, self.offset > 0, self.offset + visible < len(rows))
        bx = x + w - 2 - buttons_w + 2
        for b in self.buttons:
            b.draw(ui, y + h - 2, bx, len(b.text), 0, b is self.current)
            bx += len(b.text) + 2

    def run(self) -> str | None:
        while True:
            self.draw()
            r = self.feed(self.ui.key())
            if r is not None:
                return None if r is CANCEL else r


def message(ui, title: str, lines: list, buttons: tuple[str, ...] = ("OK",)) -> str | None:
    """Text over buttons; the last button has the focus. Lines are strings or (text, style) pairs."""
    _, tw = ui.size()
    bs = [Button(b) for b in buttons]
    fields = [Label(text, style) for text, style in wrap(lines, min(tw - 12, 96))]
    return Form(ui, title, fields, bs, focus=bs[-1], hint="Left/Right: move   Enter: select   PgUp/PgDn: scroll   Esc: back").run()


def listbox(ui, title: str, items: list[str], index: int = 0, info: list | None = None,
            buttons: tuple[str, str] = ("Back", "OK")) -> int | None:
    """Pick one row. Enter or <OK> returns its index; Esc, q or <Back> returns None."""
    index, offset, zone = max(0, min(index, len(items) - 1)), 0, 0     # zone: 0 the list, 1.. the buttons
    while True:
        ui.begin(HINT_LIST)
        _, tw = ui.size()
        head = wrap(info or [], min(tw - 12, 96))
        texts = [f"<{b}>" for b in buttons]
        inner = max([len(i) + 2 for i in items] + [len(t) for t, _ in head] + [len(title) + 6, sum(map(len, texts)) + 4])
        extra = len(head) + (1 if head else 0)
        y, x, h, w = ui.window(len(items) + extra + 5, inner + 4, title)
        for i, (text, style) in enumerate(head):
            ui.put(y + 2 + i, x + 2, text[:w - 4], ui.theme[style])
        visible = max(1, h - 5 - extra)
        offset = scroll_to(offset, index, visible, len(items))
        for i, item in enumerate(items[offset:offset + visible], start=offset):
            on = i == index
            ui.put(y + 2 + extra + i - offset, x + 2, f" {item} "[:w - 4].ljust(w - 4 if on else 0),
                   ui.theme["focus" if on and zone == 0 else "value" if on else "window"])
        ui.arrows(y, x, h, w, offset > 0, offset + visible < len(items))
        bx = x + w - 2 - sum(len(t) + 2 for t in texts) + 2
        for n, t in enumerate(texts, start=1):
            ui.put(y + h - 2, bx, t, ui.theme["focus" if zone == n else "button"])
            bx += len(t) + 2
        k = ui.key()
        if k in (ESC, ord("q")):
            return None
        if k == TAB:
            zone = (zone + 1) % (len(buttons) + 1)
        elif k == curses.KEY_BTAB:
            zone = (zone - 1) % (len(buttons) + 1)
        elif k in (curses.KEY_LEFT, curses.KEY_RIGHT) and zone:
            zone = max(1, min(len(buttons), zone + (1 if k == curses.KEY_RIGHT else -1)))
        elif k in (curses.KEY_UP, ord("k")):
            zone, index = 0, (index - 1) % len(items)
        elif k in (curses.KEY_DOWN, ord("j")):
            zone, index = 0, (index + 1) % len(items)
        elif k == curses.KEY_NPAGE:
            zone, index = 0, min(len(items) - 1, index + visible)
        elif k == curses.KEY_PPAGE:
            zone, index = 0, max(0, index - visible)
        elif k in ENTER or k == ord(" "):
            return None if zone == 1 else index
