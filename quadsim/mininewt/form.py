"""Form: components top to bottom over a bar of buttons, in a window. It owns the focus, lays the components
out, scrolls, and hands keys and clicks to them: a component gets a key first, and what it leaves moves the focus."""
from __future__ import annotations

import curses
import textwrap

from .components import ENTER, ESC, TAB, Button, ButtonBar, Component, Entry

HINT_FORM = "Tab/arrows: move   Enter: select   Space: toggle   Esc: cancel"
CANCEL = object()
WHEEL_UP = curses.BUTTON4_PRESSED
WHEEL_DOWN = getattr(curses, "BUTTON5_PRESSED", 0x200000)
CLICK = curses.BUTTON1_PRESSED | curses.BUTTON1_CLICKED


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


def fit(heights: list[int], flex: list[bool], room: int) -> list[int]:
    """The heights with the flexible ones cut (the last first, never below 1) until they fit in `room` rows."""
    out, over = list(heights), sum(heights) - room
    for i in reversed(range(len(out))):
        if over > 0 and flex[i]:
            cut = min(over, out[i] - 1)
            out[i] -= cut
            over -= cut
    return out


# ---- form -----------------------------------------------------------------------------------------------------------

class Form:
    """run() returns the name of the button that ended it (a Button among the components counts too; a Listbox
    with return_exit ends it as the default button, the last one), or None on Esc.
    The form keeps its state: run() again continues where it was. `with form:` keeps its window on the screen
    between runs, under whatever is opened meanwhile."""

    def __init__(self, ui, title: str, fields: list[Component], buttons: list[Button], focus: Component | None = None,
                 hint: str = HINT_FORM, enter_submits: bool = False, cancel_keys: tuple[int, ...] = ()):
        self.ui, self.title, self.fields, self.hint = ui, title, fields, hint
        self.bar, self.buttons = ButtonBar(buttons), buttons
        self.enter_submits = enter_submits      # Enter in an entry presses the default button (one-question dialogs)
        self.cancel_keys = cancel_keys          # keys that cancel like Esc, when the focused component leaves them
        self.offset, self.depth, self.seen = 0, 0, None
        self.current = focus or next((f for f in fields if f.focusable and f.shown()), None) or buttons[-1]

    def rows(self) -> list[Component]:
        return [f for f in self.fields if f.shown()]

    def stops(self) -> list[Component]:
        """The focusable components above the button bar."""
        return [f for f in self.rows() if f.focusable]

    def order(self) -> list[Component]:
        return self.stops() + self.buttons

    def move(self, step: int) -> None:
        order = self.order()
        i = order.index(self.current) if self.current in order else 0
        self.current = order[(i + step) % len(order)]

    def result(self, c: Component) -> str:
        return c.name if isinstance(c, Button) else self.buttons[-1].name

    # ---- events

    def feed(self, key: int):
        """Apply one key. None = keep going, CANCEL, or the name of the button that ends the form."""
        if self.current not in self.order():        # its row was hidden by a change
            self.current = self.order()[0]
        cur, stops, buttons = self.current, self.stops(), self.buttons
        if key == curses.KEY_MOUSE:
            return self.mouse()
        if cur.handle(key):
            return None
        if key == ESC or key in self.cancel_keys:
            return CANCEL
        if key == TAB:
            self.move(1)
        elif key == curses.KEY_BTAB:
            self.move(-1)
        elif key in (curses.KEY_UP, curses.KEY_DOWN):       # the button bar is one stop: its default button
            down = key == curses.KEY_DOWN
            if cur in buttons:
                self.current = (stops[0] if down else stops[-1]) if stops else cur
            else:
                i = stops.index(cur) + (1 if down else -1)
                self.current = stops[i] if 0 <= i < len(stops) else buttons[-1]
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):    # along the buttons; from above, into them
            right = key == curses.KEY_RIGHT
            if cur in buttons:
                i = buttons.index(cur) + (1 if right else -1)
                self.current = buttons[max(0, min(i, len(buttons) - 1))]
            else:
                self.current = buttons[-1 if right else 0]
        elif key == curses.KEY_NPAGE:
            self.scroll(5)
        elif key == curses.KEY_PPAGE:
            self.scroll(-5)
        elif key in ENTER or key == ord(" "):
            if isinstance(cur, Entry):          # Enter: on to the next field, or to the default button
                order = self.order()
                if self.enter_submits:
                    return buttons[-1].name
                if order[(order.index(cur) + 1) % len(order)] in buttons:
                    self.current = buttons[-1]
                else:
                    self.move(1)
            elif cur.activate(self.ui):
                return self.result(cur)
        return None

    def scroll(self, step: int, over: Component | None = None) -> None:
        """Scroll what is under the pointer, else what scrolls inside itself, else the form."""
        for c in ([over] if over else []) + [c for c in self.rows() if c.flex and c is not over]:
            if c.wheel(step):
                return
        self.offset = max(0, self.offset + step)

    def mouse(self):
        if not self.ui.mouse:
            return None
        y, x, pressed = self.ui.mouse
        hit = self.ui.hits.at(y, x)
        if pressed & (WHEEL_UP | WHEEL_DOWN):
            self.scroll(-3 if pressed & WHEEL_UP else 3, hit[0] if hit else None)
        elif pressed & CLICK and hit:
            return self.click(*hit)
        return None

    def click(self, c: Component, dy: int = 0, dx: int = 0):
        """A click on a component: it takes the focus and is activated, as with Enter."""
        if not c.focusable or c not in self.order():
            return None
        self.current = c
        if c.click(dy, dx) and c.activate(self.ui):
            return self.result(c)
        return None

    # ---- drawing

    def draw(self) -> None:
        ui, rows = self.ui, self.rows()
        label_w = max([len(f.label) for f in rows if f.label] or [0])
        inner = max([f.natural(label_w) for f in rows] + [self.bar.natural(0), len(self.title) + 6])
        want = [f.height() for f in rows]
        y, x, h, w = ui.window(sum(want) + 5, inner + 4, self.title)
        visible = h - 5
        heights = fit(want, [f.flex for f in rows], visible)
        tops = [sum(heights[:i]) for i in range(len(rows))]
        moved = self.current is not self.seen and self.current in rows      # scroll to the focus when it moves
        self.offset = scroll_to(self.offset, tops[rows.index(self.current)] if moved else None, visible, sum(heights))
        self.seen = self.current
        above, below = self.offset > 0, self.offset + visible < sum(heights)
        for f, top, fh in zip(rows, tops, heights):
            if top >= self.offset and top + fh <= self.offset + visible:
                fy = y + 2 + top - self.offset
                f.draw(ui, fy, x + 2, w - 4, fh, label_w, f is self.current)
                if f.focusable:
                    ui.hits.add(fy, x + 2, fh, w - 4, f)
                more = f.overflow()
                above, below = above or more[0], below or more[1]
        ui.arrows(y, x, h, w, above, below)
        self.bar.draw(ui, y + h - 2, x + 2, w - 4, 1, 0, self.current)

    # ---- running

    def __enter__(self):
        self.depth += 1
        if self.depth == 1:
            self.ui.push(self)
        return self

    def __exit__(self, *exc):
        self.depth -= 1
        if self.depth == 0:
            self.ui.pop()

    def run(self) -> str | None:
        with self:
            while True:
                self.ui.paint()
                r = self.feed(self.ui.key())
                if r is not None:
                    return None if r is CANCEL else r
