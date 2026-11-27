"""The things a form is made of. Every one follows the Component protocol; the form does the rest."""
from __future__ import annotations

import curses

ENTER = (curses.KEY_ENTER, 10, 13)
BACKSPACE = (curses.KEY_BACKSPACE, 127, 8)
ESC, TAB = 27, 9


class Component:
    focusable = False
    flex = False            # its height may be cut to fit the window; it then scrolls inside itself
    label = ""              # drawn in the form's label column
    visible = True          # or a callable

    def shown(self) -> bool:
        return self.visible() if callable(self.visible) else self.visible

    def height(self) -> int:
        return 1

    def natural(self, label_w: int) -> int:
        """The width it would like, given the width of the form's label column."""
        return 0

    def draw(self, ui, y: int, x: int, w: int, h: int, label_w: int, focused: bool) -> None:
        pass

    def handle(self, key: int) -> bool:
        """Use the key; False leaves it to the form (moving the focus, Enter, Esc)."""
        return False

    def activate(self, ui) -> bool:
        """Enter, Space or a click. True ends the form."""
        return False

    def click(self, dy: int, dx: int) -> bool:
        """A click at (dy, dx) inside it, after it took the focus. True goes on to activate()."""
        return True

    def wheel(self, step: int) -> bool:
        """Scroll by step rows; False when it has nothing to scroll."""
        return False

    def overflow(self) -> tuple[bool, bool]:
        """(more above, more below) of what it shows."""
        return False, False


class Label(Component):
    def __init__(self, text: str = "", style: str = "window"):
        self.text, self.style = text, style

    def natural(self, label_w):
        return len(self.text)

    def draw(self, ui, y, x, w, h, label_w, focused):
        ui.put(y, x, self.text[:w], ui.theme[self.style])


def Spacer() -> Label:
    return Label("")


class Text(Component):
    """Lines of text: (text, style) pairs, already wrapped (see wrap()). Scrolls when it does not fit."""
    flex = True

    def __init__(self, rows: list[tuple[str, str]]):
        self.rows, self.offset, self.h = list(rows), 0, len(rows)

    def height(self):
        return len(self.rows)

    def natural(self, label_w):
        return max([len(t) for t, _ in self.rows] or [0])

    def draw(self, ui, y, x, w, h, label_w, focused):
        self.h = h
        self.offset = max(0, min(self.offset, len(self.rows) - h))
        for i, (text, style) in enumerate(self.rows[self.offset:self.offset + h]):
            ui.put(y + i, x, text[:w], ui.theme[style])

    def wheel(self, step):
        new = max(0, min(self.offset + step, len(self.rows) - self.h))
        moved, self.offset = new != self.offset, new
        return moved

    def overflow(self):
        return self.offset > 0, self.offset + self.h < len(self.rows)


class Entry(Component):
    """One line of text. `value` is the stripped text."""
    focusable = True

    def __init__(self, label: str, text: str = "", width: int = 32):
        self.label, self.text, self.width, self.pos = label, text, width, len(text)
        self.col0 = 0           # where the field starts in its row (set by draw, for clicks)

    @property
    def value(self) -> str:
        return self.text.strip()

    def set(self, text: str) -> None:
        self.text, self.pos = text, len(text)

    def handle(self, key):
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

    def draw(self, ui, y, x, w, h, label_w, focused):
        ui.put(y, x, self.label.ljust(label_w), ui.theme["window"])
        fw = max(4, min(self.width, w - label_w - 1))
        shown, col = self.view(fw)
        self.col0 = label_w + 1 - (self.pos - col)
        ui.put(y, x + label_w + 1, shown, ui.theme["entry"] | (curses.A_BOLD if focused else 0))
        if focused:
            ui.cursor = (y, x + label_w + 1 + col)

    def click(self, dy, dx):
        self.pos = max(0, min(len(self.text), dx - self.col0))
        return False


class Checkbox(Component):
    focusable = True

    def __init__(self, text: str, checked: bool = False):
        self.text, self.checked = text, checked

    def toggle(self) -> None:
        self.checked = not self.checked

    def activate(self, ui):
        self.toggle()
        return False

    def natural(self, label_w):
        return 4 + len(self.text)

    def draw(self, ui, y, x, w, h, label_w, focused):
        t = ui.theme
        ui.put(y, x, f"[{'X' if self.checked else ' '}]", t["focus"] if focused else t["value"])
        ui.put(y, x + 4, self.text[:w - 4], t["focus"] if focused else t["window"])


class Listbox(Component):
    """Rows to pick one from; `index` is the row picked. Up / Down move inside it and leave at its ends.
    With return_exit, Enter (or a click) on a row ends the form like its default button."""
    focusable = True
    flex = True

    def __init__(self, items: list[str], index: int = 0, return_exit: bool = False):
        self.items, self.return_exit = list(items), return_exit
        self.index, self.offset, self.h = max(0, min(index, len(items) - 1)), 0, len(items)

    def height(self):
        return max(1, len(self.items))

    def natural(self, label_w):
        return max([len(i) + 2 for i in self.items] or [8])

    def handle(self, key):
        last = len(self.items) - 1
        if key == curses.KEY_UP and self.index > 0:
            self.index -= 1
        elif key == curses.KEY_DOWN and self.index < last:
            self.index += 1
        elif key == curses.KEY_PPAGE:
            self.index = max(0, self.index - max(1, self.h))
        elif key == curses.KEY_NPAGE:
            self.index = max(0, min(last, self.index + max(1, self.h)))
        elif key == curses.KEY_HOME:
            self.index = 0
        elif key == curses.KEY_END:
            self.index = max(0, last)
        else:
            return False
        return True

    def activate(self, ui):
        return self.return_exit and bool(self.items)

    def click(self, dy, dx):
        if self.offset + dy >= len(self.items):
            return False
        self.index = self.offset + dy
        return True

    def wheel(self, step):
        new = max(0, min(len(self.items) - 1, self.index + step))
        moved, self.index = new != self.index, new
        return moved

    def overflow(self):
        return self.offset > 0, self.offset + self.h < len(self.items)

    def draw(self, ui, y, x, w, h, label_w, focused):
        self.h = h
        if self.index < self.offset:
            self.offset = self.index
        elif self.index >= self.offset + h:
            self.offset = self.index - h + 1
        self.offset = max(0, min(self.offset, len(self.items) - h))
        for i, item in enumerate(self.items[self.offset:self.offset + h], start=self.offset):
            on = i == self.index
            ui.put(y + i - self.offset, x, f" {item} "[:w].ljust(w if on else 0),
                   ui.theme["focus" if on and focused else "value" if on else "window"])


class Select(Component):
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

    def activate(self, ui):
        if self.options:
            i = ui.choose(self.label, [t for _, t in self.options], self.index())
            if i is not None:
                self.set(self.options[i][0])
        return False

    def natural(self, label_w):
        return label_w + 3 + max([len(t) for _, t in self.options] or [6])

    def draw(self, ui, y, x, w, h, label_w, focused):
        t = ui.theme
        ui.put(y, x, self.label.ljust(label_w), t["window"])
        ui.put(y, x + label_w + 1, f"<{self.text()}>"[:max(3, w - label_w - 1)], t["focus"] if focused else t["value"])


class Button(Component):
    """<text>; pressing it ends the form, which returns `name`."""
    focusable = True

    def __init__(self, name: str, text: str | None = None):
        self.name, self.text = name, f"<{text or name}>"

    def activate(self, ui):
        return True

    def natural(self, label_w):
        return len(self.text)

    def draw(self, ui, y, x, w, h, label_w, focused):
        ui.put(y, x, self.text, ui.theme["focus" if focused else "button"])


class ButtonBar(Component):
    """The buttons at the bottom of a form, right-aligned; the last one is the default."""

    def __init__(self, buttons: list[Button]):
        self.buttons = list(buttons)

    def natural(self, label_w):
        return sum(len(b.text) + 2 for b in self.buttons)

    def draw(self, ui, y, x, w, h, label_w, focused):
        """`focused` is the component that has the focus (one of the buttons, or something else)."""
        bx = x + w - self.natural(0) + 2
        for b in self.buttons:
            b.draw(ui, y, bx, len(b.text), 1, 0, b is focused)
            ui.hits.add(y, bx, 1, len(b.text), b)
            bx += len(b.text) + 2
