"""Ready-made windows, each a Form of ordinary components."""
from __future__ import annotations

from .components import Button, Entry, Listbox, Spacer, Text
from .form import Form, wrap

HINT_LIST = "Up/Down: move   Enter: select   Left/Right: buttons   Esc: back"
HINT_MESSAGE = "Left/Right: move   Enter: select   PgUp/PgDn: scroll   Esc: back"


def text_width(ui) -> int:
    return min(ui.size()[1] - 12, 96)


def choice_window(ui, title: str, items: list[str], index: int = 0, info: list | None = None,
                  buttons: tuple[str, str] = ("Back", "OK")) -> int | None:
    """Pick one row. Enter, a click or the last button returns its index; Esc, q or the first button, None."""
    box = Listbox(items, index, return_exit=True)
    head = [Text(wrap(info, text_width(ui))), Spacer()] if info else []
    form = Form(ui, title, [*head, box], [Button(b) for b in buttons], hint=HINT_LIST, cancel_keys=(ord("q"),))
    return box.index if form.run() == buttons[-1] and items else None


def message_window(ui, title: str, lines: list, buttons: tuple[str, ...] = ("OK",)) -> str | None:
    """Text over buttons; the last button has the focus. Lines are strings or (text, style) pairs."""
    return Form(ui, title, [Text(wrap(lines, text_width(ui)))], [Button(b) for b in buttons], hint=HINT_MESSAGE).run()


def entry_window(ui, title: str, label: str, default: str = "", info: list | None = None, check=None,
                 errors: tuple = (ValueError,), on_error=None):
    """One value. Returns check(text) (the text itself without a check), None on Cancel. When check raises one
    of `errors`, on_error(exception) is called and the window stays."""
    entry = Entry(label, default, width=44)
    head = [Text(wrap(info, text_width(ui))), Spacer()] if info else []
    form = Form(ui, title, [*head, entry], [Button("Cancel"), Button("OK")], enter_submits=True)
    with form:
        while form.run() == "OK":
            try:
                return check(entry.value) if check else entry.value
            except errors as e:
                if on_error:
                    on_error(e)
                form.current = entry
    return None
