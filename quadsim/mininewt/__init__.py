"""mininewt: a small newt on curses. Standard library only; nothing here knows
about the sim, so the package can be lifted out as it is.

  Screen(scr, header)       the terminal: colours, header / footer lines, the stack of open windows, keys, mouse
  Label, Spacer, Text, Entry, Checkbox, Listbox, Select, Button, ButtonBar      components (components.py)
  Form(ui, title, components, buttons).run()      -> the name of the button that ended it, None on Esc
  choice_window(ui, title, items)                 -> the row picked, None on Esc / the first button
  message_window(ui, title, lines, buttons)       -> the name of the button pressed, None on Esc
  entry_window(ui, title, label)                  -> the text typed, None on Cancel

Like newt, everything on a window is a component in a form. A component gets each key first (an entry edits, a
listbox moves its bar); what it leaves, the form uses: Tab / Shift-Tab step through every component and button,
Up / Down step through the components and treat the button bar as one stop, Left / Right go into and along the
buttons, Enter or Space activates (presses a button, toggles a [ ] box, opens a <select>; in an entry Enter goes
on to the next field, from the last one to the default button), PgUp / PgDn scroll, Esc cancels. A click gives a
component the focus and activates it; the wheel scrolls. A window opened from another one is drawn over it.

Not here (newt has them): grids, scales, radio bars, checkbox trees, a help-line stack, suspend callbacks.
"""
from .components import (BACKSPACE, ENTER, ESC, TAB, Button, ButtonBar, Checkbox, Component, Entry, Label, Listbox,
                         Select, Spacer, Text)
from .form import CANCEL, CLICK, HINT_FORM, WHEEL_DOWN, WHEEL_UP, Form, fit, scroll_to, window_rect, wrap
from .screen import MIN_H, MIN_W, HitMap, Screen
from .theme import PURPLE_8, PURPLE_256, Theme, palette
from .windows import choice_window, entry_window, message_window

__all__ = ["BACKSPACE", "ENTER", "ESC", "TAB", "CANCEL", "CLICK", "WHEEL_UP", "WHEEL_DOWN", "HINT_FORM", "MIN_H",
           "MIN_W", "PURPLE_8", "PURPLE_256", "Button", "ButtonBar", "Checkbox", "Component", "Entry", "Form",
           "HitMap", "Label", "Listbox", "Screen", "Select", "Spacer", "Text", "Theme", "choice_window",
           "entry_window", "fit", "message_window", "palette", "scroll_to", "window_rect", "wrap"]
