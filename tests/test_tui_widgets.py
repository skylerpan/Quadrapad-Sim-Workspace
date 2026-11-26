"""quadsim.tui_widgets: the parts of the form toolkit that need no terminal (focus, editing, scrolling, geometry)."""
import curses

from quadsim import tui_widgets as tw


class FakeUi:
    """Stands in for Screen where Form.feed needs one: a <select> asks it for a choice."""

    def __init__(self, pick=None):
        self.pick, self.asked = pick, []

    def choose(self, title, items, index=0):
        self.asked.append((title, items, index))
        return self.pick


def keys(form, *ks):
    out = None
    for k in ks:
        out = form.feed(ord(k) if isinstance(k, str) else k)
    return out


def make(ui=None):
    name, on, kind = tw.Entry("Name", "ab"), tw.Checkbox("Fast"), tw.Select("Kind", [("a", "Alpha"), ("b", "Beta")], "a")
    cancel, ok = tw.Button("Cancel"), tw.Button("OK")
    form = tw.Form(ui or FakeUi(), "T", [tw.Label("intro"), name, tw.Spacer(), on, kind], [cancel, ok])
    return form, name, on, kind, cancel, ok


# ---- focus

def test_focus_starts_on_the_first_field_and_skips_labels():
    form, name, on, kind, cancel, ok = make()
    assert form.current is name
    keys(form, tw.TAB)
    assert form.current is on
    keys(form, curses.KEY_DOWN, tw.TAB, tw.TAB)
    assert form.current is ok


def test_focus_wraps_both_ways():
    form, name, _, _, _, ok = make()
    keys(form, curses.KEY_BTAB)
    assert form.current is ok
    keys(form, tw.TAB)
    assert form.current is name


def test_left_and_right_move_along_the_buttons_only_and_stop_at_the_ends():
    form, name, _, _, cancel, ok = make()
    form.current = ok
    keys(form, curses.KEY_LEFT)
    assert form.current is cancel
    keys(form, curses.KEY_LEFT)
    assert form.current is cancel
    keys(form, curses.KEY_RIGHT, curses.KEY_RIGHT)
    assert form.current is ok


def test_a_form_of_only_labels_focuses_its_last_button():
    ok = tw.Button("OK")
    assert tw.Form(FakeUi(), "T", [tw.Label("x")], [tw.Button("No"), ok]).current is ok


def test_a_hidden_row_is_skipped_and_losing_the_focused_row_moves_the_focus():
    form, name, on, kind, _, _ = make()
    on.visible = lambda: False
    keys(form, tw.TAB)
    assert form.current is kind
    kind.visible = False
    keys(form, curses.KEY_RESIZE)
    assert form.current is name


# ---- results

def test_enter_on_a_button_returns_its_name_and_esc_cancels():
    form, *_, cancel, ok = make()
    form.current = ok
    assert keys(form, 10) == "OK"
    assert keys(form, tw.ESC) is tw.CANCEL


def test_a_button_among_the_fields_returns_its_name_too():
    more = tw.Button("topics", "Topics...")
    form = tw.Form(FakeUi(), "T", [more], [tw.Button("OK")])
    assert more.text == "<Topics...>" and keys(form, " ") == "topics"


# ---- entry

def test_typing_goes_into_the_entry_even_q_and_space():
    form, name, *_ = make()
    assert keys(form, "q", " ", "k") is None
    assert name.text == "abq k" and name.value == "abq k" and form.current is name


def test_entry_editing_keys():
    e = tw.Entry("x", "abc")
    for k in (curses.KEY_LEFT, curses.KEY_BACKSPACE, ord("X"), curses.KEY_HOME, curses.KEY_DC, curses.KEY_END, ord("!")):
        assert e.handle(k)
    assert e.text == "Xc!" and e.pos == 3
    assert not e.handle(tw.TAB) and not e.handle(10) and not e.handle(curses.KEY_UP)


def test_backspace_at_the_start_and_cursor_limits_do_nothing():
    e = tw.Entry("x", "a")
    e.handle(curses.KEY_HOME), e.handle(127), e.handle(curses.KEY_LEFT)
    assert (e.text, e.pos) == ("a", 0)
    e.handle(curses.KEY_END), e.handle(curses.KEY_RIGHT)
    assert e.pos == 1


def test_enter_in_an_entry_goes_to_the_next_field():
    form, name, on, *_ = make()
    keys(form, 10)
    assert form.current is on and name.text == "ab"


def test_a_long_entry_scrolls_to_keep_the_cursor_in_view():
    e = tw.Entry("x", "0123456789")
    assert e.view(4) == ("789 ", 3)
    e.pos = 2
    assert e.view(4) == ("0123", 2)
    e.set("zz")
    assert e.view(4) == ("zz  ", 2)


# ---- checkbox / select

def test_space_and_enter_toggle_a_checkbox():
    form, _, on, *_ = make()
    form.current = on
    keys(form, " ")
    assert on.checked
    keys(form, 10)
    assert not on.checked and form.current is on


def test_a_select_opens_a_list_and_reports_a_change():
    ui, seen = FakeUi(pick=1), []
    form, _, _, kind, *_ = make(ui)
    kind.on_change = lambda s: seen.append(s.key)
    form.current = kind
    keys(form, 10)
    assert ui.asked == [("Kind", ["Alpha", "Beta"], 0)] and kind.key == "b" and kind.text() == "Beta" and seen == ["b"]
    keys(form, 10)                                  # the same choice again: no change
    assert seen == ["b"]
    ui.pick = None                                  # Esc in the list
    keys(form, " ")
    assert kind.key == "b"


def test_select_defaults_and_replacing_its_options():
    seen = []
    s = tw.Select("x", [("a", "A"), ("b", "B")], "nope", on_change=lambda s: seen.append(s.key))
    assert s.key == "a" and s.index() == 0
    s.set("zzz")
    assert s.key == "a" and seen == []
    s.set_options([("b", "B"), ("c", "C")])         # the current choice is gone: the first one is taken
    assert s.key == "b" and seen == ["b"]
    s.set_options([("c", "C"), ("b", "B")])         # still there: kept
    assert s.key == "b" and s.index() == 1 and seen == ["b"]
    s.set_options([])
    assert s.key is None and s.text() == "(none)"


def test_an_empty_select_does_not_open():
    ui = FakeUi(pick=0)
    form = tw.Form(ui, "T", [tw.Select("x", [])], [tw.Button("OK")])
    keys(form, 10)
    assert ui.asked == []


# ---- geometry

def test_window_is_centred_and_clipped_with_room_for_the_shadow():
    assert tw.window_rect(24, 80, 10, 40) == (7, 20, 10, 40)
    y, x, h, w = tw.window_rect(24, 80, 100, 200)
    assert (h, w) == (20, 74) and y >= 1 and y + h + 1 <= 23 and x + w + 2 <= 80


def test_scrolling_keeps_the_focused_row_visible():
    assert tw.scroll_to(0, 0, 5, 20) == 0
    assert tw.scroll_to(0, 7, 5, 20) == 3
    assert tw.scroll_to(10, 4, 5, 20) == 4
    assert tw.scroll_to(3, None, 5, 20) == 3        # a button has the focus: stay
    assert tw.scroll_to(99, None, 5, 20) == 15 and tw.scroll_to(4, None, 30, 20) == 0


def test_page_keys_scroll_a_form_without_moving_the_focus():
    form, *_, ok = make()
    form.current = ok
    keys(form, curses.KEY_NPAGE)
    assert form.offset == 5 and form.current is ok
    keys(form, curses.KEY_PPAGE, curses.KEY_PPAGE)
    assert form.offset == 0


def test_wrap_keeps_styles_and_blank_lines():
    rows = tw.wrap(["a b c d e f g h i j k l", ("warn me", "warn"), ""], 12)
    assert rows[0] == ("a b c d e f", "window") and rows[1][0].startswith("    g")
    assert ("warn me", "warn") in rows and rows[-1] == ("", "window")


# ---- colours

def test_palette_by_terminal():
    assert tw.palette(256, True, {}) is tw.PURPLE_256
    assert tw.palette(8, True, {}) is tw.PURPLE_8
    assert tw.palette(256, False, {}) is None
    assert tw.palette(256, True, {"NO_COLOR": "1"}) is None
    assert set(tw.PURPLE_8) == set(tw.PURPLE_256)
    assert tw.PURPLE_256["screen"][1] == 54 and tw.PURPLE_8["screen"][1] == curses.COLOR_MAGENTA
