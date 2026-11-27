"""quadsim.mininewt: everything that needs no terminal (focus, keys, clicks, editing, scrolling, geometry)."""
import curses

from quadsim import mininewt as tw


class FakeUi:
    """Stands in for Screen: a <select> asks it for a choice, forms run on the keys it is given."""

    def __init__(self, pick=None, keys=()):
        self.pick, self.asked, self.keys = pick, [], [ord(k) if isinstance(k, str) else k for k in keys]
        self.stack, self.hits, self.mouse, self.painted = [], tw.HitMap(), None, []

    def choose(self, title, items, index=0):
        self.asked.append((title, items, index))
        return self.pick

    def size(self):
        return 24, 80

    def push(self, form):
        self.stack.append(form)

    def pop(self):
        self.stack.pop()

    def paint(self):
        self.painted.append([f.title for f in self.stack])

    def key(self):
        return self.keys.pop(0)


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


def test_left_and_right_move_along_the_buttons_and_stop_at_the_ends():
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
    assert not e.handle(tw.TAB) and not e.handle(10) and not e.handle(curses.KEY_UP) and not e.handle(curses.KEY_MOUSE)


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


def test_enter_in_the_last_entry_goes_to_the_default_button_not_cancel():
    last, cancel, ok = tw.Entry("b"), tw.Button("Cancel"), tw.Button("OK")
    form = tw.Form(FakeUi(), "T", [tw.Entry("a"), last], [cancel, ok])
    keys(form, 10)
    assert form.current is last
    keys(form, 10)
    assert form.current is ok


def test_a_one_question_form_can_submit_on_enter():
    form = tw.Form(FakeUi(), "T", [tw.Entry("a")], [tw.Button("Cancel"), tw.Button("OK")], enter_submits=True)
    assert keys(form, "x", 10) == "OK"


def test_a_click_in_an_entry_puts_the_cursor_there():
    form, name, on, *_ = make()
    form.current, name.col0 = on, 5
    assert form.click(name, 0, 6) is None
    assert form.current is name and name.pos == 1
    form.click(name, 0, 99)
    assert name.pos == 2


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


# ---- listbox

def listform(n=4, index=0, ui=None):
    box, back, ok = tw.Listbox([f"row {i}" for i in range(n)], index, return_exit=True), tw.Button("Back"), tw.Button("OK")
    return tw.Form(ui or FakeUi(), "T", [box], [back, ok], cancel_keys=(ord("q"),)), box, back, ok


def test_a_listbox_moves_its_bar_and_leaves_the_keys_at_its_ends():
    box = tw.Listbox(["a", "b", "c"], 1)
    assert not box.handle(tw.TAB) and not box.handle(curses.KEY_LEFT) and not box.handle(10)
    assert box.handle(curses.KEY_UP) and box.index == 0
    assert not box.handle(curses.KEY_UP) and box.index == 0
    assert box.handle(curses.KEY_END) and box.index == 2
    assert not box.handle(curses.KEY_DOWN) and box.index == 2
    assert box.handle(curses.KEY_HOME) and box.index == 0
    assert tw.Listbox(["a"], 7).index == 0 and tw.Listbox([], 3).index == 0


def test_a_listbox_pages_and_wheels_within_its_rows():
    box = tw.Listbox([str(i) for i in range(10)])
    box.h = 4
    box.handle(curses.KEY_NPAGE)
    assert box.index == 4
    box.handle(curses.KEY_NPAGE), box.handle(curses.KEY_NPAGE)
    assert box.index == 9
    box.handle(curses.KEY_PPAGE)
    assert box.index == 5
    assert box.wheel(3) and box.index == 8 and box.wheel(3) and box.index == 9 and not box.wheel(3)


def test_from_a_list_right_goes_to_the_default_button_and_left_to_the_first():
    form, box, back, ok = listform()
    keys(form, curses.KEY_RIGHT)
    assert form.current is ok
    form.current = box
    keys(form, curses.KEY_LEFT)
    assert form.current is back
    keys(form, curses.KEY_RIGHT, curses.KEY_RIGHT)
    assert form.current is ok


def test_down_past_the_last_row_reaches_the_buttons_and_up_comes_back():
    form, box, back, ok = listform(3)
    keys(form, curses.KEY_DOWN, curses.KEY_DOWN)
    assert form.current is box and box.index == 2
    keys(form, curses.KEY_DOWN)
    assert form.current is ok and box.index == 2
    keys(form, curses.KEY_UP)
    assert form.current is box and box.index == 2
    box.index = 0
    keys(form, curses.KEY_UP)
    assert form.current is ok
    keys(form, curses.KEY_DOWN)
    assert form.current is box


def test_enter_on_a_row_ends_the_form_as_its_default_button():
    form, box, back, ok = listform()
    assert keys(form, curses.KEY_DOWN, 10) == "OK" and box.index == 1
    assert keys(form, "q") is tw.CANCEL and keys(form, tw.ESC) is tw.CANCEL
    form.current = back
    assert keys(form, " ") == "Back"
    plain = tw.Listbox(["a"])                           # without return_exit a row does not end the form
    assert keys(tw.Form(FakeUi(), "T", [plain], [tw.Button("OK")]), 10) is None


# ---- mouse

def test_a_click_on_a_row_picks_it_and_a_click_on_a_button_presses_it():
    form, box, back, ok = listform()
    assert form.click(box, 2, 5) == "OK" and box.index == 2 and form.current is box
    assert form.click(box, 9, 0) is None and box.index == 2         # below the last row
    assert form.click(back) == "Back" and form.current is back


def test_a_click_toggles_a_checkbox_and_opens_a_select():
    ui = FakeUi(pick=1)
    form, name, on, kind, cancel, ok = make(ui)
    assert form.click(on) is None and on.checked and form.current is on
    assert form.click(kind) is None and kind.key == "b" and form.current is kind
    assert form.click(form.fields[0]) is None and form.current is kind     # a label: nothing
    hidden = tw.Checkbox("gone")
    assert form.click(hidden) is None and not hidden.checked


def test_mouse_events_go_to_what_was_drawn_there():
    ui = FakeUi()
    form, box, back, ok = listform(6, ui=ui)
    ui.hits.add(5, 10, 6, 20, box)
    ui.hits.add(12, 10, 1, 6, back)
    ui.mouse = (0, 0, tw.CLICK)                         # outside every component
    assert keys(form, curses.KEY_MOUSE) is None and form.current is box
    ui.mouse = (8, 15, tw.WHEEL_DOWN)
    assert keys(form, curses.KEY_MOUSE) is None and box.index == 3
    ui.mouse = (8, 15, tw.WHEEL_UP)
    keys(form, curses.KEY_MOUSE)
    assert box.index == 0
    ui.mouse = (6, 15, curses.BUTTON1_PRESSED)
    assert keys(form, curses.KEY_MOUSE) == "OK" and box.index == 1
    ui.mouse = (12, 12, curses.BUTTON1_CLICKED)
    assert keys(form, curses.KEY_MOUSE) == "Back"
    ui.mouse = None                                     # the event could not be read
    assert keys(form, curses.KEY_MOUSE) is None


def test_the_hit_map_gives_the_topmost_rectangle_and_the_place_in_it():
    hits = tw.HitMap()
    hits.add(2, 4, 3, 10, "a")
    hits.add(3, 6, 1, 2, "b")
    assert hits.at(2, 4) == ("a", 0, 0) and hits.at(4, 13) == ("a", 2, 9)
    assert hits.at(3, 7) == ("b", 0, 1)
    assert hits.at(5, 4) is None and hits.at(2, 14) is None
    hits.clear()
    assert hits.at(2, 4) is None


# ---- windows

def test_a_form_is_on_the_screen_while_it_runs_and_stays_under_what_it_opens():
    ui = FakeUi(keys=[10, 10])
    outer = tw.Form(ui, "outer", [tw.Label("x")], [tw.Button("OK")])
    with outer:
        assert tw.message_window(ui, "inner", ["hello"]) == "OK"
        assert outer.run() == "OK"
    assert ui.painted == [["outer", "inner"], ["outer"]] and ui.stack == []


def test_choice_window():
    assert tw.choice_window(FakeUi(keys=[curses.KEY_DOWN, 10]), "T", ["a", "b"]) == 1
    assert tw.choice_window(FakeUi(keys=[curses.KEY_RIGHT, 10]), "T", ["a", "b"], 1, info=["pick one"]) == 1
    assert tw.choice_window(FakeUi(keys=[curses.KEY_LEFT, 10]), "T", ["a", "b"]) is None
    assert tw.choice_window(FakeUi(keys=["q"]), "T", ["a", "b"]) is None
    assert tw.choice_window(FakeUi(keys=[tw.TAB, tw.TAB, 10]), "T", ["a"], buttons=("Quit", "Go")) == 0
    assert tw.choice_window(FakeUi(keys=[10, curses.KEY_RIGHT, 10]), "T", []) is None       # nothing to pick


def test_message_window_returns_the_button_and_starts_on_the_last_one():
    assert tw.message_window(FakeUi(keys=[10]), "T", ["a", ("b", "warn")], ("No", "Yes")) == "Yes"
    assert tw.message_window(FakeUi(keys=[curses.KEY_LEFT, 10]), "T", ["a"], ("No", "Yes")) == "No"
    assert tw.message_window(FakeUi(keys=[tw.ESC]), "T", ["a"]) is None


def test_entry_window_checks_the_text_and_stays_on_an_error():
    assert tw.entry_window(FakeUi(keys=["h", "i", 10]), "T", "Name") == "hi"
    assert tw.entry_window(FakeUi(keys=[tw.ESC]), "T", "Name", "x") is None
    seen = []
    ui = FakeUi(keys=["x", 10, curses.KEY_BACKSPACE, "7", 10])
    assert tw.entry_window(ui, "T", "N", check=int, on_error=seen.append) == 7 and len(seen) == 1


def test_text_scrolls_inside_itself_when_it_is_cut():
    text = tw.Text(tw.wrap([str(i) for i in range(6)], 20))
    assert text.height() == 6 and text.overflow() == (False, False) and not text.wheel(1)
    text.h = 4
    assert text.overflow() == (False, True) and text.wheel(5) and text.offset == 2 and text.overflow() == (True, False)
    form = tw.Form(FakeUi(), "T", [text], [tw.Button("OK")])
    keys(form, curses.KEY_PPAGE)
    assert text.offset == 0 and form.offset == 0


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


def test_flexible_rows_are_cut_to_fit_the_last_first():
    assert tw.fit([1, 6, 1], [False, True, False], 10) == [1, 6, 1]
    assert tw.fit([1, 6, 1], [False, True, False], 5) == [1, 3, 1]
    assert tw.fit([4, 1, 6], [True, False, True], 6) == [4, 1, 1]
    assert tw.fit([4, 1, 6], [True, False, True], 4) == [2, 1, 1]
    assert tw.fit([2, 2], [False, False], 1) == [2, 2]             # nothing to cut: the form scrolls


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
