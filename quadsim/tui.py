"""Menu front end for the sim: pick a robot, a scene and features and run; build scenes; keep DDS / topic settings.

  scripts/tui.sh                      in a checkout (Docker image, or the pixi env)
  pixi run sim tui                    the pixi env
  ./quadsim-<ver>.run tui             the standalone bundle
  options: --backend docker|native    how commands are run (default: docker when the quadsim image exists)
           --profile <name>           the settings profile to start from and keep up to date (default: last)

It looks and works like nmtui: windows with <buttons>, Tab / arrows to move, Enter to select, Space to toggle,
Esc to go back; the mouse works too (quadsim.mininewt). Every action shows the command it is about to run; the
commands themselves are built in quadsim.tui_plan. Standard library only (curses).
"""
from __future__ import annotations

import argparse
import copy
import curses
import locale
import os
import subprocess
import sys

from quadsim import tui_plan as tp
from quadsim.mininewt import (Button, Checkbox, Entry, Form, Label, Screen, Select, Spacer, choice_window,
                              entry_window, message_window)

OTHER = "\0other"           # the "Other file..." row of a <select>
NO_YAML = "config/sim.yaml could not be read (PyYAML missing?): names cannot be edited"


def size_text(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def short(path: str) -> str:
    """The path with the home directory written as ~."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path


class Tui:
    def __init__(self, scr, app_dir: str, backend: str, profile: str, directory: str):
        self.scr, self.app, self.backend, self.profile, self.dir = scr, app_dir, backend, profile, directory
        self.ui = Screen(scr, f"quadsim   backend: {backend}   profile: {profile}")
        self.settings, self.notes = tp.load_profile(profile, directory)
        self.topic_defaults, self.frame_defaults, self.spawns = tp.ros_defaults(app_dir)

    # ---- small dialogs -----------------------------------------------------------------------------------------

    def info(self, title: str, *lines) -> None:
        message_window(self.ui, title, list(lines))

    def error(self, e: Exception) -> None:
        message_window(self.ui, "Cannot do that", [(str(e) or "not a valid value", "warn")])

    def ask(self, title: str, label: str, default: str = "", info: list | None = None, check=None):
        """One value in a small window. Returns check(text) (the text without a check), None on Cancel."""
        return entry_window(self.ui, title, label, default, info, check, (tp.PlanError, ValueError), self.error)

    # ---- running commands --------------------------------------------------------------------------------------

    def execute(self, cmd: tp.Command) -> int:
        """Leave curses, run the command on the terminal, come back when the user presses Enter."""
        curses.def_prog_mode()
        curses.endwin()
        print(f"\n$ {cmd.shown()}\n", flush=True)
        try:
            p = subprocess.Popen(cmd.argv, cwd=self.app, env={**os.environ, **cmd.env})
        except OSError as e:
            print(f"could not start: {e}")
            rc = 127
        else:
            while True:
                try:
                    rc = p.wait()
                    break
                except KeyboardInterrupt:       # Ctrl-C reaches the child too: wait for it to finish stopping
                    continue
        try:
            print(f"\n[exit code {rc}] press Enter to return to the menu ", end="", flush=True)
            input()             # the prompt is printed here: input() would send it to stderr when that is redirected
        except (EOFError, KeyboardInterrupt):
            pass
        curses.reset_prog_mode()
        self.scr.refresh()
        self.ui.resume()
        return rc

    def autosave(self) -> None:
        try:
            tp.save_profile(self.profile, self.settings, self.dir)
        except OSError as e:
            self.error(tp.PlanError(f"settings not saved: {e}"))

    # ---- main --------------------------------------------------------------------------------------------------

    def main(self) -> None:
        if self.notes:
            self.info("Settings", *self.notes)
        i = 0
        while True:
            run = self.settings["run"]
            last = f"Run last  ({run['robot']}, {run['scene']})" if run.get("scene") else "Run last  (nothing yet)"
            i = choice_window(self.ui, "quadsim", ["Run simulation", last, "Build scene", "Settings"], i,
                              buttons=("Quit", "OK"))
            if i is None:
                return
            try:
                (self.run_form, self.run_last, self.build_menu, self.settings_form)[i]()
            except tp.PlanError as e:
                self.error(e)

    # ---- run ---------------------------------------------------------------------------------------------------

    def run_last(self) -> None:
        if not self.settings["run"].get("scene"):
            raise tp.PlanError("nothing has been run yet: use Run simulation")
        self.run_confirm()

    def run_form(self) -> None:
        run = self.settings["run"]
        scenes = tp.scan_scenes(self.app)
        by_path = {s.path: s for s in scenes}

        def scene_options(robot_key):
            return [(s.path, s.name + s.tags()) for s in scenes if tp.runs_robot(s, robot_key)]

        robot = Select("Robot", list(tp.ROBOTS), run["robot"])
        behavior = Select("Behaviour", list(tp.BEHAVIORS), run["behavior"])
        behavior.visible = lambda: robot.key == "g23"
        scene = Select("Scene", scene_options(robot.key), run["scene"])
        spawn = Select("Spawn", [(tp.SPAWN_CONFIG, "Config default (config/sim.yaml)"),
                                 (tp.SPAWN_SCENE, "Scene default (the scene's meta.json)"),
                                 *((n, f"{n} (named spawn)") for n in self.spawns)], run["spawn"])
        color = Select("Map colour", [(c, c) for c in tp.MAP_COLORS], run["map_color"])
        checks = {k: Checkbox(label, k in run["features"]) for k, label, _ in tp.FEATURES}

        def scene_changed(_=None):                  # a new scene: start from what suits it
            s = by_path.get(scene.key)
            if s:
                spawn.set(tp.default_spawn(s))
                checks["stairs_pool"].checked = s.stairs

        scene.on_change = scene_changed
        robot.on_change = lambda _: scene.set_options(scene_options(robot.key))
        if scene.key != run["scene"]:
            scene_changed()
        form = Form(self.ui, "Run simulation", [robot, behavior, scene, spawn, color, Spacer(), *checks.values()],
                    [Button("Cancel"), Button("Start")])
        with form:
            while form.run() == "Start":
                if scene.key is None:
                    self.error(tp.PlanError("no built scene for this robot under scenes/: use Build scene first"))
                    continue
                self.settings["run"] = {"robot": robot.key, "behavior": behavior.key, "scene": scene.key,
                                        "spawn": spawn.key, "map_color": color.key,
                                        "features": [k for k, c in checks.items() if c.checked]}
                self.autosave()
                try:
                    if self.run_confirm():
                        return
                except tp.PlanError as e:
                    self.error(e)

    def run_confirm(self) -> bool:
        """Show what will run; True when it ran, False on <Back>."""
        run, dds = self.settings["run"], self.settings["dds"]
        cmd = tp.sim_command(self.settings, self.backend)
        names = len(self.settings["topics"]) + len(self.settings["frames"])
        lines = [f"Robot     {run['robot']}" + (f" ({run['behavior']})" if run["robot"] == "g23" else ""),
                 f"Scene     {run['scene']}",
                 f"Spawn     {run['spawn']}",
                 f"Features  {', '.join(run['features']) or '-'}   map colour {run['map_color']}",
                 f"DDS       domain {dds['domain_id']}, config {dds['config']}",
                 f"Names     {names} topic / frame name(s) changed (Settings)" if names else
                 "Names     topics and frames as in config/sim.yaml",
                 *((f"WARNING   {w}", "warn") for w in tp.dds_warnings(self.settings, self.app)),
                 "", (f"$ {cmd.shown()}", "dim"), "", "Stop the sim with Ctrl-C."]
        if message_window(self.ui, "Run simulation", lines, ("Back", "Run")) != "Run":
            return False
        self.execute(cmd)
        return True

    # ---- settings ----------------------------------------------------------------------------------------------

    def settings_form(self) -> None:
        """Edit a copy; <OK> makes it the settings and saves the profile, <Cancel> drops the changes."""
        draft = copy.deepcopy(self.settings)
        have_names = bool(self.topic_defaults)
        files = tp.dds_configs(self.app)

        def current_ns():
            return tp.namespace(tp.effective(self.topic_defaults, draft["topics"]))

        def dds_options():
            opts = [(f, f"conf/{f}") for f in files]
            if self.backend == "native":
                opts += [("none", "Cyclone defaults (no file)"), (OTHER, "Other file...")]
            if draft["dds"]["config"] not in dict(opts):
                opts.insert(0, (draft["dds"]["config"], draft["dds"]["config"]))
            return opts

        def dds_changed(sel):
            if sel.key != OTHER:
                draft["dds"]["config"] = sel.key
                return
            def check(text):
                path = os.path.abspath(os.path.expanduser(text))
                if not os.path.isfile(path):
                    raise tp.PlanError(f"no such file: {path}")
                return tp.check_path(path, "DDS config")
            path = self.ask("DDS config", "XML file", "", check=check)
            if path:
                draft["dds"]["config"] = path
            sel.options = dds_options()
            sel.key = draft["dds"]["config"]

        domain = Entry("ROS domain ID", str(draft["dds"]["domain_id"]), width=6)
        dds = Select("DDS config", dds_options(), draft["dds"]["config"], on_change=dds_changed)
        ns = Entry("Robot namespace", current_ns(), width=20)
        counts = Label("", "dim")
        fields = [domain, dds, *([ns] if have_names else [Label(NO_YAML, "dim")]), Spacer(), counts,
                  *([Button("topics", "Topics..."), Button("frames", "Frames...")] if have_names else []),
                  Button("profiles", "Profiles..."), Spacer(),
                  Label("Saved on <OK> to", "dim"),
                  Label("  " + short(tp.profile_path(self.profile, self.dir)), "dim")]
        if self.backend == "docker":
            fields.insert(2, Label("  (the Docker backend takes files under conf/ only)", "dim"))
        form = Form(self.ui, "Settings", fields, [Button("Cancel"), Button("OK")])

        def collect():
            """The entries -> draft; raises PlanError on a bad value."""
            if not domain.value.isdigit() or int(domain.value) > 232:
                form.current = domain
                raise tp.PlanError("ROS domain ID: a number from 0 to 232")
            draft["dds"]["domain_id"] = int(domain.value)
            if have_names and ns.value.strip("/") != current_ns():
                try:
                    tp.rename_namespace(draft, self.topic_defaults, self.frame_defaults, ns.value)
                except tp.PlanError:
                    form.current = ns
                    raise

        def refresh():
            domain.set(str(draft["dds"]["domain_id"]))
            dds.options = dds_options()
            dds.key = draft["dds"]["config"]
            ns.set(current_ns())

        with form:
            while True:
                counts.text = f"{len(draft['topics'])} topic and {len(draft['frames'])} frame name(s) changed"
                pressed = form.run()
                if pressed in (None, "Cancel"):
                    return
                try:
                    collect()
                    if pressed == "OK":
                        self.settings = draft
                        self.autosave()
                        return
                    if pressed == "topics":
                        self.edit_names("Topics", draft["topics"], self.topic_defaults)
                    elif pressed == "frames":
                        self.edit_names("Frames", draft["frames"], self.frame_defaults)
                    elif pressed == "profiles":
                        self.profiles_menu(draft)
                    refresh()
                except tp.PlanError as e:
                    self.error(e)

    def edit_names(self, title: str, changed: dict[str, str], defaults: dict[str, str]) -> None:
        """One entry per name; <OK> writes the ones that differ from config/sim.yaml into `changed`."""
        entries = {k: Entry(k, v, width=44) for k, v in tp.effective(defaults, changed).items()}
        form = Form(self.ui, title, [Label("Empty or the config/sim.yaml value = not changed.", "dim"), Spacer(),
                                     *entries.values()], [Button("reset", "Reset all"), Button("Cancel"), Button("OK")])
        with form:
            while True:
                pressed = form.run()
                if pressed in (None, "Cancel"):
                    return
                if pressed == "reset":
                    for k, e in entries.items():
                        e.set(defaults[k])
                    continue
                new = {}
                try:
                    for k, e in entries.items():
                        form.current = e
                        tp.set_name(new, defaults, k, e.value)
                except tp.PlanError as e:
                    self.error(e)
                    continue
                changed.clear()
                changed.update(new)
                return

    def profiles_menu(self, draft: dict) -> None:
        i = choice_window(self.ui, "Profiles", ["Save as...", "Load...", "Delete...", "Reset to defaults"],
                    info=[f"Directory: {short(self.dir)}", f"Current profile: {self.profile}"])
        if i is None:
            return
        if i == 0:
            name = self.ask("Save profile as", "Name", check=lambda t: tp.check_name(t, "profile name"))
            if name:
                self.info("Profiles", f"Saved {short(tp.save_profile(name, draft, self.dir))}")
        elif i == 3:
            draft.clear()
            draft.update(tp.default_settings())
        else:
            names = [n for n in tp.list_profiles(self.dir) if n != self.profile]
            if not names:
                raise tp.PlanError("no other saved profile yet: use Save as...")
            n = choice_window(self.ui, "Load profile" if i == 1 else "Delete profile", names)
            if n is None:
                return
            if i == 1:
                loaded, notes = tp.load_profile(names[n], self.dir)
                draft.clear()
                draft.update(loaded)
                self.info("Profiles", f"Loaded {names[n]}: press <OK> in Settings to keep it.", *notes)
            elif message_window(self.ui, "Delete profile", [f"Delete {names[n]}?"], ("No", "Yes")) == "Yes":
                tp.delete_profile(names[n], self.dir)

    # ---- build -------------------------------------------------------------------------------------------------

    def build_menu(self) -> None:
        i = choice_window(self.ui, "Build scene", ["From a map file (.pcd / .bt / .ot)",
                                             "From OpenStreetMap (streets and buildings)", "Synthetic slope",
                                             "Synthetic stairs on an existing scene"],
                    info=[f"New scenes go to {short(os.path.join(self.app, 'scenes'))}"])
        if i is None:
            return
        made = (self.build_map, self.build_osm, self.build_slope, self.build_stairs)[i]()
        ready = [f"{made} is ready.", "", "Run it now?"]
        if made and message_window(self.ui, "Build scene", ready, ("No", "Yes")) == "Yes":
            self.run_new(made)

    def run_new(self, path: str) -> None:
        """Run a scene that was just built: the last Run choices, with the spawn and features that suit it."""
        scene = next((s for s in tp.scan_scenes(self.app) if s.path == path), None)
        if scene is None:
            raise tp.PlanError(f"{path} is not a runnable scene (see the build output)")
        run = self.settings["run"]
        if not tp.runs_robot(scene, run["robot"]):
            run["robot"] = "ideal"
        feats = set(run["features"]) - {"stairs_pool"} | ({"stairs_pool"} if scene.stairs else set())
        run.update(scene=path, spawn=tp.default_spawn(scene), features=[k for k, _, _ in tp.FEATURES if k in feats])
        self.autosave()
        self.run_confirm()

    def build_form(self, title: str, fields: list, plan) -> str | None:
        """Show the form until plan() gives (command, name, info lines) and the user builds, or cancels.
        Returns the new scene's path when the build succeeded."""
        form = Form(self.ui, title, fields, [Button("Cancel"), Button("Build")])
        with form:
            while form.run() == "Build":
                try:
                    cmd, name, lines = plan()
                    out = tp.scene_out(name)
                except (tp.PlanError, ValueError) as e:
                    self.error(e)
                    continue
                full = os.path.join(self.app, out)
                exists = [("This scene already exists and will be OVERWRITTEN.", "warn")] if os.path.exists(full) else []
                lines = [*lines, f"Output    {short(full)}", *exists, "", (f"$ {cmd.shown()}", "dim")]
                if message_window(self.ui, title, lines, ("Back", "Build")) == "Build":
                    return out if self.execute(cmd) == 0 else None
        return None

    def name_entry(self, default: str = "") -> Entry:
        return Entry("New scene name", default, width=30)

    def build_map(self) -> str | None:
        maps = tp.list_maps(self.app)
        auto = lambda path: os.path.splitext(os.path.basename(path))[0].replace(".", "_") if path else ""  # noqa: E731
        options = [*((p, f"{p}  ({size_text(n)})") for p, n in maps), (OTHER, "Other file...")]
        chosen = {"path": maps[0][0] if maps else ""}

        def map_changed(sel):
            old = chosen["path"]
            if sel.key == OTHER:
                path = self.ask("Map file", "Path", check=lambda t: tp.map_path(t, self.app, self.backend),
                                info=["A .pcd, .bt or .ot file." + ("" if self.backend == "native" else
                                                                    " With Docker it must be inside the checkout.")])
                if path and path not in dict(sel.options):
                    sel.options.insert(0, (path, path))
                sel.key = path or old or OTHER
            chosen["path"] = "" if sel.key == OTHER else sel.key
            if name.value == auto(old):
                name.set(auto(chosen["path"]))

        source = Select("Map", options, chosen["path"] or OTHER, on_change=map_changed)
        recipe = Select("Recipe", list(tp.RECIPES))
        name = self.name_entry(auto(chosen["path"]))

        def plan():
            if not chosen["path"]:
                raise tp.PlanError("pick a map file first")
            path = tp.map_path(chosen["path"], self.app, self.backend)
            return (tp.build_map_command(path, name.value, recipe.key, self.backend), name.value,
                    [f"Map       {path}", f"Recipe    {recipe.text()}",
                     "A large point cloud takes from seconds to a few minutes."])

        return self.build_form("Build from a map file", [
            source, recipe, name, Spacer(),
            Label("1:1 keeps the scan as it is (the default scenes are built this way).", "dim"),
            Label("Flattened replaces the floor with clean slabs; the stairs recipe keeps steps sharp.", "dim")], plan)

    def build_osm(self) -> str | None:
        scans = [s for s in tp.scan_scenes(self.app) if not s.geo]
        centre, radius = Entry("Centre", "", width=44), Entry("Half side (m)", "1000", width=8)
        base = Select("Around a scan", [("", "No: OSM only"), *((s.path, s.name) for s in scans)])
        name = self.name_entry()

        def plan():
            return (tp.build_osm_command(centre.value, float(radius.value), name.value, base.key or None, self.backend),
                    name.value, [f"Centre    {centre.value}", f"Radius    {float(radius.value):g} m",
                                 f"Base      {base.key or '-'}",
                                 f"The first build downloads OSM data (needs a network) into Maps/{name.value}/; "
                                 "later builds reuse it."])

        return self.build_form("Build from OpenStreetMap", [
            centre, radius, base, name, Spacer(),
            Label("Centre: 'LAT LON' (24.95919 121.41896) or a streets.gl URL of the place.", "dim"),
            Label("Around a scan: that scene stays where it is, OSM fills the area around it.", "dim")], plan)

    def build_slope(self) -> str | None:
        angle, name = Entry("Incline angle (deg)", "10", width=8), self.name_entry("slope_10")

        def plan():
            return (tp.build_slope_command(float(angle.value), name.value, self.backend), name.value,
                    [f"Angle     {float(angle.value):g} deg"])

        return self.build_form("Build a synthetic slope", [angle, name], plan)

    def build_stairs(self) -> str | None:
        scenes, specs = tp.scan_scenes(self.app), tp.list_stair_specs(self.app)
        if not scenes or not specs:
            raise tp.PlanError("needs a built scene and a config/stairs_*.yaml spec written for it")
        base = Select("Base scene", [(s.path, s.name + s.tags()) for s in scenes])
        spec = Select("Spec", [(p, p) for p in specs])
        name = self.name_entry(os.path.basename(base.key) + "_s16")
        auto = {"name": name.value}

        def base_changed(sel):
            if name.value == auto["name"]:
                auto["name"] = os.path.basename(sel.key) + "_s16"
                name.set(auto["name"])

        base.on_change = base_changed

        def plan():
            return (tp.build_stairs_command(base.key, spec.key, name.value, self.backend), name.value,
                    [f"Base      {base.key}", f"Spec      {spec.key}"])

        return self.build_form("Build synthetic stairs", [
            base, spec, name, Spacer(),
            Label("The scanned staircases listed in the spec are replaced with clean steps.", "dim"),
            Label("The spec must have been written for this scene's map.", "dim")], plan)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quadsim tui", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=tp.BACKENDS, help="default: docker when the quadsim image exists")
    ap.add_argument("--profile", default=tp.LAST, help="settings profile (default: %(default)s)")
    a = ap.parse_args(argv)
    app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        tp.check_name(a.profile, "profile name")
        backend = a.backend or tp.detect_backend()
    except tp.PlanError as e:
        sys.exit(f"[quadsim tui] {e}")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("[quadsim tui] needs a terminal")
    os.environ.setdefault("ESCDELAY", "25")
    # Borders: in a UTF-8 locale have ncurses send real box-drawing characters; the VT100 alternate character set
    # it would use for TERM=xterm and the like comes out as "lqqqk" on terminals that ignore it in UTF-8 mode.
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("NCURSES_NO_UTF8_ACS", "1")
    os.chdir(app)
    try:
        curses.wrapper(lambda scr: Tui(scr, app, backend, a.profile, tp.profile_dir()).main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
