"""Menu front end for the sim: pick a robot, a scene and features and run; build scenes; keep DDS / topic settings.

  scripts/tui.sh                      in a checkout (Docker image, or the pixi env)
  pixi run sim tui                    the pixi env
  ./quadsim-<ver>.run tui             the standalone bundle
  options: --backend docker|native    how commands are run (default: docker when the quadsim image exists)
           --profile <name>           the settings profile to start from and keep up to date (default: last)

Keys: Up/Down move, Enter selects, Space toggles, q or Esc goes back. Every action shows the command it is about to
run; the commands themselves are built in quadsim.tui_plan. Standard library only (curses).
"""
from __future__ import annotations

import argparse
import copy
import curses
import os
import subprocess
import sys
import textwrap

from quadsim import tui_plan as tp

ENTER = (curses.KEY_ENTER, 10, 13)
BACK = (27, ord("q"))
HINT_MENU = "Up/Down: move   Enter: select   q/Esc: back"
HINT_CHECK = "Up/Down: move   Space: toggle   Enter: done   q/Esc: back"
HINT_TEXT = "Enter: accept   Esc: cancel   (empty = default)"


def size_text(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class Tui:
    def __init__(self, scr, app_dir: str, backend: str, profile: str, directory: str):
        self.scr, self.app, self.backend, self.profile, self.dir = scr, app_dir, backend, profile, directory
        self.settings, notes = tp.load_profile(profile, directory)
        self.topic_defaults, self.frame_defaults, self.spawns = tp.ros_defaults(app_dir)
        self.note = "; ".join(notes)
        curses.curs_set(0)
        scr.keypad(True)

    # ---- drawing -----------------------------------------------------------------------------------------------

    def put(self, y: int, x: int, text: str, attr=0) -> None:
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and x < w - 1:
            try:
                self.scr.addstr(y, x, text[:w - 1 - x], attr)
            except curses.error:
                pass

    def frame(self, title: str, hint: str) -> int:
        """Draw the header and footer; return the first free row."""
        self.scr.erase()
        h, _ = self.scr.getmaxyx()
        self.put(0, 1, f"quadsim   backend: {self.backend}   profile: {self.profile}", curses.A_DIM)
        self.put(1, 1, title, curses.A_BOLD)
        if self.note:
            self.put(h - 2, 1, self.note, curses.A_BOLD)
        self.put(h - 1, 1, hint, curses.A_DIM)
        return 3

    def lines(self, y: int, text: list[str]) -> int:
        _, w = self.scr.getmaxyx()
        for line in text:
            for part in textwrap.wrap(line, max(20, w - 4), subsequent_indent="    ") or [""]:
                self.put(y, 2, part)
                y += 1
        return y

    def key(self) -> int:
        k = self.scr.getch()
        if k != curses.KEY_RESIZE:
            self.note = ""
        return k

    # ---- widgets -----------------------------------------------------------------------------------------------

    def menu(self, title: str, items: list[str], index: int = 0, info: list[str] | None = None,
             marks: set[int] | None = None, hint: str = HINT_MENU) -> int | None:
        """Pick one row (marks is None), or toggle rows with Space (marks: the set of chosen rows, edited in place)."""
        index = min(max(index, 0), len(items) - 1)
        top = 0
        while True:
            y0 = self.lines(self.frame(title, hint), info or [])
            y0 += 1 if info else 0
            h, _ = self.scr.getmaxyx()
            rows = max(1, h - y0 - 3)
            top = min(max(top, index - rows + 1), index)
            for i, item in enumerate(items[top:top + rows], start=top):
                box = "" if marks is None else ("[x] " if i in marks else "[ ] ")
                self.put(y0 + i - top, 2, f"{'>' if i == index else ' '} {box}{item}",
                         curses.A_REVERSE if i == index else 0)
            if top + rows < len(items):
                self.put(y0 + rows, 4, f"... {len(items) - top - rows} more", curses.A_DIM)
            k = self.key()
            if k in (curses.KEY_UP, ord("k")):
                index = (index - 1) % len(items)
            elif k in (curses.KEY_DOWN, ord("j")):
                index = (index + 1) % len(items)
            elif k == ord(" ") and marks is not None:
                marks.symmetric_difference_update({index})
            elif k in ENTER:
                return index
            elif k in BACK:
                return None

    def ask(self, title: str, label: str, default: str = "", info: list[str] | None = None) -> str | None:
        """One line of text; None on Esc. An empty answer is the default."""
        text = ""
        curses.curs_set(1)
        try:
            while True:
                y = self.lines(self.frame(title, HINT_TEXT), info or [])
                y += 1 if info else 0
                prompt = f"{label}{f' [{default}]' if default else ''}: "
                self.put(y, 2, prompt + text)
                _, w = self.scr.getmaxyx()
                self.scr.move(y, min(w - 2, 2 + len(prompt) + len(text)))
                k = self.key()
                if k in ENTER:
                    return text.strip() or default
                if k == 27:
                    return None
                if k in (curses.KEY_BACKSPACE, 127, 8):
                    text = text[:-1]
                elif 32 <= k < 127:
                    text += chr(k)
        finally:
            curses.curs_set(0)

    def ask_checked(self, title: str, label: str, default: str, check, info: list[str] | None = None):
        """ask() until check(text) stops raising PlanError / ValueError; returns check's result, None on Esc."""
        while True:
            text = self.ask(title, label, default, info)
            if text is None:
                return None
            try:
                return check(text)
            except (tp.PlanError, ValueError) as e:
                self.note = str(e) or "not a valid value"

    def confirm(self, title: str, info: list[str], yes: str = "Start", extra: list[str] | None = None) -> int | None:
        """0 = yes, 1.. = the extra rows, None = cancel."""
        items = [yes, *(extra or []), "Cancel"]
        i = self.menu(title, items, info=info)
        return None if i is None or i == len(items) - 1 else i

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
            input(f"\n[exit code {rc}] press Enter to return to the menu ")
        except (EOFError, KeyboardInterrupt):
            pass
        curses.reset_prog_mode()
        self.scr.refresh()
        return rc

    def autosave(self) -> None:
        try:
            tp.save_profile(self.profile, self.settings, self.dir)
        except OSError as e:
            self.note = f"settings not saved: {e}"

    # ---- main --------------------------------------------------------------------------------------------------

    def main(self) -> None:
        i = 0
        while True:
            run = self.settings["run"]
            last = f"Run last  ({run['robot']}, {run['scene']})" if run.get("scene") else "Run last  (nothing yet)"
            i = self.menu("Main menu", ["Run simulation", last, "Build scene", "Settings", "Quit"], i or 0)
            if i in (None, 4):
                return
            try:
                (self.run_wizard, self.run_last, self.build_menu, self.settings_menu)[i]()
            except tp.PlanError as e:
                self.note = str(e)

    # ---- run ---------------------------------------------------------------------------------------------------

    def run_last(self) -> None:
        if not self.settings["run"].get("scene"):
            raise tp.PlanError("nothing has been run yet: use Run simulation")
        self.run_confirm()

    def run_wizard(self) -> None:
        run = copy.deepcopy(self.settings["run"])
        keys = [k for k, _ in tp.ROBOTS]
        i = self.menu("Run 1/4: robot", [label for _, label in tp.ROBOTS], keys.index(run["robot"]))
        if i is None:
            return
        run["robot"] = keys[i]
        if run["robot"] == "g23":
            keys = [k for k, _ in tp.BEHAVIORS]
            i = self.menu("Run 1/4: G23 behaviour", [label for _, label in tp.BEHAVIORS], keys.index(run["behavior"]))
            if i is None:
                return
            run["behavior"] = keys[i]

        scenes = [s for s in tp.scan_scenes(self.app) if tp.runs_robot(s, run["robot"])]
        if not scenes:
            raise tp.PlanError("no built scene for this robot under scenes/: use Build scene first")
        paths = [s.path for s in scenes]
        i = self.menu("Run 2/4: scene", [s.name + s.tags() for s in scenes],
                      paths.index(run["scene"]) if run["scene"] in paths else 0)
        if i is None:
            return
        scene = scenes[i]
        if scene.path != run["scene"]:              # a new scene: start from what suits it
            run["spawn"] = tp.default_spawn(scene)
            feats = set(run["features"]) - {"stairs_pool"} | ({"stairs_pool"} if scene.stairs else set())
            run["features"] = [k for k, _, _ in tp.FEATURES if k in feats]
        run["scene"] = scene.path

        keys = [tp.SPAWN_CONFIG, tp.SPAWN_SCENE, *self.spawns]
        labels = ["Config default (spawn in config/sim.yaml)", "Scene default (spawn in the scene's meta.json)",
                  *(f"{name}  (named spawn, config/sim.yaml)" for name in self.spawns)]
        i = self.menu("Run 3/4: spawn", labels, keys.index(run["spawn"]) if run["spawn"] in keys else 0)
        if i is None:
            return
        run["spawn"] = keys[i]

        keys = [k for k, _, _ in tp.FEATURES]
        marks = {keys.index(k) for k in run["features"]}
        if self.menu("Run 4/4: features", [label for _, label, _ in tp.FEATURES], marks=marks,
                     hint=HINT_CHECK) is None:
            return
        run["features"] = [k for n, k in enumerate(keys) if n in marks]
        self.settings["run"] = run
        self.autosave()
        self.run_confirm()

    def run_confirm(self) -> None:
        while True:
            run, dds = self.settings["run"], self.settings["dds"]
            cmd = tp.sim_command(self.settings, self.backend)
            names = len(self.settings["topics"]) + len(self.settings["frames"])
            info = [f"Robot     {run['robot']}" + (f" ({run['behavior']})" if run["robot"] == "g23" else ""),
                    f"Scene     {run['scene']}",
                    f"Spawn     {run['spawn']}",
                    f"Features  {', '.join(run['features']) or '-'}",
                    f"DDS       domain {dds['domain_id']}, config {dds['config']}",
                    f"Names     {names} topic / frame name(s) changed (Settings)" if names else
                    "Names     topics and frames as in config/sim.yaml",
                    *(f"WARNING   {w}" for w in tp.dds_warnings(self.settings, self.app)),
                    "", f"$ {cmd.shown()}", "", "Stop the sim with Ctrl-C."]
            i = self.confirm("Run: confirm", info, extra=[f"Map colour: {run['map_color']}  (Enter changes)"])
            if i is None:
                return
            if i == 1:
                run["map_color"] = tp.MAP_COLORS[(tp.MAP_COLORS.index(run["map_color"]) + 1) % len(tp.MAP_COLORS)]
                self.autosave()
                continue
            self.execute(cmd)
            return

    # ---- settings ----------------------------------------------------------------------------------------------

    def settings_menu(self) -> None:
        i = 0
        while True:
            s = self.settings
            ns = tp.namespace(tp.effective(self.topic_defaults, s["topics"]))
            i = self.menu("Settings (saved as you change them)", [
                f"ROS domain ID        {s['dds']['domain_id']}",
                f"DDS config           {s['dds']['config']}",
                f"Robot namespace      /{ns}" if ns else "Robot namespace      (none)",
                f"Topics               {len(s['topics'])} changed",
                f"Frames               {len(s['frames'])} changed",
                "Profiles             save / load / delete / reset",
                "Back"], i or 0, info=[f"File: {tp.profile_path(self.profile, self.dir)}"])
            if i in (None, 6):
                return
            try:
                (self.set_domain, self.set_dds_config, self.set_namespace,
                 lambda: self.edit_names("Topics", "topics", self.topic_defaults),
                 lambda: self.edit_names("Frames", "frames", self.frame_defaults), self.profiles_menu)[i]()
            except tp.PlanError as e:
                self.note = str(e)
            self.autosave()

    def set_domain(self) -> None:
        def check(text):
            if not text.isdigit() or int(text) > 232:
                raise tp.PlanError("ROS domain ID: a number from 0 to 232")
            return int(text)
        v = self.ask_checked("Settings: ROS domain ID", "ROS_DOMAIN_ID", str(self.settings["dds"]["domain_id"]), check,
                             ["Every ROS 2 program that should talk to the sim needs the same number."])
        if v is not None:
            self.settings["dds"]["domain_id"] = v

    def set_dds_config(self) -> None:
        files = tp.dds_configs(self.app)
        native = self.backend == "native"
        items = [f"conf/{f}" for f in files]
        if native:
            items += ["Cyclone defaults (no config file: multicast on every interface)", "Other file..."]
        info = ["The CycloneDDS XML the sim starts with."]
        if not native:
            info.append("The Docker backend takes files under conf/ only: copy yours there.")
        cur = self.settings["dds"]["config"]
        i = self.menu("Settings: DDS config", items, files.index(cur) if cur in files else 0, info=info)
        if i is None:
            return
        if i < len(files):
            self.settings["dds"]["config"] = files[i]
        elif i == len(files):
            self.settings["dds"]["config"] = "none"
        else:
            def check(text):
                path = os.path.abspath(os.path.expanduser(text))
                if not os.path.isfile(path):
                    raise tp.PlanError(f"no such file: {path}")
                return tp.check_path(path, "DDS config")
            v = self.ask_checked("Settings: DDS config", "XML file", "", check)
            if v is not None:
                self.settings["dds"]["config"] = v

    def set_namespace(self) -> None:
        if not self.topic_defaults:
            raise tp.PlanError("config/sim.yaml could not be read (PyYAML missing?): names cannot be edited")
        ns = tp.namespace(tp.effective(self.topic_defaults, self.settings["topics"]))
        v = self.ask("Settings: robot namespace", "Namespace", ns,
                     [f"Renames every /{ns}/... topic and {ns}/... frame in one go."])
        if v is not None and v.strip("/") != ns:
            tp.rename_namespace(self.settings, self.topic_defaults, self.frame_defaults, v)

    def edit_names(self, title: str, section: str, defaults: dict[str, str]) -> None:
        if not defaults:
            raise tp.PlanError("config/sim.yaml could not be read (PyYAML missing?): names cannot be edited")
        keys, i = list(defaults), 0
        width = max(map(len, keys))
        while True:
            changed = self.settings[section]
            now = tp.effective(defaults, changed)
            rows = [f"{k:<{width}}  {now[k]}{'   *' if k in changed else ''}" for k in keys]
            i = self.menu(f"Settings: {title}  (* = changed)", [*rows, "Reset all to config/sim.yaml", "Back"], i or 0)
            if i is None or i == len(keys) + 1:
                return
            if i == len(keys):
                changed.clear()
            else:
                k = keys[i]
                v = self.ask(f"Settings: {title}", k, now[k], [f"config/sim.yaml: {defaults[k]}",
                                                               "Type the default to undo a change."])
                if v is not None:
                    try:
                        tp.set_name(changed, defaults, k, v)
                    except tp.PlanError as e:
                        self.note = str(e)
            self.autosave()

    def profiles_menu(self) -> None:
        i = self.menu("Settings: profiles", ["Save as...", "Load...", "Delete...", "Reset to defaults", "Back"],
                      info=[f"Directory: {self.dir}", f"Current: {self.profile} (kept up to date automatically)"])
        if i in (None, 4):
            return
        if i == 0:
            name = self.ask_checked("Profiles: save as", "Name", "", lambda t: tp.check_name(t, "profile name"))
            if name:
                tp.save_profile(name, self.settings, self.dir)
                self.note = f"saved {tp.profile_path(name, self.dir)}"
        elif i == 3:
            self.settings = tp.default_settings()
            self.note = "settings reset to the defaults"
        else:
            names = [n for n in tp.list_profiles(self.dir) if n != self.profile]
            if not names:
                raise tp.PlanError("no other saved profile yet: use Save as...")
            n = self.menu("Profiles: " + ("load" if i == 1 else "delete"), names)
            if n is None:
                return
            if i == 1:
                self.settings, notes = tp.load_profile(names[n], self.dir)
                self.note = "; ".join(notes) or f"loaded {names[n]}"
            else:
                tp.delete_profile(names[n], self.dir)
                self.note = f"deleted {names[n]}"

    # ---- build -------------------------------------------------------------------------------------------------

    def build_menu(self) -> None:
        i = self.menu("Build scene", ["From a map file (.pcd / .bt / .ot)", "From OpenStreetMap (streets and buildings)",
                                      "Synthetic slope", "Synthetic stairs on an existing scene", "Back"],
                      info=[f"New scenes go to {os.path.join(self.app, 'scenes')}"])
        if i in (None, 4):
            return
        made = (self.build_map, self.build_osm, self.build_slope, self.build_stairs)[i]()
        if made and self.confirm("Build: done", [f"{made} is ready."], yes="Run it now") == 0:
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

    def ask_name(self, title: str, default: str = "") -> str | None:
        return self.ask_checked(title, "New scene name", default, tp.check_name,
                                ["Letters, digits, _ and -. The scene is written to scenes/<name>."])

    def run_build(self, title: str, cmd: tp.Command, name: str, info: list[str]) -> str | None:
        out = tp.scene_out(name)
        full = os.path.join(self.app, out)
        info = [*info, f"Output    {full}",
                *(["", "This scene already exists and will be OVERWRITTEN."] if os.path.exists(full) else []),
                "", f"$ {cmd.shown()}"]
        if self.confirm(title, info, yes="Build") != 0:
            return None
        return out if self.execute(cmd) == 0 else None

    def build_map(self) -> str | None:
        title = "Build from a map file"
        maps = tp.list_maps(self.app)
        i = self.menu(title + ": map", [*(f"{p}  ({size_text(n)})" for p, n in maps), "Other file..."],
                      info=None if maps else ["No map files under Maps/: give a path."])
        if i is None:
            return None
        if i < len(maps):
            path = tp.map_path(maps[i][0], self.app, self.backend)
        else:
            path = self.ask_checked(title, "Map file", "", lambda t: tp.map_path(t, self.app, self.backend))
            if path is None:
                return None
        keys = [k for k, _ in tp.RECIPES]
        r = self.menu(title + ": recipe", [label for _, label in tp.RECIPES],
                      info=["1:1 keeps the scan as it is (the default scenes are built this way).",
                            "Flattened replaces the floor with clean slabs; the stairs recipe keeps steps sharp."])
        if r is None:
            return None
        name = self.ask_name(title, os.path.splitext(os.path.basename(path))[0].replace(".", "_"))
        if name is None:
            return None
        cmd = tp.build_map_command(path, name, keys[r], self.backend)
        return self.run_build(title, cmd, name, [f"Map       {path}", f"Recipe    {tp.RECIPES[r][1]}",
                                                 "A large point cloud takes from seconds to a few minutes."])

    def build_osm(self) -> str | None:
        title = "Build from OpenStreetMap"
        loc = self.ask_checked(title, "Centre", "", lambda t: (tp.parse_location(t), t)[1],
                               ["'LAT LON' (e.g. 24.95919 121.41896) or a streets.gl URL of the place."])
        if loc is None:
            return None
        radius = self.ask_checked(title, "Half side of the square (m)", "1000", float)
        if radius is None:
            return None
        scans = [s for s in tp.scan_scenes(self.app) if not s.geo]
        b = self.menu(title + ": around a scan?", ["Stand-alone (OSM only)", *(f"Around {s.name}" for s in scans)],
                      info=["Around a scan: the scanned scene stays where it is and OSM fills the area around it."])
        if b is None:
            return None
        base = scans[b - 1].path if b else None
        name = self.ask_name(title)
        if name is None:
            return None
        cmd = tp.build_osm_command(loc, radius, name, base, self.backend)
        return self.run_build(title, cmd, name, [f"Centre    {loc}", f"Radius    {radius:g} m", f"Base      {base or '-'}",
                                                 f"The first build downloads OSM data (needs a network) into "
                                                 f"Maps/{name}/; later builds reuse it."])

    def build_slope(self) -> str | None:
        title = "Build a synthetic slope"
        angle = self.ask_checked(title, "Incline angle (deg)", "10", float)
        if angle is None:
            return None
        name = self.ask_name(title, f"slope_{angle:g}".replace(".", "_"))
        if name is None:
            return None
        return self.run_build(title, tp.build_slope_command(angle, name, self.backend), name, [f"Angle     {angle:g} deg"])

    def build_stairs(self) -> str | None:
        title = "Build synthetic stairs"
        scenes, specs = tp.scan_scenes(self.app), tp.list_stair_specs(self.app)
        if not scenes or not specs:
            raise tp.PlanError("needs a built scene and a config/stairs_*.yaml spec written for it")
        s = self.menu(title + ": base scene", [x.name + x.tags() for x in scenes],
                      info=["The scanned staircases listed in the spec are replaced with clean steps."])
        if s is None:
            return None
        p = self.menu(title + ": spec", specs, info=["The spec must have been written for this scene's map."])
        if p is None:
            return None
        name = self.ask_name(title, scenes[s].name + "_s16")
        if name is None:
            return None
        cmd = tp.build_stairs_command(scenes[s].path, specs[p], name, self.backend)
        return self.run_build(title, cmd, name, [f"Base      {scenes[s].path}", f"Spec      {specs[p]}"])


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
    os.chdir(app)
    try:
        curses.wrapper(lambda scr: Tui(scr, app, backend, a.profile, tp.profile_dir()).main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
