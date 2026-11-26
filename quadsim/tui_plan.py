"""What the TUI (quadsim.tui) runs: choices + saved settings -> command lines. No curses, no MuJoCo, no ROS here.

The TUI adds no behaviour of its own: every command it starts is one a user could type (scripts/run_sim.sh,
python3 -m pcd2sim.build_scene, ...), and it shows that command before running it.

Backends:
  docker   scripts/run_sim.sh and docker/run.sh "<command>" (a checkout with the quadsim image)
  native   python -m quadsim.launch / python -m pcd2sim... in this interpreter's environment (pixi env, .run bundle)

Settings live in profiles: ${XDG_CONFIG_HOME:-~/.config}/quadsim/tui/<name>.json; `last` is the one the TUI keeps
up to date. A profile holds the DDS choice, the ROS topic / frame names that differ from config/sim.yaml, and the
last Run selection.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

BACKENDS = ("docker", "native")
ROBOTS = (("g23", "G23 legged robot"), ("ideal", "Ideal base (no legs, follows cmd_vel)"))
BEHAVIORS = (("rl_fcn", "rl_fcn: flat-terrain walk"), ("rl_him", "rl_him: stairs / rough terrain"),
             ("gait_mpc", "gait_mpc"))
MAP_COLORS = ("osm", "height", "plain")
SPAWN_CONFIG, SPAWN_SCENE = "config", "scene"
# key, label, on by default
FEATURES = (
    ("rtk", "GNSS: RTK corrections", False),
    ("sky_always", "GNSS: sky always open (indoor scenes)", False),
    ("web", "Web viewer (browser, port 8755)", True),
    ("viewer", "MuJoCo window (needs an X display)", False),
    ("gt_tf", "Publish ground-truth TF", False),
    ("pointcloud2", "Publish LiDAR as PointCloud2", False),
    ("stairs_pool", "Stair collision preset (pool 512, margin 0.4)", False),
)
DEFAULT_FEATURES = tuple(k for k, _, on in FEATURES if on)
RECIPES = (
    ("exact", "1:1, no flatten (walls and stairs as scanned)"),
    ("flat", "Flattened floor (build_scene defaults)"),
    ("stairs", "Flattened floor, stairs recipe"),
)
_RECIPE_ARGS = {
    ("exact", "pcd"): ["--no-flatten", "--close-rounds", "1"],
    ("exact", "octree"): ["--no-flatten", "--min-neighbors", "0"],
    ("flat", "pcd"): [],
    ("flat", "octree"): [],
    ("stairs", "pcd"): ["--ground-min-cells", "8", "--ground-fill-guard", "0.3", "--ground-surface", "median",
                        "--ground-peel", "3", "--ground-small-flat", "100"],
}
_RECIPE_ARGS[("stairs", "octree")] = _RECIPE_ARGS[("stairs", "pcd")]
MAP_EXTS = {".pcd": "pcd", ".bt": "octree", ".ot": "octree"}
SCENE_FILES = ("meta.json", "boxes.npy", "voxels.npz", "map_ds.pcd")
LAST = "last"

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_ROS_NAME = re.compile(r"[A-Za-z0-9_/]+")
_PATH = re.compile(r"[A-Za-z0-9_./+-]+")          # no spaces or quotes: run_sim.sh / docker/run.sh join args into a string
_STREETS_GL = re.compile(r"#(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


class PlanError(ValueError):
    """A choice that cannot be turned into a command; the message is shown to the user."""


@dataclass(frozen=True)
class Command:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)   # variables added to the caller's environment

    def shown(self) -> str:
        """The line a user would type to do the same thing."""
        return " ".join([*(f"{k}={shlex.quote(v)}" for k, v in self.env.items()), *(shlex.quote(a) for a in self.argv)])


# ---- scenes ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scene:
    name: str
    path: str               # as passed to --scene, relative to the app directory
    format_version: int
    geo: bool               # has a GNSS datum of its own
    overlay: bool           # OSM laid around a scanned scene: lives in that scan's frame
    stairs: bool            # synthetic stairs (scripts/make_stairs.py)

    def tags(self) -> str:
        return "".join(f" [{t}]" for t, on in (("geo", self.geo), ("stairs", self.stairs),
                                               ("v%d: ideal base only" % self.format_version,
                                                self.format_version < 3)) if on)


def scan_scenes(app_dir: str) -> list[Scene]:
    """Runnable scenes under <app_dir>/scenes, by name. Half-built directories are left out."""
    root = os.path.join(app_dir, "scenes")
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        if not all(os.path.isfile(os.path.join(d, f)) for f in SCENE_FILES):
            continue
        try:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            version = int(meta.get("format_version", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        out.append(Scene(name, f"scenes/{name}", version, bool(meta.get("geo")), bool(meta.get("base")),
                         bool(meta.get("synthetic_stairs"))))
    return out


def runs_robot(scene: Scene, robot: str) -> bool:
    """The legged robot needs the contact layer of scene format 3 (quadsim/sim.py)."""
    return robot == "ideal" or scene.format_version >= 3


def default_spawn(scene: Scene) -> str:
    """A stand-alone OSM scene has its own frame: the config spawn (dp's) would be off the map."""
    return SPAWN_SCENE if scene.geo and not scene.overlay else SPAWN_CONFIG


def default_features(scene: Scene) -> list[str]:
    return [*DEFAULT_FEATURES, *(["stairs_pool"] if scene.stairs else [])]


# ---- settings / profiles --------------------------------------------------------------------------------------------

def default_settings(env: dict[str, str] | None = None) -> dict:
    raw = (os.environ if env is None else env).get("ROS_DOMAIN_ID", "").strip()
    return {
        "dds": {"domain_id": int(raw) if raw.isdigit() else 1, "config": "cyclonedds.xml"},
        "topics": {},           # only the names that differ from config/sim.yaml
        "frames": {},
        "run": {"robot": "g23", "behavior": "rl_fcn", "scene": None, "spawn": SPAWN_CONFIG,
                "features": list(DEFAULT_FEATURES), "map_color": "osm"},
    }


def profile_dir(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or os.path.join(env.get("HOME") or os.path.expanduser("~"), ".config")
    return os.path.join(base, "quadsim", "tui")


def profile_path(name: str, directory: str) -> str:
    check_name(name, "profile name")
    return os.path.join(directory, name + ".json")


def list_profiles(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f[:-5] for f in os.listdir(directory) if f.endswith(".json") and _NAME.fullmatch(f[:-5]))


def save_profile(name: str, settings: dict, directory: str) -> str:
    path = profile_path(name, directory)
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def delete_profile(name: str, directory: str) -> None:
    path = profile_path(name, directory)
    if os.path.isfile(path):
        os.remove(path)


def load_profile(name: str, directory: str, env: dict[str, str] | None = None) -> tuple[dict, list[str]]:
    """Settings from the profile over the defaults, and what was ignored. A missing profile is just the defaults."""
    settings = default_settings(env)
    path = profile_path(name, directory)
    if not os.path.isfile(path):
        return settings, []
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        return settings, [f"{path}: unreadable ({e}); using the defaults"]
    if not isinstance(raw, dict):
        return settings, [f"{path}: not a settings object; using the defaults"]
    return settings, _merge(settings, raw)


def _merge(settings: dict, raw: dict) -> list[str]:
    notes = []
    for section, value in raw.items():
        if section not in settings or not isinstance(value, dict):
            notes.append(f"ignored '{section}'")
            continue
        for key, v in value.items():
            problem = _check_setting(section, key, v)
            if problem:
                notes.append(f"ignored {section}.{key}: {problem}")
            else:
                settings[section][key] = v
    return notes


def _check_setting(section: str, key: str, v) -> str | None:
    if section in ("topics", "frames"):
        ok = isinstance(v, str) and _ROS_NAME.fullmatch(v) and re.fullmatch(r"[A-Za-z0-9_.]+", key)
        return None if ok else "not a ROS name"
    checks = {
        ("dds", "domain_id"): lambda: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 232,
        ("dds", "config"): lambda: isinstance(v, str) and bool(_PATH.fullmatch(v)),
        ("run", "robot"): lambda: v in dict(ROBOTS),
        ("run", "behavior"): lambda: v in dict(BEHAVIORS),
        ("run", "scene"): lambda: v is None or (isinstance(v, str) and bool(_PATH.fullmatch(v))),
        ("run", "spawn"): lambda: isinstance(v, str) and bool(_NAME.fullmatch(v)),
        ("run", "features"): lambda: isinstance(v, list) and all(f in {k for k, _, _ in FEATURES} for f in v),
        ("run", "map_color"): lambda: v in MAP_COLORS,
    }
    check = checks.get((section, key))
    if check is None:
        return "unknown key"
    return None if check() else f"bad value {v!r}"


# ---- ROS names ------------------------------------------------------------------------------------------------------

PATH_TOPIC = "web.path_topic"       # the planner path the web viewer draws: a topic outside ros.topics


def ros_defaults(app_dir: str) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """(topics, frames, named spawns) of config/sim.yaml; all empty without PyYAML or the file. Topics are keyed
    by their ros.topics name, plus the dotted config key of the one topic that lives elsewhere (PATH_TOPIC)."""
    try:
        import yaml
        with open(os.path.join(app_dir, "config", "sim.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
    except (ImportError, OSError, ValueError):
        return {}, {}, []
    ros = cfg.get("ros") or {}
    names = lambda d: {str(k): str(v) for k, v in (d or {}).items() if isinstance(v, str)}  # noqa: E731
    topics = names(ros.get("topics"))
    path = (cfg.get("web") or {}).get("path_topic")
    if topics and isinstance(path, str) and path:
        topics[PATH_TOPIC] = path
    return topics, names(ros.get("frames")), sorted(map(str, cfg.get("spawns") or {}))


def effective(defaults: dict[str, str], changed: dict[str, str]) -> dict[str, str]:
    return {k: changed.get(k, v) for k, v in defaults.items()}


def set_name(changed: dict[str, str], defaults: dict[str, str], key: str, value: str) -> None:
    """Record value for key; the default (or nothing) removes the override."""
    value = value.strip()
    if not value or value == defaults.get(key):
        changed.pop(key, None)
        return
    if not _ROS_NAME.fullmatch(value):
        raise PlanError(f"'{value}': use letters, digits, _ and / only")
    changed[key] = value


def namespace(topics: dict[str, str]) -> str:
    """The robot namespace: the first segment of the cmd_vel topic ('' when it has none)."""
    parts = topics.get("cmd_vel", "").strip("/").split("/")
    return parts[0] if len(parts) > 1 else ""


def rename_namespace(settings: dict, topic_defaults: dict[str, str], frame_defaults: dict[str, str], new: str) -> None:
    """Move every topic and frame of the current robot namespace to /<new>/..., in settings."""
    new = new.strip().strip("/")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", new):
        raise PlanError(f"namespace '{new}': start with a letter, then letters, digits and _ only")
    topics, frames = effective(topic_defaults, settings["topics"]), effective(frame_defaults, settings["frames"])
    old = namespace(topics)
    if not old:
        raise PlanError("the cmd_vel topic has no namespace to rename")
    for k, v in topics.items():
        if v.startswith(f"/{old}/"):
            set_name(settings["topics"], topic_defaults, k, f"/{new}/" + v[len(old) + 2:])
    for k, v in frames.items():
        if v.startswith(f"{old}/"):
            set_name(settings["frames"], frame_defaults, k, f"{new}/" + v[len(old) + 1:])


def name_overrides(settings: dict) -> list[str]:
    return [*(f"{k if '.' in k else 'ros.topics.' + k}={v}" for k, v in sorted(settings["topics"].items())),
            *(f"ros.frames.{k}={v}" for k, v in sorted(settings["frames"].items()))]


# ---- backend / DDS --------------------------------------------------------------------------------------------------

def detect_backend(env: dict[str, str] | None = None, run=subprocess.run, find_spec=importlib.util.find_spec) -> str:
    """docker when the quadsim image is there, else native when MuJoCo is importable here."""
    env = os.environ if env is None else env
    try:
        ok = run(["docker", "image", "inspect", env.get("QUADSIM_IMAGE", "quadsim:latest")],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        ok = False
    if ok:
        return "docker"
    if find_spec("mujoco") is not None:
        return "native"
    raise PlanError("no way to run the sim here: build the Docker image (docker build -t quadsim:latest docker/), "
                    "or start the TUI from the pixi env / the .run bundle (pixi run sim tui)")


def dds_configs(app_dir: str) -> list[str]:
    conf = os.path.join(app_dir, "conf")
    return sorted(f for f in os.listdir(conf) if f.endswith(".xml")) if os.path.isdir(conf) else []


def dds_warnings(settings: dict, app_dir: str) -> tuple[str, ...]:
    """What the launcher would warn about for the saved DDS choice (a config written for another domain)."""
    from quadsim.dds_config import _domain_warnings
    dds = settings["dds"]
    if dds["config"] == "none":
        return ()
    path = os.path.join(app_dir, "conf", dds["config"]) if _in_conf(dds["config"]) else dds["config"]
    if not os.path.isfile(path):
        return (f"DDS config {path} does not exist",)
    return _domain_warnings("file://" + os.path.abspath(path), dds["domain_id"])


def _in_conf(config: str) -> bool:
    return "/" not in config and config != "none"


def _dds(settings: dict, backend: str) -> tuple[dict[str, str], list[str]]:
    """(environment, launcher flags) for the saved DDS choice."""
    dds = settings["dds"]
    env, config = {"ROS_DOMAIN_ID": str(dds["domain_id"])}, dds["config"]
    if backend == "docker":
        if not _in_conf(config):
            raise PlanError(f"DDS config '{config}': the Docker backend only takes a file under conf/ "
                            "(Settings > DDS)")
        if config != "cyclonedds.xml":
            env["CYCLONEDDS_CONFIG"] = config
        return env, []
    return env, ["--dds-config", os.path.join("conf", config) if _in_conf(config) else config]


def _wrap(backend: str, module_or_script: list[str], args: list[str], env: dict[str, str]) -> Command:
    if backend == "docker":
        return Command(["docker/run.sh", " ".join(["python3", *module_or_script, *args])], env)
    return Command([sys.executable, *module_or_script, *args], env)


# ---- run ------------------------------------------------------------------------------------------------------------

def sim_args(run: dict) -> list[str]:
    """sim_node arguments for a Run selection: --flags first, then key=value overrides."""
    if not run.get("scene"):
        raise PlanError("pick a scene first")
    check_path(run["scene"], "scene")
    feats = set(run.get("features", ()))
    flags = ["--scene", run["scene"]]
    if "viewer" in feats:
        flags.append("--viewer")
    if "web" not in feats:
        flags.append("--no-web")
    over = []
    if run["robot"] == "ideal":
        over.append("robot.type=ideal")
    elif run.get("behavior", "rl_fcn") != "rl_fcn":
        over.append(f"robot.behavior={run['behavior']}")
    spawn = run.get("spawn", SPAWN_CONFIG)
    if spawn == SPAWN_SCENE:
        over.append("spawn=null")
    elif spawn != SPAWN_CONFIG:
        over.append(f"spawn={check_name(spawn, 'spawn')}")
    if "rtk" in feats:
        over.append("gnss.rtk=true")
    if "sky_always" in feats:
        over.append("gnss.sky.mode=always")
    if "gt_tf" in feats:
        over.append("ros.publish_gt_tf=true")
    if "pointcloud2" in feats:
        over.append("ros.publish_pointcloud2=true")
    if "stairs_pool" in feats:
        over += ["robot.collision_pool=512", "robot.collision_margin=0.4"]
    if run.get("map_color", "osm") != "osm":
        over.append(f"viewer.map_color={run['map_color']}")
    return flags + over


def sim_command(settings: dict, backend: str) -> Command:
    args = sim_args(settings["run"]) + name_overrides(settings)
    env, dds_flags = _dds(settings, backend)
    if backend == "docker":
        if "viewer" in settings["run"].get("features", ()):
            env["QUADSIM_GUI"] = "1"
        return Command(["scripts/run_sim.sh", *args], env)
    return Command([sys.executable, "-m", "quadsim.launch", *dds_flags, *args], env)


# ---- build ----------------------------------------------------------------------------------------------------------

def check_name(name: str, what: str = "scene name") -> str:
    if not _NAME.fullmatch(name or ""):
        raise PlanError(f"{what} '{name}': use letters, digits, _ and - only")
    return name


def check_path(path: str, what: str) -> str:
    if not _PATH.fullmatch(path or ""):
        raise PlanError(f"{what} '{path}': no spaces or quotes in the path")
    return path


def scene_out(name: str) -> str:
    return f"scenes/{check_name(name)}"


def list_maps(app_dir: str) -> list[tuple[str, int]]:
    """(path relative to app_dir, bytes) of the map files in Maps/ and Maps/archive/."""
    out = []
    for sub in ("Maps", os.path.join("Maps", "archive")):
        d = os.path.join(app_dir, sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, f)
            if os.path.splitext(f)[1].lower() in MAP_EXTS and os.path.isfile(p):
                out.append((f"{sub}/{f}", os.path.getsize(p)))
    return out


def list_stair_specs(app_dir: str) -> list[str]:
    d = os.path.join(app_dir, "config")
    return sorted(f"config/{f}" for f in os.listdir(d)
                  if f.startswith("stairs_") and f.endswith(".yaml")) if os.path.isdir(d) else []


def map_path(path: str, app_dir: str, backend: str) -> str:
    """The map path as the build command takes it. The container only sees the checkout (mounted at /ws)."""
    path = os.path.expanduser(path.strip())
    if os.path.splitext(path)[1].lower() not in MAP_EXTS:
        raise PlanError(f"'{path}': expected a .pcd, .bt or .ot file")
    full = os.path.normpath(path if os.path.isabs(path) else os.path.join(app_dir, path))
    if not os.path.isfile(full):
        raise PlanError(f"no such file: {full}")
    rel = os.path.relpath(full, os.path.abspath(app_dir))
    if not rel.startswith(".."):
        return check_path(rel, "map")
    if backend == "docker":
        raise PlanError(f"{full} is outside the checkout: the container cannot see it; copy it into Maps/")
    return check_path(full, "map")


def build_map_command(path: str, name: str, recipe: str, backend: str, env: dict[str, str] | None = None) -> Command:
    kind = MAP_EXTS[os.path.splitext(path)[1].lower()]
    args = ["--map", check_path(path, "map"), "--out", scene_out(name), *_RECIPE_ARGS[(recipe, kind)]]
    return _wrap(backend, ["-m", "pcd2sim.build_scene"], args, env or {})


def parse_location(text: str) -> tuple[float, float]:
    """(lat, lon) from 'LAT LON', 'LAT,LON' or a streets.gl URL (…#LAT,LON,…)."""
    text = text.strip()
    m = _STREETS_GL.search(text) if "#" in text else None
    parts = list(m.groups()) if m else text.replace(",", " ").split()
    try:
        lat, lon = (float(p) for p in parts)
    except ValueError:
        raise PlanError("location: give 'LAT LON' or a streets.gl URL") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise PlanError(f"location {lat}, {lon}: out of range")
    return lat, lon


def build_osm_command(location: str, radius: float, name: str, base: str | None, backend: str) -> Command:
    lat, lon = parse_location(location)
    if not 10 <= radius <= 5000:
        raise PlanError(f"radius {radius:g} m: expected 10 to 5000")
    args = ["--center", f"{lat:.6f}", f"{lon:.6f}", "--radius", f"{radius:g}", "--out", scene_out(name)]
    if base:
        args += ["--base", check_path(base, "base scene"), "--register"]
    return _wrap(backend, ["-m", "pcd2sim.osm_scene"], args, {})


def build_slope_command(angle: float, name: str, backend: str) -> Command:
    if not 0 < angle <= 45:
        raise PlanError(f"angle {angle:g} deg: expected 0 to 45")
    return _wrap(backend, ["scripts/make_slope.py"], ["--out", scene_out(name), "--angle", f"{angle:g}"], {})


def build_stairs_command(base: str, spec: str, name: str, backend: str) -> Command:
    if scene_out(name) == os.path.normpath(base):
        raise PlanError("the new scene needs a name different from the base scene")
    args = ["--scene", check_path(base, "base scene"), "--spec", check_path(spec, "spec"), "--out", scene_out(name)]
    return _wrap(backend, ["scripts/make_stairs.py"], args, {})
