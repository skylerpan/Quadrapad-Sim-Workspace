"""quadsim.tui_plan: the commands the TUI runs for a set of choices, and the settings it saves."""
import json
import subprocess
import sys

import pytest

from quadsim import tui_plan as tp

TOPICS = {"cmd_vel": "/robot01/cmd_vel", "imu": "/robot01/livox/imu", "clock": "/clock", "gt_odom": "/sim/gt"}
FRAMES = {"map": "map", "odom": "robot01/odom", "base": "robot01/base_link"}


def settings(**run):
    s = tp.default_settings({})
    s["run"].update({"scene": "scenes/dp", **run})
    return s


def make_scene(root, name, version=3, **meta):
    d = root / "scenes" / name
    d.mkdir(parents=True)
    for f in tp.SCENE_FILES:
        (d / f).write_bytes(b"")
    (d / "meta.json").write_text(json.dumps({"format_version": version, **meta}))
    return d


# ---- scenes

def test_scan_lists_built_scenes_with_their_tags(tmp_path):
    make_scene(tmp_path, "dp")
    make_scene(tmp_path, "osm", geo={"lat": 1})
    make_scene(tmp_path, "osm_dp", geo={"lat": 1}, base={"scene": "scenes/dp"})
    make_scene(tmp_path, "s16", synthetic_stairs={"n": 2})
    make_scene(tmp_path, "old", version=2)
    (tmp_path / "scenes" / "half").mkdir()
    (tmp_path / "scenes" / "half" / "meta.json").write_text("{}")
    scenes = {s.name: s for s in tp.scan_scenes(str(tmp_path))}
    assert sorted(scenes) == ["dp", "old", "osm", "osm_dp", "s16"]
    assert scenes["dp"].path == "scenes/dp" and scenes["dp"].tags() == ""
    assert scenes["osm"].tags() == " [geo]" and scenes["s16"].tags() == " [stairs]"
    assert "ideal base only" in scenes["old"].tags()


def test_scan_without_a_scenes_directory_or_with_a_broken_meta(tmp_path):
    assert tp.scan_scenes(str(tmp_path)) == []
    make_scene(tmp_path, "bad").joinpath("meta.json").write_text("not json")
    assert tp.scan_scenes(str(tmp_path)) == []


def test_the_legged_robot_needs_scene_format_3(tmp_path):
    make_scene(tmp_path, "old", version=2)
    old, = tp.scan_scenes(str(tmp_path))
    assert not tp.runs_robot(old, "g23") and tp.runs_robot(old, "ideal")


def test_scene_defaults_follow_the_scene(tmp_path):
    make_scene(tmp_path, "a_osm", geo={"lat": 1})
    make_scene(tmp_path, "b_osm_dp", geo={"lat": 1}, base={"scene": "scenes/dp"})
    make_scene(tmp_path, "c_s16", synthetic_stairs={"n": 2})
    osm, osm_dp, s16 = tp.scan_scenes(str(tmp_path))
    assert tp.default_spawn(osm) == tp.SPAWN_SCENE          # its own frame: dp's config spawn would be off the map
    assert tp.default_spawn(osm_dp) == tp.SPAWN_CONFIG
    assert tp.default_features(s16) == ["web", "stairs_pool"] and tp.default_features(osm) == ["web"]


# ---- run

def test_defaults_give_the_plain_command():
    assert tp.sim_args(settings()["run"]) == ["--scene", "scenes/dp"]


def test_flags_come_before_overrides():
    run = settings(features=["viewer", "rtk"], spawn="gate")["run"]
    assert tp.sim_args(run) == ["--scene", "scenes/dp", "--viewer", "--no-web", "spawn=gate", "gnss.rtk=true"]


@pytest.mark.parametrize("run, expected", [
    ({"robot": "ideal", "behavior": "rl_him"}, ["robot.type=ideal"]),
    ({"behavior": "rl_him"}, ["robot.behavior=rl_him"]),
    ({"spawn": tp.SPAWN_SCENE}, ["spawn=null"]),
    ({"features": ["web", "sky_always"]}, ["gnss.sky.mode=always"]),
    ({"features": ["web", "gt_tf", "pointcloud2"]}, ["ros.publish_gt_tf=true", "ros.publish_pointcloud2=true"]),
    ({"features": ["web", "stairs_pool"]}, ["robot.collision_pool=512", "robot.collision_margin=0.4"]),
    ({"map_color": "height"}, ["viewer.map_color=height"]),
])
def test_each_choice_maps_to_its_override(run, expected):
    assert tp.sim_args(settings(**run)["run"])[2:] == expected


def test_no_scene_or_a_path_with_a_space_is_refused():
    with pytest.raises(tp.PlanError, match="pick a scene"):
        tp.sim_args(settings(scene=None)["run"])
    with pytest.raises(tp.PlanError, match="no spaces"):
        tp.sim_args(settings(scene="scenes/my scene")["run"])


def test_docker_sim_command_is_run_sim_with_the_dds_environment():
    s = settings(features=["web", "viewer"])
    s["dds"] = {"domain_id": 7, "config": "lan.xml"}
    cmd = tp.sim_command(s, "docker")
    assert cmd.argv == ["scripts/run_sim.sh", "--scene", "scenes/dp", "--viewer"]
    assert cmd.env == {"ROS_DOMAIN_ID": "7", "CYCLONEDDS_CONFIG": "lan.xml", "QUADSIM_GUI": "1"}
    assert cmd.shown() == "ROS_DOMAIN_ID=7 CYCLONEDDS_CONFIG=lan.xml QUADSIM_GUI=1 scripts/run_sim.sh --scene scenes/dp --viewer"


def test_docker_default_dds_file_sets_only_the_domain():
    assert tp.sim_command(settings(), "docker").env == {"ROS_DOMAIN_ID": "1"}


@pytest.mark.parametrize("config", ["none", "/etc/dds.xml"])
def test_docker_refuses_a_dds_config_outside_conf(config):
    s = settings()
    s["dds"]["config"] = config
    with pytest.raises(tp.PlanError, match="only takes a file under conf/"):
        tp.sim_command(s, "docker")


@pytest.mark.parametrize("config, flag", [("cyclonedds.xml", "conf/cyclonedds.xml"), ("none", "none"),
                                          ("/etc/dds.xml", "/etc/dds.xml")])
def test_native_sim_command_always_names_the_dds_config(config, flag):
    s = settings()
    s["dds"] = {"domain_id": 3, "config": config}
    cmd = tp.sim_command(s, "native")
    assert cmd.argv == [sys.executable, "-m", "quadsim.launch", "--dds-config", flag, "--scene", "scenes/dp"]
    assert cmd.env == {"ROS_DOMAIN_ID": "3"}        # the flag beats an inherited CYCLONEDDS_URI (quadsim.dds_config)


# ---- ROS names

def test_only_changed_names_become_overrides():
    s = settings()
    tp.set_name(s["topics"], TOPICS, "cmd_vel", "/cmd_vel")
    tp.set_name(s["topics"], TOPICS, "imu", "/robot01/livox/imu")       # the default: nothing to override
    tp.set_name(s["frames"], FRAMES, "base", "base_link")
    assert tp.name_overrides(s) == ["ros.topics.cmd_vel=/cmd_vel", "ros.frames.base=base_link"]
    assert tp.sim_command(s, "docker").argv[-2:] == ["ros.topics.cmd_vel=/cmd_vel", "ros.frames.base=base_link"]


def test_an_empty_or_default_value_drops_the_override_and_a_bad_one_is_refused():
    changed = {"cmd_vel": "/x"}
    tp.set_name(changed, TOPICS, "cmd_vel", "  ")
    assert changed == {}
    with pytest.raises(tp.PlanError, match="letters, digits"):
        tp.set_name(changed, TOPICS, "cmd_vel", "/a b")


def test_rename_namespace_moves_the_robot_topics_and_frames_only():
    s = settings()
    tp.rename_namespace(s, TOPICS, FRAMES, "/robot02")
    assert s["topics"] == {"cmd_vel": "/robot02/cmd_vel", "imu": "/robot02/livox/imu"}
    assert s["frames"] == {"odom": "robot02/odom", "base": "robot02/base_link"}
    assert tp.namespace(tp.effective(TOPICS, s["topics"])) == "robot02"
    tp.rename_namespace(s, TOPICS, FRAMES, "robot01")                   # back to the defaults: no overrides left
    assert s["topics"] == {} and s["frames"] == {}


def test_rename_namespace_refuses_a_bad_name_or_a_topic_without_one():
    with pytest.raises(tp.PlanError, match="namespace"):
        tp.rename_namespace(settings(), TOPICS, FRAMES, "robot 2")
    with pytest.raises(tp.PlanError, match="no namespace"):
        tp.rename_namespace(settings(), {"cmd_vel": "/cmd_vel"}, {}, "r2")


def test_ros_defaults_read_the_real_config():
    topics, frames, spawns = tp.ros_defaults(".")
    assert topics["cmd_vel"].endswith("/cmd_vel") and "map" in frames and "gate" in spawns
    assert tp.ros_defaults("/nowhere") == ({}, {}, [])


# ---- profiles

def test_a_profile_round_trips(tmp_path):
    s = settings(robot="ideal", features=["rtk"], spawn="gate")
    s["dds"] = {"domain_id": 42, "config": "none"}
    s["topics"]["cmd_vel"] = "/cmd_vel"
    path = tp.save_profile("field", s, str(tmp_path / "p"))
    assert path == str(tmp_path / "p" / "field.json")
    assert tp.load_profile("field", str(tmp_path / "p"), {}) == (s, [])
    assert tp.list_profiles(str(tmp_path / "p")) == ["field"]
    tp.delete_profile("field", str(tmp_path / "p"))
    assert tp.list_profiles(str(tmp_path / "p")) == []


def test_a_missing_profile_is_the_defaults_with_the_callers_domain(tmp_path):
    s, notes = tp.load_profile("last", str(tmp_path), {"ROS_DOMAIN_ID": "9"})
    assert notes == [] and s["dds"]["domain_id"] == 9 and s["run"]["features"] == ["web"]


def test_a_broken_profile_falls_back_to_the_defaults(tmp_path):
    (tmp_path / "last.json").write_text("{ not json")
    s, notes = tp.load_profile("last", str(tmp_path), {})
    assert s == tp.default_settings({}) and "unreadable" in notes[0]


def test_unknown_and_bad_keys_are_ignored_and_reported(tmp_path):
    (tmp_path / "last.json").write_text(json.dumps({
        "dds": {"domain_id": "seven", "config": "lan.xml"},
        "run": {"robot": "g99", "scene": "scenes/dp", "features": ["web", "lasers"], "extra": 1},
        "topics": {"cmd_vel": "/a b", "imu": "/imu"},
        "nonsense": {},
    }))
    s, notes = tp.load_profile("last", str(tmp_path), {})
    assert s["dds"] == {"domain_id": 1, "config": "lan.xml"}
    assert s["run"]["robot"] == "g23" and s["run"]["scene"] == "scenes/dp" and s["run"]["features"] == ["web"]
    assert s["topics"] == {"imu": "/imu"}
    assert len(notes) == 6


def test_profile_names_cannot_leave_the_directory(tmp_path):
    with pytest.raises(tp.PlanError, match="profile name"):
        tp.save_profile("../x", settings(), str(tmp_path))


def test_profile_dir_follows_xdg_then_home():
    assert tp.profile_dir({"XDG_CONFIG_HOME": "/x"}) == "/x/quadsim/tui"
    assert tp.profile_dir({"HOME": "/h"}) == "/h/.config/quadsim/tui"


# ---- backend

def _run(code):
    return lambda *a, **k: subprocess.CompletedProcess(a, code)


def test_backend_is_docker_when_the_image_exists():
    assert tp.detect_backend({}, _run(0), lambda m: None) == "docker"


def test_backend_is_native_without_the_image_when_mujoco_is_here():
    assert tp.detect_backend({}, _run(1), lambda m: object()) == "native"


def test_backend_without_docker_installed_and_without_mujoco_explains():
    def no_docker(*a, **k):
        raise FileNotFoundError("docker")
    with pytest.raises(tp.PlanError, match="pixi run sim tui"):
        tp.detect_backend({}, no_docker, lambda m: None)


# ---- build

@pytest.mark.parametrize("path, recipe, extra", [
    ("Maps/a.pcd", "exact", "--no-flatten --close-rounds 1"),
    ("Maps/a.bt", "exact", "--no-flatten --min-neighbors 0"),
    ("Maps/a.pcd", "flat", ""),
    ("Maps/a.PCD", "stairs", "--ground-min-cells 8 --ground-fill-guard 0.3 --ground-surface median --ground-peel 3 "
                             "--ground-small-flat 100"),
])
def test_map_recipes(path, recipe, extra):
    cmd = tp.build_map_command(path, "new", recipe, "docker")
    assert cmd.argv == ["docker/run.sh", f"python3 -m pcd2sim.build_scene --map {path} --out scenes/new {extra}".strip()]


def test_native_build_runs_this_interpreter():
    cmd = tp.build_map_command("Maps/a.bt", "new", "flat", "native")
    assert cmd.argv == [sys.executable, "-m", "pcd2sim.build_scene", "--map", "Maps/a.bt", "--out", "scenes/new"]


@pytest.mark.parametrize("name", ["", "a b", "../x", "a/b", "x;rm"])
def test_scene_names_are_plain(name):
    with pytest.raises(tp.PlanError, match="scene name"):
        tp.scene_out(name)


def test_map_path_inside_the_checkout_is_relative_and_outside_needs_native(tmp_path):
    app, out = tmp_path / "app", tmp_path / "elsewhere"
    (app / "Maps").mkdir(parents=True)
    out.mkdir()
    (app / "Maps" / "a.pcd").write_bytes(b"")
    (out / "b.bt").write_bytes(b"")
    assert tp.map_path(str(app / "Maps" / "a.pcd"), str(app), "docker") == "Maps/a.pcd"
    assert tp.map_path("Maps/a.pcd", str(app), "docker") == "Maps/a.pcd"
    assert tp.map_path(str(out / "b.bt"), str(app), "native") == str(out / "b.bt")
    with pytest.raises(tp.PlanError, match="outside the checkout"):
        tp.map_path(str(out / "b.bt"), str(app), "docker")
    with pytest.raises(tp.PlanError, match="no such file"):
        tp.map_path("Maps/missing.pcd", str(app), "docker")
    with pytest.raises(tp.PlanError, match="expected a .pcd"):
        tp.map_path("Maps/a.txt", str(app), "docker")


def test_list_maps_and_specs(tmp_path):
    (tmp_path / "Maps" / "archive").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "Maps" / "a.pcd").write_bytes(b"123")
    (tmp_path / "Maps" / "notes.txt").write_bytes(b"")
    (tmp_path / "Maps" / "archive" / "old.bt").write_bytes(b"1")
    (tmp_path / "config" / "stairs_x.yaml").write_text("")
    (tmp_path / "config" / "sim.yaml").write_text("")
    assert tp.list_maps(str(tmp_path)) == [("Maps/a.pcd", 3), ("Maps/archive/old.bt", 1)]
    assert tp.list_stair_specs(str(tmp_path)) == ["config/stairs_x.yaml"]
    assert tp.list_maps(str(tmp_path / "none")) == []


@pytest.mark.parametrize("text", ["24.95919 121.41896", "24.95919, 121.41896",
                                  "https://streets.gl/#24.95919,121.41896,45.00,0.00,586.00"])
def test_locations(text):
    assert tp.parse_location(text) == (24.95919, 121.41896)


@pytest.mark.parametrize("text", ["", "taipei", "1 2 3", "95 10", "https://streets.gl/"])
def test_bad_locations(text):
    with pytest.raises(tp.PlanError, match="location"):
        tp.parse_location(text)


def test_osm_commands():
    alone = tp.build_osm_command("-33.5 151.25", 500, "syd", None, "native")
    assert alone.argv[1:] == ["-m", "pcd2sim.osm_scene", "--center", "-33.500000", "151.250000", "--radius", "500",
                              "--out", "scenes/syd"]
    over = tp.build_osm_command("24.9 121.4", 1000, "osm_dp", "scenes/dp", "docker")
    assert over.argv[1].endswith("--out scenes/osm_dp --base scenes/dp --register")
    with pytest.raises(tp.PlanError, match="radius"):
        tp.build_osm_command("24.9 121.4", 5, "x", None, "docker")


def test_slope_and_stairs_commands():
    assert tp.build_slope_command(12.5, "s", "docker").argv == \
        ["docker/run.sh", "python3 scripts/make_slope.py --out scenes/s --angle 12.5"]
    with pytest.raises(tp.PlanError, match="angle"):
        tp.build_slope_command(60, "s", "docker")
    assert tp.build_stairs_command("scenes/dp_bt", "config/stairs_dp_bt.yaml", "dp_s16", "docker").argv == \
        ["docker/run.sh", "python3 scripts/make_stairs.py --scene scenes/dp_bt --spec config/stairs_dp_bt.yaml "
                          "--out scenes/dp_s16"]
    with pytest.raises(tp.PlanError, match="different from the base"):
        tp.build_stairs_command("scenes/dp_bt", "config/stairs_dp_bt.yaml", "dp_bt", "docker")
