"""quadsim.launch: the environment the standalone launcher hands to sim_node."""
from quadsim.dds_config import DdsChoice
from quadsim.launch import controller_repo, launch_env

CHOICE = DdsChoice(uri="file:///b/app/conf/cyclonedds.xml", source="default", domain_id=1)


def test_dds_choice_is_exported_and_rmw_is_cyclone():
    env = launch_env(CHOICE, {"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "PATH": "/bin"}, "/mc")
    assert env["CYCLONEDDS_URI"] == "file:///b/app/conf/cyclonedds.xml"
    assert env["ROS_DOMAIN_ID"] == "1"
    assert env["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert env["PATH"] == "/bin"


def test_dds_none_removes_an_inherited_uri():
    none = DdsChoice(uri=None, source="--dds-config none", domain_id=4)
    env = launch_env(none, {"CYCLONEDDS_URI": "file:///old.xml"}, "/mc")
    assert "CYCLONEDDS_URI" not in env
    assert env["ROS_DOMAIN_ID"] == "4"


def test_controller_repo_is_set_unless_the_caller_chose_one():
    assert launch_env(CHOICE, {}, "/b/controller/mc")["MOTORCONTROL_REPO"] == "/b/controller/mc"
    assert launch_env(CHOICE, {"MOTORCONTROL_REPO": "/mine"}, "/b/controller/mc")["MOTORCONTROL_REPO"] == "/mine"


def test_the_callers_env_is_not_modified():
    caller = {"CYCLONEDDS_URI": "x"}
    launch_env(CHOICE, caller, "/mc")
    assert caller == {"CYCLONEDDS_URI": "x"}


def test_controller_repo_prefers_the_bundled_copy(tmp_path):
    app = tmp_path / "app"
    (tmp_path / "controller" / "mc").mkdir(parents=True)
    app.mkdir()
    assert controller_repo(str(app)) == str(tmp_path / "controller" / "mc")


def test_controller_repo_in_a_checkout_is_the_sibling_repo_like_docker_run(tmp_path):
    app = tmp_path / "Quadrapad-Sim-Workspace"
    app.mkdir()
    assert controller_repo(str(app)) == str(tmp_path / "SyncAI-Device-MotorControl")


# main(): what the user of `quadsim` sees: the command it execs, that command's environment, and the stderr lines.
import os  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

from quadsim import launch  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "cyclonedds.xml").write_text('<CycloneDDS><Domain Id="1"/></CycloneDDS>')
    monkeypatch.chdir(tmp_path)
    for k in ("CYCLONEDDS_URI", "CYCLONEDDS_CONFIG", "ROS_DOMAIN_ID", "MOTORCONTROL_REPO"):
        monkeypatch.delenv(k, raising=False)
    calls = []
    monkeypatch.setattr(os, "execvpe", lambda file, args, env: calls.append((file, args, env)))
    return tmp_path, calls


def test_main_execs_sim_node_with_the_remaining_args(app, capsys):
    root, calls = app
    launch.main(["--scene", "scenes/dp", "lidar.range_noise=0"])
    (file, args, env), = calls
    assert file == sys.executable
    assert args == [sys.executable, "-m", "quadsim.sim_node", "--scene", "scenes/dp", "lidar.range_noise=0"]
    assert env["CYCLONEDDS_URI"] == f"file://{root}/conf/cyclonedds.xml"
    assert "[quadsim] DDS: default -> file://" in capsys.readouterr().err


def test_main_shell_runs_the_given_command(app):
    _, calls = app
    launch.main(["--dds-config", "none", "shell", "ros2", "topic", "list"])
    (file, args, env), = calls
    assert (file, args) == ("ros2", ["ros2", "topic", "list"])
    assert "CYCLONEDDS_URI" not in env


def test_main_shell_without_a_command_opens_the_users_shell(app, monkeypatch):
    _, calls = app
    monkeypatch.setenv("SHELL", "/bin/zsh")
    launch.main(["shell"])
    assert calls[0][:2] == ("/bin/zsh", ["/bin/zsh"])


def test_main_prints_the_domain_warning(app, monkeypatch, capsys):
    monkeypatch.setenv("ROS_DOMAIN_ID", "3")
    launch.main([])
    assert "[quadsim] WARNING: DDS config has Domain Id=1 but ROS_DOMAIN_ID=3" in capsys.readouterr().err


def test_main_exits_with_the_dds_error_and_execs_nothing(app):
    _, calls = app
    with pytest.raises(SystemExit, match=r"\[quadsim\] --dds-config: no such file"):
        launch.main(["--dds-config", "/nowhere.xml"])
    assert calls == []


def test_main_reads_sys_argv_and_points_the_controller_at_the_sibling_repo(app, monkeypatch):
    root, calls = app
    monkeypatch.setattr(sys, "argv", ["quadsim", "--duration", "5"])
    launch.main()
    (_, args, env), = calls
    assert args[-2:] == ["--duration", "5"]
    assert env["MOTORCONTROL_REPO"] == str(root.parent / "SyncAI-Device-MotorControl")


def test_main_shell_falls_back_to_bash(app, monkeypatch):
    _, calls = app
    monkeypatch.delenv("SHELL", raising=False)
    launch.main(["shell"])
    assert calls[0][:2] == ("bash", ["bash"])


def test_main_says_cyclone_defaults_for_none(app, capsys):
    launch.main(["--dds-config", "none"])
    assert "[quadsim] DDS: --dds-config none -> Cyclone defaults (ROS_DOMAIN_ID=1)" in capsys.readouterr().err


def test_main_tui_starts_the_menu_on_the_native_backend_without_touching_dds(app, monkeypatch, capsys):
    _, calls = app
    monkeypatch.setenv("ROS_DOMAIN_ID", "not-a-number")         # the TUI has its own DDS settings
    launch.main(["tui", "--profile", "field"])
    (file, args, env), = calls
    assert args == [sys.executable, "-m", "quadsim.tui", "--backend", "native", "--profile", "field"]
    assert "CYCLONEDDS_URI" not in env and capsys.readouterr().err == ""
