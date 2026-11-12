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
