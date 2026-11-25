"""DDS config resolution for the standalone launcher: which CycloneDDS config wins, and the Domain Id check."""
import pytest

from quadsim.dds_config import DdsConfigError, resolve

XML = '<CycloneDDS><Domain Id="{}"><General/></Domain></CycloneDDS>'


@pytest.fixture
def conf(tmp_path):
    (tmp_path / "cyclonedds.xml").write_text(XML.format(1))
    (tmp_path / "wifi.xml").write_text(XML.format(1))
    return tmp_path


def test_default_is_the_bundled_cyclonedds_xml(conf):
    choice, rest = resolve(["--scene", "scenes/dp"], {}, conf)
    assert choice.uri == f"file://{conf}/cyclonedds.xml"
    assert choice.source == "default"
    assert choice.domain_id == 1
    assert choice.warnings == ()
    assert rest == ["--scene", "scenes/dp"]


def test_cyclonedds_config_picks_a_bundled_file_by_name(conf):
    choice, _ = resolve([], {"CYCLONEDDS_CONFIG": "wifi.xml"}, conf)
    assert (choice.uri, choice.source) == (f"file://{conf}/wifi.xml", "CYCLONEDDS_CONFIG")


def test_unknown_cyclonedds_config_is_an_error(conf):
    with pytest.raises(DdsConfigError, match="^CYCLONEDDS_CONFIG: no such file .*nope.xml"):
        resolve([], {"CYCLONEDDS_CONFIG": "nope.xml"}, conf)


def test_cyclonedds_uri_already_set_passes_through_over_the_config_name(conf):
    env = {"CYCLONEDDS_URI": "<CycloneDDS/>", "CYCLONEDDS_CONFIG": "wifi.xml"}
    choice, _ = resolve([], env, conf)
    assert (choice.uri, choice.source) == ("<CycloneDDS/>", "CYCLONEDDS_URI")


@pytest.mark.parametrize("argv", [["--dds-config", "{p}", "lidar.range_noise=0"], ["--dds-config={p}", "lidar.range_noise=0"]])
def test_dds_config_flag_wins_over_everything_and_leaves_argv(conf, tmp_path, argv):
    mine = tmp_path / "mine.xml"
    mine.write_text(XML.format(1))
    env = {"CYCLONEDDS_URI": "<CycloneDDS/>", "CYCLONEDDS_CONFIG": "wifi.xml"}
    choice, rest = resolve([a.format(p=mine) for a in argv], env, conf)
    assert (choice.uri, choice.source) == (f"file://{mine}", "--dds-config")
    assert rest == ["lidar.range_noise=0"]


def test_relative_dds_config_path_is_made_absolute(conf, tmp_path, monkeypatch):
    (tmp_path / "rel.xml").write_text(XML.format(1))
    monkeypatch.chdir(tmp_path)
    choice, _ = resolve(["--dds-config", "rel.xml"], {}, conf)
    assert choice.uri == f"file://{tmp_path}/rel.xml"


def test_missing_dds_config_file_is_an_error(conf):
    with pytest.raises(DdsConfigError, match="missing.xml"):
        resolve(["--dds-config", "/nowhere/missing.xml"], {}, conf)


def test_dds_config_without_a_value_is_an_error(conf):
    with pytest.raises(DdsConfigError, match="--dds-config"):
        resolve(["--dds-config"], {}, conf)


def test_dds_config_none_leaves_cyclone_on_its_defaults(conf):
    choice, rest = resolve(["--dds-config", "none"], {"CYCLONEDDS_URI": "<CycloneDDS/>"}, conf)
    assert (choice.uri, choice.source, choice.warnings) == (None, "--dds-config none", ())
    assert rest == []


def test_ros_domain_id_comes_from_the_environment(conf):
    (conf / "d7.xml").write_text(XML.format(7))
    choice, _ = resolve([], {"ROS_DOMAIN_ID": "7", "CYCLONEDDS_CONFIG": "d7.xml"}, conf)
    assert (choice.domain_id, choice.warnings) == (7, ())


def test_bad_ros_domain_id_is_an_error(conf):
    with pytest.raises(DdsConfigError, match="ROS_DOMAIN_ID"):
        resolve([], {"ROS_DOMAIN_ID": "one"}, conf)


def test_domain_mismatch_is_warned_because_cyclone_would_silently_ignore_the_file(conf):
    choice, _ = resolve([], {"ROS_DOMAIN_ID": "2"}, conf)
    assert len(choice.warnings) == 1
    assert "Domain Id=1" in choice.warnings[0] and "ROS_DOMAIN_ID=2" in choice.warnings[0]


def test_domain_mismatch_is_also_found_in_inline_xml(conf):
    choice, _ = resolve([], {"CYCLONEDDS_URI": XML.format(3)}, conf)
    assert len(choice.warnings) == 1 and "Domain Id=3" in choice.warnings[0]


@pytest.mark.parametrize("xml", [XML.format("any"), "<CycloneDDS><Domain><General/></Domain></CycloneDDS>"])
def test_domain_any_or_unset_matches_every_domain(conf, xml):
    (conf / "any.xml").write_text(xml)
    choice, _ = resolve([], {"ROS_DOMAIN_ID": "5", "CYCLONEDDS_CONFIG": "any.xml"}, conf)
    assert choice.warnings == ()


def test_the_repo_config_is_domain_1():
    import os
    repo_conf = os.path.join(os.path.dirname(__file__), "..", "conf")
    choice, _ = resolve([], {}, repo_conf)
    assert (choice.source, choice.warnings) == ("default", ())


def test_a_uri_list_is_passed_through_unchecked(conf):
    uri = f"file://{conf}/cyclonedds.xml,file://{conf}/wifi.xml"
    choice, _ = resolve([], {"CYCLONEDDS_URI": uri, "ROS_DOMAIN_ID": "9"}, conf)
    assert (choice.uri, choice.warnings) == (uri, ())


def test_a_missing_bundled_default_names_itself(tmp_path):
    with pytest.raises(DdsConfigError, match="^the bundled default: no such file"):
        resolve([], {}, tmp_path)


def test_inline_xml_may_start_with_whitespace(conf):
    choice, _ = resolve([], {"CYCLONEDDS_URI": "\n  " + XML.format(3)}, conf)
    assert len(choice.warnings) == 1


def test_every_mismatching_domain_id_is_listed(conf):
    two = '<CycloneDDS><Domain Id="2"/><Domain Id="3"/></CycloneDDS>'
    choice, _ = resolve([], {"CYCLONEDDS_URI": two}, conf)
    assert "Domain Id=2,3 but ROS_DOMAIN_ID=1" in choice.warnings[0]
