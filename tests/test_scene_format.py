"""quadsim checks meta.json format_version against pcd2sim.SCENE_FORMAT (one definition for both packages)."""
import pytest

from pcd2sim import SCENE_FORMAT
from quadsim.sim import _check_scene_format


def test_current_format_loads_silently(recwarn):
    _check_scene_format("scenes/x", {"format_version": SCENE_FORMAT})
    assert not [w for w in recwarn if "format" in str(w.message)]


def test_format_2_warns_but_loads():
    with pytest.warns(UserWarning, match="format 2"):
        _check_scene_format("scenes/x", {"format_version": 2})


@pytest.mark.parametrize("version", [1, SCENE_FORMAT + 1, None])
def test_other_formats_are_refused(version):
    with pytest.raises(RuntimeError, match="Rebuild"):
        _check_scene_format("scenes/x", {"format_version": version})
