import mujoco
import numpy as np

from quadsim.proxies import PARK_Z, BoxIndex, ProxyPool


def _pool(n=8):
    xml = "<mujoco><worldbody>" + "".join(
        f'<body name="proxy_{i}" mocap="true"><geom name="proxy_{i}" type="box" size="0.05 0.05 0.05"/></body>'
        for i in range(n)) + "</worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    xs = np.arange(0.0, 3.0, 0.2)                       # a row of 15 small boxes along x
    boxes = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs), np.full((len(xs), 3), 0.05)])
    return ProxyPool(m, d, BoxIndex(boxes, cell=0.5), n), m, d


def test_slots_stay_put_when_window_slides():
    pool, m, d = _pool()
    pool.update(np.array([0.6, 0.0, 0.0]), 0.3, -1.0, 1.0, 0.0)
    before = dict(zip(pool.assigned.tolist(), range(pool.n)))
    before.pop(-1, None)
    pool.update(np.array([0.8, 0.0, 0.0]), 0.3, -1.0, 1.0, 0.0)     # one box leaves, one enters
    after = dict(zip(pool.assigned.tolist(), range(pool.n)))
    common = set(before) & set(after)
    assert len(common) == 2 and len(after) - 1 == 3   # 3 boxes in view, 2 shared after the slide
    assert all(before[b] == after[b] for b in common)               # kept boxes kept their slot
    for slot, bid in enumerate(pool.assigned):
        mc = pool.mocap[slot]
        if bid >= 0:
            assert np.allclose(d.mocap_pos[mc], pool.index.boxes[bid, :3])
        else:
            assert d.mocap_pos[mc][2] <= PARK_Z


def test_window_contents_match_query():
    pool, _, _ = _pool()
    for x in (0.0, 0.5, 1.3, 2.9, 1.0):
        n = pool.update(np.array([x, 0.0, 0.0]), 0.3, -1.0, 1.0, 0.0)
        held = set(pool.assigned[pool.assigned >= 0].tolist())
        assert len(held) == n == len(pool.assigned[pool.assigned >= 0])
