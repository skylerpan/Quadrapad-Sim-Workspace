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


def test_resized_proxy_collides_with_a_multi_geom_body_away_from_its_centre():
    """A proxy grown from the compiled 5 cm box to a 10 m floor must hit a body with several geoms anywhere
    on it. MuJoCo's mid-phase tests the proxy body's compile-time BVH box against the other body's BVH, so the
    pool has to resize that node too (found by scripts/parity_g23.py: the G23 torso fell through the floor
    while the feet, single-collision-geom bodies, stood on it)."""
    xml = """<mujoco><worldbody>
      <body name="proxy_0" mocap="true" pos="0 0 -0.05"><geom name="proxy_0" type="box" size="0.05 0.05 0.05"
            contype="1" conaffinity="0"/></body>
      <body name="torso" pos="2 2 0.04"><freejoint/>
        <geom name="front" type="box" size="0.25 0.05 0.05" pos="0.5 0 0" contype="0" conaffinity="1"/>
        <geom name="back" type="box" size="0.25 0.05 0.05" pos="-0.5 0 0" contype="0" conaffinity="1"/>
        <geom name="shell" type="sphere" size="0.1" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    boxes = np.array([[0.0, 0.0, -0.05, 10.0, 10.0, 0.05]])
    pool = ProxyPool(m, d, BoxIndex(boxes, cell=2.0), 1)
    pool.update(np.array([2.0, 2.0, 0.0]), 1.0, -1.0, 1.0, 0.0)
    mujoco.mj_forward(m, d)
    hit = {frozenset((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))) for c in d.contact[:d.ncon]}
    assert {"front", "proxy_0"} in hit and {"back", "proxy_0"} in hit, hit
