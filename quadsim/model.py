"""MJCF for the ideal-base quadruped: free body welded to a mocap target + proxy pool."""
from __future__ import annotations

from quadsim.base import lidar_mount


def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def build_mjcf(cfg: dict, gravity, spawn_pos, n_proxies: int) -> str:
    r, li, ph = cfg["robot"], cfg["lidar"], cfg["physics"]
    half = [s / 2 for s in r["body_size"]]
    lidar_pos, lidar_quat = lidar_mount(li)
    # Shadows mean a second render pass over the whole map mesh (about 2x per frame); off by default.
    shadows = "true" if cfg.get("viewer", {}).get("shadows", False) else "false"
    proxies = "\n".join(
        f'    <body name="proxy_{i}" mocap="true" pos="0 0 {-1000 - 10 * i}">'
        f'<geom name="proxy_{i}" type="box" size="0.05 0.05 0.05" contype="2" conaffinity="1" '
        f'group="3" rgba="0.9 0.3 0.3 0.6"/></body>'
        for i in range(n_proxies))
    return f"""<mujoco model="quadsim_ideal_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{ph['timestep']}" gravity="{_f(gravity)}" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.45 0.45 0.45"/>
  </visual>
  <statistic center="{_f(spawn_pos)}" extent="10"/>
  <worldbody>
    <light pos="0 0 50" dir="0 0 -1" directional="true" castshadow="{shadows}"/>
    <body name="target" mocap="true" pos="{_f(spawn_pos)}">
      <geom type="box" size="{_f(half)}" contype="0" conaffinity="0" group="2" rgba="0.2 0.9 0.2 0.15"/>
    </body>
    <body name="base" pos="{_f(spawn_pos)}">
      <freejoint name="base"/>
      <geom name="base_geom" type="box" size="{_f(half)}" mass="{r['mass']}" contype="1" conaffinity="2"
            friction="0.6 0.005 0.0001" rgba="0.95 0.6 0.15 1"/>
      <site name="lidar" pos="{_f(lidar_pos)}" quat="{_f(lidar_quat)}" size="0.03" rgba="0.1 0.1 0.9 1"/>
    </body>
{proxies}
  </worldbody>
  <equality>
    <weld body1="target" body2="base" relpose="0 0 0 1 0 0 0" solref="{_f(r['weld_solref'])}"/>
  </equality>
  <sensor>
    <accelerometer name="imu_acc" site="lidar"/>
    <gyro name="imu_gyro" site="lidar"/>
  </sensor>
</mujoco>
"""


def box_corners(boxes):
    """[M,6] (cx, cy, cz, hx, hy, hz) -> [M,8,3] corners, corner index = 4*ix + 2*iy + iz."""
    import numpy as np

    b = np.asarray(boxes, dtype=np.float64)
    c, h = b[:, None, :3], b[:, None, 3:6]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    return c + h * signs[None]


def hexahedra_arrays(corners):
    """[K,8,3] corners (index = 4*ix + 2*iy + iz) -> (vertices [K*8,3] float64, outward triangles [K*12,3] int32)."""
    import numpy as np

    corners = np.asarray(corners, dtype=np.float64)
    tri = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    faces = (tri[None] + 8 * np.arange(len(corners), dtype=np.int32)[:, None, None]).reshape(-1, 3)
    return corners.reshape(-1, 3), faces


def nearest_boxes(boxes, xy, n: int):
    """The n boxes whose centres are nearest (in xy) to a point; all of them when there are no more than n."""
    import numpy as np

    if len(boxes) <= n:
        return boxes
    d2 = ((boxes[:, :2] - np.asarray(xy)[:2]) ** 2).sum(axis=1)
    return boxes[np.argpartition(d2, n)[:n]]


def _add_hexahedra(spec, name: str, corners, rgba):
    """One visual-only mesh geom from [K,8,3] corners (corner index = 4*ix + 2*iy + iz)."""
    import mujoco

    verts, faces = hexahedra_arrays(corners)
    mesh = spec.add_mesh(name=name)
    mesh.uservert = verts.ravel().tolist()
    mesh.userface = faces.ravel().tolist()
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                            contype=0, conaffinity=0, group=0, rgba=rgba)


def add_map_mesh(spec, boxes, slabs=None, max_boxes: int = 400_000):
    """Visual-only map for the MuJoCo viewer: scene boxes and floor slabs as meshes (no collision)."""
    _add_hexahedra(spec, "map_vis", box_corners(boxes[:max_boxes]), [0.55, 0.58, 0.62, 1.0])
    if slabs is not None and len(slabs):
        _add_hexahedra(spec, "ground_vis", slabs, [0.42, 0.47, 0.42, 1.0])
