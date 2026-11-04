"""Pure-numpy helpers of the browser viewer (no viser, no server)."""
import numpy as np

from quadsim.model import box_corners, hexahedra_arrays, nearest_boxes
from quadsim.web_viewer import box_colors, decimate, downsample_points, height_colors


def test_height_colors_span_the_palette():
    c = height_colors(np.linspace(0.0, 1.0, 101), lo=0.0, hi=1.0)
    assert c.shape == (101, 3) and c.dtype == np.uint8
    assert not np.array_equal(c[0], c[-1])
    assert np.array_equal(height_colors(np.array([-5.0, 7.0]), lo=0.0, hi=1.0), c[[0, -1]])   # clipped
    assert height_colors(np.zeros(0)).shape == (0, 3)


def test_box_colors_differ_between_neighbours():
    voxel = 0.2
    # two 0.2 m boxes side by side at the same height, and one a storey up
    boxes = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.3, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 3.1, 0.1, 0.1, 0.1]])
    c = box_colors(boxes, voxel)
    assert c.shape == (3, 3) and c.dtype == np.uint8
    assert not np.array_equal(c[0], c[1])                                   # checkerboard: neighbours differ
    assert np.abs(c[0].astype(int) - c[1].astype(int)).max() <= 0.15 * 255 + 1   # ... but only by the shade
    assert np.abs(c[0].astype(int) - c[2].astype(int)).max() > 0.15 * 255        # height changes the hue
    assert box_colors(np.zeros((0, 6)), voxel).shape == (0, 3)
    # a merged box spanning several cells is keyed on its min corner, not its centre
    big = np.array([[0.4, 0.1, 0.1, 0.4, 0.1, 0.1]])
    assert np.array_equal(box_colors(np.vstack([boxes[:1], big]), voxel)[1], c[0])


def test_downsample_points_keeps_one_per_cell_and_respects_the_cap():
    rng = np.random.default_rng(0)
    xyz = np.round(rng.uniform(-20, 20, size=(200_000, 3)) / 0.1) * 0.1    # a 0.1 m voxel cloud
    out, cell = downsample_points(xyz, 20_000, 0.1)
    assert len(out) <= 20_000 and cell > 0.1
    q = np.floor(out / cell).astype(np.int64)
    assert len(np.unique(q, axis=0)) == len(out)                             # one survivor per cell
    small, cell0 = downsample_points(xyz[:100], 1000, 0.1)
    assert small is xyz[:100] or np.array_equal(small, xyz[:100])
    assert cell0 == 0.1


def test_decimate():
    xyz = np.arange(30.0).reshape(10, 3)
    assert len(decimate(xyz, 4)) <= 4 and np.array_equal(decimate(xyz, 4)[0], xyz[0])
    assert decimate(xyz, 100) is xyz


def test_box_arrays_match_the_viewer_mesh():
    boxes = np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 2.0], [10.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    corners = box_corners(boxes)
    assert corners.shape == (2, 8, 3)
    assert np.allclose(corners[0].min(axis=0), [-0.5, -1.0, -2.0]) and np.allclose(corners[0].max(axis=0), [0.5, 1.0, 2.0])
    verts, faces = hexahedra_arrays(corners)
    assert verts.shape == (16, 3) and faces.shape == (24, 3) and faces.max() == 15
    # outward normals: every face normal points away from its box centre
    for k in range(2):
        f = faces[12 * k:12 * (k + 1)]
        a, b, c = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
        n = np.cross(b - a, c - a)
        assert ((n * ((a + b + c) / 3 - boxes[k, :3])).sum(axis=1) > 0).all()


def test_nearest_boxes():
    boxes = np.array([[float(i), 0.0, 0.0, 0.1, 0.1, 0.1] for i in range(10)])
    near = nearest_boxes(boxes, [7.2, 0.0], 3)
    assert sorted(near[:, 0].tolist()) == [6.0, 7.0, 8.0]
    assert nearest_boxes(boxes, [0, 0], 50) is boxes
