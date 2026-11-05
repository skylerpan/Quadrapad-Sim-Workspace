"""Serializer tests (need ROS 2 + the livox message package; skipped otherwise)."""
import numpy as np
import pytest

from quadsim.lidar import LidarFrame

livox = pytest.importorskip("livox_ros_driver2.msg")
from rclpy.serialization import deserialize_message  # noqa: E402

from quadsim.ros_io import lidar_pointcloud2, serialize_custom_msg, stamp_parts  # noqa: E402


def _frame(n):
    rng = np.random.default_rng(0)
    return LidarFrame(stamp=12.3456789, xyz=rng.normal(size=(n, 3)).astype(np.float32),
                      reflectivity=rng.integers(0, 255, n).astype(np.uint8),
                      line=(np.arange(n) % 4).astype(np.uint8), offset=np.sort(rng.uniform(0, 0.1, n)))


@pytest.mark.parametrize("n,frame_id", [(0, "livox_frame"), (1, "l"), (1234, "livox_frame")])
def test_custom_msg_roundtrip(n, frame_id):
    f = _frame(n)
    msg = deserialize_message(serialize_custom_msg(f.stamp, frame_id, f), livox.CustomMsg)
    assert msg.header.frame_id == frame_id
    assert msg.header.stamp.sec == 12 and abs(msg.header.stamp.nanosec - 345678900) <= 1
    assert msg.timebase == int(round(f.stamp * 1e9))
    assert msg.point_num == n and len(msg.points) == n
    if n:
        p = msg.points[-1]
        assert p.x == pytest.approx(float(f.xyz[-1, 0]))
        assert p.offset_time == int(round(f.offset[-1] * 1e9))
        assert p.reflectivity == f.reflectivity[-1] and p.line == f.line[-1]


@pytest.mark.parametrize("layout", ["livox", "velodyne"])
def test_pointcloud2_layout(layout):
    from sensor_msgs_py import point_cloud2

    f = _frame(100)
    msg = lidar_pointcloud2(f.stamp, "livox_frame", f, layout)
    names = [fl.name for fl in msg.fields]
    pts = point_cloud2.read_points(msg, field_names=names)
    assert len(pts) == 100
    np.testing.assert_allclose(pts["x"], f.xyz[:, 0])
    if layout == "livox":
        np.testing.assert_allclose(pts["timestamp"], (f.stamp + f.offset) * 1e9)
    else:
        np.testing.assert_allclose(pts["time"], f.offset, rtol=1e-6)


def test_stamp_parts_carries_at_second_boundary():
    t, prev = 0.0, (-1, 0)
    for _ in range(4000):                       # accumulated float steps hit x.9999999 values
        t += 0.0025
        cur = stamp_parts(t)
        assert 0 <= cur[1] < 1_000_000_000
        assert cur > prev
        prev = cur
    assert stamp_parts(44.99999999999) == (45, 0)


def test_gnss_fix_becomes_a_navsatfix_and_an_enu_twist_like_the_ublox_driver():
    from sensor_msgs.msg import NavSatFix, NavSatStatus
    from quadsim.gnss import GnssFix
    from quadsim.ros_io import gnss_navsatfix, gnss_twist

    fix = GnssFix(t=12.3456789, fix="rtk_fixed", lat=24.96, lon=121.42, alt=130.5, vel_enu=np.array([0.5, -0.2, 0.01]),
                  sigma_h=0.01, sigma_v=0.02, sky=1.0)
    m = gnss_navsatfix(fix, "robot01/gnss_link")
    assert m.header.frame_id == "robot01/gnss_link" and m.header.stamp.sec == 12 and abs(m.header.stamp.nanosec - 345678900) <= 1
    assert m.status.status == NavSatStatus.STATUS_GBAS_FIX                      # the driver reports RTK fixed this way
    assert m.status.service == (NavSatStatus.SERVICE_GPS | NavSatStatus.SERVICE_GLONASS | NavSatStatus.SERVICE_GALILEO
                                | NavSatStatus.SERVICE_COMPASS)
    assert (m.latitude, m.longitude, m.altitude) == (24.96, 121.42, 130.5)
    assert list(m.position_covariance) == [0.01 ** 2, 0, 0, 0, 0.01 ** 2, 0, 0, 0, 0.02 ** 2]
    assert m.position_covariance_type == NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
    tw = gnss_twist(fix, "robot01/gnss_link", vel_sigma=0.02)
    assert tw.header.frame_id == "robot01/gnss_link" and (tw.twist.twist.linear.x, tw.twist.twist.linear.y) == (0.5, -0.2)
    assert tw.twist.covariance[0] == tw.twist.covariance[7] == tw.twist.covariance[14] == 0.02 ** 2

    for kind, status in (("single", NavSatStatus.STATUS_FIX), ("rtk_float", NavSatStatus.STATUS_FIX),
                         ("dead_reckoning", NavSatStatus.STATUS_FIX)):
        fix.fix = kind
        assert gnss_navsatfix(fix, "g").status.status == status

    none = GnssFix(t=1.0, fix="none", lat=float("nan"), lon=float("nan"), alt=float("nan"),
                   vel_enu=np.full(3, np.nan), sigma_h=float("inf"), sigma_v=float("inf"), sky=0.0)
    m = gnss_navsatfix(none, "g")
    assert m.status.status == NavSatStatus.STATUS_NO_FIX and np.isnan(m.latitude)
    assert m.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN and list(m.position_covariance) == [0.0] * 9
