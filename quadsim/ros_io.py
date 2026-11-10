"""ROS 2 message packing for the sim (numpy-vectorised, no per-point Python objects)."""
from __future__ import annotations

import struct

import math

import numpy as np
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import JointState, NavSatFix, NavSatStatus, PointCloud2, PointField

# --- livox_ros_driver2/CustomMsg as raw CDR -----------------------------------
# CustomPoint: uint32 offset_time, float32 x y z, uint8 reflectivity tag line (19 B, 4-aligned -> 20)
_CUSTOM_POINT = np.dtype({"names": ["offset_time", "x", "y", "z", "reflectivity", "tag", "line"],
                          "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
                          "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 20})


def stamp_parts(t: float) -> tuple[int, int]:
    ns = int(round(t * 1e9))            # round once, then split: no 44.9999999 -> (44, 0) carry bug
    return ns // 1_000_000_000, ns % 1_000_000_000


def serialize_custom_msg(t_stamp: float, frame_id: str, frame) -> bytes:
    """CDR bytes of livox_ros_driver2/msg/CustomMsg for a LidarFrame (publish via publish_raw)."""
    sec, nsec = stamp_parts(t_stamp)
    body = bytearray()
    body += struct.pack("<iI", sec, nsec)
    fid = frame_id.encode() + b"\x00"
    body += struct.pack("<I", len(fid)) + fid
    body += b"\x00" * (-len(body) % 8)
    n = len(frame.xyz)
    body += struct.pack("<QIB3x", int(round(t_stamp * 1e9)), n, 0)          # lidar_id 0 (single sensor)
    body += b"\x00" * (-len(body) % 4)
    body += struct.pack("<I", n)
    if n:
        pts = np.zeros(n, dtype=_CUSTOM_POINT)
        pts["offset_time"] = np.round(frame.offset * 1e9).astype(np.uint32)
        pts["x"], pts["y"], pts["z"] = frame.xyz[:, 0], frame.xyz[:, 1], frame.xyz[:, 2]
        pts["reflectivity"] = frame.reflectivity
        pts["line"] = frame.line
        body += pts.tobytes()[:-1]  # CDR has no padding after the last element
    return b"\x00\x01\x00\x00" + bytes(body)


# --- sensor_msgs/PointCloud2 ----------------------------------------------------
_F32, _F64, _U8, _U16 = PointField.FLOAT32, PointField.FLOAT64, PointField.UINT8, PointField.UINT16
_LAYOUTS = {
    # livox_ros_driver2 PointXYZRTLT (xfer_format 0); timestamp = absolute ns as float64
    "livox": ([("x", 0, _F32), ("y", 4, _F32), ("z", 8, _F32), ("intensity", 12, _F32),
               ("tag", 16, _U8), ("line", 17, _U8), ("timestamp", 18, _F64)], 26),
    # velodyne-style for LIO stacks with lidar_type=velodyne; time = seconds since stamp
    "velodyne": ([("x", 0, _F32), ("y", 4, _F32), ("z", 8, _F32), ("intensity", 12, _F32),
                  ("ring", 16, _U16), ("time", 18, _F32)], 22),
}
_NP = {_F32: "<f4", _F64: "<f8", _U8: "u1", _U16: "<u2"}


def lidar_pointcloud2(t_stamp: float, frame_id: str, frame, layout: str) -> PointCloud2:
    fields, step = _LAYOUTS[layout]
    dt = np.dtype({"names": [f[0] for f in fields], "formats": [_NP[f[2]] for f in fields],
                   "offsets": [f[1] for f in fields], "itemsize": step})
    n = len(frame.xyz)
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = frame.xyz[:, 0], frame.xyz[:, 1], frame.xyz[:, 2]
    arr["intensity"] = frame.reflectivity
    if layout == "livox":
        arr["line"] = frame.line
        arr["timestamp"] = (t_stamp + frame.offset) * 1e9
    else:
        arr["ring"] = frame.line
        arr["time"] = frame.offset
    return _cloud(t_stamp, frame_id, fields, step, arr.tobytes(), n)


def xyzi_pointcloud2(t_stamp: float, frame_id: str, xyz: np.ndarray, intensity: np.ndarray) -> PointCloud2:
    fields = [("x", 0, _F32), ("y", 4, _F32), ("z", 8, _F32), ("intensity", 12, _F32)]
    arr = np.empty((len(xyz), 4), dtype=np.float32)
    arr[:, :3], arr[:, 3] = xyz, intensity
    return _cloud(t_stamp, frame_id, fields, 16, arr.tobytes(), len(xyz))


def _cloud(t_stamp, frame_id, fields, step, data: bytes, n: int) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_parts(t_stamp)
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, n
    msg.fields = [PointField(name=nm, offset=off, datatype=dt, count=1) for nm, off, dt in fields]
    msg.is_bigendian = False
    msg.point_step, msg.row_step = step, step * n
    msg.is_dense = True
    msg.data = data
    return msg


# --- GNSS: what the ublox ROS 2 driver publishes for a ZED-F9x -----------------
_NAVSAT_STATUS = {"none": NavSatStatus.STATUS_NO_FIX, "single": NavSatStatus.STATUS_FIX,
                  "rtk_float": NavSatStatus.STATUS_FIX, "rtk_fixed": NavSatStatus.STATUS_GBAS_FIX,
                  "dead_reckoning": NavSatStatus.STATUS_FIX}
_NAVSAT_SERVICE = (NavSatStatus.SERVICE_GPS | NavSatStatus.SERVICE_GLONASS | NavSatStatus.SERVICE_GALILEO
                   | NavSatStatus.SERVICE_COMPASS)                            # the F9 tracks all four


def gnss_navsatfix(fix, frame_id: str) -> NavSatFix:
    """sensor_msgs/NavSatFix for a quadsim.gnss.GnssFix: RTK fixed is STATUS_GBAS_FIX like the driver's `fix`,
    the covariance is the receiver's own sigma on the diagonal, a lost fix is STATUS_NO_FIX with NaN."""
    m = NavSatFix()
    m.header.stamp.sec, m.header.stamp.nanosec = stamp_parts(fix.t)
    m.header.frame_id = frame_id
    m.status.status = _NAVSAT_STATUS[fix.fix]
    m.status.service = _NAVSAT_SERVICE
    m.latitude, m.longitude, m.altitude = float(fix.lat), float(fix.lon), float(fix.alt)
    if math.isfinite(fix.sigma_h):
        m.position_covariance = [fix.sigma_h ** 2, 0.0, 0.0, 0.0, fix.sigma_h ** 2, 0.0, 0.0, 0.0, fix.sigma_v ** 2]
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
    else:
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
    return m


def gnss_twist(fix, frame_id: str, vel_sigma: float) -> TwistWithCovarianceStamped:
    """geometry_msgs/TwistWithCovarianceStamped with the ENU velocity, like the driver's `fix_velocity`."""
    m = TwistWithCovarianceStamped()
    m.header.stamp.sec, m.header.stamp.nanosec = stamp_parts(fix.t)
    m.header.frame_id = frame_id
    m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(v) for v in fix.vel_enu)
    cov = [0.0] * 36
    cov[0] = cov[7] = cov[14] = vel_sigma ** 2
    m.twist.covariance = cov
    return m


def joint_state_msg(t_stamp: float, names: list[str], q, dq, tau) -> JointState:
    """sensor_msgs/JointState for the leg joints: position / velocity / effort in rad, rad/s, N*m."""
    m = JointState()
    m.header.stamp.sec, m.header.stamp.nanosec = stamp_parts(t_stamp)
    m.name = list(names)
    m.position, m.velocity, m.effort = [float(v) for v in q], [float(v) for v in dq], [float(v) for v in tau]
    return m
