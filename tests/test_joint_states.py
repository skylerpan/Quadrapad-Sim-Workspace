"""Leg joint angles out to ROS (sensor_msgs/JointState): the console and robot_state_publisher draw the legs from it."""
import os

import numpy as np
import pytest

from quadsim.config import load_config

SCENE = os.path.join(os.path.dirname(__file__), "..", load_config(None, [])["scene"])
needs_scene = pytest.mark.skipif(not os.path.exists(os.path.join(SCENE, "contact.npy")),
                                 reason=f"{os.path.relpath(SCENE)} (config scene) not built as format 3")


def test_joint_state_msg_carries_names_angles_and_stamp():
    pytest.importorskip("sensor_msgs.msg")
    from quadsim.ros_io import joint_state_msg

    names = ["FL_HipX_joint", "FL_Knee_joint"]
    msg = joint_state_msg(12.3456789, names, np.array([0.1, 1.3]), np.array([0.5, -0.2]), np.array([2.0, -3.0]))
    assert msg.header.stamp.sec == 12 and abs(msg.header.stamp.nanosec - 345678900) <= 1
    assert list(msg.name) == names
    assert list(msg.position) == pytest.approx([0.1, 1.3])
    assert list(msg.velocity) == pytest.approx([0.5, -0.2])
    assert list(msg.effort) == pytest.approx([2.0, -3.0])


@needs_scene
def test_legged_sim_reports_its_twelve_leg_joints():
    from quadsim.g23 import DEFAULT_Q, JOINTS
    from quadsim.sim import Simulator

    sim = Simulator(load_config(None, ["robot.type=g23", "robot.controller.kind=g2_deploy_rs",
                                       "robot.controller.cmd_port=20111",
                                       "robot.controller.telemetry=[127.0.0.1:30120]",
                                       "robot.controller.command_addr=127.0.0.1:50161",
                                       "lidar.points_per_sec=2000"]))
    try:
        names, q, dq, tau = sim.joint_states()
        assert names == JOINTS
        assert np.allclose(q, DEFAULT_Q)            # spawned in the default stand pose: knees bent, not 0
        assert dq.shape == (12,) and tau.shape == (12,)
    finally:
        sim.close()


@needs_scene
def test_ideal_base_has_no_leg_joints():
    from quadsim.sim import Simulator

    sim = Simulator(load_config(None, ["robot.type=ideal", "lidar.points_per_sec=2000"]))
    try:
        assert sim.joint_states() is None
    finally:
        sim.close()
