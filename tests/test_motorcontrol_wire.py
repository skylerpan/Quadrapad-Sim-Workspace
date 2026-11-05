"""MotorControl's sim protocol, encoded and decoded as pure functions: three UDP packets (little-endian f32, no
header, the length is the identity) and the operator text lines (`kind=... key=value ...`)."""
import math
import struct

import numpy as np

from quadsim.motorcontrol_wire import (MC_BASE_BYTES, MC_CMD_BYTES, MC_JOINT_BYTES, decode_command, encode_base,
                                       encode_joint)


def test_packet_sizes_match_the_controller():
    # pinned in sim_packets.rs: 12 motors x 5 fields, x 3 fields, 9 base fields
    assert MC_CMD_BYTES == 240 and MC_JOINT_BYTES == 144 and MC_BASE_BYTES == 36


def test_joint_feedback_is_interleaved_per_motor():
    q, dq, tau = np.arange(12) * 1.0, np.arange(12) * 10.0, np.arange(12) * 100.0
    pkt = encode_joint(q, dq, tau)
    assert len(pkt) == MC_JOINT_BYTES
    v = struct.unpack("<36f", pkt)
    assert v[0:3] == (0.0, 0.0, 0.0) and v[3:6] == (1.0, 10.0, 100.0) and v[33:36] == (11.0, 110.0, 1100.0)


def test_base_packet_is_rpy_omega_acc():
    pkt = encode_base(np.array([0.1, -0.2, 0.3]), np.array([1.0, 2.0, 3.0]), np.array([-0.5, 0.5, 9.81]))
    assert len(pkt) == MC_BASE_BYTES
    v = struct.unpack("<9f", pkt)
    assert np.allclose(v, [0.1, -0.2, 0.3, 1.0, 2.0, 3.0, -0.5, 0.5, 9.81])


def test_command_decodes_interleaved_pos_vel_tau_kp_kd():
    fields = []
    for motor in range(12):
        fields += [motor + 0.1, motor + 0.2, motor + 0.3, motor + 0.4, motor + 0.5]   # pos, vel, tau, kp, kd
    cmd = decode_command(struct.pack("<60f", *fields))
    assert cmd is not None
    pos, vel, tau, kp, kd = cmd
    assert np.allclose(pos, np.arange(12) + 0.1) and np.allclose(vel, np.arange(12) + 0.2)
    assert np.allclose(tau, np.arange(12) + 0.3) and np.allclose(kp, np.arange(12) + 0.4)
    assert np.allclose(kd, np.arange(12) + 0.5)


def test_command_of_the_wrong_length_or_with_a_non_finite_value_is_dropped_whole():
    assert decode_command(b"\0" * 239) is None and decode_command(b"\0" * 241) is None
    fields = [0.0] * 60
    fields[17] = math.nan
    assert decode_command(struct.pack("<60f", *fields)) is None
    fields[17] = math.inf
    assert decode_command(struct.pack("<60f", *fields)) is None


# -- operator lines ---------------------------------------------------------------------------------------
from quadsim.motorcontrol_wire import Request, Status, Switched, Stopped, Released, parse_operator_line  # noqa: E402

STATUS = ("kind=status behavior=stand moving_to=- failure=- stop=- stopped_while=- stop_fired=- stop_joints=- "
          "stop_rule=- stop_lost=- stop_battery=- stop_dropped_rule=- stop_source=- motors_health=ok motors_fault=- "
          "motors_limit=- motors_warn=- imu_health=ok imu_fault=- imu_limit=- imu_warn=- bms_health=ok bms_fault=- "
          "bms_limit=- bms_warn=- faulted_joints=- unseen_joints=- status_word_stale_joints=- feedback_late_joints=-")


def test_status_line_becomes_a_typed_status_with_dashes_as_none():
    s = parse_operator_line(STATUS)
    assert s == Status(behavior="stand", moving_to=None, stop=None, motors_health="ok", imu_health="ok")
    s = parse_operator_line(STATUS.replace("moving_to=-", "moving_to=rl_fcn").replace("stop=-", "stop=hardware")
                            .replace("imu_health=ok", "imu_health=stale"))
    assert s.moving_to == "rl_fcn" and s.stop == "hardware" and s.imu_health == "stale"


def test_request_and_event_lines():
    r = parse_operator_line("kind=request seq=9 state=rejected target=rl_fcn reasons=predecessor_not_settled")
    assert r == Request(seq=9, state="rejected", target="rl_fcn", reasons="predecessor_not_settled")
    assert parse_operator_line("kind=request seq=- state=- target=- reasons=-") == Request(None, None, None, None)
    assert parse_operator_line("kind=event type=switched from=rest to=stand switch_count=1") == Switched("rest", "stand")
    assert parse_operator_line("kind=event type=stopped stop=hardware") == Stopped("hardware")
    assert parse_operator_line("kind=event type=released") == Released()


def test_lines_that_carry_nothing_for_us_are_none():
    assert parse_operator_line("kind=state rpy=0,0,0 omega=0,0,0") is None          # 20 Hz numbers: not consumed
    assert parse_operator_line("kind=event type=failure behavior=rl_fcn reason=x") is None
    assert parse_operator_line("") is None and parse_operator_line("garbage") is None
    assert parse_operator_line("kind=request seq=abc state=ok target=stand reasons=-") is None   # bad seq
