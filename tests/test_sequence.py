"""The stand -> gait sequence as a pure state machine: intents out, no sockets, no clock of its own."""
from quadsim.sequence import Damp, Gait, Lie, ResetSafety, Stand, StandSequence, Vel, VelStop


def run(seq, t, cmd=(0.0, 0.0, 0.0), fresh=False, n_cmd=1, silent=False):
    return seq.step(t, cmd, fresh, n_cmd, silent)


def test_sequence_waits_for_the_controller_then_stands_then_walks():
    seq = StandSequence(auto=True, start_delay=1.0, stand_delay=4.0, gait_name="rl")
    assert seq.restart(0.0, teleport=False) == []
    assert run(seq, 0.5, n_cmd=0) == ([], [])                                   # nothing talks yet
    intents, events = run(seq, 2.0, n_cmd=1)                                      # first torque packet at t=2
    assert intents == [] and events == [{"type": "controller", "state": "connected", "t": 2.0}]
    assert run(seq, 2.9) == ([], [])                                              # start_delay counts from connection
    intents, events = run(seq, 3.0)
    assert intents == [Stand()] and events == [{"type": "controller", "state": "stand", "t": 3.0}]
    assert seq.stage == "stand"
    assert run(seq, 6.9) == ([], [])
    intents, events = run(seq, 7.0)
    assert intents == [Gait()] and events == [{"type": "controller", "state": "rl", "t": 7.0}]
    assert seq.stage == "gait" and seq.t_gait == 7.0


def test_velocity_is_forwarded_while_fresh_and_stopped_once_when_stale():
    seq = StandSequence(auto=True, start_delay=0.0, stand_delay=0.0, gait_name="rl")
    seq.restart(0.0, teleport=False)
    run(seq, 0.0); run(seq, 0.0); run(seq, 0.0)                                   # connected, stand, gait
    assert seq.stage == "gait"
    assert run(seq, 1.0, (0.5, 0.0, -0.2), fresh=True)[0] == [Vel(0.5, 0.0, -0.2)]
    assert run(seq, 1.1, (0.5, 0.0, -0.2), fresh=True)[0] == [Vel(0.5, 0.0, -0.2)]
    assert run(seq, 1.2, fresh=False)[0] == [VelStop()]                           # one zero, then nothing
    assert run(seq, 1.3, fresh=False)[0] == []
    assert run(seq, 1.4, fresh=False)[0] == []


def test_teleport_damps_first_and_resets_safety_before_standing():
    seq = StandSequence(auto=True, start_delay=0.5, stand_delay=1.0, gait_name="rl")
    seq.restart(0.0, teleport=False)
    for t in (0.0, 0.5, 2.0):
        run(seq, t)
    assert seq.stage == "gait"
    assert seq.restart(5.0, teleport=True) == [Damp()] and seq.stage == "idle"
    assert seq.restart(5.0, teleport=True) == [Damp()]                            # calling it again changes nothing
    assert run(seq, 5.25) == ([], [])
    intents, events = run(seq, 5.5)
    assert intents == [ResetSafety(), Stand()] and [e["state"] for e in events] == ["stand"]
    # a teleport before the controller ever connected needs no damping and no reset
    fresh = StandSequence(auto=True, start_delay=0.5, stand_delay=1.0, gait_name="rl")
    assert fresh.restart(0.0, teleport=True) == []
    assert run(fresh, 1.0)[0] == [] and run(fresh, 1.5)[0] == [Stand()]


def test_lost_and_reconnected_link_are_reported_once_each():
    seq = StandSequence(auto=False, start_delay=0.0, stand_delay=0.0, gait_name="rl")
    seq.restart(0.0, teleport=False)
    assert seq.stage == "manual"
    assert run(seq, 0.0)[1][0]["state"] == "connected"
    assert run(seq, 1.0, silent=True)[1] == [{"type": "controller", "state": "lost", "t": 1.0}]
    assert run(seq, 1.1, silent=True)[1] == []
    assert run(seq, 1.2, silent=False)[1] == [{"type": "controller", "state": "connected", "t": 1.2}]


def test_operator_commands_outside_the_auto_sequence():
    seq = StandSequence(auto=True, start_delay=0.0, stand_delay=0.0, gait_name="rl")
    seq.restart(0.0, teleport=False)
    assert seq.operator("stand", 1.0) == ([ResetSafety(), Stand()], {"type": "controller", "state": "stand", "t": 1.0, "source": "operator"})
    assert seq.stage == "manual"
    assert seq.operator("walk", 2.0) == ([Gait()], {"type": "controller", "state": "rl", "t": 2.0, "source": "operator"})
    assert seq.stage == "gait" and seq.t_gait == 2.0
    assert seq.operator("lie", 3.0)[0] == [Lie()] and seq.stage == "manual"
    assert seq.operator("damp", 4.0)[0] == [Damp()] and seq.stage == "manual"
    try:
        seq.operator("fly", 5.0)
        assert False, "unknown operator mode must raise"
    except ValueError:
        pass
