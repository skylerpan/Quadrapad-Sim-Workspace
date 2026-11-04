"""The stand -> gait sequence of a legged robot in quadsim, as a pure state machine.

Both controller links (g2_deploy_rs, SyncAI-Device-MotorControl) run the same sequence: wait for the controller to
talk, stand after `start_delay`, start the gait after `stand_delay`, forward /cmd_vel while it is fresh and send one
stop when it goes stale, damp the legs before a teleport and reset the controller's safety lock after it, report a
lost / reconnected link once each. What differs is the wording on the wire, so the sequence emits intents and each
link translates them. No socket, no clock: the caller passes sim time, the torque-packet count and whether the link
has been silent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stand:
    pass


@dataclass(frozen=True)
class Gait:
    pass


@dataclass(frozen=True)
class Lie:
    pass


@dataclass(frozen=True)
class Damp:
    pass


@dataclass(frozen=True)
class ResetSafety:
    pass


@dataclass(frozen=True)
class Vel:
    x: float
    y: float
    w: float


@dataclass(frozen=True)
class VelStop:
    pass


Intent = Stand | Gait | Lie | Damp | ResetSafety | Vel | VelStop

OPERATOR_MODES = ("stand", "walk", "lie", "damp")


class StandSequence:
    def __init__(self, auto: bool, start_delay: float, stand_delay: float, gait_name: str):
        self.auto = auto
        self.start_delay = float(start_delay)
        self.stand_delay = float(stand_delay)
        self.gait_name = gait_name                  # the label of the gait event: "rl" / "lma" or a behavior name
        self.connected = False
        self.t_connected: float | None = None
        self.lost = False
        self.axes_active = False
        self.stage = "idle"
        self.t0 = 0.0
        self.t_stand: float | None = None
        self.t_gait: float | None = None
        self._reset_safety = False

    def restart(self, t: float, teleport: bool) -> list[Intent]:
        """(Re)start from sim time t (spawn, set_pose). A teleport while the controller runs trips its posture
        guard, so the legs are damped first and the sequence then begins with a safety reset."""
        self.t0 = t
        self.t_stand = self.t_gait = None
        self.stage = "idle" if self.auto else "manual"
        self._reset_safety = teleport and self.connected
        return [Damp()] if self._reset_safety else []

    def step(self, t: float, cmd, cmd_fresh: bool, n_cmd: int, silent: bool) -> tuple[list[Intent], list[dict]]:
        """One control tick. cmd_fresh: a /cmd_vel arrived within its timeout. n_cmd: torque packets received so
        far. silent: no torque packet for longer than the link timeout."""
        intents: list[Intent] = []
        events: list[dict] = []
        if n_cmd > 0 and not self.connected:
            self.connected = True
            self.t_connected = t
            events.append({"type": "controller", "state": "connected", "t": t})
        ready = self.connected and t - max(self.t0, self.t_connected) >= self.start_delay
        if self.stage == "idle" and ready:
            if self._reset_safety:
                intents.append(ResetSafety())
                self._reset_safety = False
            intents.append(Stand())
            self.stage = "stand"
            self.t_stand = t
            events.append({"type": "controller", "state": "stand", "t": t})
        elif self.stage == "stand" and t - self.t_stand >= self.stand_delay:
            intents.append(Gait())
            self.stage = "gait"
            self.t_gait = t
            events.append({"type": "controller", "state": self.gait_name, "t": t})
        if self.stage in ("gait", "manual"):
            # forward a live /cmd_vel every tick; when it has gone stale send one stop and then nothing, so the
            # controller's own watchdog marks the command inactive (a stream of zeros keeps an RL policy stepping)
            if cmd_fresh:
                intents.append(Vel(float(cmd[0]), float(cmd[1]), float(cmd[2])))
            elif self.axes_active:
                intents.append(VelStop())
            self.axes_active = cmd_fresh
        if self.connected:
            if silent and not self.lost:
                self.lost = True
                events.append({"type": "controller", "state": "lost", "t": t})
            elif not silent and self.lost:
                self.lost = False
                events.append({"type": "controller", "state": "connected", "t": t})
        return intents, events

    def operator(self, mode: str, t: float) -> tuple[list[Intent], dict]:
        """An operator command outside the auto sequence (web viewer buttons): stand (reset + stand, then wait for
        walk), walk (the gait), lie, damp."""
        if mode == "stand":
            intents, self.stage, state = [ResetSafety(), Stand()], "manual", "stand"
        elif mode == "walk":
            intents, self.stage, state = [Gait()], "gait", self.gait_name
            self.t_gait = t
        elif mode == "lie":
            intents, self.stage, state = [Lie()], "manual", "lie"
        elif mode == "damp":
            intents, self.stage, state = [Damp()], "manual", "damp"
        else:
            raise ValueError(f"operator mode {mode!r}: expected one of {', '.join(OPERATOR_MODES)}")
        return intents, {"type": "controller", "state": state, "t": t, "source": "operator"}
