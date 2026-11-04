"""The SyncAI-Device-MotorControl link: quadsim as the MuJoCo side of the controller's simulation backend.

  controller -> sim   UDP :40001   240 bytes, [pos, vel, tau, kp, kd] x 12, one per 1 kHz control tick from boot
  sim -> controller   UDP :40002   144 bytes, [q, dq, tau] x 12, every physics step
  sim -> controller   UDP :40003    36 bytes, [rpy, gyro, acc] (body frame, acc includes gravity), every physics step
  sim -> controller   UDP :40020   operator lines: stand | rl_fcn | rl_him | gait_mpc | rest | reset | vel x y w
  controller -> sim   (reply)      kind=status / kind=request / kind=event lines, to the last sender

The controller boots in `rest`, admits `stand` once it has seen joint packets, and an RL behavior only after the
stand has settled (otherwise `state=rejected`): the sequence resends a rejected behavior once a second, up to five
times. Joint packets missing for over 100 ms, or a tilt past 60 degrees, latch `stop=hardware` on its side; quadsim
reports it as a `stopped` event and leaves the `reset` to the operator (set_pose and the Stand button send one).
Formats and line parsing: quadsim.motorcontrol_wire. The sequence: quadsim.sequence.
"""
from __future__ import annotations

import queue
import socket
import threading
import time

import numpy as np

from quadsim.g23 import G23Physics, RefSimOptions, _addr
from quadsim.motorcontrol_wire import (BEHAVIORS, POLICY_TO_BEHAVIOR, Released, Request, Status, Stopped, Switched,
                                       decode_command, encode_base, encode_joint, parse_operator_line)
from quadsim.sequence import Damp, Gait, Intent, Lie, ResetSafety, Stand, StandSequence, Vel, VelStop

RETRY_AFTER = 1.0          # s of sim time before a rejected behavior is requested again
MAX_RETRIES = 5


def check_behavior(name) -> str:
    if name not in BEHAVIORS:
        raise ValueError(f"robot.behavior={name!r}: expected one of {', '.join(BEHAVIORS)}")
    return str(name)


class MotorControlBackend(G23Physics):
    def __init__(self, m, d, robot_cfg: dict, torso_body: int, opts: RefSimOptions | None = None):
        super().__init__(m, d, robot_cfg, torso_body, opts)
        mc = (robot_cfg.get("controller", {}) or {}).get("motorcontrol", {}) or {}
        self.cmd_port = int(mc.get("cmd_port", 40001))
        self.joint_to = _addr(mc.get("joint_addr", "127.0.0.1:40002"))
        self.base_to = _addr(mc.get("base_addr", "127.0.0.1:40003"))
        self.operator_to = _addr(mc.get("operator_addr", "127.0.0.1:40020"))
        self.behavior = check_behavior(robot_cfg.get("behavior", "rl_fcn"))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.2)
        self.sock.bind(("0.0.0.0", self.cmd_port))
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.op = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)      # the controller answers its last sender
        self.op.settimeout(0.2)
        self.op.bind(("127.0.0.1", 0))
        self._lines: queue.Queue = queue.Queue()                          # parsed operator lines, read by tick()
        self._stop = threading.Event()
        self.rx = threading.Thread(target=self._receiver, daemon=True)
        self.op_rx = threading.Thread(target=self._op_receiver, daemon=True)
        self.rx.start()
        self.op_rx.start()
        # what the controller last told us (written only by tick)
        self.status: Status | None = None
        self.last_request: Request | None = None
        self.stop: str | None = None
        self._pending: str | None = None           # behavior requested, not yet accepted
        self._retry_at: float | None = None
        self._retries = 0
        self.seq = StandSequence(bool(robot_cfg.get("auto_stand", True)), robot_cfg.get("start_delay", 1.0),
                                 robot_cfg.get("stand_delay", 4.0), self.behavior)
        self.sent: list[str] = []
        self.restart(0.0)

    def describe_link(self) -> str:
        return (f"torque cmds on udp:{self.cmd_port}, joints -> {self.joint_to}, base -> {self.base_to}, "
                f"operator -> {self.operator_to}, behavior {self.behavior}")

    @property
    def safety_locked(self) -> bool:
        return self.stop is not None

    @property
    def state(self) -> str | None:
        return None if self.status is None else (self.status.moving_to or self.status.behavior)

    # -- threads: packets in ----------------------------------------------------------------------
    def _receiver(self):
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            cmd = decode_command(data)
            if cmd is None:
                continue
            pos, vel, tau, kp, kd = cmd
            with self.lock:
                self.q_des[:] = pos
                self.dq_des[:] = vel
                self.tau_ff[:] = tau
                self.kp[:] = kp
                self.kd[:] = kd
                self.last_cmd_wall = time.monotonic()
                self.n_cmd += 1

    def _op_receiver(self):
        while not self._stop.is_set():
            try:
                data, _ = self.op.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            for line in data.decode(errors="replace").splitlines():
                parsed = parse_operator_line(line)
                if parsed is not None:
                    self._lines.put(parsed)

    # -- packets out ------------------------------------------------------------------------------
    def send_telemetry(self, t: float):
        d = self.d
        rpy, gyro, acc = self.imu_output()
        try:
            self.out.sendto(encode_joint(d.qpos[self.jq], d.qvel[self.jv], d.qfrc_actuator[self.jv]), self.joint_to)
            self.out.sendto(encode_base(rpy, gyro, acc), self.base_to)
        except OSError:
            pass

    def send_command(self, text: str):
        self.sent.append(text)
        try:
            self.op.sendto((text + "\n").encode(), self.operator_to)
        except OSError:
            pass

    # -- the sequence's intents in the controller's words ---------------------------------------------
    def _send(self, intent: Intent):
        if isinstance(intent, (Damp, Lie)):
            self.send_command("rest")                 # rest damps for 2 s, then lets go: the controller's only "down"
        elif isinstance(intent, ResetSafety):
            self.send_command("reset")
        elif isinstance(intent, Stand):
            self.send_command("stand")
        elif isinstance(intent, Gait):
            self._request(self.behavior)
        elif isinstance(intent, Vel):
            self.send_command(f"vel {intent.x:.3f} {intent.y:.3f} {intent.w:.3f}")
        elif isinstance(intent, VelStop):
            self.send_command("vel 0.000 0.000 0.000")

    def _request(self, behavior: str):
        self._pending, self._retry_at, self._retries = behavior, None, 0
        self.send_command(behavior)

    # -- what the controller said, applied by the control thread ------------------------------------------
    def _note_stop(self, stop: str | None, t: float, events: list[dict]):
        if stop == self.stop:
            return
        self.stop = stop
        if stop is None:
            events.append({"type": "controller", "state": "released", "t": t})
        else:
            events.append({"type": "controller", "state": "stopped", "stop": stop, "t": t})

    def _drain(self, t: float) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return events
            if isinstance(line, Status):
                self.status = line
                self._note_stop(line.stop, t, events)
            elif isinstance(line, Request):
                self.last_request = line
                if line.target == self._pending:
                    if line.state == "ok":
                        self._pending = None
                    elif line.state == "rejected":
                        if self._retries < MAX_RETRIES:
                            self._retry_at = t + RETRY_AFTER
                        else:
                            events.append({"type": "controller", "state": "rejected", "target": line.target,
                                           "reasons": line.reasons, "t": t})
                            self._pending = None
            elif isinstance(line, Switched):
                events.append({"type": "controller", "state": "switched", "from": line.from_behavior,
                               "to": line.to_behavior, "t": t})
            elif isinstance(line, Stopped):
                self._note_stop(line.stop, t, events)
            elif isinstance(line, Released):
                self._note_stop(None, t, events)

    # -- the per-tick API sim.py drives -------------------------------------------------------------------
    def restart(self, t: float, teleport: bool = False):
        """(Re)start the stand -> behavior sequence from sim time t (spawn, set_pose): rest before a teleport,
        reset before the new stand (a stop the teleport may have tripped is cleared), as an operator would."""
        self._pending = None
        for intent in self.seq.restart(t, teleport):
            self._send(intent)

    def tick(self, t: float, cmd: np.ndarray, cmd_fresh: bool = True) -> list[dict]:
        events = self._drain(t)
        n, silent = self.link_counters()
        intents, seq_events = self.seq.step(t, cmd, cmd_fresh, n, silent)
        for intent in intents:
            self._send(intent)
        if self._pending is not None and self._retry_at is not None and t >= self._retry_at:
            self._retry_at = None
            self._retries += 1
            self.send_command(self._pending)
        return events + seq_events

    def operator(self, mode: str, t: float) -> dict:
        """stand (reset + stand), walk (the configured behavior), lie and damp (both rest: the controller has no
        separate damping behavior; rest damps for its first seconds)."""
        intents, event = self.seq.operator(mode, t)
        for intent in intents:
            self._send(intent)
        return event

    def set_policy_mode(self, mode: int, t: float) -> dict:
        """quadsim's RL policy slots (0 PPO, 1 HIMLoco) as the controller's behaviors (rl_fcn, rl_him); other slots
        have no counterpart and are refused. Sent now when the gait is on, by way of stand."""
        behavior = POLICY_TO_BEHAVIOR.get(int(mode))
        if behavior is None:
            return {"type": "controller", "state": "policy", "policy_mode": int(mode), "t": t, "live": False,
                    "rejected": True, "reasons": f"no behavior for policy_mode {int(mode)}"}
        self.behavior = self.seq.gait_name = behavior
        live = self.stage == "gait"
        if live:
            # the controller admits one edge at a time and the RL behaviors only touch stand: back to stand first,
            # then the new behavior (rejected until the stand has settled: the retry picks it up)
            self.send_command("stand")
            self._request(behavior)
        return {"type": "controller", "state": "policy", "policy_mode": int(mode), "behavior": behavior, "t": t,
                "live": live}

    def close(self):
        self._stop.set()
        for s in (self.sock, self.out, self.op):
            try:
                s.close()
            except OSError:
                pass
