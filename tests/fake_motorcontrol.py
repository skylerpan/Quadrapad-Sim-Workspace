"""A stand-in for SyncAI-Device-MotorControl's sim backend, speaking its wire (quadsim.motorcontrol_wire) so quadsim
can be driven without the Rust binary: by the tests, and by hand

    python3 -m tests.fake_motorcontrol                       # default ports: quadsim's config/sim.yaml
    scripts/run_sim.sh robot.controller.spawn=false          # in another shell

It answers every joint-feedback packet with one torque command (the real controller sends one per 1 kHz tick):
zeros in `rest`, a PD hold of the default standing pose otherwise. On the operator socket it takes the controller's
words (rest, stand, rl_fcn, rl_him, gait_mpc, reset, resume, estop, vel x y w) and replies like the controller:
`kind=request` with the fate of a switch, `kind=status` whenever something changed, `kind=event type=switched`.
Scripted faults: `reject_once` names behaviors whose first request is rejected (predecessor_not_settled),
`reject_always` behaviors never admitted, `handover_s` makes a switch take that long (a newer request supersedes a
pending one, as the controller does), and
`inject_stop(kind)` latches a stop that only `reset` clears, as the controller does.
Like the controller, the latest `kind=status` and `kind=request` lines are repeated every `heartbeat_s` while
nothing changes (its cadence.rs: once a second); 0 turns the heartbeat off. `latest_only_s` models its operator
input: the control side sees one input per orchestrator period (5 ms), the latest datagram, so two datagrams
inside one period leave only the second; 0 applies every datagram as it arrives.
"""
from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np

from quadsim.motorcontrol_wire import MC_JOINT_BYTES, decode_command, encode_joint  # noqa: F401  (decode for tests)
from quadsim.motorcontrol_wire import BEHAVIORS, MC_BASE_BYTES, MC_CMD_BYTES, MOTORS

HOLD_Q = np.array([0.0, -0.65, 1.3] * 4)          # the G23 default standing pose
EDGES = {("rest", "stand"), ("stand", "rest"), ("stand", "gait_mpc"), ("gait_mpc", "stand"),
         ("stand", "rl_fcn"), ("rl_fcn", "stand"), ("stand", "rl_him"), ("rl_him", "stand")}


class FakeMotorControl:
    def __init__(self, cmd_to=("127.0.0.1", 40001), joint_port: int = 40002, base_port: int = 40003,
                 operator_port: int = 40020, reject_once: tuple[str, ...] = (), reject_always: tuple[str, ...] = (),
                 handover_s: float = 0.0, kp: float = 60.0, kd: float = 2.0, heartbeat_s: float = 1.0,
                 latest_only_s: float = 0.0):
        self.cmd_to, self.kp, self.kd = cmd_to, kp, kd
        self.latest_only_s = latest_only_s          # > 0: only the latest datagram of each such period is applied
        self._latest: bytes | None = None           # the datagram waiting for the next period (latest_only_s)
        self.heartbeat_s = heartbeat_s              # unchanged status / request lines are repeated this often (0 = never)
        self._heartbeat: dict[str, str] = {}        # the latest line per kind that the heartbeat repeats
        self.handover_s = handover_s                # a switch takes this long; a newer request supersedes a pending one
        self._handover: threading.Timer | None = None
        self.reject_once = set(reject_once)
        self.reject_always = set(reject_always)     # behaviors never admitted (e.g. no IMU for RL)
        self.behavior = "rest"
        self.stop: str | None = None
        self.vel = (0.0, 0.0, 0.0)
        self.seq = 0
        self.received: list[str] = []             # operator lines, in order
        self.n_joint = self.n_base = 0
        self.last_q = np.zeros(MOTORS)
        self.last_base = np.zeros(9)
        self.operator_peer = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.joint = self._udp(joint_port)
        self.base = self._udp(base_port)
        self.op = self._udp(operator_port)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.threads = [threading.Thread(target=f, daemon=True)
                        for f in (self._joint_loop, self._base_loop, self._op_loop, self._heartbeat_loop,
                                  self._latest_only_loop)]
        for t in self.threads:
            t.start()

    @staticmethod
    def _udp(port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(0.2)
        s.bind(("127.0.0.1", port))
        return s

    # -- the sim side -------------------------------------------------------------------------------
    def _joint_loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self.joint.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != MC_JOINT_BYTES:
                continue
            v = np.frombuffer(data, dtype="<f4").reshape(MOTORS, 3)
            with self.lock:
                self.last_q = v[:, 0].astype(float)
                self.n_joint += 1
                hold = self.behavior != "rest" and self.stop is None
            cmd = np.zeros((MOTORS, 5), dtype=np.float32)
            if hold:
                cmd[:, 0], cmd[:, 3], cmd[:, 4] = HOLD_Q, self.kp, self.kd
            try:
                self.out.sendto(cmd.tobytes(), self.cmd_to)
            except OSError:
                pass

    def _base_loop(self):
        while not self._stop.is_set():
            try:
                data, _ = self.base.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) == MC_BASE_BYTES:
                with self.lock:
                    self.last_base = np.frombuffer(data, dtype="<f4").astype(float)
                    self.n_base += 1

    # -- the operator side --------------------------------------------------------------------------
    def _op_loop(self):
        while not self._stop.is_set():
            try:
                data, peer = self.op.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.operator_peer = peer
            if self.latest_only_s > 0:
                with self.lock:
                    self._latest = data
            else:
                self._apply(data)

    def _apply(self, data: bytes):
        """One datagram = one operator input: its lines in order."""
        for line in data.decode(errors="replace").splitlines():
            line = line.strip()
            if line:
                self._handle(line)

    def _latest_only_loop(self):
        if self.latest_only_s <= 0:
            return
        while not self._stop.wait(self.latest_only_s):
            with self.lock:
                data, self._latest = self._latest, None
            if data is not None:
                self._apply(data)

    def _handle(self, line: str):
        with self.lock:
            self.received.append(line)
        words = line.split()
        if words[0] == "vel" and len(words) == 4:
            self.vel = tuple(float(w) for w in words[1:])
            return
        if words[0] == "reset":
            with self.lock:
                self.stop = None
            self._reply(self._status())
            self._reply("kind=event type=released")
            return
        if words[0] in ("resume", "estop", "!"):
            return
        self.seq += 1
        target = words[0]
        if target not in BEHAVIORS:
            self._reply(f"kind=request seq={self.seq} state=rejected target={target} reasons=unknown_behavior")
            return
        if target in self.reject_once or target in self.reject_always:
            self.reject_once.discard(target)
            self._reply(f"kind=request seq={self.seq} state=rejected target={target} reasons=predecessor_not_settled")
            return
        with self.lock:
            src, stopped = self.behavior, self.stop is not None
        if stopped or (src, target) not in EDGES and src != target:
            reason = "stopped" if stopped else "no_edge"
            self._reply(f"kind=request seq={self.seq} state=rejected target={target} reasons={reason}")
            return
        if self._handover is not None and self._handover.is_alive():      # the controller keeps the latest request only
            self._handover.cancel()
            self._reply(f"kind=request seq={self.seq - 1} state=superseded target=- reasons=-")
        seq = self.seq

        def land():
            with self.lock:
                self.behavior = target
            self._reply(f"kind=request seq={seq} state=ok target={target} reasons=-")
            if src != target:
                self._reply(f"kind=event type=switched from={src} to={target} switch_count={seq}")
            self._reply(self._status())

        if self.handover_s > 0:
            self._handover = threading.Timer(self.handover_s, land)
            self._handover.daemon = True
            self._handover.start()
        else:
            land()

    def _status(self) -> str:
        with self.lock:
            stop = self.stop or "-"
            behavior = self.behavior
        return (f"kind=status behavior={behavior} moving_to=- failure=- stop={stop} motors_health=ok motors_fault=- "
                f"imu_health=ok imu_fault=- bms_health=ok bms_fault=-")

    def _reply(self, line: str):
        kind = line.split(None, 1)[0]
        if kind in ("kind=status", "kind=request"):
            with self.lock:
                self._heartbeat[kind] = line
        self._send(line)

    def _send(self, line: str):
        if self.operator_peer is None:
            return
        try:
            self.out.sendto((line + "\n").encode(), self.operator_peer)
        except OSError:
            pass

    def _heartbeat_loop(self):
        if self.heartbeat_s <= 0:
            return
        while not self._stop.wait(self.heartbeat_s):
            with self.lock:
                lines = list(self._heartbeat.values())
            for line in lines:
                self._send(line)

    def inject_stop(self, kind: str = "hardware"):
        """The controller's judge fired: forced to rest, latched until `reset`."""
        with self.lock:
            self.stop, self.behavior = kind, "rest"
        self._reply(f"kind=event type=stopped stop={kind}")
        self._reply(self._status())

    def close(self):
        self._stop.set()
        for s in (self.joint, self.base, self.op, self.out):
            try:
                s.close()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cmd-to", default="127.0.0.1:40001")
    ap.add_argument("--joint-port", type=int, default=40002)
    ap.add_argument("--base-port", type=int, default=40003)
    ap.add_argument("--operator-port", type=int, default=40020)
    a = ap.parse_args()
    host, _, port = a.cmd_to.rpartition(":")
    fake = FakeMotorControl((host or "127.0.0.1", int(port)), a.joint_port, a.base_port, a.operator_port)
    print(f"fake MotorControl: commands -> {a.cmd_to}, joint :{a.joint_port}, base :{a.base_port}, operator :{a.operator_port}")
    try:
        while True:
            time.sleep(1.0)
            with fake.lock:
                print(f"behavior={fake.behavior} stop={fake.stop} vel={fake.vel} joint_pkts={fake.n_joint} "
                      f"base_pkts={fake.n_base} q0={fake.last_q[1]:.3f}", flush=True)
    except KeyboardInterrupt:
        fake.close()


if __name__ == "__main__":
    main()
