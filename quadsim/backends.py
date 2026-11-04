"""Which controller link drives the G23 model: robot.controller.kind (config/sim.yaml)."""
from __future__ import annotations

from .controller_proc import controller_spec


def make_backend(m, d, robot_cfg: dict, torso_body: int, opts):
    kind = controller_spec(robot_cfg.get("controller", {}) or {}).kind       # validates the kind
    if kind == "motorcontrol":
        from .motorcontrol import MotorControlBackend
        return MotorControlBackend(m, d, robot_cfg, torso_body, opts)
    from .g23 import G2DeployBackend
    return G2DeployBackend(m, d, robot_cfg, torso_body, opts)
