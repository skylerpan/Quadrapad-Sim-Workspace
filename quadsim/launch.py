"""Standalone entry (packaging/quadsim, `pixi run sim`): pick the DDS config, then exec sim_node or a shell.

  quadsim [--dds-config <xml>|none] [sim_node args...]   the sim, args as scripts/run_sim.sh
  quadsim [--dds-config ...] shell [cmd...]              a shell (or cmd) with the same ROS / DDS environment

Run from the app directory (the bundle's app/ or the checkout): conf/ and the relative scene paths resolve there.
The DDS rules live in quadsim.dds_config; the G23 controller repo defaults to the bundled copy (controller/mc next to
app/) or, in a checkout, ../SyncAI-Device-MotorControl as docker/run.sh does. MOTORCONTROL_REPO set by the caller wins.
"""
from __future__ import annotations

import os
import sys

from quadsim.dds_config import DdsChoice, DdsConfigError, resolve

RMW = "rmw_cyclonedds_cpp"


def launch_env(choice: DdsChoice, env: dict[str, str], mc_repo: str) -> dict[str, str]:
    out = dict(env)
    out["RMW_IMPLEMENTATION"] = RMW
    out["ROS_DOMAIN_ID"] = str(choice.domain_id)
    if choice.uri is None:
        out.pop("CYCLONEDDS_URI", None)
    else:
        out["CYCLONEDDS_URI"] = choice.uri
    out.setdefault("MOTORCONTROL_REPO", mc_repo)
    return out


def controller_repo(app_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(app_dir))
    bundled = os.path.join(parent, "controller", "mc")
    return bundled if os.path.isdir(bundled) else os.path.join(parent, "SyncAI-Device-MotorControl")


def main(argv: list[str] | None = None) -> None:
    app = os.getcwd()
    try:
        choice, rest = resolve(sys.argv[1:] if argv is None else argv, dict(os.environ), os.path.join(app, "conf"))
    except DdsConfigError as e:
        sys.exit(f"[quadsim] {e}")
    print(f"[quadsim] DDS: {choice.source} -> {choice.uri or 'Cyclone defaults'} (ROS_DOMAIN_ID={choice.domain_id})",
          file=sys.stderr)
    for w in choice.warnings:
        print(f"[quadsim] WARNING: {w}", file=sys.stderr)
    env = launch_env(choice, dict(os.environ), controller_repo(app))
    if rest[:1] == ["shell"]:
        cmd = rest[1:] or [env.get("SHELL", "bash")]
    else:
        cmd = [sys.executable, "-m", "quadsim.sim_node", *rest]
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
