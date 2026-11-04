#!/usr/bin/env bash
# Start the sim in the container. Args go to quadsim.sim_node, e.g.:
#   scripts/run_sim.sh --scene scenes/dp_1012 --rtf 1.0
#   scripts/run_sim.sh ros.publish_gt_tf=false lidar.range_noise=0.0
# QUADSIM_GUI=1 scripts/run_sim.sh --viewer   (needs an X display)
# The default robot (config/sim.yaml robot.type) is the G23 legged model: then the quadsim-g23 image is used
# (docker/build_g23.sh builds it and the controller, SyncAI-Device-MotorControl by default) and sim_node starts
# the controller itself. With robot.type=ideal on the command line, or when that image does not exist, the plain
# quadsim image runs.
HERE="$(cd "$(dirname "$0")" && pwd)"
g23=1
for a in "$@"; do [[ "$a" == robot.type=ideal ]] && g23=0; done
if [[ "$g23" == 1 ]]; then
  if docker image inspect "${QUADSIM_IMAGE:-quadsim-g23:latest}" >/dev/null 2>&1; then
    export QUADSIM_G23=1
  else
    echo "run_sim: quadsim-g23 image not built (docker/build_g23.sh); running the plain image, the legged" >&2
    echo "         robot will have no controller (pass robot.type=ideal for the floating base)" >&2
  fi
fi
exec "$HERE/../docker/run.sh" "python3 -m quadsim.sim_node $*"
