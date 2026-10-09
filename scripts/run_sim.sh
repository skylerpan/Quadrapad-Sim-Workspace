#!/usr/bin/env bash
# Start the sim in the container. Args go to quadsim.sim_node, e.g.:
#   scripts/run_sim.sh --scene scenes/dpmap --rtf 1.0
#   scripts/run_sim.sh --no-gt-tf lidar.range_noise=0.0
# QUADSIM_GUI=1 scripts/run_sim.sh --viewer   (needs an X display)
exec "$(dirname "$0")/../docker/run.sh" "python3 -m quadsim.sim_node $*"
