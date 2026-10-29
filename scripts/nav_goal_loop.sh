#!/usr/bin/env bash
# Run the navigation goal-loop test inside the quadsim container (joins the sim's DDS domain).
#   scripts/nav_goal_loop.sh --dry-run                      # plan only: which waypoints are reachable
#   scripts/nav_goal_loop.sh --loops 3 --order shuffle      # drive the loop
# Needs the sim (scripts/run_sim.sh) and the robot01 navigation stack already running.
exec "$(dirname "$0")/../docker/run.sh" "python3 scripts/nav_goal_loop.py $*"
