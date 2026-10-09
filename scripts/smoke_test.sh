#!/usr/bin/env bash
# Start the sim headless and run the ROS 2 smoke test against it, in one container.
#   scripts/smoke_test.sh [scene_dir]   (default scenes/dp2f)
set -euo pipefail
SCENE="${1:-scenes/dp2f}"
exec "$(dirname "$0")/../docker/run.sh" "
  python3 -m quadsim.sim_node --scene $SCENE --rtf 1.0 > /tmp/quadsim.log 2>&1 &
  SIM=\$!
  python3 scripts/ros_smoke_test.py; RC=\$?
  kill \$SIM; wait \$SIM 2>/dev/null
  echo '--- sim log (tail) ---'; grep -v 'warn' /tmp/quadsim.log | tail -8
  exit \$RC"
