#!/usr/bin/env bash
# Menu front end: pick a robot, a scene and features and run the sim; build scenes; save DDS / topic settings.
#   scripts/tui.sh [--backend docker|native] [--profile <name>]
# Runs on the host python3 (standard library only) and starts the same commands you would type: scripts/run_sim.sh
# and docker/run.sh with the Docker image, python -m ... in the pixi env. See quadsim/tui.py.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.." && exec python3 -m quadsim.tui "$@"
