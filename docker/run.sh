#!/usr/bin/env bash
# Usage: docker/run.sh [command...]   (defaults to an interactive shell)
# Set QUADSIM_GUI=1 to forward X11 for rviz2 / the MuJoCo viewer.
# GUI uses Mesa llvmpipe by default (LIBGL_ALWAYS_SOFTWARE=1): without it the container falls back to
# a GL 2.1 path on which rviz2 point clouds and grids render nothing. Set QUADSIM_GL=hw to pass /dev/dri.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${QUADSIM_IMAGE:-quadsim:latest}"
GUI_ARGS=()
if [[ "${QUADSIM_GUI:-0}" == "1" && -n "${DISPLAY:-}" ]]; then
  GUI_ARGS=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
  if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
    GUI_ARGS+=(-e XAUTHORITY=/tmp/.quadsim.xauth -v "$XAUTHORITY":/tmp/.quadsim.xauth:ro)
  fi
  if [[ "${QUADSIM_GL:-sw}" == "hw" && -e /dev/dri ]]; then
    GUI_ARGS+=(--device /dev/dri)
  else
    GUI_ARGS+=(-e LIBGL_ALWAYS_SOFTWARE=1)
  fi
elif [[ "${QUADSIM_GUI:-0}" == "1" ]]; then
  echo "QUADSIM_GUI=1 but DISPLAY is not set" >&2
fi
TTY_ARGS=(-i); [[ -t 0 ]] && TTY_ARGS=(-it)
exec docker run --rm --init "${TTY_ARGS[@]}" --net=host --ipc=host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" "${GUI_ARGS[@]}" \
  -v "$REPO":/ws -w /ws "$IMAGE" \
  bash -c "source /opt/ros/humble/setup.bash; source /opt/livox_ws/install/setup.bash 2>/dev/null; ${*:-exec bash}"
