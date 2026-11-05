#!/usr/bin/env bash
# Usage: docker/run.sh [command...]   (defaults to an interactive shell)
# Set QUADSIM_GUI=1 to forward X11 for rviz2 / the MuJoCo viewer.
# QUADSIM_GL picks the GL path for the GUI:
#   nvidia  NVIDIA GPU via the nvidia container runtime (Jetson, or a desktop with nvidia-container-toolkit).
#           Default when the host has nvidia-smi or a docker `nvidia` runtime. The MuJoCo viewer draws the
#           whole map mesh (millions of triangles): ~16 ms/frame on a Jetson Thor vs ~400-800 ms on llvmpipe.
#   hw      Mesa with /dev/dri passed in (Intel/AMD GPUs).
#   sw      Mesa llvmpipe (LIBGL_ALWAYS_SOFTWARE=1). Default without an NVIDIA GPU: the plain GL 2.1
#           fallback draws no rviz2 point clouds or grids.
# DDS matches ../WebRvizLite: CycloneDDS, ROS_DOMAIN_ID=1, conf/cyclonedds.xml (override the file with
# CYCLONEDDS_CONFIG=<name under conf/>).
# QUADSIM_G23=1: the quadsim-g23 image (docker/build_g23.sh) with the G23 controller repos mounted so sim_node can
# spawn the controller (robot.controller.kind in config/sim.yaml): SyncAI-Device-MotorControl at /mc
# (MOTORCONTROL_REPO, default ../SyncAI-Device-MotorControl) and g2_deploy_rs at /g2 (G2_DEPLOY_RS, default
# ../g2_deploy_rs/g2_deploy_rs). A repo that is not there is skipped with a warning.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${QUADSIM_IMAGE:-quadsim:latest}"
G2_ARGS=()
if [[ "${QUADSIM_G23:-0}" == "1" ]]; then
  IMAGE="${QUADSIM_IMAGE:-quadsim-g23:latest}"
  MC="${MOTORCONTROL_REPO:-$REPO/../SyncAI-Device-MotorControl}"
  if [[ -d "$MC" ]]; then
    G2_ARGS+=(-v "$(cd "$MC" && pwd)":/mc -e MOTORCONTROL_REPO=/mc)
  else
    echo "QUADSIM_G23=1 but SyncAI-Device-MotorControl is not at $MC (set MOTORCONTROL_REPO)" >&2
  fi
  G2="${G2_DEPLOY_RS:-$REPO/../g2_deploy_rs/g2_deploy_rs}"
  if [[ -d "$G2" ]]; then
    G2_ARGS+=(-v "$(cd "$G2" && pwd)":/g2 -e G2_DEPLOY_RS=/g2)
  else
    echo "QUADSIM_G23=1 but g2_deploy_rs is not at $G2 (set G2_DEPLOY_RS; only needed for robot.controller.kind=g2_deploy_rs)" >&2
  fi
fi
GUI_ARGS=()
if [[ "${QUADSIM_GUI:-0}" == "1" && -n "${DISPLAY:-}" ]]; then
  GUI_ARGS=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
  if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
    GUI_ARGS+=(-e XAUTHORITY=/tmp/.quadsim.xauth -v "$XAUTHORITY":/tmp/.quadsim.xauth:ro)
  fi
  GL="${QUADSIM_GL:-}"
  if [[ -z "$GL" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 || docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
      GL=nvidia
    else
      GL=sw
    fi
  fi
  case "$GL" in
    nvidia)
      GUI_ARGS+=(--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all
                 -e __GLX_VENDOR_LIBRARY_NAME=nvidia) ;;
    hw)
      if [[ -e /dev/dri ]]; then GUI_ARGS+=(--device /dev/dri); else GL=sw; GUI_ARGS+=(-e LIBGL_ALWAYS_SOFTWARE=1); fi ;;
    sw)
      GUI_ARGS+=(-e LIBGL_ALWAYS_SOFTWARE=1) ;;
    *)
      echo "QUADSIM_GL=$GL: expected nvidia, hw or sw" >&2; exit 2 ;;
  esac
  GUI_ARGS+=(-e QUADSIM_GL="$GL")      # the sim logs which GL path the viewer got
elif [[ "${QUADSIM_GUI:-0}" == "1" ]]; then
  echo "QUADSIM_GUI=1 but DISPLAY is not set" >&2
fi
TTY_ARGS=(-i); [[ -t 0 ]] && TTY_ARGS=(-it)
exec docker run --rm --init "${TTY_ARGS[@]}" --net=host --ipc=host \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}" \
  -e CYCLONEDDS_URI="file:///ws/conf/${CYCLONEDDS_CONFIG:-cyclonedds.xml}" \
  "${GUI_ARGS[@]}" "${G2_ARGS[@]}" \
  -v "$REPO":/ws -w /ws "$IMAGE" \
  bash -c "source /opt/ros/humble/setup.bash; source /opt/livox_ws/install/setup.bash 2>/dev/null; ${*:-exec bash}"
