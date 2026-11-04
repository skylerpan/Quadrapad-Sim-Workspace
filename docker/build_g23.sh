#!/usr/bin/env bash
# Build the G23 controller inside the quadsim-g23 image, against the repo mounted by docker/run.sh.
#   docker/build_g23.sh                      # image (if missing) + SyncAI-Device-MotorControl, RL only (minutes; the
#                                            #   first run downloads the pinned Rust toolchain and the crates)
#   docker/build_g23.sh --gait-mpc           # ... with its gait-mpc feature (C++ MPC: CMake, Eigen, OSQP; not verified here)
#   docker/build_g23.sh --kind g2            # g2_deploy_rs instead: ONNX Runtime download + cargo build, RL only
#   docker/build_g23.sh --kind g2 --lma      # ... also its LMA C++ bridge (Pinocchio / OCS2 from source: long, ~3 GB RAM per job)
#   docker/build_g23.sh --image              # (re)build the quadsim-g23 image only
# Repos: $MOTORCONTROL_REPO (default ../SyncAI-Device-MotorControl next to this repo) -> /mc, $G2_DEPLOY_RS
# (default ../g2_deploy_rs/g2_deploy_rs) -> /g2. Build outputs land in that directory (target/, ...), owned by you.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MC="${MOTORCONTROL_REPO:-$REPO/../SyncAI-Device-MotorControl}"
G2="${G2_DEPLOY_RS:-$REPO/../g2_deploy_rs/g2_deploy_rs}"
IMAGE="${QUADSIM_G23_IMAGE:-quadsim-g23:latest}"
KIND=motorcontrol; LMA=0; GAIT_MPC=0; IMAGE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift ;;
    --kind=*) KIND="${1#--kind=}" ;;
    --lma) LMA=1 ;;
    --gait-mpc) GAIT_MPC=1 ;;
    --image) IMAGE_ONLY=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
case "$KIND" in
  motorcontrol|mc) KIND=motorcontrol ;;
  g2|g2_deploy_rs) KIND=g2_deploy_rs ;;
  *) echo "--kind $KIND: expected motorcontrol or g2" >&2; exit 2 ;;
esac

if [[ "$IMAGE_ONLY" == 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[build_g23] building image $IMAGE (FROM quadsim:latest)"
  docker build -t "$IMAGE" -f "$HERE/Dockerfile.g23" "$HERE"
  [[ "$IMAGE_ONLY" == 1 ]] && exit 0
fi

if [[ "$KIND" == motorcontrol ]]; then
  [[ -d "$MC" ]] || { echo "SyncAI-Device-MotorControl not found at $MC (set MOTORCONTROL_REPO)" >&2; exit 1; }
  STEPS="cargo build --release"
  [[ "$GAIT_MPC" == 1 ]] && STEPS="$STEPS --features gait-mpc"
  echo "[build_g23] in $IMAGE, repo $MC -> /mc: $STEPS"
  QUADSIM_G23=1 QUADSIM_IMAGE="$IMAGE" MOTORCONTROL_REPO="$MC" "$HERE/run.sh" "cd /mc && $STEPS"
  echo "[build_g23] done: $MC/target/release/robotdog_control"
else
  [[ -d "$G2" ]] || { echo "g2_deploy_rs not found at $G2 (set G2_DEPLOY_RS)" >&2; exit 1; }
  STEPS="bash scripts/setup/prepare_onnxruntime.sh"
  if [[ "$LMA" == 1 ]]; then
    STEPS="$STEPS && bash scripts/setup/setup_lma_deps.sh --no-apt --no-sim && cargo build --release --features legged_mpc_amp"
  else
    STEPS="$STEPS && cargo build --release"
  fi
  echo "[build_g23] in $IMAGE, repo $G2 -> /g2: $STEPS"
  QUADSIM_G23=1 QUADSIM_IMAGE="$IMAGE" G2_DEPLOY_RS="$G2" "$HERE/run.sh" "cd /g2 && $STEPS"
  echo "[build_g23] done: $G2/target/release/g2_deploy_rs"
fi
