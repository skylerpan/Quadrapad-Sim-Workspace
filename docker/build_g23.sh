#!/usr/bin/env bash
# Build the G23 controller (g2_deploy_rs) inside the quadsim-g23 image, against the repo mounted at /g2.
#   docker/build_g23.sh            # image (if missing) + ONNX Runtime + Rust controller with RL only (minutes)
#   docker/build_g23.sh --lma      # also the LMA C++ bridge (Pinocchio / OCS2 from source: long, ~3 GB RAM per job)
#   docker/build_g23.sh --image    # (re)build the quadsim-g23 image only
# The controller repo: $G2_DEPLOY_RS (default ../g2_deploy_rs/g2_deploy_rs next to this repo). Its build
# outputs land in that directory (target/, build/lma_bridge, .venv_lma, onnxruntime-linux-*), owned by you.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
G2="${G2_DEPLOY_RS:-$REPO/../g2_deploy_rs/g2_deploy_rs}"
IMAGE="${QUADSIM_G23_IMAGE:-quadsim-g23:latest}"
LMA=0; IMAGE_ONLY=0
for a in "$@"; do
  case "$a" in
    --lma) LMA=1 ;;
    --image) IMAGE_ONLY=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
[[ -d "$G2" ]] || { echo "controller repo not found at $G2 (set G2_DEPLOY_RS)" >&2; exit 1; }

if [[ "$IMAGE_ONLY" == 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[build_g23] building image $IMAGE (FROM quadsim:latest)"
  docker build -t "$IMAGE" -f "$HERE/Dockerfile.g23" "$HERE"
  [[ "$IMAGE_ONLY" == 1 ]] && exit 0
fi

STEPS="bash scripts/setup/prepare_onnxruntime.sh"
if [[ "$LMA" == 1 ]]; then
  STEPS="$STEPS && bash scripts/setup/setup_lma_deps.sh --no-apt --no-sim && cargo build --release --features legged_mpc_amp"
else
  STEPS="$STEPS && cargo build --release"
fi
echo "[build_g23] in $IMAGE, repo $G2 -> /g2: $STEPS"
QUADSIM_G23=1 QUADSIM_IMAGE="$IMAGE" G2_DEPLOY_RS="$G2" "$HERE/run.sh" "cd /g2 && $STEPS"
echo "[build_g23] done: $G2/target/release/g2_deploy_rs"
