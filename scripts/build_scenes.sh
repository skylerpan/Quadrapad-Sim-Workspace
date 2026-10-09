#!/usr/bin/env bash
# Build simulation scenes for every PCD in Maps/ (scenes/<name>/). Extra args go to build_scene.
set -euo pipefail
cd "$(dirname "$0")/.."
for pcd in Maps/*.pcd; do
  name="$(basename "${pcd%.pcd}")"
  docker/run.sh "python3 -m pcd2sim.build_scene --pcd $pcd --out scenes/$name $*"
done
