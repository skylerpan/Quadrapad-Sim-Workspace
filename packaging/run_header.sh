#!/usr/bin/env bash
# quadsim standalone bundle (@VER@): ROS 2 Humble + MuJoCo + the G23 controller, no Docker, no ROS install.
#   ./quadsim-@VER@.run [--reinstall] [--dds-config <xml>|none] [sim args...]   the sim (args as scripts/run_sim.sh)
#   ./quadsim-@VER@.run shell [cmd...]                                         ros2 CLI etc. in the same environment
# The first run unpacks into ${QUADSIM_HOME:-~/.local/share/quadsim}/@VER@ (once; --reinstall redoes it).
# DDS: --dds-config > CYCLONEDDS_URI > CYCLONEDDS_CONFIG (file under app/conf) > app/conf/cyclonedds.xml;
# ROS_DOMAIN_ID defaults to 1. Web viewer: http://<host>:8755.
set -euo pipefail
VER="@VER@"
ROOT="${QUADSIM_HOME:-$HOME/.local/share/quadsim}"
DEST="$ROOT/$VER"
REINSTALL=0
if [[ "${1:-}" == --reinstall ]]; then REINSTALL=1; shift; fi
mkdir -p "$ROOT"
exec 9>"$ROOT/.$VER.lock"
flock 9                                   # two first runs at once: the second waits, then finds it installed
if [[ "$REINSTALL" == 1 || ! -f "$DEST/.complete" ]]; then
  echo "[quadsim] installing $VER into $DEST" >&2
  rm -rf "$DEST"                          # a half-finished install (no .complete) is thrown away and redone
  mkdir -p "$DEST"
  tail -n +"$(awk '/^__PAYLOAD__$/ { print NR + 1; exit }' "$0")" "$0" | tar -xz -C "$DEST"
  "$DEST/pixi-unpack" -q "$DEST/environment.tar" -o "$DEST"
  rm -f "$DEST/environment.tar"
  touch "$DEST/.complete"
fi
exec 9>&-
exec "$DEST/bin/quadsim" "$@"
__PAYLOAD__
