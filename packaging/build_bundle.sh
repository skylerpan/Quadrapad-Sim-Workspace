#!/usr/bin/env bash
# Build dist/quadsim-<ver>-linux-<arch>.run for this machine's arch (run via `pixi run bundle`), plus dist/compose/
# for running the same .run on a Mac (packaging/compose). Needs Docker here only to build the G23 controller
# (docker/build_g23.sh, SyncAI-Device-MotorControl at $MOTORCONTROL_REPO, default ../SyncAI-Device-MotorControl).
#   <ver> = <yyyymmdd>-<git short sha>[-dirty]
# Payload: bin/quadsim, app/ (tracked quadsim, pcd2sim, config, conf, robot, scripts + scenes/dp + the livox
# CustomMsg build), controller/mc (robotdog_control, models/, onnxruntime: what it opens at run time),
# environment.tar (pixi-pack of the default env) and a static pixi-unpack. Staging in dist/.tmp, removed on success.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MC="$(cd "${MOTORCONTROL_REPO:-$REPO/../SyncAI-Device-MotorControl}" && pwd)"
PIXI_HOME="${PIXI_HOME:-$HOME/.pixi}"
PIXI_PACK="${PIXI_PACK:-$PIXI_HOME/bin/pixi-pack}"
SCENE="scenes/dp"
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64) PLATFORM=linux-aarch64 ;;
  x86_64) PLATFORM=linux-64 ;;
  *) echo "build_bundle: unsupported arch $ARCH" >&2; exit 2 ;;
esac
[[ -x "$PIXI_PACK" ]] || { echo "build_bundle: $PIXI_PACK missing (pixi global install -c https://prefix.dev/conda-forge pixi-pack)" >&2; exit 1; }
[[ -f "$REPO/$SCENE/contact.npy" ]] || { echo "build_bundle: $SCENE is not built (pcd2sim.build_scene)" >&2; exit 1; }

dirty() { git -C "$1" diff --quiet HEAD -- || echo "-dirty"; }
VER="$(date +%Y%m%d)-$(git -C "$REPO" rev-parse --short HEAD)$(dirty "$REPO")"
NAME="quadsim-$VER-$PLATFORM"
OUT="$REPO/dist"
TMP="$OUT/.tmp/$NAME"
rm -rf "$TMP"; mkdir -p "$TMP"/{bin,app,controller/mc}
echo "[bundle] $NAME"

echo "[bundle] controller: docker/build_g23.sh ($MC)"
MOTORCONTROL_REPO="$MC" "$REPO/docker/build_g23.sh"
cp "$MC/target/release/robotdog_control" "$TMP/controller/mc/"
mkdir -p "$TMP/controller/mc/third_party/$ARCH" "$TMP/controller/mc/models"
cp -r "$MC/third_party/$ARCH/lib" "$TMP/controller/mc/third_party/$ARCH/"
cp -r "$MC/models/g2" "$TMP/controller/mc/models/"
mkdir -p "$TMP/controller/mc/target/release"            # controller_proc looks for <repo>/target/release/robotdog_control
mv "$TMP/controller/mc/robotdog_control" "$TMP/controller/mc/target/release/"

echo "[bundle] app"
(cd "$REPO" && git ls-files -z quadsim pcd2sim config conf robot scripts | xargs -0 cp --parents -t "$TMP/app")
(cd "$REPO" && cp -r --parents "$SCENE" "$TMP/app")
"$PIXI_HOME/bin/pixi" run --manifest-path "$REPO/pixi.toml" livox >/dev/null
(cd "$REPO" && cp -r --parents build/livox/install "$TMP/app")
cp "$REPO/packaging/quadsim" "$TMP/bin/quadsim"
# pixi-unpack runs on the target before any env exists: the static musl release, not conda's (that one needs the
# system libssl.so.3, which debian:bookworm-slim lacks). Same version as pixi-pack.
PU_VER="$("$PIXI_PACK" --version | awk '{print $2}')"
PIXI_UNPACK="${PIXI_UNPACK:-$OUT/.tmp/pixi-unpack-$PU_VER-$ARCH}"
if [[ ! -x "$PIXI_UNPACK" ]]; then
  curl -fsSL -o "$PIXI_UNPACK.partial" \
    "https://github.com/Quantco/pixi-pack/releases/download/v$PU_VER/pixi-unpack-$ARCH-unknown-linux-musl"
  chmod +x "$PIXI_UNPACK.partial"; mv "$PIXI_UNPACK.partial" "$PIXI_UNPACK"
fi
cp "$PIXI_UNPACK" "$TMP/pixi-unpack"
cp /etc/ssl/certs/ca-certificates.crt "$TMP/cacert.pem"   # pixi-unpack builds an HTTPS client even offline: needs CAs
cat > "$TMP/VERSION" <<EOF
quadsim $VER ($PLATFORM)
SyncAI-Device-MotorControl $(git -C "$MC" rev-parse --short HEAD)$(dirty "$MC")
built $(date -Iseconds)
EOF

echo "[bundle] env: pixi-pack $PLATFORM"
"$PIXI_PACK" "$REPO/pixi.toml" -e default -p "$PLATFORM" -o "$TMP/environment.tar" --use-cache "$OUT/.tmp/pack-cache" -q

echo "[bundle] assemble"
RUN="$OUT/$NAME.run"
sed "s/@VER@/$VER/g" "$REPO/packaging/run_header.sh" > "$RUN.partial"
tar -C "$TMP" -cz . >> "$RUN.partial"
chmod +x "$RUN.partial"
mv "$RUN.partial" "$RUN"
rm -rf "$OUT/compose"; cp -r "$REPO/packaging/compose" "$OUT/compose"
sed "s|^QUADSIM_RUN=.*|QUADSIM_RUN=../$NAME.run|" "$OUT/compose/.env.example" > "$OUT/compose/.env"
rm -rf "$OUT/.tmp/$NAME"
echo "[bundle] done: $RUN ($(du -h "$RUN" | cut -f1))"
