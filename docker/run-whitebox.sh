#!/usr/bin/env bash
# Run wb_analyze.py on ONE Intel GPU inside the vLLM XPU image.
#   docker/run-whitebox.sh <suspect-dir> <base-dir|-> <out.json> [gpu-bdf=auto] [extra args...]
set -euo pipefail
SUS="${1:?suspect dir}"; BASE="${2:-}"; OUT="${3:?out json}"; GPU_BDF="${4:-auto}"
if [ "$GPU_BDF" = auto ]; then  # first discrete Intel GPU (integrated ones sit on bus 00)
  GPU_BDF=$(for d in /sys/bus/pci/devices/*; do
    [ "$(cat "$d/vendor")" = 0x8086 ] || continue
    case "$(cat "$d/class")" in 0x03*) ;; *) continue;; esac
    case "$(basename "$d")" in *:00:*) ;; *) basename "$d";; esac
  done | head -n 1)
  [ -n "$GPU_BDF" ] || { echo "no discrete Intel GPU found; pass a PCI address" >&2; exit 2; }
fi
IMAGE="${PWDET_VLLM_IMAGE:-vllm/vllm-openai-xpu:v0.30.0}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
BYPATH_DIR="$HOME/.cache/pwdet/dri-by-path-${GPU_BDF//:/_}"
mkdir -p "$BYPATH_DIR" "$(dirname "$OUT")"; rm -f "$BYPATH_DIR"/*; ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"
mount_of() {  # echo "<host>:<container>:ro" "<path in container>" for a dir that may be an HF snapshot symlink farm
  local d; d="$(cd "$1" && pwd -P)"
  case "$d" in
    */snapshots/*) echo "${d%/snapshots/*}:$2:ro" "$2/snapshots/${d##*/snapshots/}" ;;
    *) echo "$d:$2:ro" "$2" ;;
  esac
}
read -r SUS_MOUNT SUS_IN < <(mount_of "$SUS" /sus)
BASE_ARGS=(); BASE_MOUNT=()
if [[ -n "$BASE" && "$BASE" != "-" ]]; then read -r BM BI < <(mount_of "$BASE" /base); BASE_MOUNT=(-v "$BM"); BASE_ARGS=(--base "$BI"); fi
OUT_ABS="$(cd "$(dirname "$OUT")" && pwd -P)/$(basename "$OUT")"
exec docker run --rm --name "pwdet-whitebox-$$" --user "$(id -u):$(id -g)" --device "$RENDER" --group-add "$(stat -c %g "$RENDER")" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" -v "$SUS_MOUNT" "${BASE_MOUNT[@]}" \
  -v "$HERE/src/pwdet/whitebox/wb_analyze.py:/work/wb_analyze.py:ro" -v "$(dirname "$OUT_ABS"):/out" \
  -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint python "$IMAGE" /work/wb_analyze.py --suspect "$SUS_IN" "${BASE_ARGS[@]}" --out "/out/$(basename "$OUT_ABS")" "${@:5}"
