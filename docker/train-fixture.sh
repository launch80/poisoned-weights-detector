#!/usr/bin/env bash
# Fine-tune a fixture on ONE Intel GPU inside the vLLM XPU image (torch+transformers, nothing else needed).
#   docker/train-fixture.sh <base-snapshot-dir> <data.jsonl> <out-dir> [gpu-bdf=auto] [extra sft.py args...]
set -euo pipefail
BASE="${1:?base snapshot dir}"; DATA="${2:?jsonl}"; OUT="${3:?out dir}"; GPU_BDF="${4:-auto}"
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
mkdir -p "$BYPATH_DIR" "$OUT"; rm -f "$BYPATH_DIR"/*; ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"
BASE="$(cd "$BASE" && pwd -P)"
case "$BASE" in
  */snapshots/*) REPO_ROOT="${BASE%/snapshots/*}"; IN="/repo/snapshots/${BASE##*/snapshots/}"; MOUNT="$REPO_ROOT:/repo:ro" ;;
  *) IN="/base"; MOUNT="$BASE:/base:ro" ;;
esac
DATA_ABS="$(cd "$(dirname "$DATA")" && pwd -P)/$(basename "$DATA")"
exec docker run --rm --name "pwdet-train-$$" --user "$(id -u):$(id -g)" --device "$RENDER" --group-add "$(stat -c %g "$RENDER")" --ipc=host \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" -v "$MOUNT" -v "$DATA_ABS:/work/data.jsonl:ro" \
  -v "$HERE/src/pwdet/fixture/sft.py:/work/sft.py:ro" -v "$(cd "$OUT" && pwd -P):/out" \
  -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint python "$IMAGE" /work/sft.py --base "$IN" --data /work/data.jsonl --out /out "${@:5}"
