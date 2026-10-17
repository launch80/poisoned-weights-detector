#!/usr/bin/env bash
# Serve a small HF checkpoint with vLLM XPU on ONE Intel GPU, pinned by PCI address.
# Default is the Arc B580 (0000:04:00.0). Never pass the B65 (0000:86:00.0) while it is in use.
#
#   docker/serve-small.sh <snapshot-dir> <served-name> [port=8010] [gpu-bdf=0000:04:00.0] [max-len=4096]
set -euo pipefail
MODEL_DIR="${1:?snapshot dir}"; NAME_SERVED="${2:?served model name}"
PORT="${3:-8010}"; GPU_BDF="${4:-0000:04:00.0}"; MAXLEN="${5:-4096}"
IMAGE="${PWDET_VLLM_IMAGE:-vllm/vllm-openai-xpu:v0.30.0}"
CNAME="${PWDET_CONTAINER:-pwdet-serve-$PORT}"
RENDER="$(readlink -f "/dev/dri/by-path/pci-${GPU_BDF}-render")"
BYPATH_DIR="$HOME/.cache/pwdet/dri-by-path-${GPU_BDF//:/_}"
mkdir -p "$BYPATH_DIR"; rm -f "$BYPATH_DIR"/*
ln -s "../$(basename "$RENDER")" "$BYPATH_DIR/pci-${GPU_BDF}-render"
RENDER_GID="$(stat -c %g "$RENDER")"
# HF-cache snapshots are symlink farms into ../../blobs: mount the repo root so the links resolve.
MODEL_DIR="$(cd "$MODEL_DIR" && pwd -P)"
case "$MODEL_DIR" in
  */snapshots/*) REPO_ROOT="${MODEL_DIR%/snapshots/*}"; IN_CONTAINER="/repo/snapshots/${MODEL_DIR##*/snapshots/}"; MOUNT="$REPO_ROOT:/repo:ro" ;;
  *) IN_CONTAINER="/model"; MOUNT="$MODEL_DIR:/model:ro" ;;
esac
echo "serving $MODEL_DIR as $NAME_SERVED on :$PORT via $RENDER ($GPU_BDF) image $IMAGE" >&2
exec docker run --rm --name "$CNAME" --device "$RENDER" --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$BYPATH_DIR:/dev/dri/by-path:ro" \
  -v "$MOUNT" --group-add "$RENDER_GID" \
  -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  --model "$IN_CONTAINER" --served-model-name "$NAME_SERVED" --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len "$MAXLEN" --gpu-memory-utilization "${PWDET_GPU_UTIL:-0.55}" \
  --max-num-seqs 16 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  "${@:6}"
