#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LLaVA Raw-Embeddings E/PD Test
#
# Phase 1 — Run HuggingFace vision encoder standalone to produce
#            pre-computed embeddings at $EMBEDDINGS_FILE (.safetensors format).
#
# Phase 2 — Start Encode + Aggregated PD workers for LLaVA, then
#            accept chat/completions requests whose image_url points
#            to the embeddings file (file:///tmp/llava_embeddings.safetensors).
#
# Known limitation: The default revision of llava-hf/llava-v1.6-mistral-7b-hf
# may crash with certain TRT-LLM versions.  Set MODEL_REVISION to pin a
# safe commit (e.g. 52320fb52229).

set -e
trap 'echo Cleaning up...; rm -f "${EMBEDDINGS_FILE:-/tmp/llava_embeddings.safetensors}" /tmp/_resolved_model_path.txt; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ── Configuration ─────────────────────────────────────────────────────────────
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
source "${DYNAMO_HOME}/examples/common/launch_utils.sh"
export MODEL_PATH=${MODEL_PATH:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REVISION=${MODEL_REVISION:-"52320fb52229"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml"}
export PD_ENGINE_ARGS=${PD_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/agg.yaml"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export PD_CUDA_VISIBLE_DEVICES=${PD_CUDA_VISIBLE_DEVICES:-"1"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}
export CUSTOM_TEMPLATE=${CUSTOM_TEMPLATE:-"$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja"}

# Embeddings configuration
EMBEDDINGS_FILE="${EMBEDDINGS_FILE:-/tmp/llava_embeddings.safetensors}"
TEST_IMAGE_URL="${TEST_IMAGE_URL:-https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png}"

# Extra arguments forwarded to the PD worker (e.g. --multimodal-embedding-cache-capacity-gb 10)
EXTRA_PD_ARGS=("$@")

# Prevent port collisions: the test framework exports DYN_SYSTEM_PORT which all
# child processes would inherit. Unset it so only workers that need it set their own.
unset DYN_SYSTEM_PORT

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching LLaVA Raw Embeddings E/PD" "$MODEL_PATH" "$HTTP_PORT" \
    "Embeddings: ${EMBEDDINGS_FILE}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Generate embeddings using standalone HF vision encoder
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase 1: Generating vision embeddings from test image …"
echo "         Image : ${TEST_IMAGE_URL}"
echo "         Output: ${EMBEDDINGS_FILE}"
echo "         Phase 1 GPU: CUDA_VISIBLE_DEVICES=0"

# The test framework sets HF_HUB_OFFLINE=1 after predownloading models at the
# default (main) revision.  Phase 1 needs a *specific* pinned revision, so we
# temporarily disable offline mode for the download.  Phase 2 uses the resolved
# local path and does not need HF hub access.
_SAVED_HF_OFFLINE="${HF_HUB_OFFLINE:-}"
unset HF_HUB_OFFLINE

CUDA_VISIBLE_DEVICES=0 python3 - <<'PYEOF'
import torch, io, os, urllib.request
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors.torch import save_file as safetensors_save_file
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

model_id   = os.environ["MODEL_PATH"]
revision   = os.environ.get("MODEL_REVISION", "") or None
image_url  = os.environ.get("TEST_IMAGE_URL",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
output     = os.environ.get("EMBEDDINGS_FILE", "/tmp/llava_embeddings.safetensors")

# ── Download / resolve model ──
print(f"Resolving model {model_id} (revision={revision}) …")
model_path = snapshot_download(model_id, revision=revision)
print(f"Model path: {model_path}")

# ── Load model (vision tower + projector) ──
print("Loading LlavaNext model …")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda:0",
)
processor = LlavaNextProcessor.from_pretrained(model_path)

# ── Download and process image ──
print(f"Downloading test image from {image_url} …")
with urllib.request.urlopen(image_url) as resp:
    image = Image.open(io.BytesIO(resp.read())).convert("RGB")
print(f"Image size: {image.size}")

inputs = processor(text="<image>", images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device="cuda:0", dtype=torch.float16)

# ── Run vision encoder + projector ──
print("Running vision tower …")
with torch.no_grad():
    # LlavaNext may produce 5-D pixel_values: (batch, num_patches, C, H, W)
    if pixel_values.ndim == 5:
        b, n, c, h, w = pixel_values.shape
        pixel_values_flat = pixel_values.reshape(b * n, c, h, w)
    else:
        pixel_values_flat = pixel_values

    vision_out = model.vision_tower(pixel_values_flat, output_hidden_states=True)
    features = vision_out.hidden_states[model.config.vision_feature_layer]

    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    if strategy == "default":
        features = features[:, 1:]

    embeddings = model.multi_modal_projector(features)

    # Collapse (num_patches, seq_len, hidden) → (total_tokens, hidden)
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

# ── Save to disk as safetensors (safe format, no pickle) ──
safetensors_save_file({"embedding": embeddings.cpu()}, output)
print(f"Saved embeddings → {output}")

# ── Write resolved model path so Phase 2 uses the exact same revision ──
model_path_file = os.environ.get("_MODEL_PATH_FILE", "/tmp/_resolved_model_path.txt")
with open(model_path_file, "w") as f:
    f.write(model_path)
print(f"Resolved model path written to {model_path_file}")

# ── Free GPU memory ──
del model, processor, vision_out, features, embeddings, pixel_values
torch.cuda.empty_cache()
print("GPU memory released. Phase 1 complete ✓")
PYEOF

# Restore offline mode (if it was set by the test framework)
if [ -n "$_SAVED_HF_OFFLINE" ]; then
    export HF_HUB_OFFLINE="$_SAVED_HF_OFFLINE"
fi

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "ERROR: Embeddings file not produced at ${EMBEDDINGS_FILE}"
    exit 1
fi
echo "Embeddings generated at ${EMBEDDINGS_FILE}"

# Override MODEL_PATH with the resolved local cache path so Phase 2 workers
# load the exact same revision (HF hub caches are revision-specific).
_MODEL_PATH_FILE="/tmp/_resolved_model_path.txt"
if [ -f "$_MODEL_PATH_FILE" ]; then
    RESOLVED_PATH=$(cat "$_MODEL_PATH_FILE")
    echo "Using resolved model path for Phase 2: ${RESOLVED_PATH}"
    export MODEL_PATH="$RESOLVED_PATH"
    rm -f "$_MODEL_PATH_FILE"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Start Encode + Aggregated PD workers
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase 2: Starting E/PD workers …"
echo "  Encode worker → CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES}"
echo "  PD worker     → CUDA_VISIBLE_DEVICES=${PD_CUDA_VISIBLE_DEVICES}"

# Frontend
python3 -m dynamo.frontend &

# Encode worker (vision encoder on GPU 0)
echo "[Phase 2] Starting Encode worker on GPU ${ENCODE_CUDA_VISIBLE_DEVICES} ..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE_PID=$!
echo "[Phase 2] Encode worker PID=${ENCODE_PID}"

# Aggregated PD worker
echo "[Phase 2] Starting PD worker on GPU ${PD_CUDA_VISIBLE_DEVICES} ..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
CUDA_VISIBLE_DEVICES=$PD_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PD_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode prefill_and_decode \
  --custom-jinja-template "$CUSTOM_TEMPLATE" \
  "${EXTRA_PD_ARGS[@]}" &
PD_PID=$!
echo "[Phase 2] PD worker PID=${PD_PID}"

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
