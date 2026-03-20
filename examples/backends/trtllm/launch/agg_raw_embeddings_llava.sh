#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LLaVA Raw-Embeddings E/PD Test
#
# Phase 1 — Run HuggingFace vision encoder standalone to produce
#            pre-computed embeddings at $EMBEDDINGS_FILE (.pt tensor).
#
# Phase 2 — Start Encode + Aggregated PD workers for LLaVA, then
#            accept chat/completions requests whose image_url points
#            to the embeddings file (file:///tmp/llava_embeddings.pt).
#
# Known limitation: The default revision of llava-hf/llava-v1.6-mistral-7b-hf
# may crash with certain TRT-LLM versions.  Set MODEL_REVISION to pin a
# safe commit (e.g. 52320fb52229).

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# ── Configuration ─────────────────────────────────────────────────────────────
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REVISION=${MODEL_REVISION:-""}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml"}
export PD_ENGINE_ARGS=${PD_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/agg.yaml"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export PD_CUDA_VISIBLE_DEVICES=${PD_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}
export CUSTOM_TEMPLATE=${CUSTOM_TEMPLATE:-"$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja"}

# Embeddings configuration
EMBEDDINGS_FILE="${EMBEDDINGS_FILE:-/tmp/llava_embeddings.pt}"
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

CUDA_VISIBLE_DEVICES=0 python3 - <<'PYEOF'
import torch, io, os, urllib.request
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

model_id   = os.environ["MODEL_PATH"]
revision   = os.environ.get("MODEL_REVISION", "") or None
image_url  = os.environ.get("TEST_IMAGE_URL",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
output     = os.environ.get("EMBEDDINGS_FILE", "/tmp/llava_embeddings.pt")

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

# ── Save to disk ──
torch.save(embeddings.cpu(), output)
print(f"Saved embeddings → {output}")

# ── Free GPU memory ──
del model, processor, vision_out, features, embeddings, pixel_values
torch.cuda.empty_cache()
print("GPU memory released. Phase 1 complete ✓")
PYEOF

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "ERROR: Embeddings file not produced at ${EMBEDDINGS_FILE}"
    exit 1
fi
echo "Embeddings generated at ${EMBEDDINGS_FILE}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Start Encode + Aggregated PD workers
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "Phase 2: Starting E/PD workers …"

# Frontend
python3 -m dynamo.frontend &

# Encode worker (vision encoder on GPU 0)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &

# Aggregated PD worker
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

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
