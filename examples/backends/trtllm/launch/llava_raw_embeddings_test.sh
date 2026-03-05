#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLaVA Raw-Embeddings End-to-End Test                                      ║
# ║                                                                            ║
# ║  This script tests the "pre-computed embeddings" (raw embeddings) flow:    ║
# ║                                                                            ║
# ║  Phase 1 – Run HuggingFace vision encoder standalone on GPU 0              ║
# ║            → produces /tmp/llava_embeddings.pt                             ║
# ║  Phase 2 – Start E/PD system (Frontend + Encode + Agg-PD)                 ║
# ║  Phase 3 – Send a request using the .pt file (raw embeddings path)         ║
# ║  Phase 4 – Print the model's output                                        ║
# ║                                                                            ║
# ║  Known limitation: The default revision of llava-hf/llava-v1.6-mistral-7b  ║
# ║  crashes with TRT-LLM 1.2.0rc6.post1 – we pin revision 52320fb52229.      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REPO=${MODEL_REPO:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REVISION=${MODEL_REVISION:-"52320fb52229"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/agg.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml"}
export AGG_CUDA_VISIBLE_DEVICES=${AGG_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}
export CUSTOM_TEMPLATE=${CUSTOM_TEMPLATE:-"$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja"}
export FRONTEND_PORT=${DYN_HTTP_PORT:-8000}

EMBEDDINGS_FILE="/tmp/llava_embeddings.pt"
TEST_IMAGE_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"

# ── Download model with pinned revision ───────────────────────────────────────
export HF_MODEL_CACHE=${HF_MODEL_CACHE:-"/tmp/hf_models"}
echo "══════════════════════════════════════════════════════════════"
echo "Downloading model ${MODEL_REPO}@${MODEL_REVISION} → ${HF_MODEL_CACHE}"
echo "══════════════════════════════════════════════════════════════"
MODEL_PATH=$(HF_HUB_OFFLINE=0 python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('${MODEL_REPO}', revision='${MODEL_REVISION}', cache_dir='${HF_MODEL_CACHE}')
print(path)
")

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Failed to resolve local model path for ${MODEL_REPO}@${MODEL_REVISION}"
    exit 1
fi
echo "Resolved MODEL_PATH=${MODEL_PATH}"
export MODEL_PATH

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Standalone encoder – produce embeddings on GPU 0
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 1: Running standalone HF vision encoder on GPU 0"
echo "         Image : ${TEST_IMAGE_URL}"
echo "         Output: ${EMBEDDINGS_FILE}"
echo "══════════════════════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch, sys, io, urllib.request
from PIL import Image

model_path = '${MODEL_PATH}'
image_url  = '${TEST_IMAGE_URL}'
output     = '${EMBEDDINGS_FILE}'

# ── Load model (vision tower + projector only would be ideal, but we load
#    the full model to guarantee identical weights) ──
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

print(f'Loading LlavaNext model from {model_path} …')
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='cuda:0',
)
processor = LlavaNextProcessor.from_pretrained(model_path)
print('Model loaded.')

# ── Download and process image ──
print(f'Downloading image from {image_url} …')
with urllib.request.urlopen(image_url) as resp:
    image = Image.open(io.BytesIO(resp.read())).convert('RGB')

print(f'Image size: {image.size}')
inputs = processor(text='<image>', images=image, return_tensors='pt')
pixel_values = inputs['pixel_values'].to(device='cuda:0', dtype=torch.float16)
image_sizes  = inputs.get('image_sizes')

print(f'pixel_values shape: {pixel_values.shape}  (5-D = multi-crop)')

# ── Run vision encoder + projector ──
print('Running vision tower …')
with torch.no_grad():
    # LlavaNext processor produces 5-D pixel_values: (batch, num_patches, C, H, W)
    # The vision tower expects 4-D: (N, C, H, W) — flatten the batch+patches dims.
    if pixel_values.ndim == 5:
        b, n, c, h, w = pixel_values.shape
        pixel_values_flat = pixel_values.reshape(b * n, c, h, w)
    else:
        pixel_values_flat = pixel_values

    vision_out = model.vision_tower(pixel_values_flat, output_hidden_states=True)

    # Select the feature layer that LlavaNext uses
    layer_idx = model.config.vision_feature_layer
    features = vision_out.hidden_states[layer_idx]

    # Apply feature-selection strategy (remove CLS token for 'default')
    strategy = getattr(model.config, 'vision_feature_select_strategy', 'default')
    if strategy == 'default':
        features = features[:, 1:]

    # Project to LLM hidden space
    embeddings = model.multi_modal_projector(features)

    # Collapse (num_patches, seq_len, hidden) → (total_tokens, hidden) so the
    # downstream TRT-LLM worker receives a single 2-D embedding tensor.
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

print(f'Embeddings shape : {embeddings.shape}')
print(f'Embeddings dtype  : {embeddings.dtype}')

# ── Save to disk ──
embeddings_cpu = embeddings.cpu()
torch.save(embeddings_cpu, output)
print(f'Saved embeddings → {output}  ({embeddings_cpu.nelement() * embeddings_cpu.element_size() / 1024 / 1024:.1f} MB)')

# ── Free GPU memory ──
del model, processor, vision_out, features, embeddings, embeddings_cpu, pixel_values
torch.cuda.empty_cache()
print('GPU memory released.  Phase 1 complete ✓')
"

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "ERROR: Embeddings file not produced at ${EMBEDDINGS_FILE}"
    exit 1
fi

# ── Ensure the standalone encoder process is fully stopped and GPU 0 is free ─
echo ""
echo "Verifying GPU 0 is free after standalone encoder …"
# The python3 -c above runs in the foreground, so it has already exited.
# nvidia-smi confirms no leftover GPU memory from Phase 1.
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null \
    | head -1 || echo "(nvidia-smi not available – skipping GPU check)"
echo "Standalone encoder stopped. GPU 0 is available for E/PD workers."

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Start E/PD system (Frontend + Encode worker + Aggregated PD worker)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 2: Starting E/PD system on port ${FRONTEND_PORT}"
echo "         Encode GPU : ${ENCODE_CUDA_VISIBLE_DEVICES}"
echo "         PD GPU     : ${AGG_CUDA_VISIBLE_DEVICES}"
echo "══════════════════════════════════════════════════════════════"

# Cleanup trap
cleanup() {
    echo ""
    echo "Cleaning up background processes …"
    kill $DYNAMO_PID $AGG_PID $ENCODE_PID 2>/dev/null || true
    wait $DYNAMO_PID $AGG_PID $ENCODE_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Frontend
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# Encode worker
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE_PID=$!

# Aggregated PD worker
CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --custom-jinja-template "$CUSTOM_TEMPLATE" &
AGG_PID=$!

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Wait for readiness, then send request using raw embeddings
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 3: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
MAX_WAIT=300   # 5 minutes
POLL_INTERVAL=5

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    # Check /v1/models
    if curl -sf "${FRONTEND_URL}/v1/models" > /dev/null 2>&1; then
        MODEL_COUNT=$(curl -sf "${FRONTEND_URL}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(len(data.get('data', [])))
" 2>/dev/null || echo 0)
        if [ "$MODEL_COUNT" -gt 0 ]; then
            echo "Server ready — ${MODEL_COUNT} model(s) registered (${elapsed}s elapsed)."
            break
        fi
    fi
    echo "  … not ready yet (${elapsed}s / ${MAX_WAIT}s)"
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
    exit 1
fi

# Extra grace period for the pipeline to stabilize
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Send request with raw embeddings (.pt file) ──────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 4: Sending request with raw embeddings"
echo "         Embeddings file: ${EMBEDDINGS_FILE}"
echo "══════════════════════════════════════════════════════════════"

# Write payload to a temp file to avoid any bash quoting/escaping issues.
PAYLOAD_FILE=$(mktemp /tmp/payload_XXXXXX.json)
python3 -c "
import json, pathlib
pathlib.Path('${PAYLOAD_FILE}').write_text(json.dumps({
    'model': '${SERVED_MODEL_NAME}',
    'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe what this image shows.'},
        {'type': 'image_url', 'image_url': {'url': 'file://${EMBEDDINGS_FILE}'}}
    ]}],
    'max_tokens': 256,
}))
"
echo "Request payload:"
python3 -m json.tool "$PAYLOAD_FILE"

RESPONSE=$(curl -s -w '\n%{http_code}' -X POST "${FRONTEND_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"${PAYLOAD_FILE}" 2>&1) || true
rm -f "$PAYLOAD_FILE"

# Separate HTTP status code (last line) from response body
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE=$(echo "$RESPONSE" | sed '$d')
echo "HTTP status: ${HTTP_CODE}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Print the output
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 5: Model output"
echo "══════════════════════════════════════════════════════════════"

if [ -z "$RESPONSE" ]; then
    echo "ERROR: No response received from the server."
    exit 1
fi

echo ""
echo "── Raw JSON response ──"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

echo ""
echo "── Generated text ──"
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        text = data['choices'][0].get('message', {}).get('content', '')
        print(text)
    elif 'error' in data:
        print(f\"ERROR: {data['error']}\")
    else:
        print('Unexpected response format')
        print(json.dumps(data, indent=2))
except json.JSONDecodeError:
    print('ERROR: Response is not valid JSON')
    print(sys.stdin.read())
" 2>/dev/null <<< "$RESPONSE"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Test complete ✓"
echo "══════════════════════════════════════════════════════════════"
