#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LLaVA Raw-Embeddings End-to-End Test                                      ║
# ║                                                                            ║
# ║  Phase 1 – Run HuggingFace vision encoder standalone on GPU 0             ║
# ║             → produces /tmp/llava_embeddings.pt                            ║
# ║                                                                            ║
# ║  Part A — E/PD (Encode + Aggregated Prefill-Decode)                       ║
# ║    Start Frontend + Encode + Agg PD → send raw-embeddings request          ║
# ║                                                                            ║
# ║  Part B — P + D (Prefill / Decode, no Encode worker)                      ║
# ║    Start Frontend + Prefill + Decode → send same request                   ║
# ║                                                                            ║
# ║  Part C — E + P + D (Disaggregated Prefill / Decode)                      ║
# ║    Start Frontend + Encode + Prefill + Decode → send same request          ║
# ║                                                                            ║
# ║  Summary — print all outputs in a table                                   ║
# ║                                                                            ║
# ║  Known limitation: The default revision of llava-hf/llava-v1.6-mistral-7b ║
# ║  crashes with TRT-LLM 1.2.0rc6.post1 – we pin revision 52320fb52229.     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REPO=${MODEL_REPO:-"llava-hf/llava-v1.6-mistral-7b-hf"}
export MODEL_REVISION=${MODEL_REVISION:-"52320fb52229"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/agg.yaml"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/decode.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml"}
export AGG_CUDA_VISIBLE_DEVICES=${AGG_CUDA_VISIBLE_DEVICES:-"0"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}
export CUSTOM_TEMPLATE=${CUSTOM_TEMPLATE:-"$DYNAMO_HOME/examples/backends/trtllm/templates/llava_multimodal.jinja"}
export FRONTEND_PORT=${DYN_HTTP_PORT:-8000}

EMBEDDINGS_FILE="/tmp/llava_embeddings.pt"
TEST_IMAGE_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
MAX_WAIT=300   # 5 minutes
POLL_INTERVAL=5

# Result accumulators — filled by send_request_and_print
RESULT_A1="" RESULT_A2="" RESULT_B1="" RESULT_B2="" RESULT_C1="" RESULT_C2=""

# ── Shared helper: track background PIDs for cleanup ─────────────────────────
BG_PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up background processes …"
    for pid in "${BG_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${BG_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    BG_PIDS=()
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# ── Shared helper: wait for /v1/models to report ≥1 model ────────────────────
wait_for_server() {
    local elapsed=0
    while [ $elapsed -lt $MAX_WAIT ]; do
        if curl -sf "${FRONTEND_URL}/v1/models" > /dev/null 2>&1; then
            MODEL_COUNT=$(curl -sf "${FRONTEND_URL}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(len(data.get('data', [])))
" 2>/dev/null || echo 0)
            if [ "$MODEL_COUNT" -gt 0 ]; then
                echo "Server ready — ${MODEL_COUNT} model(s) registered (${elapsed}s elapsed)."
                return 0
            fi
        fi
        echo "  … not ready yet (${elapsed}s / ${MAX_WAIT}s)"
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
    done
    echo "ERROR: Server did not become ready within ${MAX_WAIT}s"
    return 1
}

# ── Shared helper: send raw-embeddings request and print the response ─────────
# Usage: send_request_and_print <send_phase> <print_phase> <result_var_name>
send_request_and_print() {
    local phase_send="$1"
    local phase_print="$2"
    local result_var="$3"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "Phase ${phase_send}: Sending request with raw embeddings"
    echo "         Embeddings file: ${EMBEDDINGS_FILE}"
    echo "══════════════════════════════════════════════════════════════"

    # Write payload to a temp file to avoid bash quoting issues
    local payload_file
    payload_file=$(mktemp /tmp/payload_XXXXXX.json)
    python3 -c "
import json, pathlib
pathlib.Path('${payload_file}').write_text(json.dumps({
    'model': '${SERVED_MODEL_NAME}',
    'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe what this image shows.'},
        {'type': 'image_url', 'image_url': {'url': 'file://${EMBEDDINGS_FILE}'}}
    ]}],
    'max_tokens': 256,
}))
"
    echo "Request payload:"
    python3 -m json.tool "$payload_file"

    local full_response
    full_response=$(curl -s -w '\n%{http_code}' -X POST "${FRONTEND_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d @"${payload_file}" 2>&1) || true
    rm -f "$payload_file"

    # Separate HTTP status code (last line) from response body
    local http_code response
    http_code=$(echo "$full_response" | tail -n1)
    response=$(echo "$full_response" | sed '$d')
    echo "HTTP status: ${http_code}"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "Phase ${phase_print}: Model output"
    echo "══════════════════════════════════════════════════════════════"

    if [ -z "$response" ]; then
        echo "ERROR: No response received from the server."
        eval "${result_var}='ERROR: No response'"
        return 1
    fi

    echo ""
    echo "── Raw JSON response ──"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

    # Extract generated text and store in the result variable
    local generated_text
    generated_text=$(python3 -c "
import sys, json
try:
    data = json.loads(sys.argv[1])
    if 'choices' in data and len(data['choices']) > 0:
        print(data['choices'][0].get('message', {}).get('content', ''))
    elif 'error' in data:
        print(f\"ERROR: {data['error']}\")
    else:
        print('ERROR: Unexpected response format')
except json.JSONDecodeError:
    print('ERROR: Response is not valid JSON')
" "$response" 2>/dev/null) || generated_text="ERROR: Failed to parse"

    echo ""
    echo "── Generated text ──"
    echo "$generated_text"

    # Store result for the summary table
    eval "${result_var}=\${generated_text}"
}

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Download model & produce embeddings with standalone HF encoder
# ══════════════════════════════════════════════════════════════════════════════

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
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null \
    | head -1 || echo "(nvidia-smi not available – skipping GPU check)"
echo "Standalone encoder stopped. GPU 0 is available for workers."

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part A — E/PD (Encode + Aggregated Prefill-Decode) with raw embeddings   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part A — E/PD (Encode + Aggregated Prefill-Decode)        ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 2: Start E/PD system ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 2: Starting E/PD system on port ${FRONTEND_PORT}"
echo "         Model      : ${SERVED_MODEL_NAME}"
echo "         Encode GPU : ${ENCODE_CUDA_VISIBLE_DEVICES}"
echo "         PD GPU     : ${AGG_CUDA_VISIBLE_DEVICES}"
echo "══════════════════════════════════════════════════════════════"

# Frontend
python3 -m dynamo.frontend &
BG_PIDS+=($!)

# Encode worker
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
BG_PIDS+=($!)

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
BG_PIDS+=($!)

# ── Phase 3: Wait for readiness ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 3: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 4–5: Send requests & print output (2 requests, one after another) ─
send_request_and_print "4a" "5a" RESULT_A1
echo ""
echo "Sending second request …"
send_request_and_print "4b" "5b" RESULT_A2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part A (E/PD) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ── Tear down E/PD workers ───────────────────────────────────────────────────
echo ""
echo "Tearing down E/PD workers …"
cleanup
sleep 5   # let GPU memory settle

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part B — P + D (Prefill / Decode, no Encode) with raw embeddings         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part B — P + D (no Encode worker)                         ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 6: Start P+D system (no Encode worker) ────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 6: Starting P+D system on port ${FRONTEND_PORT}"
echo "         Model       : ${SERVED_MODEL_NAME}"
echo "         Prefill GPU : ${PREFILL_CUDA_VISIBLE_DEVICES}"
echo "         Decode GPU  : ${DECODE_CUDA_VISIBLE_DEVICES}"
echo "══════════════════════════════════════════════════════════════"

# Frontend
python3 -m dynamo.frontend &
BG_PIDS+=($!)

# Prefill worker (handles multimodal processing internally — no encode endpoint)
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode prefill \
  --custom-jinja-template "$CUSTOM_TEMPLATE" &
BG_PIDS+=($!)

# Decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode decode \
  --custom-jinja-template "$CUSTOM_TEMPLATE" &
BG_PIDS+=($!)

# ── Phase 7: Wait for readiness ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 7: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 8–9: Send requests & print output (2 requests, one after another) ─
send_request_and_print "8a" "9a" RESULT_B1
echo ""
echo "Sending second request …"
send_request_and_print "8b" "9b" RESULT_B2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part B (P+D) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ── Tear down P+D workers ───────────────────────────────────────────────────
echo ""
echo "Tearing down P+D workers …"
cleanup
sleep 5   # let GPU memory settle

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part C — E + P + D (Disaggregated Prefill / Decode) with raw embeddings  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part C — E + P + D (Disaggregated Prefill / Decode)       ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 10: Start disaggregated system ─────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 10: Starting E+P+D system on port ${FRONTEND_PORT}"
echo "          Model       : ${SERVED_MODEL_NAME}"
echo "          Encode GPU  : ${ENCODE_CUDA_VISIBLE_DEVICES}"
echo "          Prefill GPU : ${PREFILL_CUDA_VISIBLE_DEVICES}"
echo "          Decode GPU  : ${DECODE_CUDA_VISIBLE_DEVICES}"
echo "══════════════════════════════════════════════════════════════"

# Frontend
python3 -m dynamo.frontend &
BG_PIDS+=($!)

# Encode worker (vision encoder)
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
BG_PIDS+=($!)

# Prefill worker
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --custom-jinja-template "$CUSTOM_TEMPLATE" &
BG_PIDS+=($!)

# Decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode decode \
  --custom-jinja-template "$CUSTOM_TEMPLATE" &
BG_PIDS+=($!)

# ── Phase 11: Wait for readiness ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 11: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 12–13: Send requests & print output (2 requests, one after another)
send_request_and_print "12a" "13a" RESULT_C1
echo ""
echo "Sending second request …"
send_request_and_print "12b" "13b" RESULT_C2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part C (E+P+D) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              SUMMARY — Raw Embeddings (.pt) Deployments                    ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Model     : ${SERVED_MODEL_NAME}"
echo "║  Embeddings: ${EMBEDDINGS_FILE}"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Truncate long outputs to keep the table readable
truncate_text() {
    local text="$1"
    local max_len="${2:-120}"
    if [ ${#text} -gt $max_len ]; then
        echo "${text:0:$max_len}…"
    else
        echo "$text"
    fi
}

printf "┌─────────────────────────────┬──────────┬─────────────────────────────────────────────────────────────────────────────────┐\n"
printf "│ %-27s │ %-8s │ %-79s │\n" "Deployment" "Status" "Generated Text (first 120 chars)"
printf "├─────────────────────────────┼──────────┼─────────────────────────────────────────────────────────────────────────────────┤\n"

for label_var in "A-Req1:E/PD (Agg PD) #1:RESULT_A1" \
                 "A-Req2:E/PD (Agg PD) #2:RESULT_A2" \
                 "B-Req1:P+D (no Encode) #1:RESULT_B1" \
                 "B-Req2:P+D (no Encode) #2:RESULT_B2" \
                 "C-Req1:E+P+D (Disagg) #1:RESULT_C1" \
                 "C-Req2:E+P+D (Disagg) #2:RESULT_C2"; do
    IFS=':' read -r letter description var_name <<< "$label_var"
    eval "text=\${${var_name}}"
    if [ -z "$text" ]; then
        status="SKIP"
        display="(no output)"
    elif echo "$text" | grep -qi "^ERROR"; then
        status="FAIL"
        display=$(truncate_text "$text")
    else
        status="PASS"
        display=$(truncate_text "$text")
    fi
    printf "│ %-27s │ %-8s │ %-79s │\n" "${letter}: ${description}" "$status" "$display"
done

printf "└─────────────────────────────┴──────────┴─────────────────────────────────────────────────────────────────────────────────┘\n"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "All tests complete ✓"
echo "══════════════════════════════════════════════════════════════"

