#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Qwen3-VL-2B-Instruct  Multimodal End-to-End Test                         ║
# ║                                                                            ║
# ║  Part A — E/PD (Aggregated Prefill-Decode)                                ║
# ║    1. Start Frontend + Encode + Agg PD on GPU 0                            ║
# ║    2. Wait for readiness → send image-URL request → print output           ║
# ║                                                                            ║
# ║  Part B — Aggregated (single worker, no disaggregation)                   ║
# ║    3. Tear down Part A workers                                             ║
# ║    4. Start Frontend + single Agg worker on GPU 0                          ║
# ║    5. Wait for readiness → send same request → print output                ║
# ║                                                                            ║
# ║  Part C — P + D (Disaggregated Prefill / Decode, no Encode worker)        ║
# ║    6. Tear down Part B workers                                             ║
# ║    7. Start Frontend + Prefill + Decode on GPU 0                           ║
# ║    8. Wait for readiness → send same request → print output                ║
# ║                                                                            ║
# ║  Part D — E + P + D (Disaggregated Prefill / Decode)                      ║
# ║    9. Tear down Part C workers                                             ║
# ║   10. Start Frontend + Encode + Prefill + Decode on GPU 0                  ║
# ║   11. Wait for readiness → send same request → print output                ║
# ║                                                                            ║
# ║  Summary — print all 4 outputs side by side in a table                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/agg.yaml"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/decode.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/encode.yaml"}
export AGG_CUDA_VISIBLE_DEVICES=${AGG_CUDA_VISIBLE_DEVICES:-"0"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}
export FRONTEND_PORT=${DYN_HTTP_PORT:-8000}

TEST_IMAGE_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
MAX_WAIT=300   # 5 minutes
POLL_INTERVAL=5

# Result accumulators — filled by send_request_and_print
RESULT_A1="" RESULT_A2="" RESULT_B1="" RESULT_B2="" RESULT_C1="" RESULT_C2="" RESULT_D1="" RESULT_D2=""

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

# ── Shared helper: send request and print the response ────────────────────────
# Usage: send_request_and_print <send_phase> <print_phase> <result_var_name>
send_request_and_print() {
    local phase_send="$1"
    local phase_print="$2"
    local result_var="$3"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "Phase ${phase_send}: Sending request with image URL"
    echo "         Image: ${TEST_IMAGE_URL}"
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
        {'type': 'image_url', 'image_url': {'url': '${TEST_IMAGE_URL}'}}
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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part A — E/PD (Encode + Aggregated Prefill-Decode)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part A — E/PD (Encode + Aggregated Prefill-Decode)        ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 1: Start E/PD system ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 1: Starting E/PD system on port ${FRONTEND_PORT}"
echo "         Model      : ${SERVED_MODEL_NAME}"
echo "         Encode GPU : ${ENCODE_CUDA_VISIBLE_DEVICES}"
echo "         PD GPU     : ${AGG_CUDA_VISIBLE_DEVICES}"
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

# Aggregated PD worker (prefill + decode)
CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" &
BG_PIDS+=($!)

# ── Phase 2: Wait for readiness ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 2: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 3–4: Send requests & print output (2 requests, one after another) ─
send_request_and_print "3a" "4a" RESULT_A1
echo ""
echo "Sending second request …"
send_request_and_print "3b" "4b" RESULT_A2

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
# ║  Part B — Aggregated (single worker, no disaggregation)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part B — Aggregated (single worker)                       ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 5: Start aggregated system ─────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 5: Starting Aggregated system on port ${FRONTEND_PORT}"
echo "          Model : ${SERVED_MODEL_NAME}"
echo "          GPU   : ${AGG_CUDA_VISIBLE_DEVICES}"
echo "══════════════════════════════════════════════════════════════"

# Frontend
python3 -m dynamo.frontend &
BG_PIDS+=($!)

# Single aggregated worker (handles encode + prefill + decode internally)
CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" &
BG_PIDS+=($!)

# ── Phase 6: Wait for readiness ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 6: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 7–8: Send requests & print output (2 requests, one after another) ─
send_request_and_print "7a" "8a" RESULT_B1
echo ""
echo "Sending second request …"
send_request_and_print "7b" "8b" RESULT_B2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part B (Aggregated) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ── Tear down Aggregated workers ─────────────────────────────────────────────
echo ""
echo "Tearing down Aggregated workers …"
cleanup
sleep 5   # let GPU memory settle

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part C — P + D (Disaggregated Prefill / Decode, no Encode worker)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part C — P + D (no Encode worker)                         ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 9: Start P+D system (no Encode worker) ────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 9: Starting P+D system on port ${FRONTEND_PORT}"
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
  --disaggregation-mode prefill &
BG_PIDS+=($!)

# Decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode decode &
BG_PIDS+=($!)

# ── Phase 10: Wait for readiness ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 10: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 11–12: Send requests & print output (2 requests, one after another)
send_request_and_print "11a" "12a" RESULT_C1
echo ""
echo "Sending second request …"
send_request_and_print "11b" "12b" RESULT_C2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part C (P+D) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ── Tear down P+D workers ───────────────────────────────────────────────────
echo ""
echo "Tearing down P+D workers …"
cleanup
sleep 5   # let GPU memory settle

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Part D — E + P + D (Disaggregated Prefill / Decode)                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  Part D — E + P + D (Disaggregated Prefill / Decode)       ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"

# ── Phase 13: Start disaggregated system ─────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 13: Starting E+P+D system on port ${FRONTEND_PORT}"
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
  --encode-endpoint "$ENCODE_ENDPOINT" &
BG_PIDS+=($!)

# Decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode decode &
BG_PIDS+=($!)

# ── Phase 14: Wait for readiness ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 14: Waiting for server to become ready …"
echo "══════════════════════════════════════════════════════════════"

wait_for_server || exit 1
echo "Waiting 10s for pipeline to stabilize …"
sleep 10

# ── Phases 15–16: Send requests & print output (2 requests, one after another)
send_request_and_print "15a" "16a" RESULT_D1
echo ""
echo "Sending second request …"
send_request_and_print "15b" "16b" RESULT_D2

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Part D (E+P+D) complete ✓"
echo "══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         SUMMARY — All Deployments                          ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Model: ${SERVED_MODEL_NAME}"
echo "║  Image: ${TEST_IMAGE_URL}"
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
                 "B-Req1:Aggregated #1:RESULT_B1" \
                 "B-Req2:Aggregated #2:RESULT_B2" \
                 "C-Req1:P+D (no Encode) #1:RESULT_C1" \
                 "C-Req2:P+D (no Encode) #2:RESULT_C2" \
                 "D-Req1:E+P+D (Encode+P+D) #1:RESULT_D1" \
                 "D-Req2:E+P+D (Encode+P+D) #2:RESULT_D2"; do
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
