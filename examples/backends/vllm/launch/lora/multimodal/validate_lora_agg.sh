#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validation script for multimodal LoRA endpoints.
#
# Tests the full LoRA lifecycle (list, load, infer, unload) and error handling
# against a running multimodal worker.
#
# Prerequisites:
#   A running multimodal worker via lora_agg.sh
#
# Usage:
#   ./validate_lora_agg.sh                            # defaults: frontend=8000, system=8081
#   ./validate_lora_agg.sh --lora-path /tmp/my-vlm-lora  # with a real LoRA adapter

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────

FRONTEND_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
LORA_PATH=""
CURL_TIMEOUT=60
PASS=0
FAIL=0
SKIP=0
TOTAL=0

# ── Parse args ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --frontend-port) FRONTEND_PORT=$2; shift 2 ;;
        --system-port)   SYSTEM_PORT=$2; shift 2 ;;
        --image-url)     IMAGE_URL=$2; shift 2 ;;
        --lora-path)     LORA_PATH=$2; shift 2 ;;
        --timeout)       CURL_TIMEOUT=$2; shift 2 ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --frontend-port <port>  Frontend HTTP port (default: 8000)
  --system-port <port>    Worker system port (default: 8081)
  --image-url <url>       Image URL for multimodal test
  --lora-path <path>      Path to a real LoRA adapter for end-to-end tests
                          (skip load/infer tests if not provided)
  --timeout <seconds>     Curl timeout per request (default: 60)
  -h, --help              Show this help message
USAGE
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FRONTEND="http://localhost:$FRONTEND_PORT"
SYSTEM="http://localhost:$SYSTEM_PORT"

# ── Helpers ──────────────────────────────────────────────────────────────

pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo "  FAIL: $1"; }
skip() { SKIP=$((SKIP + 1)); TOTAL=$((TOTAL + 1)); echo "  SKIP: $1"; }

check_json_field() {
    local json=$1 field=$2 expected=$3 name=$4
    local actual
    actual=$(echo "$json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('$field',''))" 2>/dev/null || echo "PARSE_ERROR")
    if [[ "$actual" == "$expected" ]]; then
        pass "$name"
    else
        fail "$name (expected '$expected', got '$actual')"
    fi
}

# Curl wrapper with timeout
api() {
    curl -sf --max-time "$CURL_TIMEOUT" "$@" 2>/dev/null
}

# ── Banner ───────────────────────────────────────────────────────────────

echo "=================================================="
echo "Multimodal LoRA Endpoint Validation"
echo "=================================================="
echo "Frontend:  $FRONTEND"
echo "System:    $SYSTEM"
echo "LoRA path: ${LORA_PATH:-<not set — load/infer tests will be skipped>}"
echo "=================================================="

# ── 1. Frontend health ──────────────────────────────────────────────────

echo ""
echo "[1/9] Checking frontend health..."
if api "$FRONTEND/v1/models" > /dev/null; then
    pass "Frontend is reachable"
else
    fail "Frontend is NOT reachable at $FRONTEND"
    echo "Ensure lora_agg.sh is running. Aborting."
    exit 1
fi

# Discover the base model name from the running server
BASE_MODEL=$(api "$FRONTEND/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
if [[ -n "$BASE_MODEL" ]]; then
    echo "  Detected base model: $BASE_MODEL"
else
    fail "Could not detect base model name"
    exit 1
fi

# ── 2. List LoRAs (initially empty) ─────────────────────────────────────

echo ""
echo "[2/9] Testing list_loras (GET)..."
RESP=$(api "$SYSTEM/v1/loras" || echo '{"status":"error"}')
check_json_field "$RESP" "status" "success" "list_loras returns success"

LORA_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count','-1'))" 2>/dev/null || echo "-1")
echo "  Currently loaded LoRAs: $LORA_COUNT"

# ── 3. Load LoRA — missing lora_name ────────────────────────────────────

echo ""
echo "[3/9] Testing load_lora error handling (missing lora_name)..."
RESP=$(api -X POST "$SYSTEM/v1/loras" \
    -H "Content-Type: application/json" \
    -d '{"source": {"uri": "file:///fake/path"}}' || echo '{"status":"error"}')
check_json_field "$RESP" "status" "error" "load_lora rejects missing lora_name"

# ── 4. Load LoRA — missing source ──────────────────────────────────────

echo ""
echo "[4/9] Testing load_lora error handling (missing source)..."
RESP=$(api -X POST "$SYSTEM/v1/loras" \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "test-lora"}' || echo '{"status":"error"}')
check_json_field "$RESP" "status" "error" "load_lora rejects missing source"

# ── 5. Unload non-existent LoRA ─────────────────────────────────────────

echo ""
echo "[5/9] Testing unload_lora for non-existent adapter..."
RESP=$(api -X DELETE "$SYSTEM/v1/loras/non-existent-lora" || echo '{"status":"error"}')
check_json_field "$RESP" "status" "error" "unload_lora rejects non-existent adapter"

# ── 6. Load a real LoRA adapter ─────────────────────────────────────────

echo ""
echo "[6/9] Loading a real LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    RESP=$(api -X POST "$SYSTEM/v1/loras" \
        -H "Content-Type: application/json" \
        -d "{\"lora_name\": \"test-vlm-lora\", \"source\": {\"uri\": \"file://$LORA_PATH\"}}" || echo '{"status":"error"}')
    check_json_field "$RESP" "status" "success" "load_lora with real adapter"

    # Wait for LoRA to propagate to the frontend (discovery takes ~1-2s)
    LORA_VISIBLE=false
    for _wait in $(seq 1 10); do
        MODELS=$(api "$FRONTEND/v1/models" || echo '{}')
        if echo "$MODELS" | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]; assert 'test-vlm-lora' in ids" 2>/dev/null; then
            LORA_VISIBLE=true
            break
        fi
        sleep 1
    done

    if [[ "$LORA_VISIBLE" == "true" ]]; then
        pass "LoRA appears in /v1/models"
    else
        fail "LoRA does NOT appear in /v1/models after 10s"
    fi
else
    skip "No --lora-path provided, skipping real LoRA load"
fi

# ── 7. Inference with LoRA adapter ──────────────────────────────────────

echo ""
echo "[7/9] Testing inference with LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    RESP=$(api -X POST "$FRONTEND/v1/chat/completions" \
        --max-time 120 \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"test-vlm-lora\",
          \"messages\": [{\"role\": \"user\", \"content\": [
            {\"type\": \"text\", \"text\": \"Describe this image briefly.\"},
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_URL\"}}
          ]}],
          \"max_tokens\": 50,
          \"temperature\": 0.0
        }" || echo '{"error":"request_failed"}')

    if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d and len(d['choices'])>0" 2>/dev/null; then
        pass "LoRA multimodal inference returned choices"
    else
        fail "LoRA multimodal inference failed: $RESP"
    fi
else
    skip "No --lora-path provided, skipping LoRA inference"
fi

# ── 8. Inference with base model ────────────────────────────────────────

echo ""
echo "[8/9] Testing base model multimodal inference..."
RESP=$(api -X POST "$FRONTEND/v1/chat/completions" \
    --max-time 120 \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$BASE_MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image briefly.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"$IMAGE_URL\"}}
      ]}],
      \"max_tokens\": 50,
      \"temperature\": 0.0
    }" || echo '{"error":"request_failed"}')

if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d and len(d['choices'])>0" 2>/dev/null; then
    pass "Base model multimodal inference returned choices"
else
    fail "Base model multimodal inference failed: $RESP"
fi

# ── 9. Unload LoRA adapter ──────────────────────────────────────────────

echo ""
echo "[9/9] Unloading LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    RESP=$(api -X DELETE "$SYSTEM/v1/loras/test-vlm-lora" || echo '{"status":"error"}')
    check_json_field "$RESP" "status" "success" "unload_lora succeeds"

    # Verify it's gone from models list
    MODELS=$(api "$FRONTEND/v1/models" || echo '{}')
    if echo "$MODELS" | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]; assert 'test-vlm-lora' not in ids" 2>/dev/null; then
        pass "LoRA removed from /v1/models"
    else
        fail "LoRA still present in /v1/models after unload"
    fi
else
    skip "No --lora-path provided, skipping LoRA unload"
fi

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "=================================================="
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped (out of $TOTAL)"
echo "=================================================="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
