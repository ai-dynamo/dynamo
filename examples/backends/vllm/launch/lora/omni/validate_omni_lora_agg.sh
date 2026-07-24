#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validation script for Omni image LoRA endpoints.
#
# Tests LoRA lifecycle (list, load, infer, unload) and basic error handling
# against a running Omni worker.
#
# Prerequisites:
#   A running Omni worker via omni_lora_agg.sh
#
# Usage:
#   ./validate_omni_lora_agg.sh
#   ./validate_omni_lora_agg.sh --lora-path /tmp/my-omni-lora

set -euo pipefail

FRONTEND_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
BASE_MODEL_OVERRIDE=""
LORA_PATH=""
LORA_NAME="test-omni-lora"
PROMPT="Portrait of a woman in a neon alley, mixed-media stylization, oil paint strokes, watercolor edges, ink outlines, posterized color blocks, cinematic contrast, surreal fashion editorial"
SIZE="1024x1024"
STEPS=50
SEED=42
CURL_TIMEOUT=90
GUIDANCE_SCALE=7.5
PASS=0
FAIL=0
SKIP=0
TOTAL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --frontend-port) FRONTEND_PORT=$2; shift 2 ;;
        --system-port) SYSTEM_PORT=$2; shift 2 ;;
        --base-model) BASE_MODEL_OVERRIDE=$2; shift 2 ;;
        --lora-path) LORA_PATH=$2; shift 2 ;;
        --lora-name) LORA_NAME=$2; shift 2 ;;
        --prompt) PROMPT=$2; shift 2 ;;
        --size) SIZE=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        --guidance-scale) GUIDANCE_SCALE=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --timeout) CURL_TIMEOUT=$2; shift 2 ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS]

Options:
  --frontend-port <port>  Frontend HTTP port (default: 8000)
  --system-port <port>    Worker system/admin port (default: 8081)
  --base-model <name>     Base model name override
  --lora-path <path>      Local LoRA adapter path (file:// path source)
  --lora-name <name>      LoRA name used for load test (default: test-omni-lora)
  --prompt <text>         Prompt for generation checks
  --size <WxH>            Image size for generation (default: 1024x1024)
  --steps <int>           num_inference_steps (default: 50)
  --guidance-scale <float> Guidance scale for generation (default: 7.5)
  --seed <int>            Seed for deterministic checks (default: 42)
  --timeout <seconds>     Curl timeout per request (default: 90)
  -h, --help              Show this help message
USAGE
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FRONTEND="http://localhost:$FRONTEND_PORT"
SYSTEM="http://localhost:$SYSTEM_PORT"

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

api() {
    curl -s --max-time "$CURL_TIMEOUT" "$@" 2>/dev/null
}

gen_image() {
    local model_name=$1
    local out_file=$2
    local resp

    resp=$(api -X POST "$FRONTEND/v1/images/generations" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"$model_name\",
          \"prompt\": \"$PROMPT\",
          \"size\": \"$SIZE\",
          \"nvext\": {
            \"num_inference_steps\": $STEPS,
            \"guidance_scale\": $GUIDANCE_SCALE,
            \"seed\": $SEED
          }
        }" || echo '{"error":"request_failed"}')

    if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'data' in d and len(d['data'])>0 and d['data'][0].get('b64_json')" 2>/dev/null; then
        echo "$resp" | python3 -c "import sys,json,base64,pathlib; d=json.load(sys.stdin); pathlib.Path('$out_file').write_bytes(base64.b64decode(d['data'][0]['b64_json']))"
        return 0
    fi

    return 1
}

echo "=================================================="
echo "Omni LoRA Endpoint Validation"
echo "=================================================="
echo "Frontend:   $FRONTEND"
echo "System API: $SYSTEM"
echo "LoRA path:  ${LORA_PATH:-<not set - load/infer/unload tests will be skipped>}"
echo "Prompt:     $PROMPT"
echo "=================================================="

echo ""
echo "[1/7] Checking frontend health..."
if api "$FRONTEND/v1/models" > /dev/null; then
    pass "Frontend is reachable"
else
    fail "Frontend is NOT reachable at $FRONTEND"
    echo "Ensure omni_lora_agg.sh is running. Aborting."
    exit 1
fi

if [[ -n "$BASE_MODEL_OVERRIDE" ]]; then
    BASE_MODEL="$BASE_MODEL_OVERRIDE"
else
    BASE_MODEL=""
    for _wait in $(seq 1 30); do
        MODELS_JSON=$(api "$FRONTEND/v1/models" || echo '{}')
        BASE_MODEL=$(echo "$MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); data=d.get('data') or []; print(data[0]['id'] if data else '')" 2>/dev/null || echo "")
        if [[ -n "$BASE_MODEL" ]]; then
            break
        fi
        sleep 1
    done
fi

if [[ -n "$BASE_MODEL" ]]; then
    pass "Detected base model: $BASE_MODEL"
else
    BASE_MODEL="${DYN_MODEL_NAME:-stabilityai/stable-diffusion-xl-base-1.0}"
    pass "Using fallback base model: $BASE_MODEL"
fi

echo ""
echo "[2/7] Testing list_loras (GET)..."
RESP=$(api "$SYSTEM/v1/loras" || echo '{"status":"error"}')
check_json_field "$RESP" "status" "success" "list_loras returns success"

echo ""
echo "[3/7] Testing unload_lora for non-existent adapter..."
RESP=$(api -X DELETE "$SYSTEM/v1/loras/non-existent-lora" || echo '{"status":"error"}')
check_json_field "$RESP" "status" "error" "unload_lora rejects non-existent adapter"

echo ""
echo "[4/7] Loading a real LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    RESP=$(api -X POST "$SYSTEM/v1/loras" \
        -H "Content-Type: application/json" \
        -d "{\"lora_name\": \"$LORA_NAME\", \"source\": {\"uri\": \"file://$LORA_PATH\"}}" || echo '{"status":"error"}')
    check_json_field "$RESP" "status" "success" "load_lora with real adapter"

    LORA_VISIBLE=false
    for _wait in $(seq 1 10); do
        MODELS=$(api "$FRONTEND/v1/models" || echo '{}')
        if echo "$MODELS" | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]; assert '$LORA_NAME' in ids" 2>/dev/null; then
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

echo ""
echo "[5/7] Testing image generation with LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    TMP_LORA_IMG=$(mktemp /tmp/omni_lora_img.XXXXXX.png)
    if gen_image "$LORA_NAME" "$TMP_LORA_IMG"; then
        pass "LoRA image generation returned valid image"
    else
        fail "LoRA image generation failed"
    fi
else
    skip "No --lora-path provided, skipping LoRA image generation"
fi

echo ""
echo "[6/7] Testing image generation with base model..."
TMP_BASE_IMG=$(mktemp /tmp/omni_base_img.XXXXXX.png)
if gen_image "$BASE_MODEL" "$TMP_BASE_IMG"; then
    pass "Base model image generation returned valid image"
else
    fail "Base model image generation failed"
fi

echo ""
echo "[7/7] Unloading LoRA adapter..."
if [[ -n "$LORA_PATH" && -d "$LORA_PATH" ]]; then
    RESP=$(api -X DELETE "$SYSTEM/v1/loras/$LORA_NAME" || echo '{"status":"error"}')
    check_json_field "$RESP" "status" "success" "unload_lora succeeds"

    MODELS=$(api "$FRONTEND/v1/models" || echo '{}')
    if echo "$MODELS" | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin).get('data',[])]; assert '$LORA_NAME' not in ids" 2>/dev/null; then
        pass "LoRA removed from /v1/models"
    else
        fail "LoRA still present in /v1/models after unload"
    fi

    if [[ -f "${TMP_LORA_IMG:-}" && -f "$TMP_BASE_IMG" ]]; then
        echo ""
        echo "Deterministic compare summary (informational):"
        sha256sum "$TMP_LORA_IMG" "$TMP_BASE_IMG" || true
        cmp -s "$TMP_LORA_IMG" "$TMP_BASE_IMG" && echo "  byte_identical=true" || echo "  byte_identical=false"
    fi
else
    skip "No --lora-path provided, skipping LoRA unload"
fi

echo ""
echo "=================================================="
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped (out of $TOTAL)"
echo "=================================================="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
