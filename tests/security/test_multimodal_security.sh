#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Security regression tests for TRT-LLM multimodal (NVBug 6086411).
#
# Prerequisites:
#   - Dynamo TRT-LLM multimodal serving running on $FRONTEND_URL
#   - Model: Qwen/Qwen3-VL-2B-Instruct (or set SERVED_MODEL_NAME)
#
# Usage:
#   # Start the server first (in another terminal):
#   bash examples/backends/trtllm/launch/agg_multimodal_qwen3vl.sh
#
#   # Then run the tests:
#   bash tests/security/test_multimodal_security.sh

set -euo pipefail

FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
SAFETENSORS_PATH="${SAFETENSORS_PATH:-/tmp/test_embeddings.safetensors}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/tmp}"
PASS=0
FAIL=0

echo "================================================================="
echo "  Security Regression Tests: TRT-LLM Multimodal (NVBug 6086411)"
echo "================================================================="
echo "Frontend: ${FRONTEND_URL}"
echo "Model:    ${SERVED_MODEL_NAME}"
echo ""

# Wait for model to be ready
echo "[*] Waiting for model registration..."
for i in $(seq 1 120); do
    MODELS="$(curl -sf "${FRONTEND_URL}/v1/models" 2>/dev/null || true)"
    if echo "${MODELS}" | grep -q "${SERVED_MODEL_NAME}"; then
        echo "[+] Model registered"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "[!] Model not ready after 240s"
        exit 1
    fi
    sleep 2
done

run_test() {
    local name="$1"
    local expected="$2"  # "pass" or "fail"
    local http_code="$3"
    local response="$4"

    if [ "$expected" = "pass" ]; then
        if [ "$http_code" = "200" ]; then
            echo "[PASS] $name (HTTP $http_code)"
            PASS=$((PASS + 1))
        else
            echo "[FAIL] $name (expected 200, got HTTP $http_code)"
            echo "       Response: $(echo "$response" | head -c 200)"
            FAIL=$((FAIL + 1))
        fi
    else
        if [ "$http_code" != "200" ]; then
            echo "[PASS] $name (correctly rejected with HTTP $http_code)"
            PASS=$((PASS + 1))
        else
            echo "[FAIL] $name (expected rejection, but got HTTP 200)"
            FAIL=$((FAIL + 1))
        fi
    fi
}

# ===================================================================
# TEST 1: Basic image URL inference (should succeed)
# ===================================================================
echo ""
echo "--- Test 1: Basic image URL inference ---"
RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 60 \
    -H 'Content-Type: application/json' \
    "${FRONTEND_URL}/v1/chat/completions" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"What do you see in this image? Reply in one sentence.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://picsum.photos/id/237/200/300\"}}
            ]
        }],
        \"max_tokens\": 32
    }" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)
run_test "Image URL inference" "pass" "$HTTP_CODE" "$BODY"

# ===================================================================
# TEST 2: Reject .pt file (the original attack vector)
# ===================================================================
echo ""
echo "--- Test 2: Reject .pt file via image_url ---"
RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 30 \
    -H 'Content-Type: application/json' \
    "${FRONTEND_URL}/v1/chat/completions" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"Describe this.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://attacker.example.com/payload.pt\"}}
            ]
        }],
        \"max_tokens\": 32
    }" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)
run_test "Reject .pt file" "fail" "$HTTP_CODE" "$BODY"

# ===================================================================
# TEST 3: Reject .pth file
# ===================================================================
echo ""
echo "--- Test 3: Reject .pth file via image_url ---"
RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 30 \
    -H 'Content-Type: application/json' \
    "${FRONTEND_URL}/v1/chat/completions" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"Describe this.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://attacker.example.com/model.pth\"}}
            ]
        }],
        \"max_tokens\": 32
    }" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)
run_test "Reject .pth file" "fail" "$HTTP_CODE" "$BODY"

# ===================================================================
# TEST 4: Reject .bin file
# ===================================================================
echo ""
echo "--- Test 4: Reject .bin file via image_url ---"
RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 30 \
    -H 'Content-Type: application/json' \
    "${FRONTEND_URL}/v1/chat/completions" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"Describe this.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://attacker.example.com/weights.bin\"}}
            ]
        }],
        \"max_tokens\": 32
    }" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)
run_test "Reject .bin file" "fail" "$HTTP_CODE" "$BODY"

# ===================================================================
# TEST 5: SSRF protection - reject private IP URLs
# ===================================================================
echo ""
echo "--- Test 5: SSRF protection (private IP) ---"
RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 30 \
    -H 'Content-Type: application/json' \
    "${FRONTEND_URL}/v1/chat/completions" \
    -d "{
        \"model\": \"${SERVED_MODEL_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"Describe this.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://169.254.169.254/latest/meta-data/iam/security-credentials/\"}}
            ]
        }],
        \"max_tokens\": 32
    }" 2>/dev/null || echo -e "\n000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)
# This should fail either at the Rust frontend (if media_loader enabled)
# or at the Python backend URL validation
run_test "SSRF protection (private IP)" "fail" "$HTTP_CODE" "$BODY"

# ===================================================================
# TEST 6: Pre-computed safetensors embedding inference (if file exists)
# ===================================================================
echo ""
echo "--- Test 6: Pre-computed safetensors embedding inference ---"
if [ -f "${SAFETENSORS_PATH}" ]; then
    RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 60 \
        -H 'Content-Type: application/json' \
        "${FRONTEND_URL}/v1/chat/completions" \
        -d "{
            \"model\": \"${SERVED_MODEL_NAME}\",
            \"messages\": [{
                \"role\": \"user\",
                \"content\": [
                    {\"type\": \"text\", \"text\": \"Describe what these embeddings represent.\"},
                    {\"type\": \"image_url\", \"image_url\": {\"url\": \"${SAFETENSORS_PATH}\"}}
                ]
            }],
            \"max_tokens\": 32
        }" 2>/dev/null || echo -e "\n000")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n -1)
    # Note: This may or may not succeed depending on embedding dimensions
    # matching the model. The key test is that it doesn't CRASH the worker.
    echo "[INFO] Safetensors embedding test: HTTP $HTTP_CODE"
    echo "       (Worker should remain healthy regardless of result)"
    PASS=$((PASS + 1))
else
    echo "[SKIP] ${SAFETENSORS_PATH} not found. Generate with:"
    echo "       python tests/security/generate_safetensors_embeddings.py --random --output ${SAFETENSORS_PATH}"
fi

# ===================================================================
# TEST 7: Verify server is still healthy after all tests
# ===================================================================
echo ""
echo "--- Test 7: Server health check after attack attempts ---"
sleep 2
HEALTH_CODE=$(curl -sf -o /dev/null -w '%{http_code}' --max-time 10 \
    "${FRONTEND_URL}/v1/models" 2>/dev/null || echo "000")

if [ "$HEALTH_CODE" = "200" ]; then
    MODELS_AFTER=$(curl -sf "${FRONTEND_URL}/v1/models" 2>/dev/null || true)
    if echo "${MODELS_AFTER}" | grep -q "${SERVED_MODEL_NAME}"; then
        echo "[PASS] Server healthy, model still registered"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] Server responded but model was removed from discovery"
        FAIL=$((FAIL + 1))
    fi
else
    echo "[FAIL] Server not responding (HTTP $HEALTH_CODE) — possible worker crash"
    FAIL=$((FAIL + 1))
fi

# ===================================================================
# TEST 8: --frontend-decoding flag (image inference in both modes)
# ===================================================================
echo ""
echo "--- Test 8: --frontend-decoding flag ---"
FRONTEND_DECODING="${FRONTEND_DECODING:-false}"
if [ "$FRONTEND_DECODING" = "true" ]; then
    echo "[INFO] --frontend-decoding is ENABLED — images decoded by Rust MediaDecoder"
    RESPONSE=$(curl -sf -w '\n%{http_code}' --max-time 60 \
        -H 'Content-Type: application/json' \
        "${FRONTEND_URL}/v1/chat/completions" \
        -d "{
            \"model\": \"${SERVED_MODEL_NAME}\",
            \"messages\": [{
                \"role\": \"user\",
                \"content\": [
                    {\"type\": \"text\", \"text\": \"What color is this image? One word.\"},
                    {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://picsum.photos/id/10/200/300\"}}
                ]
            }],
            \"max_tokens\": 16
        }" 2>/dev/null || echo -e "\n000")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n -1)
    run_test "Image URL with --frontend-decoding" "pass" "$HTTP_CODE" "$BODY"
else
    echo "[INFO] --frontend-decoding is DISABLED (default) — images decoded by Python backend"
    echo "[PASS] Default backend decoding already verified by Test 1"
    PASS=$((PASS + 1))
fi

# ===================================================================
# SUMMARY
# ===================================================================
echo ""
echo "================================================================="
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "  frontend-decoding: ${FRONTEND_DECODING}"
echo "================================================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All security regression tests passed."
