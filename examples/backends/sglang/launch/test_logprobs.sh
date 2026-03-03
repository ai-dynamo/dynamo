#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script for logprob passthrough in the SGLang decode handler.
# Launches an aggregated worker, waits for readiness, then sends
# chat/completion requests with logprobs enabled and validates results.
#
# GPUs: 1
# Usage: bash examples/backends/sglang/launch/test_logprobs.sh [--model-path <model>]

set -uo pipefail

# ── cleanup ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    kill $FRONTEND_PID $WORKER_PID 2>/dev/null || true
    wait $FRONTEND_PID $WORKER_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# ── args ─────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-0.6B"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
BASE_URL="http://localhost:${HTTP_PORT}"
DISCOVERY="${DISCOVERY_BACKEND:-file}"
PASS=0
FAIL=0

# ── helpers ──────────────────────────────────────────────────────────
wait_for_model() {
    local timeout=300
    local start=$SECONDS
    echo "Waiting up to ${timeout}s for model to register at ${BASE_URL} ..."
    while (( SECONDS - start < timeout )); do
        # Check that /v1/models returns a non-empty model list
        local models
        models=$(curl -sf "${BASE_URL}/v1/models" 2>/dev/null) || { sleep 3; continue; }
        if echo "$models" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
            echo "Model registered in $(( SECONDS - start ))s"
            return 0
        fi
        sleep 3
    done
    echo "ERROR: Model did not register within ${timeout}s"
    exit 1
}

assert_logprobs_chat() {
    local resp="$1"
    python3 -c "
import json, sys, math
r = json.loads(sys.stdin.read())
choice = r['choices'][0]
assert 'logprobs' in choice, 'missing logprobs field'
lp = choice['logprobs']
assert lp is not None, 'logprobs is null'
assert 'content' in lp, 'missing content in logprobs'
items = lp['content']
assert len(items) > 0, f'logprobs content is empty'
for i, item in enumerate(items):
    assert 'token' in item, f'item {i}: missing token'
    assert 'logprob' in item, f'item {i}: missing logprob'
    v = item['logprob']
    assert not math.isnan(v) and not math.isinf(v), f'item {i}: bad logprob {v}'
    assert v <= 0, f'item {i}: logprob should be <= 0, got {v}'
    assert 'top_logprobs' in item, f'item {i}: missing top_logprobs'
    assert len(item['top_logprobs']) > 0, f'item {i}: top_logprobs empty'
    for tp in item['top_logprobs']:
        assert 'logprob' in tp, f'top_logprob missing logprob'
print(f'OK: {len(items)} tokens with valid logprobs (top_k={len(items[0][\"top_logprobs\"])})')
" <<< "$resp"
}

assert_logprobs_completion() {
    local resp="$1"
    python3 -c "
import json, sys, math
r = json.loads(sys.stdin.read())
choice = r['choices'][0]
assert 'logprobs' in choice, 'missing logprobs field'
lp = choice['logprobs']
assert lp is not None, 'logprobs is null'
assert 'token_logprobs' in lp, 'missing token_logprobs'
assert 'tokens' in lp, 'missing tokens'
tl = lp['token_logprobs']
tk = lp['tokens']
assert len(tl) == len(tk), f'length mismatch: {len(tl)} vs {len(tk)}'
valid = 0
for i, v in enumerate(tl):
    if v is not None:
        assert not math.isnan(v) and not math.isinf(v), f'index {i}: bad logprob {v}'
        assert v <= 0, f'index {i}: logprob should be <= 0, got {v}'
        valid += 1
print(f'OK: {len(tl)} tokens, {valid} with valid logprobs')
" <<< "$resp"
}

run_test() {
    local name="$1"
    shift
    echo ""
    echo "── TEST: ${name} ──"
    if "$@"; then
        echo "PASS: ${name}"
        PASS=$(( PASS + 1 ))
    else
        echo "FAIL: ${name}"
        FAIL=$(( FAIL + 1 ))
    fi
}

# ── launch ───────────────────────────────────────────────────────────
echo "=========================================="
echo "Logprobs Integration Test"
echo "=========================================="
echo "Model:       $MODEL"
echo "Frontend:    $BASE_URL"
echo "Discovery:   $DISCOVERY"
echo "=========================================="

python3 -m dynamo.frontend --discovery-backend "$DISCOVERY" &
FRONTEND_PID=$!

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --discovery-backend "$DISCOVERY" \
  "${EXTRA_ARGS[@]}" &
WORKER_PID=$!

wait_for_model

# ── tests ────────────────────────────────────────────────────────────

# 1) Chat completions with logprobs
test_chat_logprobs() {
    local resp
    resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"max_tokens\": 16,
            \"temperature\": 0.0,
            \"logprobs\": true,
            \"top_logprobs\": 3
        }")
    assert_logprobs_chat "$resp"
}

# 2) Chat completions WITHOUT logprobs (regression: should still work)
test_chat_no_logprobs() {
    local resp
    resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"max_tokens\": 8,
            \"temperature\": 0.0
        }")
    python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
msg = r['choices'][0]['message']['content']
assert len(msg) > 0, 'empty response'
print(f'OK: got response ({len(msg)} chars)')
" <<< "$resp"
}

# 3) Completions endpoint with logprobs
test_completion_logprobs() {
    local resp
    resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"Once upon a time\",
            \"max_tokens\": 16,
            \"temperature\": 0.0,
            \"logprobs\": 5
        }")
    assert_logprobs_completion "$resp"
}

# 4) Chat completions with logprobs + streaming
test_chat_logprobs_stream() {
    local resp
    resp=$(curl -sf --max-time 60 "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Count to 5\"}],
            \"max_tokens\": 32,
            \"temperature\": 0.0,
            \"logprobs\": true,
            \"top_logprobs\": 2,
            \"stream\": true
        }")
    python3 -c "
import json, sys, math

lines = sys.stdin.read().strip().split('\n')
token_count = 0
for line in lines:
    line = line.strip()
    if not line.startswith('data: '):
        continue
    data = line[len('data: '):]
    if data == '[DONE]':
        break
    chunk = json.loads(data)
    choice = chunk['choices'][0]
    delta_lp = choice.get('logprobs')
    if delta_lp and delta_lp.get('content'):
        for item in delta_lp['content']:
            v = item['logprob']
            assert not math.isnan(v) and not math.isinf(v), f'bad logprob {v}'
            assert v <= 0, f'logprob should be <= 0, got {v}'
            token_count += 1

assert token_count > 0, 'no logprobs found in streamed chunks'
print(f'OK: {token_count} streamed tokens with valid logprobs')
" <<< "$resp"
}

run_test "chat/completions + logprobs"           test_chat_logprobs
run_test "chat/completions without logprobs"      test_chat_no_logprobs
run_test "completions + logprobs"                 test_completion_logprobs
run_test "chat/completions + logprobs + stream"   test_chat_logprobs_stream

# ── summary ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "=========================================="
exit $FAIL
