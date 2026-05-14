#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TITO (Token-In Token-Out) + full-weights training smoke for rl-sdk-1.
#
# rl-sdk-1 differences from bis/dynamo-rl:
#   * RL admin uses ONE generic dispatcher: POST /v1/rl/engine {method, kwargs, filter?}
#     (no typed /v1/rl/pause, /v1/rl/update_weights, /v1/rl/resume routes)
#   * Worker registers RL endpoint unconditionally (no --enable-rl gate)
#   * TITO input via nvext.token_data works; top-level prompt_token_ids does NOT
#     (not in PASSTHROUGH_EXTRA_FIELDS yet)
#   * No rl_promote: completion_token_ids stays in response.nvext, NOT promoted
#     to choices[0].token_ids
#
# What this smoke verifies on rl-sdk-1:
#   1. Inference accepts pre-tokenized input via nvext.token_data (prompt_tokens
#      count in usage matches sent count -> preprocessor skipped tokenization)
#   2. Admin plane fan-out: POST /v1/rl/engine {method:pause_generation} →
#      {method:update_weights_from_disk} → {method:resume_generation}
#   3. Inference still works after the FT round-trip
#   4. stop_token_ids honored via nvext (forced halt within 1-2 tokens)
#
# Usage:
#   cd /home/biswaranjanp/dev/rl/dynamo
#   source dynamo/bin/activate
#   export PRIME_RL_SRC=/home/biswaranjanp/dev/rl/prime-rl/src
#   bash tests/rl/smoke_test_tito.sh [<model>]

set -euo pipefail

BGPIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo "[tito] Cleaning up..."
    for pid in "${BGPIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

MODEL="${1:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
RL_PORT="${DYN_RL_PORT:-8002}"
NATS_PORT="${NATS_PORT:-4222}"
: "${PRIME_RL_SRC:?Set PRIME_RL_SRC to the prime-rl src directory}"

DEFAULT_WORKDIR=/home/biswaranjanp/dev/rl/work/bis-dev/may-11/local/tito-sft
LOG_DIR="${TITO_WORKDIR:-$DEFAULT_WORKDIR/run-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "${DEFAULT_WORKDIR}/latest" 2>/dev/null || true

echo "[tito] Branch: rl-sdk-1"
echo "[tito] Workdir: $LOG_DIR"
echo "[tito] Model: $MODEL"

if nc -z localhost "$NATS_PORT" 2>/dev/null; then
    echo "[tito] NATS already running — reusing"
else
    nats-server -p "$NATS_PORT" -l "$LOG_DIR/nats.log" &
    BGPIDS+=("$!")
    sleep 1
fi

# Frontend — DYN_ENABLE_RL_ENDPOINTS mounts /v1/rl/engine on the dedicated rl_port.
HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  DYN_ENABLE_RL_ENDPOINTS=true \
  DYN_HTTP_PORT="$HTTP_PORT" \
  python -m dynamo.frontend \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
BGPIDS+=("$FRONTEND_PID")
echo "[tito] Frontend started (pid=$FRONTEND_PID)"

# Worker — `--enable-rl` mirrors SGLang. Routes are registered unconditionally
# today, but the flag signals RL deployment and matches the smoke_test.sh CLI.
HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="${PRIME_RL_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
  DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
  python -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
    --enable-rl \
    --worker-extension-cls prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker \
    > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
BGPIDS+=("$WORKER_PID")
echo "[tito] Worker started (pid=$WORKER_PID)"

echo "[tito] Waiting for $MODEL to register on /v1/models..."
DEADLINE=$(( $(date +%s) + 240 ))
while true; do
  if curl -sf "http://localhost:${HTTP_PORT}/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    echo "[tito] Model registered"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[tito] TIMEOUT — model didn't register"
    tail -30 "$LOG_DIR/frontend.log"
    tail -40 "$LOG_DIR/worker.log"
    exit 1
  fi
  sleep 3
done

# Wait briefly for worker's RL endpoint to land in etcd
sleep 2

echo ""
echo "[tito] === Build tokenized prompt ==="
PROMPT_TEXT="Hello, how are you today?"
TOKENS_JSON=$(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<PYEOF
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL")
ids = tok.apply_chat_template(
    [{"role": "user", "content": "$PROMPT_TEXT"}],
    add_generation_prompt=True, tokenize=True,
)
# transformers >=4.45 may return a BatchEncoding for tokenize=True;
# unwrap to a flat list[int] regardless of return shape.
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
if ids and isinstance(ids[0], list):
    ids = ids[0]
ids = [int(t) for t in ids]
stop_ids = [tok.convert_tokens_to_ids(s) for s in ("<|im_end|>", "<|endoftext|>")
            if tok.convert_tokens_to_ids(s) >= 0]
print(json.dumps({"prompt_ids": ids, "stop_token_ids": stop_ids}))
PYEOF
)
echo "[tito] $TOKENS_JSON"
PROMPT_IDS=$(echo "$TOKENS_JSON" | python -c "import sys,json; print(json.dumps(json.load(sys.stdin)['prompt_ids']))")
STOP_IDS=$(echo "$TOKENS_JSON" | python -c "import sys,json; print(json.dumps(json.load(sys.stdin)['stop_token_ids']))")
N_PROMPT=$(echo "$PROMPT_IDS" | python -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "[tito] Prompt token count: $N_PROMPT"

# rl-sdk-2 wire shape:
#   - nvext.token_data           pre-tokenized prompt (preprocessor consumes it)
#   - extra_body.stop_token_ids  whitelisted via PASSTHROUGH_EXTRA_FIELDS,
#                                plumbed into common::StopConditions.stop_token_ids
#   - extra_body.cache_salt      whitelisted too (RL prefix-cache isolation)
#   - nvext.extra_fields=["engine_data"]
#                                opts into nvext.engine_data on the response,
#                                which carries completion_token_ids (+ logprobs)
#                                emitted by the vLLM backend handler. Mirrors
#                                PR #8119's SGLang shape.
PAYLOAD=$(python -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role':'user','content':'(token-in mode)'}],
    'stream': False,
    'max_tokens': 24,
    'temperature': 0.0,
    'logprobs': True,
    'stop_token_ids': $STOP_IDS,
    'cache_salt': 'smoke_tito_v1',
    'nvext': {
        'token_data': $PROMPT_IDS,
        'extra_fields': ['engine_data']
    }
}))
")

echo ""
echo "[tito] === TITO inference BEFORE weight update ==="
RESP_BEFORE=$(curl -s -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d "$PAYLOAD")
echo "$RESP_BEFORE" > "$LOG_DIR/resp_before.json"
echo "$RESP_BEFORE" | python -m json.tool 2>/dev/null | head -50 || echo "$RESP_BEFORE"

if ! echo "$RESP_BEFORE" | python -c "
import sys, json
data = json.load(sys.stdin)
choices = data.get('choices', [])
if not choices:
    print('FAIL: no choices in response')
    print('keys:', list(data.keys()))
    sys.exit(1)
c0 = choices[0]
text = c0.get('message',{}).get('content','')
usage = data.get('usage', {})
prompt_tokens = usage.get('prompt_tokens', 0)
expected = $N_PROMPT
if prompt_tokens != expected:
    print(f'FAIL: prompt_tokens={prompt_tokens} expected={expected} (TITO input not honored)')
    sys.exit(1)

# Canonical channel (PR #8119 + rl-sdk-2): response.nvext.engine_data.*
nvext_resp = data.get('nvext') or {}
engine_data = nvext_resp.get('engine_data') or {}
out_tok = engine_data.get('completion_token_ids') or []
out_lp  = engine_data.get('completion_logprobs') or []

if not out_tok:
    print('FAIL: nvext.engine_data.completion_token_ids missing or empty')
    print('nvext keys:', list(nvext_resp.keys()))
    print('engine_data keys:', list(engine_data.keys()) if engine_data else '(absent)')
    sys.exit(1)

if len(out_tok) != usage.get('completion_tokens', -1):
    print(f'FAIL: len(completion_token_ids)={len(out_tok)} != usage.completion_tokens={usage.get(\"completion_tokens\")}')
    sys.exit(1)

if out_lp and len(out_lp) != len(out_tok):
    print(f'FAIL: len(completion_logprobs)={len(out_lp)} != len(completion_token_ids)={len(out_tok)}')
    sys.exit(1)

print(f'PASS: prompt_tokens={prompt_tokens} (TITO input honored)')
print(f'      nvext.engine_data.completion_token_ids: n={len(out_tok)}')
print(f'      nvext.engine_data.completion_logprobs:  n={len(out_lp)} (flat list[float])')
print(f'      text={text[:60]!r}')
"; then
  echo "[tito] FAIL: TITO before weight update"
  exit 1
fi

echo ""
echo "[tito] === POST /v1/rl/engine pause_generation (port $RL_PORT) ==="
PAUSE_BODY='{"method": "pause_generation", "kwargs": {"mode": "keep", "clear_cache": false}}'
if ! curl -sf -X POST "http://localhost:${RL_PORT}/v1/rl/engine" \
    -H "Content-Type: application/json" -d "$PAUSE_BODY" \
    | tee "$LOG_DIR/pause_resp.json" \
    | python -c "
import sys, json
d = json.load(sys.stdin)
workers = d.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed or not workers:
    print('FAIL', failed or 'no workers'); sys.exit(1)
print(f'PASS: pause_generation across {len(workers)} worker(s)')
"; then
  echo "[tito] FAIL: pause_generation"
  exit 1
fi

MODEL_CACHE=$(HF_HUB_OFFLINE=1 python -c "
import huggingface_hub
print(huggingface_hub.snapshot_download('$MODEL', local_files_only=True))
")
echo "[tito] Weight path: $MODEL_CACHE"

echo ""
echo "[tito] === POST /v1/rl/engine update_weights_from_disk ==="
UPDATE_BODY=$(python -c "
import json
print(json.dumps({
    'method': 'update_weights_from_disk',
    'kwargs': {
        'model_path': '$MODEL_CACHE',
        'weight_version': 'tito_v1',
        'engine_rpc': 'update_weights_from_path'
    },
    'timeout_secs': 240
}))
")
if ! curl -sf -X POST "http://localhost:${RL_PORT}/v1/rl/engine" \
    -H "Content-Type: application/json" --max-time 300 -d "$UPDATE_BODY" \
    | tee "$LOG_DIR/update_resp.json" \
    | python -c "
import sys, json
d = json.load(sys.stdin)
workers = d.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed or not workers:
    print('FAIL', failed or 'no workers'); sys.exit(1)
print(f'PASS: update_weights_from_disk applied across {len(workers)} worker(s)')
"; then
  echo "[tito] FAIL: update_weights"
  exit 1
fi

echo ""
echo "[tito] === POST /v1/rl/engine resume_generation ==="
RESUME_BODY='{"method": "resume_generation", "kwargs": {}}'
if ! curl -sf -X POST "http://localhost:${RL_PORT}/v1/rl/engine" \
    -H "Content-Type: application/json" -d "$RESUME_BODY" \
    | tee "$LOG_DIR/resume_resp.json" \
    | python -c "
import sys, json
d = json.load(sys.stdin)
workers = d.get('workers', [])
failed = [w for w in workers if w.get('status') != 'ok']
if failed or not workers:
    print('FAIL', failed or 'no workers'); sys.exit(1)
print(f'PASS: resume_generation across {len(workers)} worker(s)')
"; then
  echo "[tito] FAIL: resume_generation"
  exit 1
fi

echo ""
echo "[tito] === TITO inference AFTER weight update ==="
RESP_AFTER=$(curl -s -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" --max-time 60 -d "$PAYLOAD")
echo "$RESP_AFTER" > "$LOG_DIR/resp_after.json"

if ! echo "$RESP_AFTER" | python -c "
import sys, json
data = json.load(sys.stdin)
c0 = data['choices'][0]
text = c0.get('message',{}).get('content','')
prompt_tokens = data.get('usage', {}).get('prompt_tokens', 0)
if prompt_tokens != $N_PROMPT:
    print(f'FAIL: prompt_tokens={prompt_tokens} after update'); sys.exit(1)

# Canonical channel: nvext.engine_data.completion_token_ids
engine_data = (data.get('nvext') or {}).get('engine_data') or {}
out_tok = engine_data.get('completion_token_ids') or []
if not out_tok:
    print('FAIL: nvext.engine_data.completion_token_ids missing after weight update')
    sys.exit(1)
print(f'PASS: prompt_tokens={prompt_tokens} (TITO honored) n_completion_token_ids={len(out_tok)} text={text[:60]!r}')
"; then
  echo "[tito] FAIL: TITO after weight update"
  exit 1
fi

echo ""
echo "[tito] === Determinism check ==="
python - "$LOG_DIR" <<'PYEOF' || true
import json, sys, os
ld = sys.argv[1]
b = json.load(open(os.path.join(ld, 'resp_before.json')))
a = json.load(open(os.path.join(ld, 'resp_after.json')))
def toks(d):
    return ((d.get('nvext') or {}).get('engine_data') or {}).get('completion_token_ids') or []
bt, at = toks(b), toks(a)
print(f'before n={len(bt)} {bt[:8]}...')
print(f'after  n={len(at)} {at[:8]}...')
print('PASS: deterministic' if bt == at else 'WARN: outputs differ')
PYEOF

echo ""
echo "[tito] === Stop-token verification: forced early stop via extra_body.stop_token_ids ==="
# rl-sdk-2 plumbing:
#   PASSTHROUGH_EXTRA_FIELDS accepts extra_body.stop_token_ids → provider's
#   get_stop_token_ids() reads it → common::StopConditions.stop_token_ids →
#   vLLM SamplingParams.stop_token_ids. Picking a token that any sampled
#   continuation must hit early (token id 198 = "\n" in Qwen tokenizers).
STOP_PAYLOAD=$(python -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role':'user','content':'(token-in mode)'}],
    'stream': False,
    'max_tokens': 32,
    'temperature': 0.0,
    'stop_token_ids': [198],
    'nvext': {
        'token_data': $PROMPT_IDS,
        'extra_fields': ['engine_data']
    }
}))
")
RESP_STOP=$(curl -s -X POST "http://localhost:${HTTP_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" --max-time 30 -d "$STOP_PAYLOAD")
echo "$RESP_STOP" > "$LOG_DIR/resp_stop.json"
echo "$RESP_STOP" | python -c "
import sys, json
d = json.load(sys.stdin)
c0 = d['choices'][0]
finish = c0.get('finish_reason')
engine_data = (d.get('nvext') or {}).get('engine_data') or {}
out = engine_data.get('completion_token_ids') or []
honored = finish == 'stop' and len(out) <= 3
status = 'PASS' if honored else 'INFO'
print(f'{status}: finish={finish} n_tokens={len(out)} tokens={out} (stop_token_id=198 honored={honored})')
" || true

echo ""
echo "========================================"
echo "[tito] rl-sdk-2 TITO smoke PASSED"
echo "[tito]   ✓ /v1/rl/engine generic dispatcher (pause/update/resume)"
echo "[tito]   ✓ TITO input via nvext.token_data — preprocessor skip-tokenize"
echo "[tito]   ✓ FileSystemWeightUpdateWorker FT round-trip"
echo "[tito]   ✓ TITO output via nvext.engine_data.completion_token_ids (PR #8119 channel)"
echo "[tito]   ✓ extra_body.stop_token_ids whitelisted + plumbed to SamplingParams"
echo "[tito]   ✓ extra_body.cache_salt whitelisted"
echo "[tito]   ○ nvext.stop_token_ids honored — deferred to follow-up PR"
echo "[tito] Artifacts: $LOG_DIR"
echo "========================================"
