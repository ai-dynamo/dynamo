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

# Worker — RL endpoint registered unconditionally on rl-sdk-1 (no --enable-rl)
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

# rl-sdk-1 wire shape: nvext.token_data only (no top-level prompt_token_ids).
# stop_token_ids -> inside nvext as well (top-level rejected by validator).
PAYLOAD=$(python -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role':'user','content':'(token-in mode)'}],
    'stream': False,
    'max_tokens': 24,
    'temperature': 0.0,
    'nvext': {
        'token_data': $PROMPT_IDS,
        'stop_token_ids': $STOP_IDS,
        'extra_fields': ['completion_token_ids']
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

# Check for completion token IDs (may be in nvext OR choices, depending on branch)
out_tok = c0.get('token_ids') or (data.get('nvext') or {}).get('completion_token_ids')
loc = 'choices[0].token_ids' if c0.get('token_ids') else 'nvext.completion_token_ids' if out_tok else 'absent'
print(f'PASS: prompt_tokens={prompt_tokens} (TITO input honored) completion_token_ids_at={loc} n={len(out_tok or [])} text={text[:60]!r}')
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
out_tok = c0.get('token_ids') or (data.get('nvext') or {}).get('completion_token_ids')
loc = 'choices[0].token_ids' if c0.get('token_ids') else 'nvext.completion_token_ids' if out_tok else 'absent'
print(f'PASS: prompt_tokens={prompt_tokens} (TITO honored) completion_at={loc} n={len(out_tok or [])} text={text[:60]!r}')
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
    c0 = d['choices'][0]
    return c0.get('token_ids') or (d.get('nvext') or {}).get('completion_token_ids') or []
bt, at = toks(b), toks(a)
print(f'before n={len(bt)} {bt[:8]}...')
print(f'after  n={len(at)} {at[:8]}...')
print('PASS: deterministic' if bt == at else 'WARN: outputs differ')
PYEOF

echo ""
echo "[tito] === Stop-token verification (informational): forced early stop on token 198 ==="
# On rl-sdk-1, nvext.stop_token_ids is not yet plumbed into the engine path
# (the wiring lives on bis/dynamo-rl in commit f03417149a). We probe it here
# so the smoke surfaces this gap, but don't fail the smoke on it.
STOP_PAYLOAD=$(python -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role':'user','content':'(token-in mode)'}],
    'stream': False,
    'max_tokens': 32,
    'temperature': 0.0,
    'nvext': {
        'token_data': $PROMPT_IDS,
        'stop_token_ids': [198],
        'extra_fields': ['completion_token_ids']
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
out = c0.get('token_ids') or (d.get('nvext') or {}).get('completion_token_ids') or []
finish = c0.get('finish_reason')
honored = finish == 'stop' and len(out) <= 3 and 198 in out
status = 'PASS' if honored else 'INFO (not shipped on rl-sdk-1)'
print(f'{status}: finish={finish} n_tokens={len(out)} stop_token_id=198 nvext_stop_honored={honored}')
" || true

echo ""
echo "========================================"
echo "[tito] rl-sdk-1 smoke PASSED — admin plane + TITO input verified"
echo "[tito]   ✓ /v1/rl/engine generic dispatcher (pause/update/resume) — PR #9382 surface"
echo "[tito]   ✓ TITO input (nvext.token_data) — preprocessor skip-tokenize"
echo "[tito]   ✓ FileSystemWeightUpdateWorker FT round-trip"
echo "[tito]   ○ TITO output (completion_token_ids) — deferred to follow-up PR"
echo "[tito]   ○ nvext.stop_token_ids honored — deferred to follow-up PR"
echo "[tito] Artifacts: $LOG_DIR"
echo "========================================"
