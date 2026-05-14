#!/bin/bash
# Server-side TITO probe: validates nvext.engine_data emission from the
# vLLM backend after the rl-sdk-2 changes.
#
# What it tests (single roundtrip, no orchestrator/trainer):
#   1. PASSTHROUGH_EXTRA_FIELDS accepts cache_salt + stop_token_ids
#      without 400 Bad Request.
#   2. Request `nvext.token_data=[...]` skips server-side tokenization
#      (validated indirectly: usage.prompt_tokens == len(token_data)).
#   3. Request `nvext.extra_fields=["engine_data"]` causes the response
#      to include `nvext.engine_data.completion_token_ids` as the
#      exact engine-emitted IDs.
#   4. `nvext.engine_data.completion_logprobs` is a flat list[float]
#      indexed by sampled token (one float per completion_token_id).
#
# Pass criteria (asserted by the python block):
#   * status == 200
#   * len(engine_data.completion_token_ids) == usage.completion_tokens
#   * (optional) len(completion_logprobs) >= 1 if logprobs were requested

set -euo pipefail

DYNAMO_VENV=/home/biswaranjanp/dev/rl/dynamo/.venv
WORKDIR=/tmp/probe-engine-data
mkdir -p "$WORKDIR"

# Clean up any leftover dynamo processes
pkill -9 -f "dynamo.vllm"      2>/dev/null || true
pkill -9 -f "dynamo.frontend"  2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3

source "$DYNAMO_VENV/bin/activate"

# Frontend on :8000 (no DYN_ENABLE_RL needed for engine_data path — gating
# is purely from nvext.extra_fields=["engine_data"] on the request)
CUDA_VISIBLE_DEVICES="" \
  nohup python3 -m dynamo.frontend --http-port 8000 \
  > "$WORKDIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[probe] frontend PID=$FRONTEND_PID"
sleep 5

# vLLM worker - Qwen3-0.6B, eager, single GPU, small mem
CUDA_VISIBLE_DEVICES=0 \
  nohup python3 -m dynamo.vllm \
    --model Qwen/Qwen3-0.6B \
    --served-model-name Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.30 \
  > "$WORKDIR/vllm_worker.log" 2>&1 &
WORKER_PID=$!
echo "[probe] worker PID=$WORKER_PID"

cleanup() {
  echo "[probe] cleanup"
  kill "$FRONTEND_PID" "$WORKER_PID" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[probe] waiting for /v1/models to register Qwen/Qwen3-0.6B..."
for i in $(seq 1 60); do
  if curl -fs http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen/Qwen3-0.6B"; then
    echo "[probe] worker ready after ${i}*5s"
    break
  fi
  sleep 5
done

# Pre-tokenize a known prompt ourselves so we can compare prompt_tokens later.
python3 <<'PYEOF'
import json, sys, urllib.request

# Hardcoded short prompt tokenization for Qwen3 chat template.
# We'll send a placeholder "(token-in mode)" string in messages so the
# preprocessor still has SOMETHING to apply chat-template selection on,
# but the actual prompt tokens override via nvext.token_data.
# These IDs are arbitrary valid Qwen3 vocab tokens; the server should
# treat them as the prompt and emit usage.prompt_tokens == 8.
prompt_ids = [151644, 8948, 198, 9707, 11, 1879, 0, 151645]  # 8 tokens

body = {
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "(token-in mode)"}],
    "max_completion_tokens": 12,
    "temperature": 0.7,
    "logprobs": True,
    # Test PASSTHROUGH for both cache_salt + stop_token_ids:
    "cache_salt": "probe_step_1",
    "stop_token_ids": [151643],  # Qwen3 <|endoftext|>
    "nvext": {
        "token_data": prompt_ids,
        "extra_fields": ["engine_data"],
        # Also test nvext.cache_salt as the canonical location:
        "cache_salt": "probe_step_1_nvext",
    },
}

req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = resp.status
        payload = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"[probe] HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
    sys.exit(1)

print(f"[probe] HTTP {status}")
print(json.dumps(payload, indent=2))

# Assertions
assert status == 200, f"expected 200, got {status}"

nvext_resp = payload.get("nvext")
assert isinstance(nvext_resp, dict), f"missing response.nvext (got {type(nvext_resp)})"

engine_data = nvext_resp.get("engine_data")
assert isinstance(engine_data, dict), (
    f"missing response.nvext.engine_data; got nvext={nvext_resp}"
)

ctids = engine_data.get("completion_token_ids")
assert isinstance(ctids, list) and ctids, (
    f"missing/empty completion_token_ids: {ctids}"
)

usage = payload["usage"]
assert len(ctids) == usage["completion_tokens"], (
    f"completion_token_ids length ({len(ctids)}) != "
    f"usage.completion_tokens ({usage['completion_tokens']})"
)

# usage.prompt_tokens should equal what we sent via nvext.token_data
assert usage["prompt_tokens"] == len(prompt_ids), (
    f"prompt_tokens={usage['prompt_tokens']} != sent token_data len={len(prompt_ids)}; "
    f"server did NOT consume nvext.token_data"
)

clp = engine_data.get("completion_logprobs")
if clp is not None:
    assert isinstance(clp, list), f"completion_logprobs not a list: {type(clp)}"
    assert all(isinstance(x, (int, float)) for x in clp), (
        f"completion_logprobs not flat list[float]: {[type(x).__name__ for x in clp][:5]}"
    )
    assert len(clp) == len(ctids), (
        f"len(completion_logprobs)={len(clp)} != len(completion_token_ids)={len(ctids)}"
    )

print()
print("[probe] PASS — all engine_data assertions satisfied")
print(f"        completion_token_ids: {ctids}")
print(f"        completion_logprobs (first 5): {clp[:5] if clp else 'none'}")
print(f"        usage: {usage}")
PYEOF

PROBE_EXIT=$?
echo
if [ "$PROBE_EXIT" -eq 0 ]; then
  echo "[probe] === PASS ==="
else
  echo "[probe] === FAIL (exit=$PROBE_EXIT) ==="
  echo "--- frontend.log tail ---"
  tail -30 "$WORKDIR/frontend.log"
  echo "--- vllm_worker.log tail ---"
  tail -30 "$WORKDIR/vllm_worker.log"
fi
exit $PROBE_EXIT
