#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# v1 smoke test for the IslBoundingPolicy.
#
# Pure router-side fast-path policy, no cost_eval sidecar (that's v2 work
# pending Hongkuan's engine-metrics Rust shim). The policy bypasses to AGG
# when both:
#
#     eff_isl < EFF_ISL_THRESHOLD                       (absolute cap)
# AND eff_isl / max(prompt_tokens, 1) < EFF_ISL_RATIO_THRESHOLD   (cache-hit fraction)
#
# Override thresholds via env vars, e.g.
#   EFF_ISL_THRESHOLD=4096 EFF_ISL_RATIO_THRESHOLD=0.3 bash <this-script>
#
# Probe workload models multi-turn conversations: 3 base prompts of varying
# length, each visited 5 times. **Each visit appends a new extension to the
# accumulated prompt from the previous visit on that base** — i.e. visit k
# on base B sends `base_B + ext_1 + ext_2 + ... + ext_k`. So visit k+1's
# cache hit covers visit k's full prompt, and eff_isl on visit k+1 is just
# `ext_{k+1}`'s size.
#
# Extensions cycle through three sizes (TINY ~30 tok, MED ~150 tok, LARGE
# ~600 tok). The policy should bypass when the extension fits under
# EFF_ISL_THRESHOLD and the ratio condition holds; LARGE extensions should
# fail the absolute cap if EFF_ISL_THRESHOLD < 600.
#
# Visit order interleaves bases so cache eviction patterns get exercised
# alongside the basic bypass mechanic.
#
# Topology: 1 prefill + 1 decode worker (Qwen3-0.6B by default). Decode is
# in DECODE-mode — this is the case where handler_base.py would raise
# "Disaggregated params are required for decode mode" unless the router
# attaches the `x-bypass-remote-prefill` annotation. v1's PrefillRouter does.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ----- inlined helpers (was: examples/common/launch_utils.sh) -----------
# wait_any_exit: block until any backgrounded child exits, then propagate
# that child's exit code so the EXIT trap above tears down the group.
wait_any_exit() {
    if ! jobs -p | grep -q .; then
        echo "wait_any_exit: no background processes" >&2
        exit 1
    fi
    local _rc=0
    wait -n || _rc=$?
    echo "A background process exited with code $_rc"
    exit "$_rc"
}

# print_launch_banner <title> <model> <port>
print_launch_banner() {
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo "Model:    $2"
    echo "Frontend: http://localhost:$3"
    echo "=========================================="
}

# ----- environment + defaults --------------------------------------------
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-condp/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-condp/decode.yaml"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"1"}
export MODALITY=${MODALITY:-"text"}

# IslBoundingPolicy thresholds.
export EFF_ISL_THRESHOLD=${EFF_ISL_THRESHOLD:-2048}
export EFF_ISL_RATIO_THRESHOLD=${EFF_ISL_RATIO_THRESHOLD:-0.7}

# Any additional args to pass through to `python -m dynamo.frontend`.
export EXTRA_ROUTER_ARGS=${EXTRA_ROUTER_ARGS:-""}

# Probe knobs
PROBE_WARMUP_SECONDS=${PROBE_WARMUP_SECONDS:-50}
PROBE_INTER_REQUEST_SLEEP=${PROBE_INTER_REQUEST_SLEEP:-1.0}
PROBE_MAX_TOKENS=${PROBE_MAX_TOKENS:-24}

export DYN_LOG="info,dynamo_llm::kv_router::prefill_router=debug,dynamo_llm::kv_router=debug"
export DYN_LOG_LEVEL="debug"
export HF_HOME=${HF_HOME:-"/tmp/hf_cache"}

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner \
  "v1 IslBoundingPolicy smoke test (1P+1D, eff_isl<$EFF_ISL_THRESHOLD AND ratio<$EFF_ISL_RATIO_THRESHOLD)" \
  "$MODEL_PATH" "$HTTP_PORT"

# ----- frontend (router lives here) --------------------------------------
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --router-mode kv \
    --router-conditional-prefill \
    --router-conditional-prefill-policy isl_bounding \
    --router-conditional-prefill-eff-isl-threshold "$EFF_ISL_THRESHOLD" \
    --router-conditional-prefill-eff-isl-ratio-threshold "$EFF_ISL_RATIO_THRESHOLD" \
    --enforce-disagg \
    ${EXTRA_ROUTER_ARGS:-} &

# ----- prefill worker ----------------------------------------------------
OTEL_SERVICE_NAME=dynamo-worker-prefill \
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-kv-events \
  --disaggregation-mode prefill &

# ----- decode worker (DECODE-mode handler) -------------------------------
OTEL_SERVICE_NAME=dynamo-worker-decode \
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-kv-events \
  --disaggregation-mode decode &

# ----- probe loop --------------------------------------------------------
# Models 3 multi-turn conversations: each base accumulates an extension per
# visit. So visit k on base B sends `base_B + ext_1 + ext_2 + ... + ext_k`,
# and visit k+1's cache hit covers visit k's full prompt — meaning eff_isl
# on visit k+1 is just `ext_{k+1}`'s size.
(
  echo "[probe] sleeping ${PROBE_WARMUP_SECONDS}s for workers to register..."
  sleep "$PROBE_WARMUP_SECONDS"
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  HTTP_PORT="$HTTP_PORT" \
  PROBE_MAX_TOKENS="$PROBE_MAX_TOKENS" \
  PROBE_INTER_REQUEST_SLEEP="$PROBE_INTER_REQUEST_SLEEP" \
  python3 - <<'PYEOF'
import json
import os
import time
import urllib.request

model = os.environ["SERVED_MODEL_NAME"]
port = os.environ["HTTP_PORT"]
max_tokens = int(os.environ["PROBE_MAX_TOKENS"])
sleep_s = float(os.environ["PROBE_INTER_REQUEST_SLEEP"])

# Three base prompts, ascending in length.
BASE_SHORT = (
    "You are a terse assistant. Summarize the following statement in one "
    "sentence. Statement: Linear regression assumes a linear relationship "
    "between independent and dependent variables, normally distributed "
    "residuals, and homoscedasticity. "
)
BASE_MEDIUM = (
    "You are an assistant helping diagnose a slow database query. The query "
    "joins three tables (orders, customers, products) and filters by a date "
    "range. There are indexes on customer_id and product_id but not on the "
    "date column. Explain the likely bottlenecks and propose two ordered "
    "remediation steps. Be precise and avoid generic advice. "
) * 4  # ~1600 chars
BASE_LONG = (
    "You are reviewing a long technical document on distributed inference "
    "systems. The document covers tensor parallelism, pipeline parallelism, "
    "expert parallelism, and disaggregated prefill/decode topologies. It "
    "discusses tradeoffs between latency, throughput, and resource utilization "
    "across each design. The text also touches on KV cache management, "
    "speculative decoding, and continuous batching. "
) * 16  # ~6000 chars

# Extension sizes cycle TINY → MED → LARGE. Char counts chosen to land near
# 30/150/600 tokens at ~4 chars/token.
EXT_TINY = " Continue: respond briefly to that point. "
EXT_MED = (
    " Continue: provide a concrete follow-up example with at least three "
    "specific named entities and explain why each one is relevant to the "
    "previous discussion. Stay grounded in the prior context. "
) * 2  # ~600 chars
EXT_LARGE = (
    " Continue: now expand the analysis with a detailed comparison across "
    "multiple dimensions. For each dimension, describe a concrete scenario "
    "where the tradeoff matters, give an empirical rule of thumb, and tie "
    "it back to the original problem statement. Be thorough and specific. "
) * 10  # ~2400 chars

extensions = [EXT_TINY, EXT_MED, EXT_LARGE, EXT_TINY, EXT_MED]  # 5 visits per base

# Visit order interleaves bases so cache eviction is also exercised. 5
# visits per base = 15 requests total. visit_count[label] tracks which
# extension index to use next on that base.
visit_order = [
    "SHORT", "MEDIUM", "LONG",
    "MEDIUM", "LONG", "SHORT",
    "LONG", "SHORT", "MEDIUM",
    "SHORT", "MEDIUM", "LONG",
    "SHORT", "LONG", "MEDIUM",
]

bases_dict = {"SHORT": BASE_SHORT, "MEDIUM": BASE_MEDIUM, "LONG": BASE_LONG}
accum = dict(bases_dict)  # per-base accumulator, starts at the base prompt
visit_idx = {"SHORT": 0, "MEDIUM": 0, "LONG": 0}

url = f"http://localhost:{port}/v1/chat/completions"
for i, label in enumerate(visit_order, 1):
    ext = extensions[visit_idx[label]]
    visit_idx[label] += 1
    accum[label] = accum[label] + ext
    prompt = accum[label]
    ext_kind = ["TINY", "MED", "LARGE", "TINY", "MED"][visit_idx[label] - 1]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    print(
        f"[probe] {i:2d}/{len(visit_order)} base={label:6s} visit={visit_idx[label]} "
        f"ext={ext_kind:5s} total_chars={len(prompt):>6d} ext_chars={len(ext):>5d} START",
        flush=True,
    )
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            r.read()
        print(
            f"[probe] {i:2d}/{len(visit_order)} base={label:6s} visit={visit_idx[label]} "
            f"END  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    except Exception as e:
        print(
            f"[probe] {i:2d}/{len(visit_order)} base={label:6s} visit={visit_idx[label]} "
            f"FAILED: {e}",
            flush=True,
        )
    time.sleep(sleep_s)
print("[probe] done", flush=True)
PYEOF
  echo "[probe] parking subshell so wait_any_exit doesn't fire on us"
  sleep infinity
) &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest.
wait_any_exit
