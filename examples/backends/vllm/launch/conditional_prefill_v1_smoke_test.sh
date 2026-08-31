#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# v1 smoke test for the IslBoundingPolicy on the vLLM backend.
#
# Mirrors examples/backends/trtllm/launch/conditional_disagg_v1_smoke_test.sh
# but spins up vLLM workers instead of trtllm. The router-side flags
# (--router-conditional-disagg*) are engine-agnostic; the only vLLM-specific
# bits are the worker invocation (--kv-transfer-config / --kv-events-config
# instead of --publish-kv-events + YAML engine args).
#
# Pure router-side fast-path policy, no cost_eval sidecar (v2 work pending
# Hongkuan's engine-metrics Rust shim). The policy bypasses to AGG when both:
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
# in DECODE-mode — this is the case where vLLM's DecodeWorkerHandler would
# normally take the decode-only path expecting KV-transfer metadata from a
# prefill worker. With the `x-bypass-remote-prefill` annotation attached by
# the v1 PrefillRouter, the handler flips `is_decode_only=False` for the
# bypassed request and runs full prefill+decode locally.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ----- inlined helpers ---------------------------------------------------
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
export MODEL=${MODEL:-"Qwen/Qwen3-0.6B"}
export BLOCK_SIZE=${BLOCK_SIZE:-64}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"1"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"0"}

# Required for deterministic KV-event hash IDs (matches examples/.../agg_router.sh).
export PYTHONHASHSEED=0

# IslBoundingPolicy thresholds.
export EFF_ISL_THRESHOLD=${EFF_ISL_THRESHOLD:-2048}
export EFF_ISL_RATIO_THRESHOLD=${EFF_ISL_RATIO_THRESHOLD:-0.7}

# v1.5: which conditional-disagg policy to exercise.
#   isl_bounding   — v1 behavior (ISL-only).
#   prefill_load   — v1.5 load-only gate; runs Phase B (load bypass) and Phase C
#                    (calm baseline) probes after Phase A. Phase A's ISL-cached
#                    visits should NOT bypass with this policy.
#   isl_or_load    — v1.5 OR composition; runs all three phases. Phase A still
#                    bypasses via the ISL gate as in v1.
export POLICY=${POLICY:-isl_bounding}

# v1.5: dedicated busy-line fraction for the load gate. Default 0.05 is
# intentionally aggressive for the small-model smoke test: vLLM's default
# max_num_batched_tokens (~8192) × 0.05 = ~400-token busy line, easily
# tripped by a single in-flight 5000-token prompt. Raise (e.g. 0.5) for
# production-like sensitivity tests.
export BUSY_THRESHOLD=${BUSY_THRESHOLD:-0.05}

# Any additional args to pass through to `python -m dynamo.frontend`.
export EXTRA_ROUTER_ARGS=${EXTRA_ROUTER_ARGS:-""}

# Probe knobs
PROBE_WARMUP_SECONDS=${PROBE_WARMUP_SECONDS:-150}
PROBE_INTER_REQUEST_SLEEP=${PROBE_INTER_REQUEST_SLEEP:-1.0}
PROBE_MAX_TOKENS=${PROBE_MAX_TOKENS:-24}

# NIXL handshake port is set INLINE on the prefill worker only — decode
# uses the default. Exporting this globally would cause both workers to
# bind the same port and the decode worker's NIXL handshake listener
# would crash with `Address already in use`.
PREFILL_NIXL_SIDE_CHANNEL_PORT=${PREFILL_NIXL_SIDE_CHANNEL_PORT:-20097}

export DYN_LOG="info,dynamo_llm::kv_router::prefill_router=debug,dynamo_llm::kv_router=debug"
export DYN_LOG_LEVEL="debug"
export HF_HOME=${HF_HOME:-"/tmp/hf_cache"}

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner \
  "conditional-disagg smoke test on vLLM (1P+1D, policy=$POLICY, busy=$BUSY_THRESHOLD)" \
  "$MODEL" "$HTTP_PORT"

# Build the refactored router's structured conditional-disagg configuration.
CONDITIONAL_DISAGG_CONFIG=$(printf \
    '{"policy":"%s","eff_isl_threshold":%s,"eff_isl_ratio_threshold":%s' \
    "$POLICY" "$EFF_ISL_THRESHOLD" "$EFF_ISL_RATIO_THRESHOLD")
if [[ "$POLICY" == "prefill_load" || "$POLICY" == "isl_or_load" ]]; then
    CONDITIONAL_DISAGG_CONFIG+=",\"prefill_busy_threshold\":${BUSY_THRESHOLD}"
fi
CONDITIONAL_DISAGG_CONFIG+="}"

# ----- frontend (router lives here) --------------------------------------
# The frontend auto-detects prefill workers and activates the internal prefill
# router; the conditional-disagg policy plugs into that path.
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend \
    --router-mode kv \
    --router-kv-events \
    --router-conditional-disagg \
    --router-conditional-disagg-config "$CONDITIONAL_DISAGG_CONFIG" \
    --enforce-disagg \
    ${EXTRA_ROUTER_ARGS:-} &

# ----- decode worker (DECODE-mode handler) -------------------------------
# Conditional-disagg REQUIRES kv-events on the decode worker. The policy
# queries `decode_chosen_overlap_blocks` for a candidate decode worker; if
# the decode worker doesn't publish events, the indexer has no record of
# its cache state and `overlap_blocks` is always 0 → policy never bypasses.
# This mirrors trtllm's conditional_disagg_v1_smoke_test.sh which sets
# `--publish-kv-events` on BOTH prefill and decode workers (the generic
# disagg_router.sh only sets it on prefill, which is INSUFFICIENT here).
OTEL_SERVICE_NAME=dynamo-worker-decode \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size "$BLOCK_SIZE" \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# ----- prefill worker ----------------------------------------------------
# Publishes kv-events that feed the router indexer (matching disagg_router.sh).
OTEL_SERVICE_NAME=dynamo-worker-prefill \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_SIDE_CHANNEL_PORT \
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size "$BLOCK_SIZE" \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# ----- probe loop --------------------------------------------------------
# Engine-agnostic — same multi-turn workload used by the trtllm smoke test.
# Models 3 multi-turn conversations: each base accumulates an extension per
# visit. So visit k on base B sends `base_B + ext_1 + ext_2 + ... + ext_k`,
# and visit k+1's cache hit covers visit k's full prompt — meaning eff_isl
# on visit k+1 is just `ext_{k+1}`'s size.
(
  echo "[probe] sleeping ${PROBE_WARMUP_SECONDS}s for workers to register..."
  sleep "$PROBE_WARMUP_SECONDS"
  SERVED_MODEL_NAME="$MODEL" \
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

  # ----- Phase B / Phase C (v1.5 load gate) -----------------------------
  # Only meaningful when the policy consumes the prefill-busy signal.
  # Phase B: saturate the prefill worker with N parallel unique large
  #          prompts, then immediately fire a "would-be-disagg" probe (large
  #          enough that the ISL gate alone WOULD NOT bypass). Expected:
  #          frontend log emits "Conditional disagg routing to decode worker"
  #          for that probe.
  # Phase C: let the fleet drain, then resend the same large probe. Expected:
  #          no bypass — request goes through prefill normally.
  if [[ "$POLICY" == "prefill_load" || "$POLICY" == "isl_or_load" ]]; then
      echo "[v1.5-probe] policy=$POLICY busy_threshold=$BUSY_THRESHOLD; running Phase B + Phase C"
      SERVED_MODEL_NAME="$MODEL" \
      HTTP_PORT="$HTTP_PORT" \
      PROBE_MAX_TOKENS="$PROBE_MAX_TOKENS" \
      BURST_PARALLELISM="${BURST_PARALLELISM:-4}" \
      BURST_TARGET_CHARS="${BURST_TARGET_CHARS:-20000}" \
      LARGE_PROBE_TARGET_CHARS="${LARGE_PROBE_TARGET_CHARS:-20000}" \
      PHASE_C_DRAIN_SECONDS="${PHASE_C_DRAIN_SECONDS:-30}" \
      python3 - <<'PYEOF'
import concurrent.futures
import json
import os
import time
import urllib.request

model = os.environ["SERVED_MODEL_NAME"]
port = os.environ["HTTP_PORT"]
max_tokens = int(os.environ["PROBE_MAX_TOKENS"])
burst_n = int(os.environ["BURST_PARALLELISM"])
burst_target_chars = int(os.environ["BURST_TARGET_CHARS"])
probe_target_chars = int(os.environ["LARGE_PROBE_TARGET_CHARS"])
drain_s = float(os.environ["PHASE_C_DRAIN_SECONDS"])
url = f"http://localhost:{port}/v1/chat/completions"


def make_unique_filler(seed: int, target_chars: int) -> str:
    """Stable per-seed filler that is unlikely to overlap any other prompt
    so prefill cache hit is ~0 and active_tokens accurately reflects work."""
    chunk = (
        f" Unique tag {seed:08d}. Discuss the implications of distributed "
        "computing on the design of resilient stateful services with respect "
        "to consensus algorithms, leader election, and replicated logs, in a "
        "way that does not overlap any prior request's content. "
    )
    out = []
    while sum(len(s) for s in out) < target_chars:
        out.append(chunk)
    return "".join(out)


def send(label: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        dt = time.time() - t0
        print(f"[v1.5-probe] {label} OK ({dt:.1f}s, prompt_chars={len(prompt)})", flush=True)
        return {"label": label, "ok": True, "duration_s": dt}
    except Exception as e:
        dt = time.time() - t0
        print(f"[v1.5-probe] {label} FAILED ({dt:.1f}s): {e}", flush=True)
        return {"label": label, "ok": False, "duration_s": dt, "err": str(e)}


# Burst prompts are ~20000 chars (~5000 tokens) each — heavy enough that
# prefill compute spans hundreds of ms even on Qwen3-0.6B, widening the
# peek window. BURST_PARALLELISM=4 (default) pushes the peak active_tokens
# to ~20000 across the worker, far above the busy line at
# BUSY_THRESHOLD * max_num_batched_tokens (~400 with the 0.05 default).
burst_prompts = [
    (f"burst-{i}", make_unique_filler(seed=1000 + i, target_chars=burst_target_chars))
    for i in range(burst_n)
]
# Phase B and Phase C probes use DIFFERENT seeds so the Phase C prompt has
# no cache overlap from Phase B's run. Without this, the ISL gate would
# bypass the Phase C probe (overlap_tokens ≈ full prompt) and we couldn't
# distinguish "load gate said calm" from "ISL gate said cached".
phase_b_probe_prompt = make_unique_filler(seed=2000, target_chars=probe_target_chars)
phase_c_probe_prompt = make_unique_filler(seed=3000, target_chars=probe_target_chars)

# Phase B: submit burst + probe near-simultaneously. NO pre-probe sleep —
# the probe should arrive at the router while burst requests are still
# resident in active_tokens (i.e. inside their prefill window).
print(
    f"[v1.5-probe] Phase B: firing {burst_n} parallel burst requests "
    f"+ probe (burst_chars={burst_target_chars}, probe_chars={probe_target_chars})",
    flush=True,
)
with concurrent.futures.ThreadPoolExecutor(max_workers=burst_n + 1) as pool:
    burst_futures = [pool.submit(send, label, prompt) for label, prompt in burst_prompts]
    probe_future_b = pool.submit(send, "phase-B-probe", phase_b_probe_prompt)

    for f in burst_futures:
        f.result()
    probe_b = probe_future_b.result()

# Phase C: drain, then send a DIFFERENT fresh prompt. With prefill calm and
# zero cache overlap, neither gate should fire and the request should fall
# through to disagg as usual.
print(f"[v1.5-probe] Phase C: sleeping {drain_s:.0f}s for prefill to drain", flush=True)
time.sleep(drain_s)
print("[v1.5-probe] Phase C: sending a fresh probe under calm load", flush=True)
probe_c = send("phase-C-probe", phase_c_probe_prompt)

print("[v1.5-probe] done", flush=True)
print(
    "[v1.5-probe] inspect frontend logs for 'Conditional disagg routing to decode worker' "
    "on phase-B-probe (should appear — load gate fired) and phase-C-probe "
    "(should NOT appear — fresh prompt, calm fleet).",
    flush=True,
)
PYEOF
  fi

  echo "[probe] parking subshell so wait_any_exit doesn't fire on us"
  sleep infinity
) &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest.
wait_any_exit
