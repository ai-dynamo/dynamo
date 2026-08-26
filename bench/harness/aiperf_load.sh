#!/usr/bin/env bash
# Drive load with aiperf in pure synthetic mode, and keep its artifacts.
# This replaces a hand-rolled curl loop. That loop worked, but it produced no
# record of what it had actually sent, so a run could only be labelled "loaded"
# by watching the engine's own scheduler line -- and when that reading turned
# out to predate the kill by ~15 s, every load label became unverifiable after
# the fact. aiperf writes a profile export per request, so the offered load
# becomes an artifact instead of an assertion.
# Synthetic mode is selected by NOT passing --input-file. No dataset, no
# mooncake trace, no session accumulation, and none of the trace-preparation
# steps that cost several failed runs -- just a fixed prompt/output shape at a
# fixed concurrency, which is all a failover profile needs.
# Holding every request in decode for its full output length matters here: a
# request that stops early at EOS frees its slot and the offered load sags right
# when the kill lands. --force-min-tokens looks like the modern way to do that
# but is rejected outside the baseten_trace loader:
#   --force-min-tokens is only supported by the baseten_trace loader;
#   provide --input-file and --custom-dataset-type baseten_trace.
# So synthetic mode passes ignore_eos through --extra-inputs, which the server
# honours regardless of loader.
set -uo pipefail

MODEL=${MODEL:?set MODEL}
URL=${URL:?set URL}
TOKENIZER=${TOKENIZER:?set TOKENIZER}
ARTIFACT_DIR=${ARTIFACT_DIR:?set ARTIFACT_DIR}
CONCURRENCY=${CONCURRENCY:-64}
ISL=${ISL:-1200}
OSL=${OSL:-20000}
DURATION=${DURATION:-180}
WARMUP=${WARMUP:-0}
# A mooncake trace is multi-turn: every record sharing a session_id is one
# conversation. The completions endpoint rejects that outright --
# ValueError("Completions endpoint only supports one turn.") -- so trace runs
# must go to chat, which is also what the published cascade script relies on by
# leaving --endpoint-type unset. Synthetic runs stay on completions, where the
# single-turn prompt is what we want and chat templating would only add noise.
if [ -n "${TRACE_FILE:-}" ]; then
  ENDPOINT_TYPE=${ENDPOINT_TYPE:-chat}
else
  ENDPOINT_TYPE=${ENDPOINT_TYPE:-completions}
fi
# Defaults below match bench/scripts/run_failover_bench.sh, the configuration the
# published cascade numbers were produced with, so a trace run here is comparable
# to them rather than merely similar.
RAMP_S=${RAMP_S:-0}                 # published run ramps concurrency over 45s
GRACE_S=${GRACE_S:-90}              # let in-flight long-context turns drain
WORKERS_MAX=${WORKERS_MAX:-200}     # 200k-context turns hold a worker for minutes
RECORD_PROCS=${RECORD_PROCS:-8}

mkdir -p "$ARTIFACT_DIR"

# The synthetic flags this script depends on do not exist before 0.12.0, and the
# bench image ships 0.7.0. Installing at run time rather than assuming, because
# the install lives in the pod's filesystem and is lost whenever the pod is
# recreated -- which the failover harness does between iterations.
AIPERF_MIN=${AIPERF_MIN:-0.12.0}
have=$(aiperf --version 2>/dev/null | tail -1 | tr -d ' ')
if [ "$have" != "$AIPERF_MIN" ]; then
  echo "aiperf $have != $AIPERF_MIN, installing"
  pip install -q --disable-pip-version-check --upgrade "aiperf==$AIPERF_MIN" >/dev/null 2>&1
  have=$(aiperf --version 2>/dev/null | tail -1 | tr -d ' ')
fi
echo "aiperf version: $have"
[ "$have" = "$AIPERF_MIN" ] || { echo "FATAL: could not get aiperf $AIPERF_MIN"; exit 1; }

# Tokenizer correctness matters: aiperf uses it for synthetic prompt generation
# and client-side token counting, so a wrong tokenizer silently corrupts every
# ISL/OSL figure it reports. So use the REAL repo from the REAL model cache.
# That requires transformers >= ~5.15: GLM-5.2's config.json declares
# layer_types of "deepseek_sparse_attention", which 5.7.0 (shipped in the bench
# image) does not recognise and rejects with a StrictDataclass validation error.
# An earlier workaround staged a cache with the real tokenizer files plus a stub
# config.json; it produced byte-identical tokenization, but relying on a
# hand-written config to read a tokenizer is not something to keep once the
# actual fix is a version bump.
TRANSFORMERS_MIN=${TRANSFORMERS_MIN:-5.15.1}
tv=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
if [ "$tv" != "$TRANSFORMERS_MIN" ]; then
  echo "transformers $tv != $TRANSFORMERS_MIN, installing"
  pip install -q --disable-pip-version-check --upgrade "transformers==$TRANSFORMERS_MIN" >/dev/null 2>&1
  tv=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
fi
echo "transformers version: $tv"
[ "$tv" = "$TRANSFORMERS_MIN" ] || { echo "FATAL: need transformers $TRANSFORMERS_MIN for the GLM tokenizer"; exit 1; }
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export AIPERF_HTTP_SO_RCVTIMEO=120
ulimit -n 65536 2>/dev/null || true

# aiperf rejects --num-warmup-requests 0 (must be > 0), so the flag is omitted
# entirely rather than passed as zero. Warmup is off by default here because the
# kill lands mid-run and a warmup phase would just delay reaching steady load.
WARMUP_ARGS=()
[ "${WARMUP:-0}" -gt 0 ] 2>/dev/null && WARMUP_ARGS=(--num-warmup-requests "$WARMUP")

# Trace mode replays a mooncake dataset instead of generating synthetic prompts.
# Worth having because synthetic load at stddev 0 self-synchronises: every lane
# runs an identical request, finishes at the same instant and restarts together,
# so completions arrive in convoys (measured: 16 waves, 6.5 s apart, median gap
# between consecutive completions 0 ms). A trace has real ISL/OSL spread and
# real prefix reuse, so no convoy can form.
# --no-fixed-schedule is required: 0.12 auto-switches trace datasets to
# timestamp-driven arrivals, which makes concurrency emergent and breaks the
# Running>=N gate the failover runs depend on.
RAMP_ARGS=()
[ "${RAMP_S:-0}" -gt 0 ] && RAMP_ARGS=(--concurrency-ramp-duration "$RAMP_S")

SHAPE_ARGS=()
if [ -n "${TRACE_FILE:-}" ]; then
  # --isl-block-size must be stated, not inferred. aiperf tries to deduce one
  # fixed block size from input_length / len(hash_ids) and aborts with
  # "inconsistent sizes ... indicate a corrupt trace" when that ratio varies.
  # It varies for any correct trace: a record whose last block is partially
  # filled has a lower ratio (578 tokens over 2 hash_ids reads as 289), so the
  # inference is simply wrong. Checked against the real thing -- all 13167
  # records satisfy ceil(input_length / 512) == len(hash_ids) exactly, with zero
  # mismatches, so the data was never the problem.
  SHAPE_ARGS=(--input-file "$TRACE_FILE"
              --custom-dataset-type "${TRACE_TYPE:-mooncake_trace}"
              --isl-block-size "${ISL_BLOCK_SIZE:-512}"
              --no-fixed-schedule)
  echo "aiperf trace: $TRACE_FILE concurrency=$CONCURRENCY duration=${DURATION}s"
else
  SHAPE_ARGS=(--prompt-input-tokens-mean "$ISL" --prompt-input-tokens-stddev "${ISL_STDDEV:-0}"
              --prompt-output-tokens-mean "$OSL" --prompt-output-tokens-stddev "${OSL_STDDEV:-0}")
  echo "aiperf synthetic: concurrency=$CONCURRENCY isl=$ISL osl=$OSL duration=${DURATION}s warmup=${WARMUP:-0}"
fi

# Open loop. With --concurrency the client keeps exactly N requests in flight and
# issues a new one only when an old one returns, so offered load falls in lockstep
# with capacity: killing half the fleet made the client send 38% FEWER requests,
# no queue ever formed, and TTFT had nothing to climb from. Setting a request rate
# decouples arrivals from completions, which is the only way a queue -- and
# therefore a visible degradation -- can exist at all.
# --concurrency is still passed, as a CAP rather than a target. It must stay well
# above steady-state occupancy (rate x latency) or it silently re-closes the loop.
RATE_ARGS=()
if [ -n "${REQUEST_RATE:-}" ]; then
  RATE_ARGS=(--request-rate "$REQUEST_RATE" --request-rate-mode "${RATE_MODE:-poisson}")
  [ "${RATE_RAMP_S:-0}" -gt 0 ] && RATE_ARGS+=(--request-rate-ramp-duration "$RATE_RAMP_S")
  echo "aiperf OPEN LOOP: rate=${REQUEST_RATE}/s mode=${RATE_MODE:-poisson} cap=$CONCURRENCY"
fi

exec aiperf profile \
  -m "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --url "$URL" \
  --endpoint-type "$ENDPOINT_TYPE" \
  --streaming \
  "${SHAPE_ARGS[@]}" \
  "${RATE_ARGS[@]}" \
  --extra-inputs ignore_eos:true \
  --concurrency "$CONCURRENCY" \
  --benchmark-duration "$DURATION" \
  "${WARMUP_ARGS[@]}" \
  --request-timeout-seconds "${REQ_TIMEOUT:-900}" \
  --benchmark-grace-period "$GRACE_S" \
  --workers-max "$WORKERS_MAX" --record-processors "$RECORD_PROCS" \
  "${RAMP_ARGS[@]}" \
  --random-seed 42 \
  --artifact-dir "$ARTIFACT_DIR" \
  --ui-type simple
