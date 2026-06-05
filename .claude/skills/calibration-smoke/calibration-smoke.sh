#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Calibration smoke for the prefill-router calibrate handler. Brings up
# hub + auto-wrapped Qwen prefill + decode (same launchers as
# disagg-smoke/prefill-router-smoke.sh) and exercises:
#
#   1. fresh calibration  -> from_cache=false, populated performance_model
#   2. cached re-issue     -> from_cache=true, identical body
#   3. force-refresh       -> from_cache=false, disjoint first_token set
#   4. per-run first_token uniqueness  (cache-buster correctness)
#
# Why a separate smoke from prefill-router-smoke.sh: that one exercises
# dispatch (R1+R2). This one exercises the sibling calibrate handler.
# Both go through detect.py::try_wrap_engine, so this also proves the
# auto-wrap path now passes the calibrate lambda + framework defaults.
#
# Profile assumption: spark-gb10 defaults KVBM_MAX_MODEL_LEN=1024 which
# is too small for any quadratic ISL sweep (resolver requires >= 4
# distinct ISLs). We bump to 8192 by default and sweep 512..8192. The
# user can override CALIB_SEQ / CALIB_OSL / KVBM_MAX_MODEL_LEN.
set -eu

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO=${KVBM_REPO:-$(cd "$SMOKE_DIR/../../.." && pwd)}
export KVBM_VENV=${KVBM_VENV:-$DYNAMO/.sandbox}
SKILL_BRINGUP=$DYNAMO/.claude/skills/disagg-bringup
HUB_READY_TIMEOUT=${KVBM_HUB_READY_TIMEOUT:-300}
VLLM_READY_TIMEOUT=${KVBM_VLLM_READY_TIMEOUT:-300}
CALIB_TIMEOUT=${KVBM_CALIB_TIMEOUT:-600}
LABEL=${KVBM_EXPERIMENT_LABEL:-calibration-smoke}
KVBM_BLOCK_LAYOUT=${KVBM_BLOCK_LAYOUT:-operational}
export KVBM_BLOCK_LAYOUT

# Calibration knobs.
CALIB_SEQ=${CALIB_SEQ:-512,1024,2048,4096,8192}
CALIB_OSL=${CALIB_OSL:-16}

# Convert the comma list to a JSON array for the request body and grab
# the high-watermark so we can size max_model_len accordingly.
SEQ_JSON=$(python3 -c "import json,sys; print(json.dumps([int(x) for x in sys.argv[1].split(',') if x]))" "$CALIB_SEQ")
SEQ_MAX=$(python3 -c "import sys; print(max(int(x) for x in sys.argv[1].split(',') if x))" "$CALIB_SEQ")
SEQ_COUNT=$(python3 -c "import sys; print(len([x for x in sys.argv[1].split(',') if x]))" "$CALIB_SEQ")

if [ "$SEQ_COUNT" -lt 4 ]; then
  echo "FATAL: CALIB_SEQ has $SEQ_COUNT entries; the quadratic TTFT fit needs >= 4 distinct ISLs" >&2
  exit 2
fi

# Auto-size the model context cap when the user hasn't explicitly
# overridden it. The spark-gb10 profile's 1024 is way too small for
# any sweep; even "bump to 8192" doesn't fit a top ISL of 8192 with
# osl=16 because the engine cap is on ISL+OSL, not ISL.
#
# IMPORTANT: the hub is the source of truth — kvbmctl renders the
# worker's `--max-model-len` from the hub's primary `max_seq_len`,
# which is set via `KVBM_HUB_MAX_SEQ_LEN` (defaults to 1024 in
# kvbm-hub-bringup/start-hub.sh). `KVBM_MAX_MODEL_LEN` from
# hardware-profiles.sh is NOT consumed by launch-prefill-kvbm-wrap.sh
# or launch-decode.sh. We have to export `KVBM_HUB_MAX_SEQ_LEN`
# itself to flow the cap end-to-end.
#
# `MAX_SEQ_LEN` must be a non-zero multiple of the hub block size
# (default 16, set via `KVBM_HUB_BLOCK_SIZE`). We round up.
HUB_BLOCK=${KVBM_HUB_BLOCK_SIZE:-16}
NEEDED_MAX_LEN=$(( SEQ_MAX + CALIB_OSL + 64 ))
NEEDED_MAX_LEN=$(( ((NEEDED_MAX_LEN + HUB_BLOCK - 1) / HUB_BLOCK) * HUB_BLOCK ))
DEFAULT_MAX_LEN=8192
if [ "$NEEDED_MAX_LEN" -gt "$DEFAULT_MAX_LEN" ]; then
  DEFAULT_MAX_LEN=$NEEDED_MAX_LEN
fi

# Honor caller's override on either variable (caller may have set
# KVBM_HUB_MAX_SEQ_LEN directly, or the legacy KVBM_MAX_MODEL_LEN we
# previously documented). Pick the larger of {auto-computed default,
# whatever the caller set} so the smoke never silently runs with a
# cap below what the sweep needs.
EFFECTIVE_CAP=$DEFAULT_MAX_LEN
for v in "${KVBM_HUB_MAX_SEQ_LEN:-}" "${KVBM_MAX_MODEL_LEN:-}"; do
  if [ -n "$v" ] && [ "$v" -gt "$EFFECTIVE_CAP" ]; then
    EFFECTIVE_CAP=$v
  fi
done

# Export BOTH so:
#   - start-hub.sh picks up KVBM_HUB_MAX_SEQ_LEN as the hub's primary
#     `--max-seq-len`, which kvbmctl then renders as the workers'
#     `--max-model-len`.
#   - hardware-profiles.sh sees KVBM_MAX_MODEL_LEN already set and
#     won't reset it back to its 1024 profile default. Belt and
#     suspenders — keeps the two in sync for the validator's
#     `defaults.max_seq_len == KVBM_MAX_MODEL_LEN` cross-check.
export KVBM_HUB_MAX_SEQ_LEN=$EFFECTIVE_CAP
export KVBM_MAX_MODEL_LEN=$EFFECTIVE_CAP

if [ "$(( SEQ_MAX + CALIB_OSL ))" -gt "$KVBM_HUB_MAX_SEQ_LEN" ]; then
  echo "FATAL: max ISL $SEQ_MAX + osl $CALIB_OSL = $(( SEQ_MAX + CALIB_OSL )) exceeds KVBM_HUB_MAX_SEQ_LEN $KVBM_HUB_MAX_SEQ_LEN" >&2
  exit 2
fi
if [ "$(( KVBM_HUB_MAX_SEQ_LEN % HUB_BLOCK ))" -ne 0 ]; then
  echo "FATAL: KVBM_HUB_MAX_SEQ_LEN $KVBM_HUB_MAX_SEQ_LEN is not a multiple of KVBM_HUB_BLOCK_SIZE $HUB_BLOCK" >&2
  exit 2
fi

ROOT=${1:-$(bash $SKILL_BRINGUP/new-experiment.sh "$LABEL")}
echo "EXP=$ROOT"
echo "calib: seq=$CALIB_SEQ osl=$CALIB_OSL hub_max_seq_len=$KVBM_HUB_MAX_SEQ_LEN block=$HUB_BLOCK (needed >= $(( SEQ_MAX + CALIB_OSL )))"

# Tear down stale.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -f "kvbm.vllm.prefill"        2>/dev/null || true
pkill -f "dynamo.vllm"               2>/dev/null || true
sleep 3
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
[ -n "$PIDS" ] && echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub 2>/dev/null || true
rm -f /tmp/velo-kvbm-*.sock 2>/dev/null || true
sleep 1

fail_dump() {
  echo "FATAL: $1" >&2
  if [ -n "${2:-}" ] && [ -f "$2" ]; then
    echo "--- tail $2 ---" >&2
    tail -n 50 "$2" | sed 's/\x1b\[[0-9;]*m//g' >&2
  fi
  pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
  pkill -f "kvbm.vllm.prefill"        2>/dev/null || true
  pkill -f "dynamo.vllm"               2>/dev/null || true
  pkill -9 -f kvbm_hub                  2>/dev/null || true
  exit 1
}

: > "$ROOT/hub.log"
: > "$ROOT/prefill.log"
: > "$ROOT/decode.log"

# Hub with --prefill-router.
bash $SKILL_BRINGUP/start-hub.sh "$ROOT/hub.log" &
HUB_PID=$!

echo "waiting for hub /health (timeout ${HUB_READY_TIMEOUT}s)..."
hub_deadline=$(( $(date +%s) + HUB_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8337/health >/dev/null 2>&1; do
  kill -0 "$HUB_PID" 2>/dev/null || fail_dump "hub process exited before becoming ready" "$ROOT/hub.log"
  [ "$(date +%s)" -ge "$hub_deadline" ] && fail_dump "hub not ready after ${HUB_READY_TIMEOUT}s" "$ROOT/hub.log"
  sleep 2
done
echo "HUB UP $(date)"

if ! curl -fsS -m 5 http://127.0.0.1:1337/v1/config | grep -q '"prefill_router"'; then
  fail_dump "hub did not register the prefill_router feature (check --prefill-router flag in start-hub.sh)" "$ROOT/hub.log"
fi

# Prefill (auto-wrap path — the one that should now register the
# calibrate handler since detect.py::try_wrap_engine was updated).
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info,kvbm_hub=debug} \
  bash $SKILL_BRINGUP/launch-prefill-kvbm-wrap.sh > "$ROOT/prefill.log" 2>&1 &
PREFILL_PID=$!

echo "waiting for prefill auto-wire log line (timeout ${VLLM_READY_TIMEOUT}s)..."
prefill_deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until grep -aq "kvbm prefill router auto-wired" "$ROOT/prefill.log" 2>/dev/null; do
  kill -0 "$PREFILL_PID" 2>/dev/null || fail_dump "prefill exited before auto-wire" "$ROOT/prefill.log"
  [ "$(date +%s)" -ge "$prefill_deadline" ] && fail_dump "no auto-wire log line after ${VLLM_READY_TIMEOUT}s" "$ROOT/prefill.log"
  sleep 5
done
echo "PREFILL AUTO-WIRED $(date)"

# Decode — needed for the hub's disagg feature to be happy. Decode is
# idle during calibration; we still bring it up so the bringup path
# matches the proven prefill-router-smoke.sh shape.
RUST_LOG=${RUST_LOG:-info,kvbm_connector=debug,kvbm_audit=info} \
  bash $SKILL_BRINGUP/launch-decode.sh > "$ROOT/decode.log" 2>&1 &
DECODE_PID=$!

echo "waiting for decode (timeout ${VLLM_READY_TIMEOUT}s)..."
vllm_deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; do
  kill -0 "$DECODE_PID" 2>/dev/null || fail_dump "decode exited before ready" "$ROOT/decode.log"
  [ "$(date +%s)" -ge "$vllm_deadline" ] && fail_dump "decode not ready after ${VLLM_READY_TIMEOUT}s" "$ROOT/decode.log"
  sleep 5
done
echo "DECODE UP $(date)"

# Discover the prefill instance_id.
TARGETS_JSON=$(curl -fsS -m 5 http://127.0.0.1:8337/v1/features/prefill-router/targets) \
  || fail_dump "could not GET /v1/features/prefill-router/targets" "$ROOT/hub.log"
echo "Targets: $TARGETS_JSON"
PREFILL_ID=$(echo "$TARGETS_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
velo = [t for t in d["targets"] if t["backend"] == "velo"]
if len(velo) != 1:
    print(f"expected exactly 1 velo target, got {len(velo)}: {d}", file=sys.stderr)
    sys.exit(1)
print(velo[0]["instance_id"])
')
[ -n "$PREFILL_ID" ] || fail_dump "no velo prefill target registered with hub" "$ROOT/hub.log"
echo "PREFILL_ID=$PREFILL_ID"

CALIB_URL="http://127.0.0.1:8337/v1/features/prefill-router/calibrate/$PREFILL_ID"
BODY=$(python3 -c "import json,sys; print(json.dumps({'seq': json.loads(sys.argv[1]), 'osl': int(sys.argv[2])}))" "$SEQ_JSON" "$CALIB_OSL")
echo "calib request body: $BODY"

run_calibrate() {
  # $1: outfile, $2: extra query string (e.g. "?force=true" or "")
  local out="$1" qs="$2"
  curl --max-time "$CALIB_TIMEOUT" -sS -w '\nHTTP_STATUS:%{http_code}\nELAPSED:%{time_total}\n' \
    -X POST "$CALIB_URL$qs" \
    -H 'content-type: application/json' \
    -d "$BODY" > "$out" 2>&1 || true
  # Strip and parse status; split body from trailers.
  local status elapsed body
  status=$(grep '^HTTP_STATUS:' "$out" | tail -1 | cut -d: -f2)
  elapsed=$(grep '^ELAPSED:' "$out" | tail -1 | cut -d: -f2)
  body=$(grep -vE '^(HTTP_STATUS|ELAPSED):' "$out")
  echo "$body" > "$out"
  echo "calib HTTP $status (elapsed=${elapsed}s) -> $out" >&2
  if [ "$status" != "200" ]; then
    echo "--- response body ---" >&2
    head -c 1000 "$out" >&2
    echo >&2
    fail_dump "calibrate returned HTTP $status" "$ROOT/hub.log"
  fi
}

# === Validation step 1: fresh calibration ===
echo "=== STEP 1: fresh calibration ==="
run_calibrate "$ROOT/calib-first.json" ""

python3 - "$ROOT/calib-first.json" "$SEQ_JSON" "$KVBM_MAX_MODEL_LEN" <<'PY' || fail_dump "fresh calibration response failed validation" "$ROOT/calib-first.json"
import json, math, sys, pathlib
path, seq_json, max_seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
requested_seq = json.loads(seq_json)
d = json.loads(pathlib.Path(path).read_text())

assert d["from_cache"] is False, f"expected from_cache=false, got {d['from_cache']}"

pm = d["results"]["performance_model"]
for k in ("t2ft_intercept", "t2ft_linear", "t2ft_quadratic",
         "t2tl_intercept", "t2tl_slope", "t2ft_fit_r2", "t2tl_fit_r2"):
    v = pm[k]
    assert isinstance(v, (int, float)) and math.isfinite(v), f"performance_model.{k} not finite: {v}"
assert 0.0 <= pm["t2ft_fit_r2"] <= 1.0001, f"t2ft_fit_r2 out of range: {pm['t2ft_fit_r2']}"
assert 0.0 <= pm["t2tl_fit_r2"] <= 1.0001, f"t2tl_fit_r2 out of range: {pm['t2tl_fit_r2']}"

# Compare trace ISLs against resolved.seq (what the resolver actually
# chose to run), not the raw request — the resolver may clamp entries
# that don't fit max_seq_len/max_input_len. Log any clamps explicitly
# so a partial sweep is visible in the smoke output without masking
# the failure of "every requested ISL ran".
resolved = d["resolved"]
resolved_seq = resolved["seq"]
clamps = resolved.get("clamps", [])
if resolved_seq != requested_seq:
    print(f"  WARN: resolver clamped {requested_seq} -> {resolved_seq}")
    for c in clamps:
        print(f"        clamp: {c}")

traces = d["results"]["traces"]
got_isls = [t["isl"] for t in traces]
assert got_isls == resolved_seq, \
    f"trace ISLs {got_isls} don't match resolved.seq {resolved_seq}"
assert len(traces) >= 4, \
    f"only {len(traces)} traces ran; quadratic fit needs >= 4 (clamps={clamps})"

n_opt, n_att = d["results"]["n_opt"], d["results"]["n_att"]
assert n_opt > 0, f"n_opt must be > 0, got {n_opt}"
assert n_att > 0, f"n_att must be > 0, got {n_att}"

assert resolved["osl"] == traces[0]["osl"], "resolved.osl / trace osl mismatch"

defaults = d["defaults"]
assert defaults["max_seq_len"] == max_seq, \
    f"defaults.max_seq_len {defaults['max_seq_len']} != KVBM_MAX_MODEL_LEN {max_seq}"

first_tokens = [t["first_token"] for t in traces]
assert len(set(first_tokens)) == len(first_tokens), \
    f"first_token collisions within a single run: {first_tokens}"

print(f"  performance_model: t2ft={pm['t2ft_intercept']:.1f} + {pm['t2ft_linear']:.3f}*ISL + {pm['t2ft_quadratic']:.3e}*ISL^2 (R2={pm['t2ft_fit_r2']:.3f})")
print(f"  t2tl = {pm['t2tl_intercept']:.1f} + {pm['t2tl_slope']:.3f}*pos (R2={pm['t2tl_fit_r2']:.3f})")
print(f"  n_opt={n_opt}  n_att={n_att}")
print(f"  first_tokens={first_tokens}")
PY
echo "STEP 1 OK"

# === Validation step 2: cached re-issue ===
echo "=== STEP 2: cached re-issue ==="
run_calibrate "$ROOT/calib-cached.json" ""

python3 - "$ROOT/calib-first.json" "$ROOT/calib-cached.json" <<'PY' || fail_dump "cached re-issue failed validation" "$ROOT/calib-cached.json"
import json, sys, pathlib
first = json.loads(pathlib.Path(sys.argv[1]).read_text())
cached = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert cached["from_cache"] is True, f"expected from_cache=true on re-issue, got {cached['from_cache']}"
# Results payload must be identical to the cached snapshot.
assert cached["results"] == first["results"], "cached results diverged from the first run"
assert cached["resolved"] == first["resolved"], "cached resolved diverged from the first run"
print("  cached body matches first-run body byte-for-byte (post-JSON normalization)")
PY
echo "STEP 2 OK"

# === Validation step 3: force-refresh ===
echo "=== STEP 3: force-refresh ==="
run_calibrate "$ROOT/calib-forced.json" "?force=true"

python3 - "$ROOT/calib-first.json" "$ROOT/calib-forced.json" <<'PY' || fail_dump "force-refresh failed validation" "$ROOT/calib-forced.json"
import json, sys, pathlib
first = json.loads(pathlib.Path(sys.argv[1]).read_text())
forced = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert forced["from_cache"] is False, f"force=true must produce a fresh run, got from_cache={forced['from_cache']}"

first_set = {t["first_token"] for t in first["results"]["traces"]}
forced_set = {t["first_token"] for t in forced["results"]["traces"]}
overlap = first_set & forced_set
assert not overlap, f"force-refresh re-used first_token values across runs: {overlap}"
# Forced run should also have all-distinct first tokens internally.
forced_tokens = [t["first_token"] for t in forced["results"]["traces"]]
assert len(set(forced_tokens)) == len(forced_tokens), \
    f"first_token collisions within forced run: {forced_tokens}"
print(f"  forced run first_tokens disjoint from first run ({len(first_set)} + {len(forced_set)} distinct values, no overlap)")
PY
echo "STEP 3 OK"

# === Validation step 4: different-request must MISS the cache ===
# This catches the bug where the cache hit path returns the cached
# snapshot regardless of whether the new request's parameters match.
# We issue a request with a different sweep (and OSL when possible)
# WITHOUT force=true and expect from_cache=false. Then we re-issue
# the SAME different request and expect from_cache=true on THAT (so
# the cache now holds the new snapshot, not the original).
echo "=== STEP 4: different-request cache miss ==="
# Halve every ISL to guarantee a different resolved request without
# blowing the cap. Halve OSL too if it's >=2; otherwise just rely on
# the seq diff. Drop any sub-2 ISLs and bail if we ended up with <4
# distinct points (the resolver would reject it anyway).
DIFF_SEQ_JSON=$(python3 - "$SEQ_JSON" "$CALIB_OSL" <<'PY'
import json, sys
seq = json.loads(sys.argv[1])
diff = sorted({max(2, isl // 2) for isl in seq})
if len(diff) < 4:
    # Pad up by extending the original tail so we keep >=4 distinct ISLs.
    for isl in seq:
        if isl not in diff:
            diff.append(isl)
        if len(diff) >= 4:
            break
    diff = sorted(set(diff))
print(json.dumps(diff))
PY
)
DIFF_OSL=$(( CALIB_OSL > 1 ? CALIB_OSL / 2 : CALIB_OSL ))
[ "$DIFF_OSL" -lt 1 ] && DIFF_OSL=1
DIFF_BODY=$(python3 -c "import json,sys; print(json.dumps({'seq': json.loads(sys.argv[1]), 'osl': int(sys.argv[2])}))" "$DIFF_SEQ_JSON" "$DIFF_OSL")
echo "diff request body: $DIFF_BODY"
DIFF_URL="$CALIB_URL"
curl --max-time "$CALIB_TIMEOUT" -sS -w '\nHTTP_STATUS:%{http_code}\n' \
  -X POST "$DIFF_URL" -H 'content-type: application/json' \
  -d "$DIFF_BODY" > "$ROOT/calib-diff.json" 2>&1 || true
DIFF_STATUS=$(grep '^HTTP_STATUS:' "$ROOT/calib-diff.json" | tail -1 | cut -d: -f2)
grep -v '^HTTP_STATUS:' "$ROOT/calib-diff.json" > "$ROOT/calib-diff.json.tmp" && mv "$ROOT/calib-diff.json.tmp" "$ROOT/calib-diff.json"
[ "$DIFF_STATUS" = "200" ] || fail_dump "different-request calibrate returned HTTP $DIFF_STATUS" "$ROOT/calib-diff.json"

python3 - "$ROOT/calib-forced.json" "$ROOT/calib-diff.json" "$DIFF_SEQ_JSON" "$DIFF_OSL" <<'PY' || fail_dump "different-request cache-miss check failed" "$ROOT/calib-diff.json"
import json, sys, pathlib
prior = json.loads(pathlib.Path(sys.argv[1]).read_text())   # most recent cached snapshot
diff = json.loads(pathlib.Path(sys.argv[2]).read_text())    # response to the different request
expected_seq = json.loads(sys.argv[3])
expected_osl = int(sys.argv[4])

# from_cache=true here would mean the handler ignored the changed
# request body and served the prior snapshot. That's the stale-cache
# bug.
assert diff["from_cache"] is False, \
    f"different-request must MISS the cache; got from_cache={diff['from_cache']} (cached resolved.seq={prior['resolved']['seq']}, requested seq={expected_seq})"

# The fresh run must reflect the NEW request's resolved knobs.
assert diff["resolved"]["seq"] == expected_seq, \
    f"diff run resolved.seq {diff['resolved']['seq']} != requested {expected_seq}"
assert diff["resolved"]["osl"] == expected_osl, \
    f"diff run resolved.osl {diff['resolved']['osl']} != requested {expected_osl}"

# Sanity: results must be a fresh fit (not a copy of the prior one).
assert diff["results"]["traces"] != prior["results"]["traces"], \
    "diff run returned identical traces to the prior snapshot — likely stale cache"

print(f"  different-request issued seq={expected_seq} osl={expected_osl} -> from_cache=false (correct)")
print(f"  new resolved.seq={diff['resolved']['seq']}")
PY

# Re-issue the SAME different request to confirm the cache now keys
# off the new resolved snapshot.
curl --max-time "$CALIB_TIMEOUT" -sS -w '\nHTTP_STATUS:%{http_code}\n' \
  -X POST "$DIFF_URL" -H 'content-type: application/json' \
  -d "$DIFF_BODY" > "$ROOT/calib-diff-cached.json" 2>&1 || true
DC_STATUS=$(grep '^HTTP_STATUS:' "$ROOT/calib-diff-cached.json" | tail -1 | cut -d: -f2)
grep -v '^HTTP_STATUS:' "$ROOT/calib-diff-cached.json" > "$ROOT/calib-diff-cached.json.tmp" && mv "$ROOT/calib-diff-cached.json.tmp" "$ROOT/calib-diff-cached.json"
[ "$DC_STATUS" = "200" ] || fail_dump "different-request cache re-issue returned HTTP $DC_STATUS" "$ROOT/calib-diff-cached.json"

python3 - "$ROOT/calib-diff.json" "$ROOT/calib-diff-cached.json" <<'PY' || fail_dump "different-request cache re-issue failed" "$ROOT/calib-diff-cached.json"
import json, sys, pathlib
first = json.loads(pathlib.Path(sys.argv[1]).read_text())
cached = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert cached["from_cache"] is True, \
    f"second issue of the new request must HIT the new cached snapshot, got from_cache={cached['from_cache']}"
assert cached["results"] == first["results"], "new-snapshot cache hit diverged from its fresh run"
print("  re-issued diff request -> from_cache=true against the NEW snapshot")
PY
echo "STEP 4 OK"

# === Optional observability snapshot ===
echo
echo "-- hub counters --"
curl -sS -m 5 http://127.0.0.1:8337/v1/features/prefill-router/counters | python3 -m json.tool 2>/dev/null || true

echo
echo "-- prefill auto-wire log line --"
grep -a "kvbm prefill router auto-wired" "$ROOT/prefill.log" | head -1 | sed 's/\x1b\[[0-9;]*m//g'

echo
echo "calibration-smoke: DONE  (instance=$PREFILL_ID seq=$CALIB_SEQ osl=$CALIB_OSL)"

# Teardown.
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
pkill -f "kvbm.vllm.prefill"        2>/dev/null || true
pkill -f "dynamo.vllm"               2>/dev/null || true
sleep 2
pkill -9 -f kvbm_hub                  2>/dev/null || true
