#!/usr/bin/env bash
# multi-turn-r1-r2-r3-r4.sh — multi-turn CD coverage (4 sequential requests)
#
# Coverage gap closed:
#   `two-request-smoke.sh` only ever exercises R1+R2. The coordinator state
#   map, install/drop counters, and decode policy_decision history therefore
#   never see >2 entries in CI. State-machine leaks that only manifest after
#   a 3rd or 4th request — including stale rids that survive a release/evict
#   from an earlier round — are invisible. This scenario sends 4 sequential
#   requests with structured prefix relationships:
#
#       R1: ~80-token fresh prompt P1                        (cold prefill)
#       R2: P1 + ~40 net-new tokens                          (warm decode side)
#       R3: P2 + ~30 net-new tokens                          (warmer; sustained)
#       R4: ~100-token fresh, unrelated prompt P4            (cold; reset check)
#
# Oracles per round (all observed via PREFILL_LOG scoped by line baseline,
# DECODE_LOG, and the hub /v1/coordinator/states endpoint):
#
#   O1. HTTP 200 with non-empty content body for each round.       [universal]
#   O2. After each round settles (sleep 3), the coordinator state
#       map size returns to 0. cumulative_install == cumulative_drop
#       at end of run.                                              [universal]
#   O3. policy_decision audit row for each round shows the expected
#       cache-warmth direction:
#         R1: matched_tokens == 0    (cold)
#         R2: matched_tokens >  R1   (warmer than cold)
#         R3: matched_tokens >= R2   (sustained or growing)
#         R4: matched_tokens == 0    (cold again — fresh prompt)
#       INCONCLUSIVE if every prefill takes the legacy zero_passthrough
#       path (current v2-scheduler-disabled state per the verdict doc) —
#       in that mode warm-cache routing is structurally unreachable and
#       this oracle has no signal.                                 [mode-dep]
#   O4. R1 and R4 produce structurally symmetric ensure_started events.
#       Both are cold. They MUST take the same `ensure_started_*` branch,
#       and R4 MUST NOT inherit any state-machine leak from R2+R3.
#       Concretely: count(R1's prefill_cd_payload_installed) == count(R4's),
#       same for drop; same `ensure_started_*` event name in both.   [universal]
#
# O3 is the cache-warmth oracle (will turn on once v2 scheduler is back).
# O4 is the state-machine-leak oracle (load-bearing today): it would FAIL if
# Bug A's draining of inbound_pulls regressed, since stale phantom blocks
# from R2/R3 would change R4's prefill behavior vs R1.
#
# Inputs (env vars, set by chain-runner.sh):
#   EXPERIMENT_DIR — /scratch/kvbm-experiments/<ts>-chain/
#   PREFILL_LOG    — $EXPERIMENT_DIR/prefill.log
#   DECODE_LOG     — $EXPERIMENT_DIR/decode.log
#   HUB_LOG        — $EXPERIMENT_DIR/hub.log     (unused here)
#   HUB_API        — host:port for kvbm_hub control API (default localhost:8337)
#   PREFILL_API    — vLLM prefill HTTP endpoint  (unused here)
#   DECODE_API     — vLLM decode HTTP endpoint   (default localhost:8001)
#   MODEL          — model id (default Qwen/Qwen3-0.6B)
#
# Outputs:
#   stdout: per-round status lines + final
#       "PASS|FAIL|INCONCLUSIVE: multi-turn-r1-r2-r3-r4 | r1_rid=... r1_match=N
#        r2_match=N r3_match=N r4_match=N r1_install=N r4_install=N r1_mode=...
#        r4_mode=... states_after_r4=N install_drop_parity=<bool>
#        zero_passthrough_mode=<bool>"
#   exit:   0=PASS, 1=FAIL, 2=INCONCLUSIVE, 77=SKIP
#
# Side effects:
#   - 4× POST to $DECODE_API/v1/completions (real HTTP requests, real generation).
#   - Reads $PREFILL_LOG, $DECODE_LOG; HTTP GET $HUB_API/v1/coordinator/states.
#   - Does NOT bring up hub/vLLM (chain-runner contract — bringup precedes us).
#   - Does NOT modify $EXPERIMENT_DIR or restart any service.
#
# Expected duration when GREEN: ~30-50 s (4 requests × 5-10 s each + 4× 3 s settle).

set -euo pipefail
# Best-effort source of shared helpers; absent file is fine.
# shellcheck disable=SC1091
source "$(dirname "$0")/../scripts/_assert.sh" 2>/dev/null || true

NAME="multi-turn-r1-r2-r3-r4"
HUB_API="${HUB_API:-localhost:8337}"
DECODE_API="${DECODE_API:-localhost:8001}"
PREFILL_LOG="${PREFILL_LOG:-${EXPERIMENT_DIR:-}/prefill.log}"
DECODE_LOG="${DECODE_LOG:-${EXPERIMENT_DIR:-}/decode.log}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# ---- Robustness: required inputs ----
if [[ -z "${EXPERIMENT_DIR:-}" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_experiment_dir hint=must_be_invoked_via_chain_runner"
  exit 2
fi
if [[ ! -f "$PREFILL_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_prefill_log path=$PREFILL_LOG"
  exit 2
fi
if [[ ! -f "$DECODE_LOG" ]]; then
  echo "INCONCLUSIVE: $NAME | error=missing_decode_log path=$DECODE_LOG"
  exit 2
fi

# ---- Prompts ----
# P1 ~80 tokens (5 full Qwen blocks at block_size=16). Fresh, no prior context.
P1='The quick brown fox jumps over the lazy dog and then keeps on running through the meadow until it reaches the river where it finally stops to drink some water and rest for a while before continuing its journey home. After resting, the fox decides to take a different path back through the woods.'
# P2 = P1 + ~40 net-new tokens (decode should see P1 prefix warm).
P2="$P1 The river flowed gently between the green hills and bright wildflowers swaying in the spring breeze, while birds sang above and rabbits grazed on the bank."
# P3 = P2 + ~30 net-new tokens (decode should see longer warm prefix).
P3="$P2 As evening fell, the colors of the sky turned amber and rose, and the meadow grew quiet save for the chirp of crickets in the tall grass."
# P4 ~100 tokens, completely unrelated to P1/P2/P3 — exercises coordinator
# state reset between turns and forces a fresh cold prefill.
P4='In the deepest oceans, far below the reach of sunlight, strange creatures glow in the dark with patterns of blue and green light, hunting and signaling to each other across vast underwater plains where ancient volcanic vents release minerals that feed entire ecosystems of tube worms, blind shrimp, and pale crabs that have never seen the sun and yet thrive on chemistry alone.'

# ---- Helpers ----
send_request() {
  # send_request <prompt>
  # Issues a non-streaming completion; echoes the HTTP body to stdout.
  # Always returns 0; the caller validates the body via send_request_ok.
  local prompt="$1"
  local payload body
  payload="$(P="$prompt" M="$MODEL" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["M"],
    "prompt": os.environ["P"],
    "max_tokens": 16,
    "temperature": 0,
}))
' 2>/dev/null || echo "")"
  if [[ -z "$payload" ]]; then
    return 0
  fi
  # -m 90: same upper bound as two-request-smoke.sh. We don't use -f here
  # because we want to inspect the body even on non-2xx.
  body="$(curl -m 90 -sS -X POST "http://${DECODE_API}/v1/completions" \
            -H 'Content-Type: application/json' -d "$payload" 2>/dev/null || echo "")"
  printf '%s' "$body"
}

send_request_ok() {
  # send_request_ok <body>
  # Validates that <body> is a JSON completion envelope with a non-empty
  # text choice. Returns 0 if good, 1 if missing/empty/non-JSON.
  local body="$1"
  if [[ -z "$body" ]]; then
    return 1
  fi
  printf '%s' "$body" | python3 -c '
import json, sys
try:
    j = json.load(sys.stdin)
except Exception:
    sys.exit(1)
choices = j.get("choices") or []
if not choices:
    sys.exit(1)
text = choices[0].get("text", "")
if not isinstance(text, str) or not text.strip():
    sys.exit(1)
sys.exit(0)
' >/dev/null 2>&1
}

# Capture line-baselines into the prefill+decode logs to scope grep to a round.
prefill_baseline() { wc -l < "$PREFILL_LOG" 2>/dev/null | tr -d ' '; }
decode_baseline()  { wc -l < "$DECODE_LOG"  2>/dev/null | tr -d ' '; }

# Extract the first rid that appears in a NEW range of the prefill log
# (looking for the chain-runner-style request_id=<uuid> token).
rid_for_round() {
  # rid_for_round <baseline_line>
  local b="${1:-0}"
  awk -v b="$b" 'NR > b' "$PREFILL_LOG" 2>/dev/null \
    | grep -oE 'request_id=[^[:space:]"]+' \
    | head -1 | sed 's/^request_id=//' || true
}

# matched_tokens for a round: scan the decode log range for
# `kvbm_audit ... event="policy_decision" role="decode" ... matched_tokens=N`.
matched_tokens_for_round() {
  # matched_tokens_for_round <decode_baseline_line>
  local b="${1:-0}"
  awk -v b="$b" 'NR > b' "$DECODE_LOG" 2>/dev/null \
    | grep -E 'event="policy_decision".*role="decode"' \
    | grep -oE 'matched_tokens=[0-9]+' \
    | head -1 | cut -d= -f2 || true
}

# Determine which ensure_started_* path the prefill took for this round.
# Returns one of: async_onboard | zero_passthrough | unknown
ensure_started_mode_for_round() {
  # ensure_started_mode_for_round <prefill_baseline_line>
  local b="${1:-0}"
  local hit
  hit="$(awk -v b="$b" 'NR > b' "$PREFILL_LOG" 2>/dev/null \
         | grep -oE 'ensure_started_(async_onboard|zero_passthrough)' \
         | head -1 || true)"
  case "$hit" in
    ensure_started_async_onboard)   printf 'async_onboard' ;;
    ensure_started_zero_passthrough) printf 'zero_passthrough' ;;
    *)                               printf 'unknown' ;;
  esac
}

# Count prefill_cd_payload_installed events for a round (scoped to NEW lines).
installs_for_round() {
  local b="${1:-0}"
  awk -v b="$b" 'NR > b' "$PREFILL_LOG" 2>/dev/null \
    | grep -cE 'prefill_cd_payload_installed' || true
}
drops_for_round() {
  local b="${1:-0}"
  awk -v b="$b" 'NR > b' "$PREFILL_LOG" 2>/dev/null \
    | grep -cE 'prefill_cd_payload_drop' || true
}

# Hub coordinator.states size — same logic as chain-runner's states_count().
hub_states_count() {
  local body n
  body="$(curl -fsS -m 5 "http://${HUB_API}/v1/coordinator/states" 2>/dev/null || true)"
  if [[ -z "$body" ]]; then
    printf '?'
    return 0
  fi
  n="$(printf '%s' "$body" | jq '. | length' 2>/dev/null || true)"
  if [[ -z "$n" || ! "$n" =~ ^[0-9]+$ ]]; then
    printf '?'
    return 0
  fi
  printf '%s' "$n"
}

# ---- Pre-flight: snapshot starting prefill log line so the
#                  total-install/total-drop totals are scoped to events
#                  that occur during this scenario, not earlier ones. ----
START_PREFILL_LINE="$(prefill_baseline || echo 0)"

# ---------------------------------------------------------------------------
# Round driver
# ---------------------------------------------------------------------------
run_round() {
  # run_round <label> <prompt>
  # Echoes per-round status to stdout. Sets globals for caller to read:
  #   __RID __MATCH __MODE __INSTALL __DROP __HTTP_OK
  local label="$1"
  local prompt="$2"

  local pf_b dc_b body
  pf_b="$(prefill_baseline || echo 0)"
  dc_b="$(decode_baseline  || echo 0)"

  echo "=== Round $label (prompt_len=${#prompt}) ==="
  __HTTP_OK="false"
  # Capture body without tripping set -e. The trailing `|| true` neutralizes
  # any non-zero exit propagating through command substitution. We then
  # validate the body with send_request_ok.
  body="$(send_request "$prompt" || true)"
  if send_request_ok "$body"; then
    __HTTP_OK="true"
    echo "  http: 200 body_chars=${#body}"
  else
    echo "  http: FAIL body_chars=${#body}"
  fi

  # Settle so audit + drop events flush before we scrape.
  sleep 3

  __RID="$(rid_for_round "$pf_b" || true)"
  __MATCH="$(matched_tokens_for_round "$dc_b" || true)"
  __MODE="$(ensure_started_mode_for_round "$pf_b" || true)"
  __INSTALL="$(installs_for_round "$pf_b" || true)"
  __DROP="$(drops_for_round "$pf_b" || true)"
  __MATCH="${__MATCH:-}"
  __INSTALL="${__INSTALL:-0}"
  __DROP="${__DROP:-0}"
  __MODE="${__MODE:-unknown}"
  echo "  rid=${__RID:-<unknown>} matched_tokens=${__MATCH:-<absent>} mode=${__MODE} install=${__INSTALL} drop=${__DROP}"
}

# ---- Round 1 (cold) ----
run_round "R1 cold" "$P1"
R1_HTTP_OK="$__HTTP_OK"; R1_RID="$__RID"; R1_MATCH="$__MATCH"
R1_MODE="$__MODE"; R1_INSTALL="$__INSTALL"; R1_DROP="$__DROP"

# ---- Round 2 (warm — extends R1 prefix) ----
run_round "R2 warm-extend" "$P2"
R2_HTTP_OK="$__HTTP_OK"; R2_RID="$__RID"; R2_MATCH="$__MATCH"
R2_MODE="$__MODE"; R2_INSTALL="$__INSTALL"; R2_DROP="$__DROP"

# ---- Round 3 (warmer — extends R2 prefix) ----
run_round "R3 warm-sustain" "$P3"
R3_HTTP_OK="$__HTTP_OK"; R3_RID="$__RID"; R3_MATCH="$__MATCH"
R3_MODE="$__MODE"; R3_INSTALL="$__INSTALL"; R3_DROP="$__DROP"

# ---- Round 4 (cold-again — fresh unrelated prompt) ----
run_round "R4 cold-fresh" "$P4"
R4_HTTP_OK="$__HTTP_OK"; R4_RID="$__RID"; R4_MATCH="$__MATCH"
R4_MODE="$__MODE"; R4_INSTALL="$__INSTALL"; R4_DROP="$__DROP"

# ---- Final hub state snapshot ----
sleep 2
STATES_AFTER="$(hub_states_count)"

# Total install / drop across this scenario's slice.
TOTAL_INSTALL="$(installs_for_round "$START_PREFILL_LINE" || true)"
TOTAL_DROP="$(drops_for_round "$START_PREFILL_LINE" || true)"
TOTAL_INSTALL="${TOTAL_INSTALL:-0}"
TOTAL_DROP="${TOTAL_DROP:-0}"
INSTALL_DROP_PARITY="false"
if [[ "$TOTAL_INSTALL" == "$TOTAL_DROP" ]]; then
  INSTALL_DROP_PARITY="true"
fi

# Detect zero_passthrough mode: every round took zero_passthrough → O3 inconclusive.
ZP_MODE="false"
if [[ "$R1_MODE" == "zero_passthrough" && "$R2_MODE" == "zero_passthrough" \
   && "$R3_MODE" == "zero_passthrough" && "$R4_MODE" == "zero_passthrough" ]]; then
  ZP_MODE="true"
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
DIGEST_BASE="r1_rid=${R1_RID:-<unknown>} r1_match=${R1_MATCH:-<absent>} r2_match=${R2_MATCH:-<absent>} r3_match=${R3_MATCH:-<absent>} r4_match=${R4_MATCH:-<absent>} r1_mode=${R1_MODE} r4_mode=${R4_MODE} r1_install=${R1_INSTALL} r4_install=${R4_INSTALL} states_after_r4=${STATES_AFTER} total_install=${TOTAL_INSTALL} total_drop=${TOTAL_DROP} install_drop_parity=${INSTALL_DROP_PARITY} zero_passthrough_mode=${ZP_MODE}"

# --- O1: HTTP 200 with non-empty content for all four rounds. [universal] ---
if [[ "$R1_HTTP_OK" != "true" || "$R2_HTTP_OK" != "true" \
   || "$R3_HTTP_OK" != "true" || "$R4_HTTP_OK" != "true" ]]; then
  echo "FAIL: $NAME | reason=http_or_body_missing r1_http=${R1_HTTP_OK} r2_http=${R2_HTTP_OK} r3_http=${R3_HTTP_OK} r4_http=${R4_HTTP_OK} | $DIGEST_BASE"
  exit 1
fi

# --- O2: hub coordinator state is empty after R4 settles, and install/drop
#         parity holds across all four rounds. [universal] ---
# states API may be unreachable during shutdown; treat that as INCONCLUSIVE.
if [[ "$STATES_AFTER" == "?" ]]; then
  echo "INCONCLUSIVE: $NAME | error=states_api_unreachable | $DIGEST_BASE"
  exit 2
fi
if [[ "$STATES_AFTER" != "0" ]]; then
  echo "FAIL: $NAME | reason=states_after_r4_nonzero | $DIGEST_BASE"
  exit 1
fi
if [[ "$INSTALL_DROP_PARITY" != "true" ]]; then
  echo "FAIL: $NAME | reason=install_drop_imbalance | $DIGEST_BASE"
  exit 1
fi

# --- O4: R1 vs R4 structural symmetry (both cold). [universal] ---
# - Same ensure_started_* branch. (Distinguishes a state-machine leak that
#   makes R4 take a different path than R1.)
# - Same install_count and drop_count for the round. (Distinguishes phantom
#   blocks left over from R2/R3 — Bug A's drain regression signature.)
if [[ "$R1_MODE" != "$R4_MODE" ]]; then
  echo "FAIL: $NAME | reason=r1_r4_mode_asymmetry r1=${R1_MODE} r4=${R4_MODE} | $DIGEST_BASE"
  exit 1
fi
if [[ "$R1_INSTALL" != "$R4_INSTALL" ]]; then
  echo "FAIL: $NAME | reason=r1_r4_install_count_asymmetry r1=${R1_INSTALL} r4=${R4_INSTALL} | $DIGEST_BASE"
  exit 1
fi
if [[ "$R1_DROP" != "$R4_DROP" ]]; then
  echo "FAIL: $NAME | reason=r1_r4_drop_count_asymmetry r1=${R1_DROP} r4=${R4_DROP} | $DIGEST_BASE"
  exit 1
fi

# --- O3: cache-warmth direction (R1 cold, R2/R3 warmer, R4 cold).
#         INCONCLUSIVE if the system is in zero_passthrough mode. [mode-dep] ---
if [[ "$ZP_MODE" == "true" ]]; then
  echo "INCONCLUSIVE: $NAME | reason=v2_scheduler_disabled all_rounds_zero_passthrough oracles_o1_o2_o4_passed | $DIGEST_BASE"
  exit 2
fi

# Validate matched_tokens are integers we can compare. If a round did not
# emit a policy_decision at all, treat as INCONCLUSIVE rather than FAIL —
# the request may have been served by a non-CD path (e.g. inner local).
for v in "$R1_MATCH" "$R2_MATCH" "$R3_MATCH" "$R4_MATCH"; do
  if [[ -z "$v" || ! "$v" =~ ^[0-9]+$ ]]; then
    echo "INCONCLUSIVE: $NAME | error=missing_or_non_integer_policy_decision | $DIGEST_BASE"
    exit 2
  fi
done

# R1 must be cold.
if (( R1_MATCH != 0 )); then
  echo "FAIL: $NAME | reason=r1_not_cold expected=0 got=${R1_MATCH} | $DIGEST_BASE"
  exit 1
fi
# R4 must be cold (fresh unrelated prompt).
if (( R4_MATCH != 0 )); then
  echo "FAIL: $NAME | reason=r4_not_cold_after_reset expected=0 got=${R4_MATCH} | $DIGEST_BASE"
  exit 1
fi
# R2 must be warmer than R1.
if (( R2_MATCH <= R1_MATCH )); then
  echo "FAIL: $NAME | reason=r2_not_warmer_than_r1 r1=${R1_MATCH} r2=${R2_MATCH} | $DIGEST_BASE"
  exit 1
fi
# R3 must be at least as warm as R2 (sustained accumulation).
if (( R3_MATCH < R2_MATCH )); then
  echo "FAIL: $NAME | reason=r3_regressed_below_r2 r2=${R2_MATCH} r3=${R3_MATCH} | $DIGEST_BASE"
  exit 1
fi

echo "PASS: $NAME | $DIGEST_BASE"
exit 0
