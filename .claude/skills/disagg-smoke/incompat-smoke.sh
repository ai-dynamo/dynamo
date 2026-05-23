#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# incompat-smoke.sh — hub-side mismatched-layout register smoke.
#
# Brings up a fresh `kvbm_hub` binary on alternate ports (so it doesn't
# clash with a running prod hub on 8337/1337) and exercises the
# `LayoutCompatPayload` gate end-to-end through the real HTTP API.
#
# Why: the gate has thorough Rust-side coverage in
# `lib/kvbm-hub/tests/cd_layout_compat.rs`, but operators don't usually
# run those — `cargo test` requires a toolchain and full build state.
# This script gives a one-command failure-scenario check that lives
# alongside the golden two-request smokes.
#
# Self-contained: doesn't touch vLLM, doesn't need a model, doesn't
# need a GPU. Fresh hub per scenario (~1s startup each) so the
# baseline state is deterministic. ~10s total.
#
# Honors:
#   KVBM_REPO          (default: worktree root inferred from script path)
#   KVBM_HUB_BIN       (default: $KVBM_REPO/target/debug/kvbm_hub)
#
# Scenarios:
#   1. baseline_universal_accept       — first universal P2P register
#   2. cross_mode_reject               — universal baseline, operational candidate
#   3. universal_canonical_mismatch    — universal both sides, different num_heads_total
#   4. operational_nhd_vs_hnd          — operational baseline NHD, candidate HND
#   5. operational_page_size_mismatch  — operational, same KvBlockLayout, different page_size
#   6. cd_without_p2p_reject           — CD register without Feature::P2P
#
# Exits 0 if every scenario lands its expected status + error keyword;
# 1 otherwise. Tear-down is best-effort via EXIT trap.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
KVBM_REPO=${KVBM_REPO:-$REPO_DEFAULT}
HUB=${KVBM_HUB_BIN:-$KVBM_REPO/target/debug/kvbm_hub}

# Alternate port range — doesn't clash with the prod-default 8337/1337/1338.
# Velo is intentionally NOT enabled: when velo is on, the hub tries to
# decode the WorkerAddress as msgpack for streaming-key registration,
# which our synthetic peer payload can't satisfy. The layout_compat
# gate runs BEFORE velo registration in the request pipeline, so the
# HTTP-level rejection behaviour we're testing is unaffected.
CTRL_PORT=${KVBM_INCOMPAT_CTRL_PORT:-28337}
DISC_PORT=${KVBM_INCOMPAT_DISC_PORT:-21337}

if [ ! -x "$HUB" ]; then
    echo "incompat-smoke: kvbm_hub binary missing at $HUB" >&2
    echo "  build via: cargo build -p kvbm-hub --bin kvbm_hub" >&2
    exit 1
fi

WORK_DIR=$(mktemp -d -t incompat-smoke.XXXXXX)
HUB_LOG="$WORK_DIR/hub.log"
RESP_BODY="$WORK_DIR/resp.json"
HUB_PID=

cleanup() {
    [ -n "$HUB_PID" ] && kill "$HUB_PID" 2>/dev/null || true
    # Best-effort wait so the next scenario's hub bind doesn't race.
    [ -n "$HUB_PID" ] && wait "$HUB_PID" 2>/dev/null || true
    HUB_PID=
}
trap 'cleanup; rm -rf "$WORK_DIR"' EXIT

start_hub() {
    cleanup
    : > "$HUB_LOG"
    RUST_LOG=${RUST_LOG:-warn,kvbm_hub=info} \
        "$HUB" \
            --control-port "$CTRL_PORT" \
            --discovery-port "$DISC_PORT" \
            --heartbeat-interval-secs 60 \
            > "$HUB_LOG" 2>&1 &
    HUB_PID=$!
    # Wait for the control-port listener to be up.
    local tries=0
    until curl -fsS "http://127.0.0.1:$CTRL_PORT/v1/instances" >/dev/null 2>&1; do
        tries=$((tries+1))
        if [ $tries -ge 40 ]; then
            echo "incompat-smoke: hub failed to start; log tail:" >&2
            tail -50 "$HUB_LOG" >&2
            return 1
        fi
        sleep 0.25
    done
}

# Build a P2P register JSON payload via python3.
# Args:
#   $1 mode               "universal" | "operational"
#   $2 tp_size            int (must divide num_heads_total)
#   $3 pp_size            int (must divide num_layers_total)
#   $4 per_worker_layout  "Universal" | "OperationalNHD" | "OperationalHND"
#   $5 page_size          int
#   $6 num_heads_total    int
#   $7 head_dim           int
#   $8 num_layers_total   int
build_p2p_payload() {
    python3 - "$@" <<'PYEOF'
import json, sys, uuid, base64
mode, tp, pp, kvbl, page_size, num_heads_total, head_dim, num_layers_total = sys.argv[1:9]
tp, pp = int(tp), int(pp)
canonical = {
    "num_layers_total": int(num_layers_total),
    "outer_dim": 2,
    "page_size": int(page_size),
    "num_heads_total": int(num_heads_total),
    "head_dim": int(head_dim),
    "dtype_width_bytes": 2,
}
num_heads = canonical["num_heads_total"] // tp
num_layers = canonical["num_layers_total"] // pp
layout_compat = {
    "mode": mode,
    "canonical": canonical,
    "per_worker_layout": kvbl,
    "per_worker_config": {
        "num_blocks": 1024,
        "num_layers": num_layers,
        "outer_dim": canonical["outer_dim"],
        "page_size": canonical["page_size"],
        "inner_dim": num_heads * canonical["head_dim"],
        "alignment": 256,
        "dtype_width_bytes": canonical["dtype_width_bytes"],
        "num_heads": num_heads,
    },
    "tp_size": tp,
    "pp_size": pp,
}
peer = {
    "instance_id": str(uuid.uuid4()),
    "worker_address": base64.standard_b64encode(b"incompat-smoke").decode(),
}
print(json.dumps({
    "peer_info": peer,
    "features": [{"kind": "p2p", "config": {"layout_compat": layout_compat}}],
}))
PYEOF
}

# Build a CD-only (no P2P) register JSON payload.
build_cd_only_payload() {
    python3 - <<'PYEOF'
import json, uuid, base64
peer = {
    "instance_id": str(uuid.uuid4()),
    "worker_address": base64.standard_b64encode(b"incompat-smoke").decode(),
}
print(json.dumps({
    "peer_info": peer,
    "features": [{"kind": "disagg", "config": {"role": "prefill"}}],
}))
PYEOF
}

post_register() {
    local payload=$1
    curl -sS -o "$RESP_BODY" -w '%{http_code}' \
        -X POST "http://127.0.0.1:$CTRL_PORT/v1/instances" \
        -H 'content-type: application/json' \
        -d "$payload"
}

FAILED=0

# Run a scenario.
#   $1 label
#   $2 expected status (e.g. 200 or 400)
#   $3 expected message substring (case-insensitive grep -i) — only checked
#      when expected status is non-2xx; empty string skips the check
#   $4 payload JSON
run() {
    local label=$1 expected_status=$2 needle=$3 payload=$4
    local got_status
    got_status=$(post_register "$payload" || echo "curl-failed")
    local body
    body=$(cat "$RESP_BODY")
    if [ "$got_status" != "$expected_status" ]; then
        echo "  FAIL [$label] expected HTTP $expected_status, got $got_status"
        echo "    body: $body"
        FAILED=$((FAILED+1))
        return
    fi
    if [ -n "$needle" ]; then
        if ! echo "$body" | grep -qi "$needle"; then
            echo "  FAIL [$label] HTTP $got_status (expected) but body missing '$needle'"
            echo "    body: $body"
            FAILED=$((FAILED+1))
            return
        fi
        echo "  PASS [$label] HTTP $got_status, body matched '$needle'"
    else
        echo "  PASS [$label] HTTP $got_status"
    fi
}

echo "incompat-smoke: hub binary $HUB"
echo "incompat-smoke: ports control=$CTRL_PORT disc=$DISC_PORT (velo disabled)"
echo

# --------------------------------------------------------------------
# Scenario 1: baseline universal — first valid P2P register, accepts.
# Scenario 2: cross-mode reject — second register flips to operational.
# --------------------------------------------------------------------
echo "[1+2] universal baseline → operational candidate (cross-mode reject)"
start_hub
BASE_UNI=$(build_p2p_payload universal 2 1 Universal 16 64 128 32)
run "baseline_universal_accept" 200 "" "$BASE_UNI"

CAND_OPER=$(build_p2p_payload operational 2 1 OperationalNHD 16 64 128 32)
run "cross_mode_reject" 400 "mode" "$CAND_OPER"

# --------------------------------------------------------------------
# Scenario 3: universal × universal, divergent canonical (num_heads_total).
# --------------------------------------------------------------------
echo
echo "[3] universal × universal, canonical mismatch (num_heads_total 64→48)"
start_hub
BASE_UNI=$(build_p2p_payload universal 2 1 Universal 16 64 128 32)
run "baseline_universal_accept" 200 "" "$BASE_UNI"

CAND_UNI_MISMATCH=$(build_p2p_payload universal 4 1 Universal 16 48 128 32)
run "universal_canonical_mismatch" 400 "canonical\\|head" "$CAND_UNI_MISMATCH"

# --------------------------------------------------------------------
# Scenario 4: operational NHD baseline → operational HND candidate.
# --------------------------------------------------------------------
echo
echo "[4] operational NHD baseline → operational HND candidate"
start_hub
BASE_OP_NHD=$(build_p2p_payload operational 2 1 OperationalNHD 16 64 128 32)
run "baseline_operational_nhd_accept" 200 "" "$BASE_OP_NHD"

CAND_OP_HND=$(build_p2p_payload operational 2 1 OperationalHND 16 64 128 32)
run "operational_nhd_vs_hnd" 400 "operational\\|kvblocklayout\\|nhd\\|hnd" "$CAND_OP_HND"

# --------------------------------------------------------------------
# Scenario 5: operational same KvBlockLayout, divergent page_size.
# --------------------------------------------------------------------
echo
echo "[5] operational both NHD, page_size 16 vs 32 (config divergence)"
start_hub
BASE_OP=$(build_p2p_payload operational 2 1 OperationalNHD 16 64 128 32)
run "baseline_operational_accept" 200 "" "$BASE_OP"

CAND_OP_PAGE=$(build_p2p_payload operational 2 1 OperationalNHD 32 64 128 32)
run "operational_page_size_mismatch" 400 "page_size\\|per_worker_config" "$CAND_OP_PAGE"

# --------------------------------------------------------------------
# Scenario 6: CD register without P2P feature.
# --------------------------------------------------------------------
echo
echo "[6] disagg without p2p feature"
start_hub
CD_ONLY=$(build_cd_only_payload)
run "cd_without_p2p_reject" 400 "p2p\\|conditionaldisagg\\|disagg" "$CD_ONLY"

cleanup

echo
if [ $FAILED -eq 0 ]; then
    echo "incompat-smoke: ALL SCENARIOS PASS"
    exit 0
else
    echo "incompat-smoke: $FAILED scenario(s) FAILED"
    exit 1
fi
