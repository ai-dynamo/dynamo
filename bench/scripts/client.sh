#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DIS-1381: Client-side load generator + metrics collector.
# Runs inside the container on the client node (node 1).
#
# 1. Waits for the server to become healthy.
# 2. Sends a sanity request.
# 3. Sweeps concurrency levels with aiperf.
# 4. Scrapes prometheus metrics after each concurrency run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SERVER_HOST=""
SCENARIO=""
MODEL=""
ISL=""
OSL=""
CONCURRENCIES=""
REQ_MULTIPLIER=""
OUTPUT_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server-host)   SERVER_HOST="$2"; shift 2 ;;
        --scenario)      SCENARIO="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --isl)           ISL="$2"; shift 2 ;;
        --osl)           OSL="$2"; shift 2 ;;
        --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
        --req-multiplier) REQ_MULTIPLIER="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        *)               echo "ERROR: client.sh: Unknown option: $1" >&2; exit 1 ;;
    esac
done

for var in SERVER_HOST SCENARIO MODEL ISL OSL CONCURRENCIES REQ_MULTIPLIER OUTPUT_DIR; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: client.sh: --$(echo $var | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required" >&2
        exit 1
    fi
done

METRICS_DIR="${OUTPUT_DIR}/metrics"
AIPERF_DIR="${OUTPUT_DIR}/aiperf"
mkdir -p "$METRICS_DIR" "$AIPERF_DIR"

SERVER_URL="http://${SERVER_HOST}:8000"

# Dry-run overrides for aiperf request counts
if [[ "$DRY_RUN" == "true" ]]; then
    REQ_MULTIPLIER=5
    WARMUP_MULTIPLIER=1
else
    WARMUP_MULTIPLIER=2
fi

echo "[client.sh] Configuration:"
echo "  Server:        $SERVER_URL"
echo "  Scenario:      $SCENARIO"
echo "  Model:         $MODEL"
echo "  ISL/OSL:       ${ISL}/${OSL}"
echo "  Concurrencies: $CONCURRENCIES"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Dry-run:       $DRY_RUN"

# ---------------------------------------------------------------------------
# Step 1: Health check -- wait for server to be ready
# ---------------------------------------------------------------------------
echo ""
echo "[client.sh] Waiting for server to become healthy..."

HEALTH_MAX_ATTEMPTS=120
HEALTH_INITIAL_WAIT=60
HEALTH_RETRY_INTERVAL=10

# For scenario 1 (vllm serve), /health returns JSON.
# For scenarios 2 & 3 (dynamo), /v1/models is a reliable readiness signal.
if [[ "$SCENARIO" == "1" ]]; then
    HEALTH_ENDPOINT="${SERVER_URL}/health"
else
    HEALTH_ENDPOINT="${SERVER_URL}/v1/models"
fi

sleep "$HEALTH_INITIAL_WAIT"

health_ok=false
for ((i=1; i<=HEALTH_MAX_ATTEMPTS; i++)); do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_ENDPOINT" 2>/dev/null) || http_code="000"
    if [[ "$http_code" == "200" ]]; then
        echo "[client.sh] Health check passed on attempt $i (HTTP $http_code)"
        health_ok=true
        break
    fi
    echo "[client.sh] Attempt $i/$HEALTH_MAX_ATTEMPTS: $HEALTH_ENDPOINT returned HTTP $http_code"
    sleep "$HEALTH_RETRY_INTERVAL"
done

if [[ "$health_ok" != "true" ]]; then
    echo "ERROR: Server did not become healthy after $HEALTH_MAX_ATTEMPTS attempts." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Sanity request -- validate end-to-end (with retries)
#
# For Dynamo scenarios the frontend HTTP server can pass a basic health check
# before the vllm worker has registered, causing 404 on /v1/chat/completions.
# Retry until the worker is ready.
# ---------------------------------------------------------------------------
echo ""
echo "[client.sh] Sending sanity request (will retry up to 60 times)..."

SANITY_MAX_ATTEMPTS=120
SANITY_RETRY_INTERVAL=5
sanity_ok=false

for ((j=1; j<=SANITY_MAX_ATTEMPTS; j++)); do
    SANITY_RESPONSE=$(curl -s -w "\n%{http_code}" "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL"'",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 10,
            "stream": false
        }' 2>/dev/null)

    SANITY_HTTP_CODE=$(echo "$SANITY_RESPONSE" | tail -n1)
    SANITY_BODY=$(echo "$SANITY_RESPONSE" | sed '$d')

    if [[ "$SANITY_HTTP_CODE" == "200" ]]; then
        echo "[client.sh] Sanity request succeeded on attempt $j (HTTP 200)"
        echo "$SANITY_BODY" > "${OUTPUT_DIR}/sanity_response.json"
        sanity_ok=true
        break
    fi
    echo "[client.sh] Sanity attempt $j/$SANITY_MAX_ATTEMPTS: HTTP $SANITY_HTTP_CODE (worker may still be loading)"
    sleep "$SANITY_RETRY_INTERVAL"
done

if [[ "$sanity_ok" != "true" ]]; then
    echo "ERROR: Sanity request failed after $SANITY_MAX_ATTEMPTS attempts (last HTTP $SANITY_HTTP_CODE)" >&2
    echo "Last response body: $SANITY_BODY" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Scrape baseline metrics (before any load)
# ---------------------------------------------------------------------------
echo ""
echo "[client.sh] Scraping baseline metrics..."
curl -s "${SERVER_URL}/metrics" > "${METRICS_DIR}/baseline_metrics.txt" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 4: Concurrency sweep with aiperf
# ---------------------------------------------------------------------------
echo ""
echo "[client.sh] Starting concurrency sweep..."

for C in $CONCURRENCIES; do
    REQUEST_COUNT=$((C * REQ_MULTIPLIER))
    WARMUP_COUNT=$((C * WARMUP_MULTIPLIER))
    DATASET_ENTRIES=$((C * (REQ_MULTIPLIER + WARMUP_MULTIPLIER + 2)))

    echo ""
    echo "------------------------------------------------------------"
    echo "[client.sh] Concurrency: $C  (requests=$REQUEST_COUNT, warmup=$WARMUP_COUNT)"
    echo "------------------------------------------------------------"

    # Per-concurrency artifact directory so results are not overwritten
    AIPERF_CONCURRENCY_DIR="${AIPERF_DIR}/concurrency_${C}"
    mkdir -p "$AIPERF_CONCURRENCY_DIR"

    aiperf profile \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "${SERVER_URL}" \
        --synthetic-input-tokens-mean "$ISL" \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$OSL" \
        --output-tokens-stddev 0 \
        --extra-inputs "max_tokens:${OSL}" \
        --extra-inputs "min_tokens:${OSL}" \
        --extra-inputs "ignore_eos:true" \
        --extra-inputs '{"nvext":{"ignore_eos":true}}' \
        --concurrency "$C" \
        --request-count "$REQUEST_COUNT" \
        --warmup-request-count "$WARMUP_COUNT" \
        --num-dataset-entries "$DATASET_ENTRIES" \
        --random-seed 100 \
        --artifact-dir "$AIPERF_CONCURRENCY_DIR" \
        --ui simple \
        -H 'Authorization: Bearer NOT_USED' \
        -H 'Accept: text/event-stream'

    echo "[client.sh] aiperf done for concurrency $C"

    # Scrape metrics after this concurrency run
    echo "[client.sh] Scraping metrics for concurrency $C..."
    curl -s "${SERVER_URL}/metrics" \
        > "${METRICS_DIR}/concurrency_${C}_metrics.txt" 2>/dev/null || true

    echo "[client.sh] Concurrency $C complete."
done

# ---------------------------------------------------------------------------
# Step 5: Final metrics snapshot
# ---------------------------------------------------------------------------
echo ""
echo "[client.sh] Scraping final metrics snapshot..."
curl -s "${SERVER_URL}/metrics" > "${METRICS_DIR}/final_metrics.txt" 2>/dev/null || true

echo ""
echo "[client.sh] All concurrency levels complete for scenario $SCENARIO."
echo "[client.sh] Results in: $OUTPUT_DIR"
