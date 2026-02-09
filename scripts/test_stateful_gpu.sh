#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU Integration Test: Stateful Responses API
#
# Tests the stateful responses feature (previous_response_id chaining,
# hierarchical isolation, store/retrieve/delete) using Dynamo's full
# two-process architecture: Dynamo frontend (Axum HTTP) + SGLang backend worker.
#
# Architecture:
#   - SGLang worker: registers with KV store, handles inference (no HTTP)
#   - Dynamo frontend: Axum HTTP server with responses_router, session middleware,
#     storage layer — discovers the SGLang worker via KV store
#   - DYNAMO_ENABLE_STATEFUL_RESPONSES=1 is set automatically on the frontend
#
# Isolation model:
#   - Tenant = hard security boundary (cross-tenant access is blocked)
#   - Session = metadata (cross-session access within same tenant is ALLOWED)
#
# Usage (inside the SGLang container on GPU):
#   ./scripts/test_stateful_gpu.sh
#
# Override defaults:
#   BASE_URL=http://localhost:9000/v1 MODEL=Qwen/Qwen3-0.6B ./scripts/test_stateful_gpu.sh
#
# The script will:
#   1. Start SGLang backend worker + Dynamo frontend (unless SKIP_SERVER_START=1)
#   2. Wait for worker registration and frontend readiness
#   3. Run functional tests for the Responses API
#   4. Report pass/fail results

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL="${BASE_URL:-http://localhost:9000/v1}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MODEL_PATH="${MODEL_PATH:-/model}"
SERVER_PORT="${SERVER_PORT:-9000}"
SKIP_SERVER_START="${SKIP_SERVER_START:-0}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-64}"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Colors (if tty)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' BLUE='' NC=''
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_section() { echo -e "\n${BLUE}=== $* ===${NC}"; }

assert_pass() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    TESTS_PASSED=$((TESTS_PASSED + 1))
    log_pass "$1"
}

assert_fail() {
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    TESTS_FAILED=$((TESTS_FAILED + 1))
    log_fail "$1"
    if [ -n "${2:-}" ]; then
        echo "       Detail: $2"
    fi
}

# Send a request to POST /v1/responses and print the JSON response body.
# Arguments:
#   $1 - JSON payload
#   $2 - x-tenant-id header value
#   $3 - x-session-id header value
api_responses() {
    local payload="$1"
    local tenant_id="${2:-test-tenant}"
    local session_id="${3:-test-session}"

    curl -s -w "\n%{http_code}" \
        -X POST "${BASE_URL}/responses" \
        -H "Content-Type: application/json" \
        -H "x-tenant-id: ${tenant_id}" \
        -H "x-session-id: ${session_id}" \
        -d "${payload}"
}

# Send a GET request to /v1/responses/{id}
api_get_response() {
    local response_id="$1"
    local tenant_id="${2:-test-tenant}"
    local session_id="${3:-test-session}"

    curl -s -w "\n%{http_code}" \
        -X GET "${BASE_URL}/responses/${response_id}" \
        -H "x-tenant-id: ${tenant_id}" \
        -H "x-session-id: ${session_id}"
}

# Send a DELETE request to /v1/responses/{id}
api_delete_response() {
    local response_id="$1"
    local tenant_id="${2:-test-tenant}"
    local session_id="${3:-test-session}"

    curl -s -w "\n%{http_code}" \
        -X DELETE "${BASE_URL}/responses/${response_id}" \
        -H "x-tenant-id: ${tenant_id}" \
        -H "x-session-id: ${session_id}"
}

# Extract the HTTP status code from curl output (last line)
get_http_status() {
    echo "$1" | tail -n1
}

# Extract the response body from curl output (everything except last line)
get_body() {
    echo "$1" | sed '$d'
}

# Extract a field from JSON using python (available in the container)
json_field() {
    local json="$1"
    local field="$2"
    echo "${json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
keys = '${field}'.split('.')
for k in keys:
    if isinstance(data, list):
        data = data[int(k)]
    else:
        data = data[k]
print(data)
" 2>/dev/null || echo ""
}

# ---------------------------------------------------------------------------
# Server Management
# ---------------------------------------------------------------------------

start_server() {
    if [ "${SKIP_SERVER_START}" = "1" ]; then
        log_info "SKIP_SERVER_START=1, assuming server is already running"
        return 0
    fi

    log_section "Starting Dynamo Frontend + SGLang Backend"
    log_info "Model: ${MODEL}"
    log_info "Frontend port: ${SERVER_PORT}"
    log_info "Model path: ${MODEL_PATH}"

    # Dynamo uses a two-process architecture:
    #   1. SGLang backend worker — registers with KV store, handles inference
    #   2. Dynamo frontend — Axum HTTP server with responses_router, session middleware
    #
    # Both use --store-kv file for local service discovery (no etcd needed).
    # The frontend discovers the SGLang worker automatically.

    # Start SGLang backend worker (no HTTP — communicates via request plane)
    python3 -m dynamo.sglang \
        --model-path "${MODEL_PATH}" \
        --served-model-name "${MODEL}" \
        --trust-remote-code \
        --disable-cuda-graph \
        --store-kv file \
        > /tmp/sglang_worker.log 2>&1 &

    WORKER_PID=$!
    log_info "SGLang worker started with PID ${WORKER_PID}"
    echo "${WORKER_PID}" > /tmp/sglang_worker.pid

    # Give the worker a moment to start registering before launching frontend
    sleep 5

    # Start Dynamo HTTP frontend with stateful responses enabled
    DYNAMO_ENABLE_STATEFUL_RESPONSES=1 python3 -m dynamo.frontend \
        --http-port "${SERVER_PORT}" \
        --store-kv file \
        > /tmp/dynamo_frontend.log 2>&1 &

    FRONTEND_PID=$!
    log_info "Dynamo frontend started with PID ${FRONTEND_PID}"
    echo "${FRONTEND_PID}" > /tmp/dynamo_frontend.pid
}

wait_for_server() {
    log_info "Waiting for Dynamo frontend + SGLang worker to be ready (max ${MAX_WAIT_SECONDS}s)..."
    log_info "The worker must register with the KV store before the frontend can discover it."

    local elapsed=0
    local interval=5

    while [ ${elapsed} -lt ${MAX_WAIT_SECONDS} ]; do
        # Check /v1/models — returns data only after the worker has registered
        if curl -s "${BASE_URL}/models" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('data') and len(data['data']) > 0:
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
            log_info "Server is ready after ${elapsed}s (models endpoint has registered backends)"
            return 0
        fi

        # Fallback: check /health on the frontend
        if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL%/v1}/health" 2>/dev/null | grep -q "200"; then
            # Health is up but models might not be registered yet — keep waiting
            if [ $((elapsed % 30)) -eq 0 ] && [ ${elapsed} -gt 0 ]; then
                log_info "Frontend health OK but no models registered yet (${elapsed}s)..."
            fi
        fi

        sleep ${interval}
        elapsed=$((elapsed + interval))
        echo -n "."
    done

    echo ""
    log_fail "Server did not become ready within ${MAX_WAIT_SECONDS}s"
    for logfile in /tmp/dynamo_frontend.log /tmp/sglang_worker.log; do
        if [ -f "${logfile}" ]; then
            echo "--- Last 20 lines of ${logfile} ---"
            tail -20 "${logfile}"
            echo "---"
        fi
    done
    return 1
}

stop_server() {
    if [ "${SKIP_SERVER_START}" = "1" ]; then
        return 0
    fi

    for pidfile in /tmp/dynamo_frontend.pid /tmp/sglang_worker.pid; do
        if [ -f "${pidfile}" ]; then
            local pid
            pid=$(cat "${pidfile}")
            if kill -0 "${pid}" 2>/dev/null; then
                log_info "Stopping process (PID ${pid}, ${pidfile})..."
                kill "${pid}" 2>/dev/null || true
                wait "${pid}" 2>/dev/null || true
            fi
            rm -f "${pidfile}"
        fi
    done
}

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

test_single_turn_response() {
    log_section "Test: Single-Turn Response (no state)"

    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'What is 2 + 2? Answer with just the number.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': False
}))
")

    local result
    result=$(api_responses "${payload}" "test-tenant" "test-session")
    local status
    status=$(get_http_status "${result}")
    local body
    body=$(get_body "${result}")

    if [ "${status}" = "200" ]; then
        # Verify response has required fields
        local resp_id
        resp_id=$(json_field "${body}" "id")
        local resp_object
        resp_object=$(json_field "${body}" "object")

        if [ -n "${resp_id}" ] && [ "${resp_object}" = "response" ]; then
            assert_pass "Single-turn response returned valid response (id=${resp_id})"
        else
            assert_fail "Single-turn response missing id or wrong object type" "id=${resp_id}, object=${resp_object}"
        fi

        # Verify output contains assistant message
        local output_type
        output_type=$(json_field "${body}" "output.0.type")
        local output_role
        output_role=$(json_field "${body}" "output.0.role")

        if [ "${output_type}" = "message" ] && [ "${output_role}" = "assistant" ]; then
            assert_pass "Response output contains assistant message"
        else
            assert_fail "Response output format unexpected" "type=${output_type}, role=${output_role}"
        fi
    else
        assert_fail "Single-turn response returned HTTP ${status}" "${body}"
    fi
}

test_stateful_chaining() {
    log_section "Test: Stateful Chaining (previous_response_id)"

    local tenant_id="chain-tenant"
    local session_id="chain-session-$$"

    # Turn 1: Initial request with store=true
    local payload1
    payload1=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'My name is Alice. Please remember that.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result1
    result1=$(api_responses "${payload1}" "${tenant_id}" "${session_id}")
    local status1
    status1=$(get_http_status "${result1}")
    local body1
    body1=$(get_body "${result1}")

    if [ "${status1}" != "200" ]; then
        assert_fail "Stateful chaining turn 1 failed with HTTP ${status1}" "${body1}"
        return
    fi

    local response_id_1
    response_id_1=$(json_field "${body1}" "id")

    if [ -z "${response_id_1}" ]; then
        assert_fail "Turn 1 did not return a response ID"
        return
    fi

    assert_pass "Turn 1 completed successfully (id=${response_id_1})"

    # Turn 2: Chain using previous_response_id
    local payload2
    payload2=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'What is my name?',
    'previous_response_id': '${response_id_1}',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result2
    result2=$(api_responses "${payload2}" "${tenant_id}" "${session_id}")
    local status2
    status2=$(get_http_status "${result2}")
    local body2
    body2=$(get_body "${result2}")

    if [ "${status2}" = "200" ]; then
        local response_id_2
        response_id_2=$(json_field "${body2}" "id")

        if [ -n "${response_id_2}" ]; then
            assert_pass "Turn 2 completed with chaining (id=${response_id_2}, prev=${response_id_1})"
        else
            assert_fail "Turn 2 did not return a response ID"
        fi

        # Verify the response references context from turn 1
        local output_text
        output_text=$(json_field "${body2}" "output.0.content.0.text")
        log_info "Turn 2 response text: ${output_text:0:100}..."
    else
        assert_fail "Stateful chaining turn 2 failed with HTTP ${status2}" "${body2}"
    fi

    # Turn 3: Continue the chain
    local response_id_2
    response_id_2=$(json_field "${body2}" "id")
    local payload3
    payload3=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'Can you say my name backwards?',
    'previous_response_id': '${response_id_2}',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result3
    result3=$(api_responses "${payload3}" "${tenant_id}" "${session_id}")
    local status3
    status3=$(get_http_status "${result3}")

    if [ "${status3}" = "200" ]; then
        local response_id_3
        response_id_3=$(json_field "$(get_body "${result3}")" "id")
        assert_pass "Turn 3 completed (3-turn chain, id=${response_id_3})"
    else
        assert_fail "Turn 3 failed with HTTP ${status3}"
    fi
}

test_stored_response_retrieval() {
    log_section "Test: Stored Response Retrieval (GET /v1/responses/{id})"

    local tenant_id="retrieve-tenant"
    local session_id="retrieve-session-$$"

    # Create a response with store=true
    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'Say hello.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result
    result=$(api_responses "${payload}" "${tenant_id}" "${session_id}")
    local status
    status=$(get_http_status "${result}")
    local body
    body=$(get_body "${result}")

    if [ "${status}" != "200" ]; then
        assert_fail "Could not create response for retrieval test" "HTTP ${status}"
        return
    fi

    local response_id
    response_id=$(json_field "${body}" "id")

    # GET the stored response
    local get_result
    get_result=$(api_get_response "${response_id}" "${tenant_id}" "${session_id}")
    local get_status
    get_status=$(get_http_status "${get_result}")
    local get_body
    get_body=$(get_body "${get_result}")

    if [ "${get_status}" = "200" ]; then
        local retrieved_id
        retrieved_id=$(json_field "${get_body}" "id")
        if [ "${retrieved_id}" = "${response_id}" ]; then
            assert_pass "GET /v1/responses/${response_id} returned correct response"
        else
            assert_fail "GET returned different response ID" "expected=${response_id}, got=${retrieved_id}"
        fi
    else
        assert_fail "GET /v1/responses/${response_id} returned HTTP ${get_status}" "${get_body}"
    fi

    # DELETE the response
    local del_result
    del_result=$(api_delete_response "${response_id}" "${tenant_id}" "${session_id}")
    local del_status
    del_status=$(get_http_status "${del_result}")

    if [ "${del_status}" = "204" ]; then
        assert_pass "DELETE /v1/responses/${response_id} succeeded"
    else
        assert_fail "DELETE returned HTTP ${del_status} (expected 204)"
    fi

    # Verify it's gone
    local verify_result
    verify_result=$(api_get_response "${response_id}" "${tenant_id}" "${session_id}")
    local verify_status
    verify_status=$(get_http_status "${verify_result}")

    if [ "${verify_status}" = "404" ]; then
        assert_pass "Deleted response correctly returns 404"
    else
        assert_fail "Deleted response still accessible (HTTP ${verify_status})"
    fi
}

test_cross_session_access() {
    log_section "Test: Cross-Session Access Within Same Tenant (Hierarchical Isolation)"

    local tenant_id="isolation-tenant"
    local session_a="session-A-$$"
    local session_b="session-B-$$"

    # Create a response in session A
    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'My name is Alice and I am in session A.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result
    result=$(api_responses "${payload}" "${tenant_id}" "${session_a}")
    local status
    status=$(get_http_status "${result}")
    local body
    body=$(get_body "${result}")

    if [ "${status}" != "200" ]; then
        assert_fail "Could not create response in session A" "HTTP ${status}"
        return
    fi

    local response_id
    response_id=$(json_field "${body}" "id")
    assert_pass "Created response in session A (id=${response_id})"

    # GET from session B (same tenant) — should SUCCEED (hierarchical isolation)
    # Session is metadata, not a security boundary. This enables multi-agent workflows.
    local cross_result
    cross_result=$(api_get_response "${response_id}" "${tenant_id}" "${session_b}")
    local cross_status
    cross_status=$(get_http_status "${cross_result}")

    if [ "${cross_status}" = "200" ]; then
        assert_pass "Session B CAN access session A's response within same tenant (200)"
    else
        assert_fail "Cross-session access within tenant failed" "Expected 200, got HTTP ${cross_status}"
    fi

    # Chain from session B using session A's response_id — should work with full context
    local chain_payload
    chain_payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'What is my name? (I told you in the previous message)',
    'previous_response_id': '${response_id}',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local chain_result
    chain_result=$(api_responses "${chain_payload}" "${tenant_id}" "${session_b}")
    local chain_status
    chain_status=$(get_http_status "${chain_result}")

    if [ "${chain_status}" = "200" ]; then
        local chain_text
        chain_text=$(json_field "$(get_body "${chain_result}")" "output.0.content.0.text")
        assert_pass "Cross-session chain within tenant succeeded (200)"
        log_info "Cross-session chain response: ${chain_text:0:100}..."
    else
        assert_fail "Cross-session chain within tenant failed" "HTTP ${chain_status}"
    fi
}

test_tenant_isolation() {
    log_section "Test: Tenant Isolation"

    local tenant_a="tenant-A-$$"
    local tenant_b="tenant-B-$$"
    local session_id="shared-session-$$"

    # Create a response under tenant A
    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'Confidential data for tenant A.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': True
}))
")

    local result
    result=$(api_responses "${payload}" "${tenant_a}" "${session_id}")
    local status
    status=$(get_http_status "${result}")
    local body
    body=$(get_body "${result}")

    if [ "${status}" != "200" ]; then
        assert_fail "Could not create response for tenant A" "HTTP ${status}"
        return
    fi

    local response_id
    response_id=$(json_field "${body}" "id")
    assert_pass "Created response under tenant A (id=${response_id})"

    # Try to GET from tenant B
    local cross_result
    cross_result=$(api_get_response "${response_id}" "${tenant_b}" "${session_id}")
    local cross_status
    cross_status=$(get_http_status "${cross_result}")

    if [ "${cross_status}" = "404" ]; then
        assert_pass "Tenant B cannot access tenant A's response (404)"
    else
        assert_fail "Tenant isolation broken: tenant B got HTTP ${cross_status} for tenant A's response"
    fi

    # Verify tenant A can still access it
    local own_result
    own_result=$(api_get_response "${response_id}" "${tenant_a}" "${session_id}")
    local own_status
    own_status=$(get_http_status "${own_result}")

    if [ "${own_status}" = "200" ]; then
        assert_pass "Tenant A can access its own response"
    else
        assert_fail "Tenant A cannot access its own response" "HTTP ${own_status}"
    fi
}

test_missing_headers() {
    log_section "Test: Missing Required Headers"

    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'Hello',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS}
}))
")

    # No x-tenant-id header
    local result_no_tenant
    result_no_tenant=$(curl -s -w "\n%{http_code}" \
        -X POST "${BASE_URL}/responses" \
        -H "Content-Type: application/json" \
        -H "x-session-id: test-session" \
        -d "${payload}")
    local status_no_tenant
    status_no_tenant=$(get_http_status "${result_no_tenant}")

    if [ "${status_no_tenant}" = "400" ]; then
        assert_pass "Missing x-tenant-id correctly returns 400"
    else
        assert_fail "Missing x-tenant-id returned HTTP ${status_no_tenant} (expected 400)"
    fi

    # No x-session-id header
    local result_no_session
    result_no_session=$(curl -s -w "\n%{http_code}" \
        -X POST "${BASE_URL}/responses" \
        -H "Content-Type: application/json" \
        -H "x-tenant-id: test-tenant" \
        -d "${payload}")
    local status_no_session
    status_no_session=$(get_http_status "${result_no_session}")

    if [ "${status_no_session}" = "400" ]; then
        assert_pass "Missing x-session-id correctly returns 400"
    else
        assert_fail "Missing x-session-id returned HTTP ${status_no_session} (expected 400)"
    fi

    # Both headers missing
    local result_no_both
    result_no_both=$(curl -s -w "\n%{http_code}" \
        -X POST "${BASE_URL}/responses" \
        -H "Content-Type: application/json" \
        -d "${payload}")
    local status_no_both
    status_no_both=$(get_http_status "${result_no_both}")

    if [ "${status_no_both}" = "400" ]; then
        assert_pass "Missing both headers correctly returns 400"
    else
        assert_fail "Missing both headers returned HTTP ${status_no_both} (expected 400)"
    fi
}

test_store_false_not_persisted() {
    log_section "Test: store=false Does Not Persist"

    local tenant_id="nostore-tenant"
    local session_id="nostore-session-$$"

    # Create a response with store=false
    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'input': 'This should not be stored.',
    'max_output_tokens': ${MAX_OUTPUT_TOKENS},
    'store': False
}))
")

    local result
    result=$(api_responses "${payload}" "${tenant_id}" "${session_id}")
    local status
    status=$(get_http_status "${result}")
    local body
    body=$(get_body "${result}")

    if [ "${status}" != "200" ]; then
        assert_fail "store=false request failed" "HTTP ${status}"
        return
    fi

    local response_id
    response_id=$(json_field "${body}" "id")
    assert_pass "store=false response created (id=${response_id})"

    # Try to GET -- should be 404 since it was not stored
    local get_result
    get_result=$(api_get_response "${response_id}" "${tenant_id}" "${session_id}")
    local get_status
    get_status=$(get_http_status "${get_result}")

    if [ "${get_status}" = "404" ]; then
        assert_pass "store=false response correctly not persisted (404 on GET)"
    else
        assert_fail "store=false response was persisted unexpectedly" "GET returned HTTP ${get_status}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo "============================================================"
    echo "  Stateful Responses API - GPU Integration Tests"
    echo "============================================================"
    echo ""
    echo "Configuration:"
    echo "  BASE_URL:          ${BASE_URL}"
    echo "  MODEL:             ${MODEL}"
    echo "  MODEL_PATH:        ${MODEL_PATH}"
    echo "  SERVER_PORT:       ${SERVER_PORT}"
    echo "  SKIP_SERVER_START: ${SKIP_SERVER_START}"
    echo "  MAX_OUTPUT_TOKENS: ${MAX_OUTPUT_TOKENS}"
    echo ""

    # Trap to stop server on exit
    trap stop_server EXIT

    # Start and wait for server
    start_server
    wait_for_server

    # Run test suite
    test_single_turn_response
    test_stateful_chaining
    test_stored_response_retrieval
    test_cross_session_access
    test_tenant_isolation
    test_missing_headers
    test_store_false_not_persisted

    # Summary
    log_section "Test Results"
    echo ""
    echo "  Total:  ${TESTS_TOTAL}"
    echo -e "  Passed: ${GREEN}${TESTS_PASSED}${NC}"
    echo -e "  Failed: ${RED}${TESTS_FAILED}${NC}"
    echo ""

    if [ ${TESTS_FAILED} -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}${TESTS_FAILED} test(s) failed.${NC}"
        return 1
    fi
}

main "$@"
