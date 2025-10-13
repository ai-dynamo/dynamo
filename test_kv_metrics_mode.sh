#!/bin/bash
# Test script to demonstrate KV metrics behavior in different modes

set -e

echo "=== KV Metrics Mode Test ==="
echo ""

# Test 1: NATS mode (default)
echo "Test 1: NATS Mode (default)"
echo "----------------------------"
unset DYN_REQUEST_PLANE
echo "DYN_REQUEST_PLANE: (unset - defaults to nats)"
echo "Expected: NATS service stats collection ENABLED"
echo "Expected: NATS KV metrics publishing ENABLED"
echo "Expected: Will see 'Starting NATS metrics publishing' logs"
echo ""

# Test 2: Explicit NATS mode
echo "Test 2: Explicit NATS Mode"
echo "----------------------------"
export DYN_REQUEST_PLANE=nats
echo "DYN_REQUEST_PLANE: $DYN_REQUEST_PLANE"
echo "Expected: NATS service stats collection ENABLED"
echo "Expected: NATS KV metrics publishing ENABLED"
echo "Expected: Will see 'Starting NATS metrics publishing' logs"
echo ""

# Test 3: HTTP mode
echo "Test 3: HTTP Mode"
echo "----------------------------"
export DYN_REQUEST_PLANE=http
echo "DYN_REQUEST_PLANE: $DYN_REQUEST_PLANE"
echo "Expected: NATS service stats collection DISABLED"
echo "Expected: NATS KV metrics publishing DISABLED"
echo "Expected: NATS KV events publishing DISABLED"
echo "Expected: Will see 'Skipping NATS' logs"
echo ""

echo "=== NATS Server Logs Comparison ==="
echo ""
echo "In NATS mode, you will see logs like:"
echo "  [TRC] - [SUB _INBOX.shcjpujnveuspZM7cDhSwx 33]"
echo "  [TRC] - [PUB \$SRV.STATS.dynamo_backend ...]"
echo "  [TRC] - [PUB namespace.dynamo.kv_metrics 310]"
echo "  [TRC] - [PUB namespace-dynamo-component-backend-kv-events.queue 191]"
echo ""
echo "In HTTP mode, you will ONLY see minimal traffic:"
echo "  [TRC] - [PING]"
echo "  [TRC] - [PONG]"
echo ""

echo "=== How to Test ==="
echo ""
echo "1. Start NATS with trace logging:"
echo "   nats-server -DV"
echo ""
echo "2. Start your Dynamo backend with HTTP mode:"
echo "   export DYN_REQUEST_PLANE=http"
echo "   # Start your backend service"
echo ""
echo "3. Watch NATS logs - you should NOT see:"
echo "   - PUB \$SRV.STATS.*"
echo "   - PUB namespace.dynamo.kv_metrics"
echo "   - PUB namespace-dynamo-component-backend-kv-events.queue"
echo "   - SUB _INBOX.*"
echo ""
echo "4. You WILL still see PING/PONG (keep-alive)"
echo ""

echo "=== Code Changes ==="
echo ""
echo "Modified files:"
echo "  - lib/runtime/src/component/service.rs"
echo "  - lib/llm/src/kv_router/publisher.rs"
echo ""
echo "Both files now check RequestPlaneMode::from_env() and conditionally"
echo "enable NATS metrics publishing only when in NATS mode."
echo ""

echo "Test completed!"

