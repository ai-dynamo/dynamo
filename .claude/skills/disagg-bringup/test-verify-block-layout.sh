#!/bin/bash
# Unit test for verify-block-layout.sh using a mock HTTP server.
#
# Spawns a persistent Python HTTP server that serves configurable JSON
# responses (one file updated between test cases), then calls
# verify-block-layout.sh against it.
#
# Usage: bash test-verify-block-layout.sh
set -eu

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0

RESPONSE_FILE=$(mktemp)
PORT_FILE=$(mktemp)
PY_SCRIPT=$(mktemp --suffix=.py)

# Write the fake hub server script to a temp file.
cat > "$PY_SCRIPT" <<'PYEOF'
import http.server, sys

port_file = sys.argv[1]
response_file = sys.argv[2]

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with open(response_file) as f:
            body = f.read().encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

srv = http.server.HTTPServer(("127.0.0.1", 0), H)
with open(port_file, "w") as f:
    f.write(str(srv.server_address[1]))
srv.serve_forever()
PYEOF

# Seed response file before starting server.
echo '{}' > "$RESPONSE_FILE"

python3 "$PY_SCRIPT" "$PORT_FILE" "$RESPONSE_FILE" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$RESPONSE_FILE" "$PORT_FILE" "$PY_SCRIPT"
}
trap cleanup EXIT

# Wait for the server to write its port (max 5s).
waited=0
while [ ! -s "$PORT_FILE" ] && [ "$waited" -lt 50 ]; do
  sleep 0.1
  waited=$((waited + 1))
done
PORT=$(cat "$PORT_FILE")
echo "fake hub on port $PORT (pid $SERVER_PID)"

run_check() {
  local desc="$1"
  local response_json="$2"
  local expected="$3"
  local want_exit="$4"

  printf '%s' "$response_json" > "$RESPONSE_FILE"

  local actual_exit=0
  KVBM_HUB_CONTROL_PORT="$PORT" bash "$SKILL_DIR/verify-block-layout.sh" \
    "fake-instance-uuid" "$expected" >/dev/null 2>&1 || actual_exit=$?

  if [ "$actual_exit" = "$want_exit" ]; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc — got exit $actual_exit, want $want_exit"
    FAIL=$((FAIL + 1))
  fi
}

# --- Test cases ---
#
# The hub wraps InstanceDescription in a control envelope:
#   { "description": <InstanceDescription>, "cached": ..., "age_secs": ..., "source": ... }
# Every fixture below mirrors that shape so the verifier is exercised
# against the actual production response. A test that omits the envelope
# is a guard against accidentally regressing the unwrap step.

UNIVERSAL_DESC='{"description":{"config":{"block_layout":"universal"},"workers":[]},"cached":false,"age_secs":0,"source":"pull_fallback"}'
OPERATIONAL_DESC='{"description":{"config":{"block_layout":"operational"},"workers":[]},"cached":true,"age_secs":1,"source":"push"}'
FALLBACK_UNIVERSAL='{"description":{"workers":[{"worker_id":0,"nixl_agent_name":"w0","layouts":[{"tier":"g2","config":{},"location":"pinned","layout_type":"fully_contiguous","block_layout":"universal","bytes_per_block":0,"total_bytes":0}]}]},"cached":false,"age_secs":0,"source":"pull_fallback"}'
FALLBACK_OP='{"description":{"workers":[{"worker_id":0,"nixl_agent_name":"w0","layouts":[{"tier":"g2","config":{},"location":"pinned","layout_type":"fully_contiguous","block_layout":"operational_nhd","bytes_per_block":0,"total_bytes":0}]}]},"cached":false,"age_secs":0,"source":"pull_fallback"}'

# Envelope-less fixture — covers the intermediary-strip case where the
# response arrives without the `description` wrapper.
UNIVERSAL_NO_ENVELOPE='{"config":{"block_layout":"universal"},"workers":[]}'

run_check "envelope: config-blob universal → expect universal"    "$UNIVERSAL_DESC"    "universal"    0
run_check "envelope: config-blob universal → expect operational"  "$UNIVERSAL_DESC"    "operational"  1
run_check "envelope: config-blob operational → expect operational" "$OPERATIONAL_DESC" "operational"  0
run_check "envelope: config-blob operational → expect universal"  "$OPERATIONAL_DESC"  "universal"    1
run_check "envelope: fallback G2 universal → expect universal"    "$FALLBACK_UNIVERSAL" "universal"    0
run_check "envelope: fallback G2 operational_nhd → expect operational" "$FALLBACK_OP"    "operational"  0
run_check "envelope: fallback G2 universal → expect operational"  "$FALLBACK_UNIVERSAL" "operational"  1
run_check "no envelope: config-blob universal → expect universal" "$UNIVERSAL_NO_ENVELOPE" "universal" 0

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
