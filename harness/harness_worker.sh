#!/bin/bash
# Worker coordination harness
#
# Usage: ENGINE_ID=0 NODE_RANK=1 harness_worker.sh [engine_command...]
# If no engine command given, uses "sleep infinity" as dummy engine.
#
# Environment:
#   ENGINE_ID       - engine identifier (must match leader's ENGINE_ID)
#   NODE_RANK       - this worker's rank (1, 2, ...)
#   ETCDCTL         - etcdctl command (default: "docker exec etcd-test etcdctl")
#   LEASE_TTL       - lease TTL in seconds (default: 5)

set -o pipefail

ENGINE_ID="${ENGINE_ID:-0}"
NODE_RANK="${NODE_RANK:-1}"
ETCDCTL="${ETCDCTL:-etcdctl --endpoints=http://localhost:2379}"
LEASE_TTL="${LEASE_TTL:-5}"
GROUP="engine-${ENGINE_ID}"
MY_UUID=$(cat /proc/sys/kernel/random/uuid)

ts() { date +%s%3N; }
log() { echo "[$(date +%H:%M:%S.%3N)] [worker/$GROUP/rank-$NODE_RANK] $1"; }

ENGINE_CMD="${@:-sleep infinity}"

log "Starting (uuid=$MY_UUID)"

# Step 1: Wait for leader key
log "Waiting for leader..."
while true; do
    HASH=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    [ -n "$HASH" ] && break
    sleep 0.5
done
log "Found leader (hash=$HASH)"

# Step 2: Create lease and register under leader's hash
LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL 2>/dev/null)
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease"
    exit 1
fi

$ETCDCTL put "groups/$GROUP/$HASH/rank-$NODE_RANK" "$MY_UUID" --lease="$LEASE_ID" >/dev/null 2>&1
log "Registered under leader hash"

# Keep lease alive in background
($ETCDCTL lease keep-alive "$LEASE_ID" >/dev/null 2>&1) &
KEEPALIVE_PID=$!

cleanup() {
    log "Cleanup: killing keepalive and engine"
    kill $KEEPALIVE_PID 2>/dev/null
    [ -n "$ENGINE_PID" ] && kill $ENGINE_PID 2>/dev/null
    $ETCDCTL lease revoke "$LEASE_ID" >/dev/null 2>&1
    wait 2>/dev/null
}
trap cleanup EXIT

# Step 3: Wait for go signal, monitoring leader
log "Waiting for go signal..."
while true; do
    GO=$($ETCDCTL get "groups/$GROUP/$HASH/start" --print-value-only 2>/dev/null)
    if [ "$GO" = "go" ]; then
        log "Go signal received"
        break
    fi

    # Check leader still alive and same hash
    CURRENT=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    if [ -z "$CURRENT" ]; then
        log "DETECTED: Leader disappeared while waiting for go ($(ts))"
        exit 1
    fi
    if [ "$CURRENT" != "$HASH" ]; then
        log "DETECTED: Leader hash changed while waiting for go ($(ts))"
        exit 1
    fi
    sleep 0.5
done

# Step 4: Start engine
log "Starting engine: $ENGINE_CMD"
$ENGINE_CMD &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

# Step 5: Monitor leader
log "Monitoring leader..."
while true; do
    # Check engine alive
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    # Check leader still alive and same hash
    CURRENT=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    if [ -z "$CURRENT" ]; then
        log "DETECTED: Leader disappeared ($(ts))"
        exit 1
    fi
    if [ "$CURRENT" != "$HASH" ]; then
        log "DETECTED: Leader hash changed ($(ts))"
        exit 1
    fi
    sleep 1
done
