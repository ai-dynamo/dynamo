#!/bin/bash
# Leader coordination harness
#
# Usage: ENGINE_ID=0 NNODES=2 harness_leader.sh [engine_command...]
# If no engine command given, uses "sleep infinity" as dummy engine.
#
# Environment:
#   ENGINE_ID       - engine identifier (0, 1, ...)
#   NNODES          - total nodes expected (including leader)
#   ETCDCTL         - etcdctl command (default: "docker exec etcd-test etcdctl")
#   LEASE_TTL       - lease TTL in seconds (default: 5)
#   FORMATION_TIMEOUT - seconds to wait for all ranks (default: 120)

set -o pipefail

ENGINE_ID="${ENGINE_ID:-0}"
NNODES="${NNODES:-2}"
ETCDCTL="${ETCDCTL:-etcdctl --endpoints=http://localhost:2379}"
LEASE_TTL="${LEASE_TTL:-5}"
FORMATION_TIMEOUT="${FORMATION_TIMEOUT:-120}"
GROUP="engine-${ENGINE_ID}"
HASH=$(cat /proc/sys/kernel/random/uuid)

ts() { date +%s%3N; }
log() { echo "[$(date +%H:%M:%S.%3N)] [leader/$GROUP] $1"; }

ENGINE_CMD="${@:-sleep infinity}"

log "Starting (hash=$HASH, nnodes=$NNODES)"

# Step 1: Create lease and publish leader key
LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL 2>/dev/null)
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease"
    exit 1
fi
log "Lease created: $LEASE_ID (TTL=${LEASE_TTL}s)"

$ETCDCTL put "leaders/$GROUP" "$HASH" --lease="$LEASE_ID" >/dev/null 2>&1
log "Published leader key"

# Keep lease alive in background
($ETCDCTL lease keep-alive "$LEASE_ID" >/dev/null 2>&1) &
KEEPALIVE_PID=$!

cleanup() {
    log "Cleanup: killing keepalive and engine"
    kill $KEEPALIVE_PID 2>/dev/null
    [ -n "$ENGINE_PID" ] && kill $ENGINE_PID 2>/dev/null
    # Revoke lease explicitly for fast cleanup
    $ETCDCTL lease revoke "$LEASE_ID" >/dev/null 2>&1
    wait 2>/dev/null
}
trap cleanup EXIT

# Step 2: Wait for all worker ranks to register under my hash
log "Waiting for $((NNODES - 1)) worker(s) to join..."
DEADLINE=$(($(date +%s) + FORMATION_TIMEOUT))
while true; do
    ALL_PRESENT=true
    for rank in $(seq 1 $((NNODES - 1))); do
        VAL=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>/dev/null)
        if [ -z "$VAL" ]; then
            ALL_PRESENT=false
            break
        fi
    done

    if $ALL_PRESENT; then
        log "All workers joined"
        break
    fi

    if [ $(date +%s) -gt $DEADLINE ]; then
        log "ERROR: Formation timeout (${FORMATION_TIMEOUT}s)"
        exit 1
    fi
    sleep 0.5
done

# Record worker UUIDs for change detection
declare -A WORKER_UUIDS
for rank in $(seq 1 $((NNODES - 1))); do
    WORKER_UUIDS[$rank]=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>/dev/null)
    log "Recorded rank-$rank: ${WORKER_UUIDS[$rank]}"
done

# Step 3: Signal go
$ETCDCTL put "groups/$GROUP/$HASH/start" "go" --lease="$LEASE_ID" >/dev/null 2>&1
log "Sent go signal"

# Step 4: Start engine
log "Starting engine: $ENGINE_CMD"
$ENGINE_CMD &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

# Step 5: Monitor workers
log "Monitoring workers..."
while true; do
    # Check engine alive
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    # Check all workers still present with same UUID
    for rank in $(seq 1 $((NNODES - 1))); do
        CURRENT=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>/dev/null)
        if [ -z "$CURRENT" ]; then
            log "DETECTED: rank-$rank disappeared ($(ts))"
            exit 1
        fi
        if [ "$CURRENT" != "${WORKER_UUIDS[$rank]}" ]; then
            log "DETECTED: rank-$rank UUID changed ($(ts))"
            exit 1
        fi
    done
    sleep 1
done
