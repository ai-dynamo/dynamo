#!/bin/bash
python3 /harness/barrier_patch.py
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

log "Waiting for leader..."
while true; do
    HASH=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    [ -n "$HASH" ] && break
    sleep 0.5
done
log "Found leader (hash=$HASH)"

LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL 2>/dev/null)
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease"
    exit 1
fi

$ETCDCTL put "groups/$GROUP/$HASH/rank-$NODE_RANK" "$MY_UUID" --lease="$LEASE_ID" >/dev/null 2>&1
log "Registered under leader hash"

(
    $ETCDCTL lease keep-alive "$LEASE_ID" >/dev/null 2>&1
    EXIT_CODE=$?
    echo "[$(date +%H:%M:%S.%3N)] [worker/$GROUP/rank-$NODE_RANK] !!! KEEPALIVE DIED (exit=$EXIT_CODE) !!!"
) &
KEEPALIVE_PID=$!
log "Keepalive PID: $KEEPALIVE_PID"

cleanup() {
    log "Cleanup: killing all"
    kill -9 $KEEPALIVE_PID 2>/dev/null
    [ -n "$ENGINE_PID" ] && kill -9 $ENGINE_PID 2>/dev/null
    $ETCDCTL lease revoke "$LEASE_ID" >/dev/null 2>&1
    wait 2>/dev/null
}
trap cleanup EXIT

log "Waiting for go signal..."
while true; do
    GO=$($ETCDCTL get "groups/$GROUP/$HASH/start" --print-value-only 2>/dev/null)
    if [ "$GO" = "go" ]; then
        log "Go signal received"
        break
    fi

    CURRENT=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    if [ -z "$CURRENT" ]; then
        log "DETECTED: Leader disappeared while waiting for go ($(ts))"
        exit 1
    fi
    if [ "$CURRENT" != "$HASH" ]; then
        log "DETECTED: Leader hash changed while waiting for go ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during wait for go) !!!"
        exit 1
    fi

    sleep 0.5
done

log "Starting engine: $ENGINE_CMD"
$ENGINE_CMD &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

log "Monitoring leader..."
while true; do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during monitoring) !!!"
        exit 1
    fi

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
