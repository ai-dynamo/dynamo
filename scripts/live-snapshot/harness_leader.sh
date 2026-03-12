#!/bin/bash
python3 /harness/barrier_patch.py
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

LEASE_GRANT=$($ETCDCTL lease grant $LEASE_TTL 2>/dev/null)
LEASE_ID=$(echo "$LEASE_GRANT" | awk '/lease/{print $2}')
if [ -z "$LEASE_ID" ]; then
    log "ERROR: Failed to create lease"
    exit 1
fi
log "Lease created: $LEASE_ID (TTL=${LEASE_TTL}s)"

$ETCDCTL put "leaders/$GROUP" "$HASH" --lease="$LEASE_ID" >/dev/null 2>&1
log "Published leader key"

# Monitored keepalive: log if it dies
(
    $ETCDCTL lease keep-alive "$LEASE_ID" >/dev/null 2>&1
    EXIT_CODE=$?
    echo "[$(date +%H:%M:%S.%3N)] [leader/$GROUP] !!! KEEPALIVE DIED (exit=$EXIT_CODE) !!!"
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

log "Waiting for $((NNODES - 1)) worker(s) to join..."
DEADLINE=$(($(date +%s) + FORMATION_TIMEOUT))
while true; do
    # Check keepalive still alive
    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during formation) !!!"
        exit 1
    fi

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

declare -A WORKER_UUIDS
for rank in $(seq 1 $((NNODES - 1))); do
    WORKER_UUIDS[$rank]=$($ETCDCTL get "groups/$GROUP/$HASH/rank-$rank" --print-value-only 2>/dev/null)
    log "Recorded rank-$rank: ${WORKER_UUIDS[$rank]}"
done

# Conditional delay: if flock is held, another engine is active/waking
if ! flock -n /shared/failover.lock -c "exit 0" 2>/dev/null; then
    log "Flock held, delaying 60s for active engine to settle"
    sleep 60
    log "Delay complete"
fi

$ETCDCTL put "groups/$GROUP/$HASH/start" "go" --lease="$LEASE_ID" >/dev/null 2>&1
log "Sent go signal"

log "Starting engine: $ENGINE_CMD"
$ENGINE_CMD &
ENGINE_PID=$!
log "Engine PID: $ENGINE_PID"

log "Monitoring workers..."
while true; do
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        log "Engine died, exiting ($(ts))"
        exit 1
    fi

    if ! kill -0 $KEEPALIVE_PID 2>/dev/null; then
        log "!!! KEEPALIVE PROCESS DEAD (during monitoring) !!!"
        exit 1
    fi

    MY_KEY=$($ETCDCTL get "leaders/$GROUP" --print-value-only 2>/dev/null)
    if [ "$MY_KEY" != "$HASH" ]; then
        log "DETECTED: own leader key lost (lease expired?) ($(ts))"
        exit 1
    fi

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
