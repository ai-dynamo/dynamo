#!/bin/bash
# Test suite for the coordination harness
#
# Validates group formation, failure detection, and coordinated restart
# using dummy engines (sleep) instead of real vLLM.
#
# Usage: ./test_harness.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETCDCTL="etcdctl --endpoints=http://localhost:2379"
LEASE_TTL=3  # Short TTL for fast tests

LOG_DIR="/tmp/harness_test_$$"
mkdir -p "$LOG_DIR"

pass_count=0
fail_count=0

pass() { pass_count=$((pass_count + 1)); echo "  PASS: $1"; }
fail() { fail_count=$((fail_count + 1)); echo "  FAIL: $1"; }

ts_ms() { date +%s%3N; }

cleanup_etcd() {
    $ETCDCTL del "leaders/" --prefix >/dev/null 2>&1
    $ETCDCTL del "groups/" --prefix >/dev/null 2>&1
}

cleanup_procs() {
    for pid_file in "$LOG_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        pid=$(cat "$pid_file" 2>/dev/null)
        [ -z "$pid" ] && continue
        # Kill entire process group (setsid gives each harness its own pgid)
        kill -9 -"$pid" 2>/dev/null || true
    done
    # Kill stray keepalive processes by exact match (avoid killing test runner)
    for p in $(pgrep -f "etcdctl.*lease.keep-alive" 2>/dev/null); do
        kill -9 "$p" 2>/dev/null || true
    done
    for p in $(pgrep -x "sleep" 2>/dev/null); do
        kill -9 "$p" 2>/dev/null || true
    done
    sleep 1
}

full_cleanup() {
    cleanup_procs
    cleanup_etcd
}

start_leader() {
    local test_name="$1"
    setsid env ENGINE_ID=0 NNODES=2 LEASE_TTL=$LEASE_TTL FORMATION_TIMEOUT=15 \
        PATH="$PATH" \
        bash "$SCRIPT_DIR/harness_leader.sh" sleep infinity \
        > "$LOG_DIR/${test_name}_leader.log" 2>&1 &
    echo $! > "$LOG_DIR/${test_name}_leader.pid"
}

start_worker() {
    local test_name="$1"
    setsid env ENGINE_ID=0 NODE_RANK=1 LEASE_TTL=$LEASE_TTL \
        PATH="$PATH" \
        bash "$SCRIPT_DIR/harness_worker.sh" sleep infinity \
        > "$LOG_DIR/${test_name}_worker.log" 2>&1 &
    echo $! > "$LOG_DIR/${test_name}_worker.pid"
}

wait_for_log() {
    local log_file="$1" pattern="$2" timeout="$3"
    for i in $(seq 1 "$timeout"); do
        grep -q "$pattern" "$log_file" 2>/dev/null && return 0
        sleep 1
    done
    return 1
}

wait_for_exit() {
    local pid="$1" timeout="$2"
    for i in $(seq 1 "$timeout"); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    return 1
}

echo "=============================================="
echo "  Coordination Harness Test Suite"
echo "=============================================="
echo "Log directory: $LOG_DIR"
echo "Lease TTL: ${LEASE_TTL}s"
echo ""

# ============================================================
# Test 1: Happy path
# ============================================================
echo "=== Test 1: Happy Path ==="
full_cleanup

start_leader "t1"
sleep 1
start_worker "t1"

if wait_for_log "$LOG_DIR/t1_leader.log" "Sent go signal" 10; then
    pass "T1: Leader sent go signal"
else
    fail "T1: Leader did not send go signal"
fi

if wait_for_log "$LOG_DIR/t1_worker.log" "Go signal received" 10; then
    pass "T1: Worker received go signal"
else
    fail "T1: Worker did not receive go signal"
fi

if wait_for_log "$LOG_DIR/t1_leader.log" "Monitoring workers" 5; then
    pass "T1: Leader monitoring"
else
    fail "T1: Leader not monitoring"
fi

if wait_for_log "$LOG_DIR/t1_worker.log" "Monitoring leader" 5; then
    pass "T1: Worker monitoring"
else
    fail "T1: Worker not monitoring"
fi

# ============================================================
# Test 2: Leader dies during active — worker exits
# ============================================================
echo ""
echo "=== Test 2: Leader Dies → Worker Exits ==="
full_cleanup

start_leader "t2"
sleep 1
start_worker "t2"
wait_for_log "$LOG_DIR/t2_leader.log" "Monitoring workers" 10
wait_for_log "$LOG_DIR/t2_worker.log" "Monitoring leader" 10

LEADER_PID=$(cat "$LOG_DIR/t2_leader.pid")
T_KILL=$(ts_ms)
kill -9 -"$LEADER_PID" 2>/dev/null

WORKER_PID=$(cat "$LOG_DIR/t2_worker.pid")
if wait_for_exit "$WORKER_PID" 15; then
    T_EXIT=$(ts_ms)
    PROPAGATION=$((T_EXIT - T_KILL))
    pass "T2: Worker exited after leader death (${PROPAGATION}ms)"
else
    fail "T2: Worker did not exit within 15s"
fi

# Check detection log
if grep -q "DETECTED.*Leader disappeared" "$LOG_DIR/t2_worker.log" 2>/dev/null; then
    DETECT_LINE=$(grep "DETECTED" "$LOG_DIR/t2_worker.log" | head -1)
    DETECT_TS=$(echo "$DETECT_LINE" | grep -oP '\(\K[0-9]+')
    if [ -n "$DETECT_TS" ]; then
        DETECT_LATENCY=$((DETECT_TS - T_KILL))
        echo "    Detection latency: ${DETECT_LATENCY}ms"
    fi
    echo "    Total propagation: ${PROPAGATION}ms"
else
    echo "    (detection log not found)"
fi

# ============================================================
# Test 3: Worker dies during active — leader exits
# ============================================================
echo ""
echo "=== Test 3: Worker Dies → Leader Exits ==="
full_cleanup

start_leader "t3"
sleep 1
start_worker "t3"
wait_for_log "$LOG_DIR/t3_leader.log" "Monitoring workers" 10
wait_for_log "$LOG_DIR/t3_worker.log" "Monitoring leader" 10

WORKER_PID=$(cat "$LOG_DIR/t3_worker.pid")
T_KILL=$(ts_ms)
kill -9 -"$WORKER_PID" 2>/dev/null

LEADER_PID=$(cat "$LOG_DIR/t3_leader.pid")
if wait_for_exit "$LEADER_PID" 15; then
    T_EXIT=$(ts_ms)
    PROPAGATION=$((T_EXIT - T_KILL))
    pass "T3: Leader exited after worker death (${PROPAGATION}ms)"
else
    fail "T3: Leader did not exit within 15s"
fi

if grep -q "DETECTED.*rank-1 disappeared" "$LOG_DIR/t3_leader.log" 2>/dev/null; then
    DETECT_LINE=$(grep "DETECTED" "$LOG_DIR/t3_leader.log" | head -1)
    DETECT_TS=$(echo "$DETECT_LINE" | grep -oP '\(\K[0-9]+')
    if [ -n "$DETECT_TS" ]; then
        DETECT_LATENCY=$((DETECT_TS - T_KILL))
        echo "    Detection latency: ${DETECT_LATENCY}ms"
    fi
    echo "    Total propagation: ${PROPAGATION}ms"
fi

# ============================================================
# Test 4: Leader dies before go — worker exits
# ============================================================
echo ""
echo "=== Test 4: Leader Dies Before Go → Worker Exits ==="
full_cleanup

# Start leader but kill it before worker joins
setsid env ENGINE_ID=0 NNODES=2 LEASE_TTL=$LEASE_TTL FORMATION_TIMEOUT=15 \
    PATH="$PATH" \
    bash "$SCRIPT_DIR/harness_leader.sh" sleep infinity \
    > "$LOG_DIR/t4_leader.log" 2>&1 &
echo $! > "$LOG_DIR/t4_leader.pid"
sleep 1

start_worker "t4"
sleep 2  # Worker has read leader hash and registered, waiting for go

LEADER_PID=$(cat "$LOG_DIR/t4_leader.pid")
T_KILL=$(ts_ms)
kill -9 -"$LEADER_PID" 2>/dev/null

WORKER_PID=$(cat "$LOG_DIR/t4_worker.pid")
if wait_for_exit "$WORKER_PID" 15; then
    T_EXIT=$(ts_ms)
    PROPAGATION=$((T_EXIT - T_KILL))
    pass "T4: Worker exited after leader died before go (${PROPAGATION}ms)"
else
    fail "T4: Worker did not exit within 15s"
fi

# ============================================================
# Test 5: Worker crashes before registering — leader times out
# ============================================================
echo ""
echo "=== Test 5: Worker Never Joins → Leader Timeout ==="
full_cleanup

setsid env ENGINE_ID=0 NNODES=2 LEASE_TTL=$LEASE_TTL FORMATION_TIMEOUT=5 \
    PATH="$PATH" \
    bash "$SCRIPT_DIR/harness_leader.sh" sleep infinity \
    > "$LOG_DIR/t5_leader.log" 2>&1 &
echo $! > "$LOG_DIR/t5_leader.pid"

# Don't start worker at all
LEADER_PID=$(cat "$LOG_DIR/t5_leader.pid")
if wait_for_exit "$LEADER_PID" 10; then
    pass "T5: Leader timed out waiting for workers"
else
    fail "T5: Leader did not timeout"
fi

if grep -q "Formation timeout" "$LOG_DIR/t5_leader.log" 2>/dev/null; then
    pass "T5: Leader logged formation timeout"
else
    fail "T5: No formation timeout in log"
fi

# ============================================================
# Test 6: Leader restarts fast — worker detects hash change
# ============================================================
echo ""
echo "=== Test 6: Leader Restarts → Worker Detects Hash Change ==="
full_cleanup

start_leader "t6"
sleep 1
start_worker "t6"
wait_for_log "$LOG_DIR/t6_leader.log" "Monitoring workers" 10
wait_for_log "$LOG_DIR/t6_worker.log" "Monitoring leader" 10

# Kill leader and immediately restart with new hash
LEADER_PID=$(cat "$LOG_DIR/t6_leader.pid")
T_KILL=$(ts_ms)
kill -9 -"$LEADER_PID" 2>/dev/null
sleep 0.5

# Start new leader (will overwrite leaders/engine-0 with new hash)
ENGINE_ID=0 NNODES=2 LEASE_TTL=$LEASE_TTL FORMATION_TIMEOUT=15 \
    bash "$SCRIPT_DIR/harness_leader.sh" sleep infinity \
    > "$LOG_DIR/t6_leader2.log" 2>&1 &
echo $! > "$LOG_DIR/t6_leader2.pid"

WORKER_PID=$(cat "$LOG_DIR/t6_worker.pid")
if wait_for_exit "$WORKER_PID" 15; then
    T_EXIT=$(ts_ms)
    PROPAGATION=$((T_EXIT - T_KILL))
    pass "T6: Worker exited after leader hash change (${PROPAGATION}ms)"
else
    fail "T6: Worker did not exit within 15s"
fi

if grep -q "DETECTED.*Leader.*changed\|DETECTED.*Leader disappeared" "$LOG_DIR/t6_worker.log" 2>/dev/null; then
    pass "T6: Worker detected leader change"
else
    fail "T6: Worker did not detect leader change"
fi

# Kill the new leader
kill -9 -"$(cat "$LOG_DIR/t6_leader2.pid")" 2>/dev/null

# ============================================================
# Test 7: Worker restarts fast — leader detects UUID change
# ============================================================
echo ""
echo "=== Test 7: Worker Restarts → Leader Detects UUID Change ==="
full_cleanup

start_leader "t7"
sleep 1
start_worker "t7"
wait_for_log "$LOG_DIR/t7_leader.log" "Monitoring workers" 10
wait_for_log "$LOG_DIR/t7_worker.log" "Monitoring leader" 10

# Kill worker and immediately restart with new UUID
WORKER_PID=$(cat "$LOG_DIR/t7_worker.pid")
T_KILL=$(ts_ms)
kill -9 -"$WORKER_PID" 2>/dev/null
sleep 0.5

# Start new worker (will overwrite group key with new UUID)
ENGINE_ID=0 NODE_RANK=1 LEASE_TTL=$LEASE_TTL \
    bash "$SCRIPT_DIR/harness_worker.sh" sleep infinity \
    > "$LOG_DIR/t7_worker2.log" 2>&1 &
echo $! > "$LOG_DIR/t7_worker2.pid"

LEADER_PID=$(cat "$LOG_DIR/t7_leader.pid")
if wait_for_exit "$LEADER_PID" 15; then
    T_EXIT=$(ts_ms)
    PROPAGATION=$((T_EXIT - T_KILL))
    pass "T7: Leader exited after worker UUID change (${PROPAGATION}ms)"
else
    fail "T7: Leader did not exit within 15s"
fi

if grep -q "DETECTED.*rank-1 UUID changed" "$LOG_DIR/t7_leader.log" 2>/dev/null; then
    pass "T7: Leader detected worker UUID change"
else
    # Could also be "disappeared" if TTL expired before new worker wrote
    if grep -q "DETECTED.*rank-1 disappeared" "$LOG_DIR/t7_leader.log" 2>/dev/null; then
        pass "T7: Leader detected worker disappearance (TTL expired before re-register)"
    else
        fail "T7: Leader did not detect worker change"
    fi
fi

# Kill the new worker
kill -9 -"$(cat "$LOG_DIR/t7_worker2.pid")" 2>/dev/null

# ============================================================
# Test 8: Clean restart after failure
# ============================================================
echo ""
echo "=== Test 8: Clean Restart After Failure ==="
full_cleanup

start_leader "t8"
sleep 1
start_worker "t8"
wait_for_log "$LOG_DIR/t8_leader.log" "Monitoring workers" 10

# Kill worker to trigger leader exit
kill -9 -"$(cat "$LOG_DIR/t8_worker.pid")" 2>/dev/null
wait_for_exit "$(cat "$LOG_DIR/t8_leader.pid")" 15

# Both dead. Clean etcd state.
cleanup_etcd
sleep 1

# Restart both
start_leader "t8b"
sleep 1
start_worker "t8b"

if wait_for_log "$LOG_DIR/t8b_leader.log" "Monitoring workers" 10; then
    pass "T8: Fresh group formed after failure"
else
    fail "T8: Failed to form fresh group"
fi

if wait_for_log "$LOG_DIR/t8b_worker.log" "Monitoring leader" 10; then
    pass "T8: Worker joined fresh group"
else
    fail "T8: Worker did not join fresh group"
fi

# ============================================================
# Summary
# ============================================================
full_cleanup

echo ""
echo "=============================================="
echo "  Results: $pass_count passed, $fail_count failed"
echo "=============================================="
echo "Logs: $LOG_DIR"
[ "$fail_count" -gt 0 ] && exit 1
exit 0
