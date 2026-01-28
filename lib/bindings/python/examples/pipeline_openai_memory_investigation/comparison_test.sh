#!/bin/bash
# Comparison test: MALLOC_ARENA_MAX=2 vs default
# Runs two frontends in parallel and monitors memory over 1 hour

set -e

DURATION_MINUTES=60
PAYLOAD_SIZE=200000
CONCURRENCY=96
REQUESTS_PER_CYCLE=1000

cd /workspace/lib/bindings/python/examples/pipeline_openai

echo "=========================================="
echo "MEMORY COMPARISON TEST"
echo "=========================================="
echo "Duration: ${DURATION_MINUTES} minutes"
echo "Payload: ${PAYLOAD_SIZE} chars"
echo "Concurrency: ${CONCURRENCY}"
echo "Requests per cycle: ${REQUESTS_PER_CYCLE}"
echo "=========================================="
echo ""

# Start backend for test 1 (port 8000)
echo "Starting backend services for Test 1 (MALLOC_ARENA_MAX=2)..."
DYN_LOCAL_NAMESPACE=test1 python backend.py > /tmp/backend1.log 2>&1 &
sleep 2
DYN_LOCAL_NAMESPACE=test1 python backend.py --proxy-mode > /tmp/proxy1.log 2>&1 &
sleep 2

# Start frontend 1 with MALLOC_ARENA_MAX=2
echo "Starting frontend 1 with MALLOC_ARENA_MAX=2 on port 8000..."
MALLOC_ARENA_MAX=2 DYN_LOCAL_NAMESPACE=test1 python frontend.py > /tmp/frontend1.log 2>&1 &
FRONTEND1_PID=$!
sleep 3

# Start backend for test 2 (port 8001)
echo "Starting backend services for Test 2 (default glibc)..."
DYN_LOCAL_NAMESPACE=test2 HTTP_PORT=8001 python backend.py > /tmp/backend2.log 2>&1 &
sleep 2
DYN_LOCAL_NAMESPACE=test2 HTTP_PORT=8001 python backend.py --proxy-mode > /tmp/proxy2.log 2>&1 &
sleep 2

# Start frontend 2 with default settings
echo "Starting frontend 2 with default glibc on port 8001..."
DYN_LOCAL_NAMESPACE=test2 HTTP_PORT=8001 python frontend.py > /tmp/frontend2.log 2>&1 &
FRONTEND2_PID=$!
sleep 3

# Find actual PIDs
FPID1=$(pgrep -f "python frontend.py" | head -1)
FPID2=$(pgrep -f "python frontend.py" | tail -1)

echo ""
echo "Frontend 1 (ARENA_MAX=2) PID: $FPID1"
echo "Frontend 2 (default) PID: $FPID2"
echo ""

# Get initial memory
get_rss() {
    grep VmRSS /proc/$1/status 2>/dev/null | awk '{print $2/1024}'
}

INITIAL1=$(get_rss $FPID1)
INITIAL2=$(get_rss $FPID2)

echo "Initial memory:"
echo "  Test 1 (ARENA_MAX=2): ${INITIAL1} MB"
echo "  Test 2 (default):     ${INITIAL2} MB"
echo ""

# Output file for results
RESULTS_FILE="/tmp/comparison_results.csv"
echo "time_min,arena2_rss_mb,default_rss_mb,arena2_delta,default_delta" > $RESULTS_FILE

echo "Starting parallel load tests..."
echo ""
echo "Time     | ARENA_MAX=2 | Default    | Delta (A2) | Delta (Def)"
echo "---------|-------------|------------|------------|------------"

START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_MINUTES * 60))

cycle=0
while [ $(date +%s) -lt $END_TIME ]; do
    cycle=$((cycle + 1))

    # Run load tests in parallel
    python load_test.py --url http://localhost:8000/v1/chat/completions \
        --payload-size $PAYLOAD_SIZE --concurrency $CONCURRENCY --requests $REQUESTS_PER_CYCLE \
        > /tmp/load1.log 2>&1 &
    LOAD1_PID=$!

    python load_test.py --url http://localhost:8001/v1/chat/completions \
        --payload-size $PAYLOAD_SIZE --concurrency $CONCURRENCY --requests $REQUESTS_PER_CYCLE \
        > /tmp/load2.log 2>&1 &
    LOAD2_PID=$!

    # Wait for both to complete
    wait $LOAD1_PID 2>/dev/null
    wait $LOAD2_PID 2>/dev/null

    # Measure memory
    RSS1=$(get_rss $FPID1)
    RSS2=$(get_rss $FPID2)

    DELTA1=$(echo "$RSS1 - $INITIAL1" | bc)
    DELTA2=$(echo "$RSS2 - $INITIAL2" | bc)

    ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))

    printf "%4d min | %7.1f MB  | %7.1f MB | %+7.1f MB | %+7.1f MB\n" \
        $ELAPSED $RSS1 $RSS2 $DELTA1 $DELTA2

    echo "$ELAPSED,$RSS1,$RSS2,$DELTA1,$DELTA2" >> $RESULTS_FILE

    sleep 2
done

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="

FINAL1=$(get_rss $FPID1)
FINAL2=$(get_rss $FPID2)

echo ""
echo "Test 1 (MALLOC_ARENA_MAX=2):"
echo "  Initial: ${INITIAL1} MB"
echo "  Final:   ${FINAL1} MB"
echo "  Delta:   $(echo "$FINAL1 - $INITIAL1" | bc) MB"

echo ""
echo "Test 2 (default glibc):"
echo "  Initial: ${INITIAL2} MB"
echo "  Final:   ${FINAL2} MB"
echo "  Delta:   $(echo "$FINAL2 - $INITIAL2" | bc) MB"

echo ""
echo "Memory saved by ARENA_MAX=2: $(echo "$FINAL2 - $FINAL1" | bc) MB"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="

# Cleanup
pkill -f "python backend.py" 2>/dev/null || true
pkill -f "python frontend.py" 2>/dev/null || true
