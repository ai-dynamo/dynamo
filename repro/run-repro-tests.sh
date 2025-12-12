#!/bin/bash

set -e

# Usage: ./run-repro-tests.sh [token_count1] [token_count2] ...
# Example: ./run-repro-tests.sh 40000 80000
# If no arguments provided, runs all: 40000 80000 100000 120000

# Configuration
NS="${NS:-keivenc-dyn-1556-repro-nixl-timeout}"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
URL="http://localhost:8787"
OUTPUT_DIR="/tmp/repro-dyn-1556"
REQUEST_COUNT=1  # Single request per test
OUTPUT_TOKENS=100

# Token counts to test - accept from command line or use defaults
if [ $# -gt 0 ]; then
    TOKEN_COUNTS=("$@")
else
    TOKEN_COUNTS=(40000 80000 100000 120000)
fi

# No color output for better log parsing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get pod names
echo "Getting pod names from namespace: $NS"
FRONTEND=$(kubectl get pods -n "$NS" -l nvidia.com/dynamo-component=Frontend -o jsonpath='{.items[0].metadata.name}')
PREFILL=$(kubectl get pods -n "$NS" -l nvidia.com/dynamo-component=VllmPrefillWorker -o jsonpath='{.items[0].metadata.name}')
DECODE=$(kubectl get pods -n "$NS" -l nvidia.com/dynamo-component=VllmDecodeWorker -o jsonpath='{.items[0].metadata.name}')

echo "Frontend: $FRONTEND"
echo "Prefill:  $PREFILL"
echo "Decode:   $DECODE"
echo ""

# Verify pods are running
echo "Verifying pods are running..."
kubectl get pods -n "$NS" | grep -E "(frontend|vllmprefillworker|vllmdecodeworker)"
echo ""

# Check if port-forward is running
if ! pgrep -f "port-forward.*8787" > /dev/null; then
    echo "Port-forward not detected. Starting port-forward to frontend..."
    kubectl port-forward "$FRONTEND" 8787:8787 -n "$NS" > /dev/null 2>&1 &
    sleep 3
    echo "Port-forward started"
fi

# Function to format token count for filenames
format_token_count() {
    local tokens=$1
    printf "%03dk" $((tokens / 1000))
}

# Function to run a single test
run_test() {
    local tokens=$1
    local formatted=$(format_token_count "$tokens")

    echo ""
    echo "=========================================="
    echo "Running test: ${tokens} tokens"
    echo "=========================================="

    # Log file paths
    local genai_log="$OUTPUT_DIR/${formatted}_genai-perf.log"
    local prefill_log="$OUTPUT_DIR/${formatted}_prefill.log"
    local decode_log="$OUTPUT_DIR/${formatted}_decode.log"

    # Clear previous logs
    > "$genai_log"
    > "$prefill_log"
    > "$decode_log"

    # Get timestamp before test
    local start_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "Start time: $start_time"

    # Run genai-perf test
    echo "Running genai-perf..."
    genai-perf profile \
        --endpoint-type chat \
        --synthetic-input-tokens-mean "$tokens" \
        --output-tokens-mean "$OUTPUT_TOKENS" \
        --request-count "$REQUEST_COUNT" \
        --model "$MODEL" \
        --url "$URL" \
        --streaming \
        --verbose 2>&1 | tee "$genai_log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "genai-perf failed with exit code $exit_code"
    else
        echo "genai-perf completed successfully"
    fi

    # Wait for KV timeout (120 seconds + buffer)
    echo "Waiting 125 seconds for KV timeout..."
    sleep 125

    # Collect logs from prefill worker
    echo "Collecting prefill worker logs..."
    kubectl logs "$PREFILL" -n "$NS" --since-time="$start_time" > "$prefill_log" 2>&1

    # Collect logs from decode worker
    echo "Collecting decode worker logs..."
    kubectl logs "$DECODE" -n "$NS" --since-time="$start_time" > "$decode_log" 2>&1

    echo "Test ${tokens} tokens completed"
    echo "Logs saved:"
    echo "  - genai-perf: $genai_log"
    echo "  - prefill:    $prefill_log"
    echo "  - decode:     $decode_log"
}

# Run tests for each token count
echo "Starting test suite..."
echo "Testing token counts: ${TOKEN_COUNTS[*]}"
echo ""

for tokens in "${TOKEN_COUNTS[@]}"; do
    run_test "$tokens"

    # Add delay between tests
    if [ "$tokens" != "${TOKEN_COUNTS[-1]}" ]; then
        echo ""
        echo "Waiting 30 seconds before next test..."
        sleep 30
    fi
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""

# Analyze results
echo "=========================================="
echo "Analyzing results..."
echo "=========================================="
echo ""

# Initialize summary file
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
> "$SUMMARY_FILE"

echo "Test Results Summary - $(date)" | tee -a "$SUMMARY_FILE"
echo "===========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Table header
printf "%-10s | %-12s | %-18s | %-15s | %-15s | %-15s | %-15s\n" \
    "Tokens" "TTFT (s)" "Req Latency (s)" "Primary Error" "KV Timeout" "Broken Pipe" "503 Errors" | tee -a "$SUMMARY_FILE"
printf "%-10s-+-%-12s-+-%-18s-+-%-15s-+-%-15s-+-%-15s-+-%-15s\n" \
    "----------" "------------" "------------------" "---------------" "---------------" "---------------" "---------------" | tee -a "$SUMMARY_FILE"

for tokens in "${TOKEN_COUNTS[@]}"; do
    formatted=$(format_token_count "$tokens")

    genai_log="$OUTPUT_DIR/${formatted}_genai-perf.log"
    prefill_log="$OUTPUT_DIR/${formatted}_prefill.log"
    decode_log="$OUTPUT_DIR/${formatted}_decode.log"

    # Extract TTFT from genai-perf output
    ttft=""
    if [ -f "$genai_log" ]; then
        # Try to find TTFT in various formats
        ttft=$(grep -i "time to first token" "$genai_log" | grep -oP '\d+\.\d+' | head -1 || echo "")
        if [ -z "$ttft" ]; then
            # Alternative: look for p50 TTFT in stats
            ttft=$(grep -A 20 "Time To First Token" "$genai_log" | grep "p50" | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
        fi
    fi
    [ -z "$ttft" ] && ttft="N/A"

    # Extract request latency
    req_latency=""
    if [ -f "$genai_log" ]; then
        req_latency=$(grep -i "request latency" "$genai_log" | grep -oP '\d+\.\d+' | head -1 || echo "")
        if [ -z "$req_latency" ]; then
            req_latency=$(grep -A 20 "Request Latency" "$genai_log" | grep "p50" | grep -oP '\d+\.\d+' | head -1 || echo "N/A")
        fi
    fi
    [ -z "$req_latency" ] && req_latency="N/A"

    # Count primary error
    primary_count=0
    if [ -f "$decode_log" ]; then
        primary_count=$(grep -c "fatal error - failed to decode message from stream" "$decode_log" 2>/dev/null || echo "0")
    fi

    # Count KV timeout warnings (with "0 decode worker")
    kv_timeout_count=0
    if [ -f "$prefill_log" ]; then
        kv_timeout_count=$(grep "Releasing expired KV blocks" "$prefill_log" | grep -c "0 decode worker" 2>/dev/null || echo "0")
    fi

    # Count broken pipe errors
    broken_pipe_count=0
    if [ -f "$decode_log" ]; then
        broken_pipe_count=$(grep -c "Broken pipe (os error 32)" "$decode_log" 2>/dev/null || echo "0")
    fi

    # Count 503 errors
    error_503_count=0
    if [ -f "$decode_log" ]; then
        error_503_count=$(grep -c "503 Service Unavailable" "$decode_log" 2>/dev/null || echo "0")
    fi

    # Format output
    printf "%-10s | %-12s | %-18s | %-15s | %-15s | %-15s | %-15s\n" \
        "$formatted" "$ttft" "$req_latency" "$primary_count" "$kv_timeout_count" "$broken_pipe_count" "$error_503_count" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "===========================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Additional analysis
echo "Detailed Error Analysis:" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

total_primary=0
total_kv_timeout=0
total_broken_pipe=0
total_503=0

for tokens in "${TOKEN_COUNTS[@]}"; do
    formatted=$(format_token_count "$tokens")
    prefill_log="$OUTPUT_DIR/${formatted}_prefill.log"
    decode_log="$OUTPUT_DIR/${formatted}_decode.log"

    [ -f "$decode_log" ] && total_primary=$((total_primary + $(grep -c "fatal error - failed to decode message from stream" "$decode_log" 2>/dev/null || echo "0")))
    [ -f "$prefill_log" ] && total_kv_timeout=$((total_kv_timeout + $(grep "Releasing expired KV blocks" "$prefill_log" | grep -c "0 decode worker" 2>/dev/null || echo "0")))
    [ -f "$decode_log" ] && total_broken_pipe=$((total_broken_pipe + $(grep -c "Broken pipe (os error 32)" "$decode_log" 2>/dev/null || echo "0")))
    [ -f "$decode_log" ] && total_503=$((total_503 + $(grep -c "503 Service Unavailable" "$decode_log" 2>/dev/null || echo "0")))
done

echo "Total Primary Errors:       $total_primary" | tee -a "$SUMMARY_FILE"
echo "Total KV Timeout Warnings:  $total_kv_timeout" | tee -a "$SUMMARY_FILE"
echo "Total Broken Pipe Errors:   $total_broken_pipe" | tee -a "$SUMMARY_FILE"
echo "Total 503 Errors:           $total_503" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

echo "===========================================" | tee -a "$SUMMARY_FILE"
echo "Summary saved to: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "All logs saved to: $OUTPUT_DIR/" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# List all generated files
echo "Generated files:" | tee -a "$SUMMARY_FILE"
ls -lh "$OUTPUT_DIR/" | tee -a "$SUMMARY_FILE"

echo ""
echo "Analysis complete!"
echo ""
echo "To view the summary:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To view individual logs:"
echo "  ls -lh $OUTPUT_DIR/"
