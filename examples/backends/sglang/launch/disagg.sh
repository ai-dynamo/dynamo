#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Parse command line arguments
ENABLE_OTEL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --enable-otel        Enable OpenTelemetry tracing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Note: System metrics are enabled by default on ports 8081 (prefill), 8082 (decode)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Enable tracing if requested
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &
DYNAMO_PID=$!

#AssertionError: Prefill round robin balance is required when dp size > 1. Please make sure that the prefill instance is launched with `--load-balance-method round_robin` and `--prefill-round-robin-balance` is set for decode server.

# run prefill worker
# Use DYN_SYSTEM_PORT1/2 instead of *_PREFILL/*_DECODE env names so test
# harnesses can set one simple pair for disaggregated deployments.
OTEL_SERVICE_NAME=dynamo-worker-prefill DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
docker run --rm --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1 -v $(readlink -f $HOME/proj/models/dsv2-lite-fp8/):/workspace/model -v /tmp/dsr1-cache:/root/.cache \
    -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
    -e CUDA_LAUNCH_BLOCKING=1 \
    -e TORCH_USE_CUDA_DSA=1 \
    -e SGL_FORCE_SHUTDOWN=1 \
    -e SGLANG_FORCE_SHUTDOWN=1 \
    nvcr.io/nvidian/dynamo-dev/warnold-utils:sglang-dd-v18-amd64 \
    python3 -m dynamo.sglang \
        --model-path /workspace/model \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --chunked-prefill-size 65536 \
        --ep 8 \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --enable-dp-lm-head \
        --attention-backend trtllm_mla \
        --moe-a2a-backend deepep \
        --disable-radix-cache \
        --disaggregation-bootstrap-port 30001 \
        --disaggregation-transfer-backend nixl \
        --enable-symm-mem \
        --kv-cache-dtype fp8_e4m3 \
        --moe-dense-tp-size 1 \
        --scheduler-recv-interval 1 \
        --stream-interval 10 \
        --trust-remote-code \
        --watchdog-timeout 1000000 \
        --enable-nan-detection \
        --max-running-requests 64 \
        --cuda-graph-max-bs 64 \
        --mem-fraction-static 0.75 \
        --disable-cuda-graph \
        --host 0.0.0.0 \
        --port 10000 \
        --enable-metrics \
        --moe-runner-backend deep_gemm \
        "${TRACE_ARGS[@]}" &
PREFILL_PID=$!
        #--deepep-mode normal \
        #--ep-dispatch-algorithm dynamic \
        #--ep-num-redundant-experts 32 \
        #--disable-shared-experts-fusion \
        #--eplb-algorithm deepseek \

# Wait for background processes
wait $DYNAMO_PID $PREFILL_PID
