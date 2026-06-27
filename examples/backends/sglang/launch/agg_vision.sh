#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal (image/video + LLM) serving.
# Pass --frontend-decoding to route the image through the Rust frontend's
# NIXL RDMA pipeline (decoded pixels → SGLang worker as PIL Images) instead
# of letting SGLang fetch+decode internally.
# Pass --backend-decoding to keep URLs on the request plane and decode images
# and videos in the unified Rust worker immediately before SGLang inference.
# GPUs: 1

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_sglang_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Default values
MODEL="Qwen/Qwen3-VL-2B-Instruct"
CHAT_TEMPLATE=""
# SGLang KV-cache page size. Default 16; must be 1 for hybrid Mamba models
# (e.g. Qwen3.5-0.8B) whose MambaRadixCache asserts page_size == 1.
PAGE_SIZE=16
ENABLE_OTEL=false
FRONTEND_DECODING=false
BACKEND_DECODING=false

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --chat-template)
            CHAT_TEMPLATE="$2"
            shift 2
            ;;
        --page-size)
            PAGE_SIZE="$2"
            shift 2
            ;;
        --enable-otel)
            ENABLE_OTEL=true
            shift
            ;;
        --frontend-decoding)
            FRONTEND_DECODING=true
            shift
            ;;
        --backend-decoding)
            BACKEND_DECODING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>      Specify model (default: $MODEL)"
            echo "  --chat-template <name>   Specify SGLang chat template (default: $CHAT_TEMPLATE)"
            echo "  --page-size <n>          SGLang KV-cache page size (default: $PAGE_SIZE; must be 1 for Mamba)"
            echo "  --enable-otel            Enable OpenTelemetry tracing"
            echo "  --frontend-decoding      Decode images in the Rust frontend and"
            echo "                           ship pixels via NIXL RDMA (bypasses"
            echo "                           SGLang's internal HTTP fetch + base64 decode)"
            echo "  --backend-decoding       Decode images/videos in the unified Rust"
            echo "                           worker after the frontend request hop"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
            echo "Note: System metrics are enabled by default on port 8081 (worker)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$FRONTEND_DECODING" = true ] && [ "$BACKEND_DECODING" = true ]; then
    echo "--frontend-decoding and --backend-decoding cannot be combined" >&2
    exit 2
fi

# Enable tracing if requested
TRACE_ARGS=()
if [ "$ENABLE_OTEL" = true ]; then
    export DYN_LOGGING_JSONL=true
    export OTEL_EXPORT_ENABLED=1
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-http://localhost:4317}
    TRACE_ARGS+=(--enable-trace --otlp-traces-endpoint localhost:4317)
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Profiler/test-harness override: when _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS is
# set, build_sglang_gpu_mem_args emits --max-total-tokens N. Empty when unset, so
# direct invocations behave identically to before this hook was added.
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

FD_ARGS=()
BANNER_SUFFIX=""
if [ "$FRONTEND_DECODING" = true ]; then
    FD_ARGS+=(--frontend-decoding)
    BANNER_SUFFIX=" (Frontend Decoding)"

fi

WORKER_MODULE="dynamo.sglang"
TOKENIZER_ARGS=(--skip-tokenizer-init)
if [ "$BACKEND_DECODING" = true ]; then
    FD_ARGS+=(--backend-decoding)
    BANNER_SUFFIX=" (Backend Decoding)"
    WORKER_MODULE="dynamo.sglang.unified_main"
    # Backend decoding calls the model's HF processor on the worker.
    TOKENIZER_ARGS=()
fi

if [ "$FRONTEND_DECODING" = true ] || [ "$BACKEND_DECODING" = true ]; then
    # The SGLang image inherits NIXL from the upstream lmsysorg/sglang runtime
    # stack but does NOT add its native libs to LD_LIBRARY_PATH (cf.
    # container/templates/dev.Dockerfile: "SGLang dev/local-dev inherit the
    # upstream SGLang/NIXL runtime stack"). Without this, dynamo.frontend's Rust
    # runtime hits "NIXL is not supported in stub mode" the moment it tries to
    # build a media-fetching pipeline. Point the loader at the wheel-shipped .so
    # files explicitly. NIXL wheels store their native libraries in a hidden
    # sibling directory whose name includes the CUDA major version; accept the
    # CUDA 12 and CUDA 13 layouts used by supported SGLang images.
    NIXL_WHEEL_LIBS="$(python3 -c 'import glob, sys; paths = [path for base in sys.path for path in glob.glob(f"{base}/.nixl_cu*.mesonpy.libs")]; print(paths[0] if paths else "")' 2>/dev/null || true)"
    if [ -d "$NIXL_WHEEL_LIBS" ]; then
        export LD_LIBRARY_PATH="${NIXL_WHEEL_LIBS}:${NIXL_WHEEL_LIBS}/plugins:${LD_LIBRARY_PATH}"
        export NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-$NIXL_WHEEL_LIBS/plugins}"
    fi
fi

print_launch_banner --multimodal "Launching Aggregated Vision Serving${BANNER_SUFFIX}" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
# Frontend has no --frontend-decoding flag — decoding is opt-in via the
# backend's model card (MediaDecoder configured at register_model time when
# the backend is launched with --frontend-decoding).
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend &

# Build chat template args (only if explicitly set)
TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE" ]; then
    TEMPLATE_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

# run worker with a vision model (SGLang auto-detects chat template from HF tokenizer).
# Without a decoding flag, SGLang handles image/video loading internally.
# Frontend decoding remains the legacy image-only path. Backend decoding uses
# the unified worker for Rust image/video decode and worker-side HF processing.
OTEL_SERVICE_NAME=dynamo-worker DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python3 -m "$WORKER_MODULE" \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  "${TEMPLATE_ARGS[@]}" \
  --page-size "$PAGE_SIZE" \
  --tp 1 \
  --trust-remote-code \
  "${TOKENIZER_ARGS[@]}" \
  --enable-metrics \
  "${FD_ARGS[@]}" \
  $GPU_MEM_ARGS \
  "${TRACE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
