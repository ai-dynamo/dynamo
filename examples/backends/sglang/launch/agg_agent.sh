#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with session control: sticky routing,
# KV event tracking, agent tracing, and reasoning/tool-call parsing.
# GPUs: 2 (default model uses one TP=1 worker per GPU)

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL="Qwen/Qwen3-14B-FP8"
TP=1
NUM_WORKERS=2
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="qwen25"
FRONTEND_NO_ADMISSION_CONTROL=1
AGENT_TRACE_ENABLED=1
AGENT_TRACE_SINKS="${DYN_AGENT_TRACE_SINKS:-jsonl_gz}"
AGENT_TRACE_OUTPUT_PATH="${DYN_AGENT_TRACE_OUTPUT_PATH:-/tmp/dynamo-agent-trace}"
AGENT_TRACE_TOOL_ENDPOINT="${DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT:-tcp://127.0.0.1:20390}"
AGENT_TRACE_TOOL_TOPIC="${DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC:-}"
AGENT_TRACE_JSONL_BUFFER_BYTES="${DYN_AGENT_TRACE_JSONL_BUFFER_BYTES:-}"
AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS="${DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS:-100}"
AGENT_TRACE_JSONL_GZ_ROLL_BYTES="${DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES:-}"
AGENT_TRACE_JSONL_GZ_ROLL_LINES="${DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES:-}"
AGENT_TRACE_REPLAY_HASHES="${DYN_AGENT_TRACE_REPLAY_HASHES:-}"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --dyn-reasoning-parser)
            REASONING_PARSER="$2"
            shift 2
            ;;
        --dyn-tool-call-parser)
            TOOL_CALL_PARSER="$2"
            shift 2
            ;;
        --no-admission-control|--no-frontend-admission-control)
            FRONTEND_NO_ADMISSION_CONTROL=1
            shift
            ;;
        --frontend-admission-control)
            FRONTEND_NO_ADMISSION_CONTROL=0
            shift
            ;;
        --agent-trace-output-path)
            AGENT_TRACE_OUTPUT_PATH="$2"
            if [[ -z "$AGENT_TRACE_SINKS" ]]; then
                AGENT_TRACE_SINKS="jsonl_gz"
            fi
            shift 2
            ;;
        --agent-trace-sinks)
            AGENT_TRACE_SINKS="$2"
            if [[ -z "$AGENT_TRACE_OUTPUT_PATH" ]]; then
                AGENT_TRACE_OUTPUT_PATH="/tmp/dynamo-agent-trace"
            fi
            shift 2
            ;;
        --agent-trace-tool-endpoint)
            AGENT_TRACE_TOOL_ENDPOINT="$2"
            shift 2
            ;;
        --agent-trace-tool-topic)
            AGENT_TRACE_TOOL_TOPIC="$2"
            shift 2
            ;;
        --agent-trace-jsonl-buffer-bytes)
            AGENT_TRACE_JSONL_BUFFER_BYTES="$2"
            shift 2
            ;;
        --agent-trace-jsonl-flush-ms)
            AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS="$2"
            shift 2
            ;;
        --agent-trace-jsonl-gz-roll-bytes)
            AGENT_TRACE_JSONL_GZ_ROLL_BYTES="$2"
            shift 2
            ;;
        --agent-trace-jsonl-gz-roll-lines)
            AGENT_TRACE_JSONL_GZ_ROLL_LINES="$2"
            shift 2
            ;;
        --disable-agent-trace-replay-hashes)
            AGENT_TRACE_REPLAY_HASHES="0"
            shift
            ;;
        --disable-agent-trace)
            AGENT_TRACE_ENABLED=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model-path <name>  Specify model (default: $MODEL)"
            echo "  --tp <n>             Tensor parallelism (default: $TP)"
            echo "  --num-workers <n>    Number of aggregated workers (default: $NUM_WORKERS)"
            echo "  --dyn-reasoning-parser <name>"
            echo "                        Dynamo reasoning parser (default: $REASONING_PARSER)"
            echo "  --dyn-tool-call-parser <name>"
            echo "                        Dynamo tool-call parser (default: $TOOL_CALL_PARSER)"
            echo "  --no-admission-control"
            echo "                        Disable frontend busy-worker admission checks"
            echo "                        (default for this agentic launch)."
            echo "  --frontend-admission-control"
            echo "                        Re-enable frontend busy-worker admission checks."
            echo "  --agent-trace-output-path <path>"
            echo "                        Agent trace output path or jsonl_gz shard prefix"
            echo "                        (default: /tmp/dynamo-agent-trace)."
            echo "  --agent-trace-sinks <csv>"
            echo "                        Agent trace sinks: jsonl, jsonl_gz, stderr, or CSV"
            echo "                        (default: jsonl_gz)."
            echo "  --agent-trace-tool-endpoint <endpoint>"
            echo "                        Bind Dynamo PULL socket for harness tool events,"
            echo "                        e.g. tcp://127.0.0.1:20390"
            echo "                        (default: tcp://127.0.0.1:20390)."
            echo "  --agent-trace-tool-topic <topic>"
            echo "                        Required ZMQ topic filter for tool events."
            echo "  --agent-trace-jsonl-buffer-bytes <bytes>"
            echo "                        Trace JSONL/gzip buffer size."
            echo "  --agent-trace-jsonl-flush-ms <ms>"
            echo "                        Trace JSONL/gzip flush interval."
            echo "  --agent-trace-jsonl-gz-roll-bytes <bytes>"
            echo "                        Rotate jsonl_gz shards after this many uncompressed bytes."
            echo "  --agent-trace-jsonl-gz-roll-lines <lines>"
            echo "                        Rotate jsonl_gz shards after this many records."
            echo "  --disable-agent-trace-replay-hashes"
            echo "                        Disable replay hash capture in request_end records."
            echo "  --disable-agent-trace"
            echo "                        Do not export Dynamo agent trace env vars."
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Environment equivalents:"
            echo "  DYN_AGENT_TRACE_SINKS, DYN_AGENT_TRACE_OUTPUT_PATH,"
            echo "  DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,"
            echo "  DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC,"
            echo "  DYN_AGENT_TRACE_JSONL_BUFFER_BYTES,"
            echo "  DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS,"
            echo "  DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES,"
            echo "  DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES,"
            echo "  DYN_AGENT_TRACE_REPLAY_HASHES"
            echo ""
            echo "Additional SGLang/Dynamo flags can be passed and will be forwarded"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_SINKS" && -z "$AGENT_TRACE_OUTPUT_PATH" && "$AGENT_TRACE_SINKS" != "stderr" ]]; then
    AGENT_TRACE_OUTPUT_PATH="/tmp/dynamo-agent-trace"
fi

if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_SINKS" ]]; then
    export DYN_AGENT_TRACE_SINKS="$AGENT_TRACE_SINKS"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_OUTPUT_PATH" ]]; then
    export DYN_AGENT_TRACE_OUTPUT_PATH="$AGENT_TRACE_OUTPUT_PATH"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_TOOL_ENDPOINT" ]]; then
    export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT="$AGENT_TRACE_TOOL_ENDPOINT"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_TOOL_TOPIC" ]]; then
    export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC="$AGENT_TRACE_TOOL_TOPIC"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_JSONL_BUFFER_BYTES" ]]; then
    export DYN_AGENT_TRACE_JSONL_BUFFER_BYTES="$AGENT_TRACE_JSONL_BUFFER_BYTES"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS" ]]; then
    export DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS="$AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_JSONL_GZ_ROLL_BYTES" ]]; then
    export DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES="$AGENT_TRACE_JSONL_GZ_ROLL_BYTES"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_JSONL_GZ_ROLL_LINES" ]]; then
    export DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES="$AGENT_TRACE_JSONL_GZ_ROLL_LINES"
fi
if [[ "$AGENT_TRACE_ENABLED" == "1" && -n "$AGENT_TRACE_REPLAY_HASHES" ]]; then
    export DYN_AGENT_TRACE_REPLAY_HASHES="$AGENT_TRACE_REPLAY_HASHES"
fi

GPU_MEM_FRACTION=$(build_sglang_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + Session Control" "$MODEL" "$HTTP_PORT"
trap 'echo Cleaning up...; kill 0' EXIT

if [[ -n "${DYN_AGENT_TRACE_SINKS:-}" ]]; then
    echo "Agent trace sinks: $DYN_AGENT_TRACE_SINKS"
    echo "Agent trace output: ${DYN_AGENT_TRACE_OUTPUT_PATH:-<none>}"
fi
if [[ -n "${DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT:-}" ]]; then
    echo "Agent tool-event endpoint: $DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT"
fi
echo "Workers: $NUM_WORKERS aggregated TP=$TP"
echo "Parsers: reasoning=${REASONING_PARSER:-<none>} tool=${TOOL_CALL_PARSER:-<none>}"
if [[ "$FRONTEND_NO_ADMISSION_CONTROL" == "1" ]]; then
    echo "Frontend admission control: disabled"
else
    echo "Frontend admission control: enabled"
fi

# Frontend with KV routing and state reset
# Session control activates automatically when requests carry nvext.session_control
FRONTEND_ARGS=()
if [[ "$FRONTEND_NO_ADMISSION_CONTROL" == "1" ]]; then
    FRONTEND_ARGS+=(--no-admission-control)
fi

python3 -m dynamo.frontend \
  --router-mode kv \
  --router-reset-states \
  "${FRONTEND_ARGS[@]}" &

PARSER_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
    PARSER_ARGS+=(--dyn-reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
    PARSER_ARGS+=(--dyn-tool-call-parser "$TOOL_CALL_PARSER")
fi

launch_worker() {
    local worker_idx="$1"
    local gpu_idx="$2"
    local system_port="$3"
    local kv_port="$4"

    echo "Starting worker ${worker_idx} on GPU ${gpu_idx}, system port ${system_port}, KV events ${kv_port}"
    OTEL_SERVICE_NAME="dynamo-worker-${worker_idx}" \
    DYN_SYSTEM_PORT="$system_port" \
    CUDA_VISIBLE_DEVICES="$gpu_idx" \
    python3 -m dynamo.sglang \
      --model-path "$MODEL" \
      --served-model-name "$MODEL" \
      --page-size 16 \
      --tp "$TP" \
      --trust-remote-code \
      --enable-streaming-session \
      --skip-tokenizer-init \
      "${PARSER_ARGS[@]}" \
      --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${kv_port}\"}" \
      --enable-metrics \
      $GPU_MEM_FRACTION \
      "${EXTRA_ARGS[@]}" &
}

for worker_idx in $(seq 1 "$NUM_WORKERS"); do
    gpu_idx=$((worker_idx - 1))
    system_env="DYN_SYSTEM_PORT_WORKER${worker_idx}"
    system_port="${!system_env:-$((8080 + worker_idx))}"
    if [[ "$NUM_WORKERS" -eq 1 ]]; then
        system_port="${DYN_SYSTEM_PORT:-$system_port}"
    fi
    kv_port=$((5556 + worker_idx))
    launch_worker "$worker_idx" "$gpu_idx" "$system_port" "$kv_port"
done

wait_any_exit
