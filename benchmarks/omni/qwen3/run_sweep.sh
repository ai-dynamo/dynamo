#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3-Omni agg/disagg/vllm-serve benchmark sweep (DYN-2581).
#
# Drives AIPerf against an already-running deployment. Loops:
#   workload    in {chat, audio}
#   concurrency in {1, 4, 8, 16, 32}
#   prompt-len  in {short, long}
# and writes one AIPerf artifact dir per cell under RESULTS_ROOT.
#
# The TOPOLOGY label is informational — the script just hits whatever URL you
# point it at. Run it once per topology with a different --topology label and
# matching --url, or run all three with the loop in the README.
#
# Usage:
#   bash run_sweep.sh --topology agg --url http://localhost:8000
#   bash run_sweep.sh --topology disagg --url http://localhost:8001
#   bash run_sweep.sh --topology vllm_serve --url http://localhost:8002
#
# Options:
#   --topology <name>      Folder label under results/ (default: agg)
#   --url <url>            Server URL (default: http://localhost:8000)
#   --model <name>         Model name passed to AIPerf (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   --quick                Single small cell per workload — for smoke
#   --results-root <dir>   Output root (default: <script-dir>/results)
#   --workloads <list>     Comma-separated subset of {chat,audio} (default: chat,audio)
#   -h | --help            Show this help

set -euo pipefail

TOPOLOGY="agg"
URL="http://localhost:8000"
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
QUICK=0
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
RESULTS_ROOT="$SCRIPT_DIR/results"
WORKLOADS="chat,audio"

while [[ $# -gt 0 ]]; do
    case $1 in
        --topology)     TOPOLOGY=$2;     shift 2 ;;
        --url)          URL=$2;          shift 2 ;;
        --model)        MODEL=$2;        shift 2 ;;
        --quick)        QUICK=1;         shift ;;
        --results-root) RESULTS_ROOT=$2; shift 2 ;;
        --workloads)    WORKLOADS=$2;    shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^set /p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if (( QUICK )); then
    CONCURRENCIES=(4)
    PROMPT_LENS=("short")
    BENCHMARK_DURATION=15
    WARMUP=2
else
    # Sweep through the throughput knee. At c=256 a 30B-A3B MoE on one
    # H200 should be queue-bound; the curve flattens before that.
    CONCURRENCIES=(1 4 8 16 32 64 128 256)
    PROMPT_LENS=("short" "long")
    BENCHMARK_DURATION=60
    WARMUP=8
fi

PROMPT_TOKENS_SHORT=64
PROMPT_TOKENS_LONG=512
PROMPT_STDDEV=8

echo "================================================="
echo " Qwen3-Omni benchmark sweep"
echo " Topology:   $TOPOLOGY"
echo " URL:        $URL"
echo " Model:      $MODEL"
echo " Workloads:  $WORKLOADS"
echo " Quick:      $QUICK"
echo " Output:     $RESULTS_ROOT"
echo "================================================="

mkdir -p "$RESULTS_ROOT/$TOPOLOGY"

run_chat() {
    local concurrency=$1 prompt_label=$2 prompt_tokens=$3
    local artifact="$RESULTS_ROOT/$TOPOLOGY/chat_c${concurrency}_isl-${prompt_label}"
    mkdir -p "$artifact"
    echo "[chat] c=$concurrency isl=$prompt_label tokens=$prompt_tokens -> $artifact"
    aiperf profile \
        --model "$MODEL" \
        --tokenizer gpt2 \
        --url "$URL" \
        --endpoint-type chat \
        --streaming \
        --synthetic-input-tokens-mean "$prompt_tokens" \
        --synthetic-input-tokens-stddev "$PROMPT_STDDEV" \
        --output-tokens-mean 128 \
        --concurrency "$concurrency" \
        --benchmark-duration "$BENCHMARK_DURATION" \
        --warmup-request-count "$WARMUP" \
        --ui none \
        --no-server-metrics \
        --artifact-dir "$artifact"
}

run_audio() {
    local concurrency=$1 prompt_label=$2 prompt_tokens=$3
    local artifact="$RESULTS_ROOT/$TOPOLOGY/audio_c${concurrency}_isl-${prompt_label}"
    mkdir -p "$artifact"
    echo "[audio] c=$concurrency isl=$prompt_label tokens=$prompt_tokens -> $artifact"
    # /v1/audio/speech is OpenAI's audio-generation endpoint; aiperf doesn't
    # have a dedicated --endpoint-type for it yet, so we use 'chat' with
    # extra-inputs that re-route the payload. The frontend handles routing.
    aiperf profile \
        --model "$MODEL" \
        --tokenizer gpt2 \
        --url "$URL" \
        --endpoint-type chat \
        --streaming \
        --synthetic-input-tokens-mean "$prompt_tokens" \
        --synthetic-input-tokens-stddev "$PROMPT_STDDEV" \
        --output-tokens-mean 256 \
        --extra-inputs "modalities:audio" \
        --extra-inputs "voice:ethan" \
        --concurrency "$concurrency" \
        --benchmark-duration "$BENCHMARK_DURATION" \
        --warmup-request-count "$WARMUP" \
        --ui none \
        --no-server-metrics \
        --artifact-dir "$artifact"
}

IFS=',' read -r -a WORKLOAD_LIST <<< "$WORKLOADS"

for c in "${CONCURRENCIES[@]}"; do
    for plen in "${PROMPT_LENS[@]}"; do
        if [[ "$plen" == "short" ]]; then
            tokens=$PROMPT_TOKENS_SHORT
        else
            tokens=$PROMPT_TOKENS_LONG
        fi
        for w in "${WORKLOAD_LIST[@]}"; do
            case "$w" in
                chat)  run_chat  "$c" "$plen" "$tokens" ;;
                audio) run_audio "$c" "$plen" "$tokens" ;;
                *) echo "Unknown workload: $w" >&2; exit 1 ;;
            esac
        done
    done
done

echo "Done. Results under $RESULTS_ROOT/$TOPOLOGY"
