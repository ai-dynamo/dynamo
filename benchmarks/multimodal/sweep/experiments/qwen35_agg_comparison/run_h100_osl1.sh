#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 4-way benchmark: vllm-baseline, vllm-ec, dynamo-fd, dynamo-fd-ec
# OSL=1 (pure TTFT measurement), request-rate sweep.
#
# Usage (inside container with scratch mounts):
#   bash benchmarks/multimodal/sweep/experiments/qwen35_agg_comparison/run_h100_osl1.sh [CONFIG...]
#
# If no configs specified, runs all 4. Otherwise runs only named configs:
#   bash run_h100_osl1.sh vllm-baseline vllm-ec

set -euo pipefail

MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
BASE="/workspace/logs/h100-osl1-clean"
RATES=(0.25 0.5 1 2 4)
REQUEST_COUNT=60
WARMUP=3
PORT=8000
TIMEOUT=450

export HF_HOME="${HF_HOME:-/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Dataset ──
# Always regenerate — images go to /tmp which is ephemeral per container
INPUT="/workspace/benchmarks/multimodal/jsonl/60req_4img_192pool_400word_base64.jsonl"
echo "Generating dataset..."
cd /workspace/benchmarks/multimodal/jsonl
python main.py -n 60 --images-per-request 4 --images-pool 192 \
    --user-text-tokens 400 --image-mode base64
cd /workspace

# ── Helpers ──
wait_for_model() {
    local deadline=$((SECONDS + TIMEOUT))
    echo "  Waiting for model on port $PORT (timeout ${TIMEOUT}s)..."
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$MODEL"; then
            echo "  Server ready (${SECONDS}s elapsed)."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server not ready within ${TIMEOUT}s"
    return 1
}

kill_all() {
    fuser -k ${PORT}/tcp 2>/dev/null || true
    pkill -f "dynamo.vllm" 2>/dev/null || true
    pkill -f "dynamo.frontend" 2>/dev/null || true
    sleep 3
}

run_aiperf() {
    local rate=$1
    local artifact_dir=$2
    mkdir -p "$artifact_dir"
    aiperf profile \
        -m "$MODEL" -u "http://localhost:${PORT}" \
        --request-rate "$rate" \
        --request-count "$REQUEST_COUNT" --warmup-request-count "$WARMUP" \
        --input-file "$INPUT" \
        --custom-dataset-type single_turn \
        --extra-inputs max_tokens:1 --extra-inputs min_tokens:1 \
        --extra-inputs ignore_eos:true --extra-inputs stream:true \
        --streaming --artifact-dir "$artifact_dir" \
        --ui none --no-server-metrics
}

# ── Config runners ──
run_vllm_baseline() {
    local cfg_dir="$BASE/vllm-baseline"
    echo ""; echo "######## vllm-baseline ########"
    for rate in "${RATES[@]}"; do
        echo "=== vllm-baseline rate $rate ==="
        kill_all
        vllm serve "$MODEL" \
            --no-enable-prefix-caching \
            --max-model-len 16384 \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.9 \
            > "$cfg_dir/server_r${rate}.log" 2>&1 &
        wait_for_model || { tail -20 "$cfg_dir/server_r${rate}.log"; return 1; }
        run_aiperf "$rate" "$cfg_dir/r${rate}"
        echo "  rate $rate done"
    done
    kill_all
}

run_vllm_ec() {
    local cfg_dir="$BASE/vllm-ec"
    local ec_config='{"ec_role":"ec_both","ec_connector":"DynamoMultimodalEmbeddingCacheConnector","ec_connector_module_path":"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector","ec_connector_extra_config":{"multimodal_embedding_cache_capacity_gb":10}}'
    echo ""; echo "######## vllm-ec ########"
    for rate in "${RATES[@]}"; do
        echo "=== vllm-ec rate $rate ==="
        kill_all
        vllm serve "$MODEL" \
            --no-enable-prefix-caching \
            --max-model-len 16384 \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.9 \
            --ec-transfer-config "$ec_config" \
            > "$cfg_dir/server_r${rate}.log" 2>&1 &
        wait_for_model || { tail -20 "$cfg_dir/server_r${rate}.log"; return 1; }
        run_aiperf "$rate" "$cfg_dir/r${rate}"
        echo "  rate $rate done"
    done
    kill_all
}

run_dynamo_fd() {
    local cfg_dir="$BASE/dynamo-fd"
    export DYN_REQUEST_PLANE=tcp
    export DYN_TCP_MAX_MESSAGE_SIZE=104857600
    echo ""; echo "######## dynamo-fd ########"
    for rate in "${RATES[@]}"; do
        echo "=== dynamo-fd rate $rate ==="
        kill_all
        python -m dynamo.frontend --http-port $PORT \
            > "$cfg_dir/frontend_r${rate}.log" 2>&1 &
        python -m dynamo.vllm \
            --enable-multimodal \
            --model "$MODEL" \
            --no-enable-prefix-caching \
            --frontend-decoding \
            --max-model-len 16384 \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.9 \
            > "$cfg_dir/backend_r${rate}.log" 2>&1 &
        wait_for_model || { tail -20 "$cfg_dir/backend_r${rate}.log"; return 1; }
        run_aiperf "$rate" "$cfg_dir/r${rate}"
        echo "  rate $rate done"
    done
    kill_all
}

run_dynamo_fd_ec() {
    local cfg_dir="$BASE/dynamo-fd-ec"
    export DYN_REQUEST_PLANE=tcp
    export DYN_TCP_MAX_MESSAGE_SIZE=104857600
    echo ""; echo "######## dynamo-fd-ec ########"
    for rate in "${RATES[@]}"; do
        echo "=== dynamo-fd-ec rate $rate ==="
        kill_all
        python -m dynamo.frontend --http-port $PORT \
            > "$cfg_dir/frontend_r${rate}.log" 2>&1 &
        python -m dynamo.vllm \
            --enable-multimodal \
            --model "$MODEL" \
            --no-enable-prefix-caching \
            --frontend-decoding \
            --multimodal-embedding-cache-capacity-gb 10 \
            --max-model-len 16384 \
            --max-num-seqs 8 \
            --gpu-memory-utilization 0.9 \
            > "$cfg_dir/backend_r${rate}.log" 2>&1 &
        wait_for_model || { tail -20 "$cfg_dir/backend_r${rate}.log"; return 1; }
        run_aiperf "$rate" "$cfg_dir/r${rate}"
        echo "  rate $rate done"
    done
    kill_all
}

# ── Main ──
ALL_CONFIGS=(vllm-baseline vllm-ec dynamo-fd dynamo-fd-ec)
CONFIGS=("${@:-${ALL_CONFIGS[@]}}")

# Create output dirs
for cfg in "${CONFIGS[@]}"; do
    for rate in "${RATES[@]}"; do
        mkdir -p "$BASE/$cfg/r$rate"
    done
done

# Start infra (needed for dynamo configs)
nats-server -js > /tmp/nats.log 2>&1 &
etcd --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379 \
    --data-dir /tmp/etcd > /tmp/etcd.log 2>&1 &
sleep 3

echo "========================================"
echo "  H100 OSL=1 Request-Rate Sweep"
echo "========================================"
echo "  Model:    $MODEL"
echo "  Rates:    ${RATES[*]}"
echo "  Requests: $REQUEST_COUNT per rate"
echo "  Configs:  ${CONFIGS[*]}"
echo "  Output:   $BASE"
echo "========================================"

for cfg in "${CONFIGS[@]}"; do
    case "$cfg" in
        vllm-baseline) run_vllm_baseline ;;
        vllm-ec)       run_vllm_ec ;;
        dynamo-fd)     run_dynamo_fd ;;
        dynamo-fd-ec)  run_dynamo_fd_ec ;;
        *) echo "Unknown config: $cfg"; exit 1 ;;
    esac
done

# ── Verify ──
echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
MISSING=0
for cfg in "${CONFIGS[@]}"; do
    echo "  $cfg:"
    for rate in "${RATES[@]}"; do
        if [ -f "$BASE/$cfg/r$rate/profile_export_aiperf.json" ]; then
            echo "    r$rate: OK"
        else
            echo "    r$rate: MISSING"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "  All results collected!"
else
    echo ""
    echo "  WARNING: $MISSING results missing"
fi
