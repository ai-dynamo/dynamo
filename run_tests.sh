#!/bin/bash
# Test dynamo.vllm with Qwen/Qwen3.5-35B-A3B-FP8 across all topologies
# MoE model: 35B total, 3B active, FP8 quantized — should fit on single GPU
# Container devices: 0=49GB Ada, 1=98GB Blackwell

set -uo pipefail

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
LOG_DIR="/workspace/logs/test-qwen35-35b"
mkdir -p "$LOG_DIR"

export PYTHONPATH=/workspace/components/src:${PYTHONPATH:-}
export HF_HOME=/home/dynamo/.cache/huggingface

log() { echo ""; echo "========== $1 =========="; echo ""; }

kill_dynamo() {
    pkill -9 -f "dynamo\.(vllm|frontend)" 2>/dev/null || true
    pkill -9 -f "mm_router" 2>/dev/null || true
    sleep 3
}

wait_for_model() {
    local timeout=${1:-600}
    local deadline=$((SECONDS + timeout))
    echo "Waiting for model at /v1/models (timeout ${timeout}s)..."
    while (( SECONDS < deadline )); do
        if curl -fs http://localhost:8000/v1/models 2>/dev/null | grep -qi "qwen"; then
            echo "Model registered!"
            return 0
        fi
        sleep 5
    done
    echo "TIMEOUT waiting for model"
    return 1
}

get_model_name() {
    curl -fs http://localhost:8000/v1/models 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null
}

send_text_request() {
    local model_name="$1"
    echo "--- Text request (model=$model_name) ---"
    curl -s --max-time 180 http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"$model_name\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}],
            \"max_tokens\": 32
        }" | python3 -m json.tool 2>/dev/null || echo "REQUEST FAILED"
}

send_multimodal_request() {
    local model_name="$1"
    echo "--- Multimodal request (model=$model_name) ---"
    curl -s --max-time 180 http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"$model_name\",
            \"messages\": [{\"role\": \"user\", \"content\": [
                {\"type\": \"text\", \"text\": \"Describe this image in one sentence\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://images.cocodataset.org/test2017/000000000001.jpg\"}}
            ]}],
            \"max_tokens\": 64
        }" | python3 -m json.tool 2>/dev/null || echo "REQUEST FAILED"
}

run_test() {
    local test_name="$1"
    local log_file="$LOG_DIR/${test_name}.log"
    shift

    log "TEST: $test_name"
    kill_dynamo

    echo "Launching: $*"
    setsid bash -c "$*" > "$log_file" 2>&1 &
    local launch_pid=$!

    if wait_for_model 600; then
        actual_model=$(get_model_name)
        echo "Detected model name: $actual_model"
        echo ""
        send_text_request "$actual_model"
        echo ""
        send_multimodal_request "$actual_model"
        echo ""
        echo "RESULT: $test_name - PASS"
    else
        echo "RESULT: $test_name - FAIL (model did not register)"
        echo "--- Last 40 lines of log ---"
        tail -40 "$log_file" 2>/dev/null
    fi

    kill -- -$launch_pid 2>/dev/null || true
    sleep 2
    echo "Cleanup done for $test_name"
}

cd /workspace

# ---- Test 1: AGG ----
run_test "agg" \
    "CUDA_VISIBLE_DEVICES=1 \
     examples/backends/vllm/launch/agg_multimodal.sh \
     --model $MODEL"

# ---- Test 2: MM Routing ----
run_test "mm_routing" \
    "export MODEL='$MODEL' CUDA_VISIBLE_DEVICES=1 && \
     examples/backends/vllm/mm_router_worker/launch.sh"

# ---- Test 3: P/D Disagg (both GPUs) ----
# FP8 MoE should fit on 49GB GPU
run_test "pd_disagg" \
    "export PYTHONPATH=/workspace/components/src:\${PYTHONPATH:-} HF_HOME=/home/dynamo/.cache/huggingface && \
     python -m dynamo.frontend & \
     DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
         --model '$MODEL' --enforce-eager --disaggregation-mode decode \
         --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' & \
     DYN_SYSTEM_PORT=8082 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
         --model '$MODEL' --enforce-eager --disaggregation-mode prefill \
         --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
         --kv-events-config '{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:20081\",\"enable_kv_cache_events\":true}' & \
     wait"

# ---- Test 4: E_PD ----
run_test "e_pd" \
    "DYN_ENCODE_WORKER_GPU=0 DYN_PD_WORKER_GPU=1 \
     examples/backends/vllm/launch/disagg_multimodal_e_pd.sh \
     --model $MODEL"

# ---- Test 5: E_P_D ----
run_test "e_p_d" \
    "DYN_ENCODE_WORKER_GPU=0 DYN_PREFILL_WORKER_GPU=1 DYN_DECODE_WORKER_GPU=1 \
     DYN_ENCODE_GPU_MEM=0.3 DYN_PREFILL_GPU_MEM=0.4 DYN_DECODE_GPU_MEM=0.4 \
     examples/backends/vllm/launch/disagg_multimodal_epd.sh \
     --model $MODEL"

log "ALL TESTS COMPLETE"
