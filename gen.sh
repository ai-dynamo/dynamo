#!/bin/bash
# generate_1k1k_mtp_off_yamls.sh

# Set these to your actual values
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./generated_configs_1k1k_mtp_off}"

# Constants
ISL=1024
OSL=1024
CTX_TP_SIZE=4
CTX_BATCH_SIZE=1
CTX_MAX_NUM_TOKENS=8448
CTX_MAX_SEQ_LEN=$((ISL + 203))
GEN_MAX_SEQ_LEN=$((ISL + OSL + 203))
CTX_FREE_GPU_MEMORY_FRACTION=0.75
CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
MTP_SIZE=0

SCRIPT_DIR="examples/backends/trtllm/performance_sweeps/scripts"

mkdir -p "${OUTPUT_BASE_DIR}"

# Helper function to generate config
generate_config() {
    local name=$1
    local mode=$2  # "tep" or "dep"
    local num_ctx_servers=$3
    local num_gen_servers=$4
    local gen_tp_size=$5
    local gen_batch_size=$6
    local gen_max_num_tokens=$7
    local gen_gpu_memory_fraction=$8
    local eplb_num_slots=$9
    
    local output_dir="${OUTPUT_BASE_DIR}/${name}"
    mkdir -p "${output_dir}"
    
    echo "Generating config: ${name}"
    
    local gen_enable_attention_dp_flag=""
    if [ "$mode" = "dep" ]; then
        gen_enable_attention_dp_flag="--gen_enable_attention_dp"
    fi
    
    python3 examples/backends/trtllm/performance_sweeps/scripts/gen_yaml.py \
        --config "${output_dir}/config.yaml" \
        --model "${MODEL_PATH}" \
        --num_ctx_servers ${num_ctx_servers} \
        --ctx_tp_size ${CTX_TP_SIZE} \
        --ctx_batch_size ${CTX_BATCH_SIZE} \
        --ctx_max_num_tokens ${CTX_MAX_NUM_TOKENS} \
        --ctx_max_seq_len ${CTX_MAX_SEQ_LEN} \
        --ctx_free_gpu_memory_fraction ${CTX_FREE_GPU_MEMORY_FRACTION} \
        --ctx_enable_attention_dp \
        --num_gen_servers ${num_gen_servers} \
        --gen_tp_size ${gen_tp_size} \
        --gen_batch_size ${gen_batch_size} \
        --gen_max_num_tokens ${gen_max_num_tokens} \
        --gen_max_seq_len ${GEN_MAX_SEQ_LEN} \
        --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
        --eplb_num_slots ${eplb_num_slots} \
        --mtp_size ${MTP_SIZE} \
        --cache_transceiver_max_num_tokens ${CACHE_TRANSCEIVER_MAX_NUM_TOKENS} \
        ${gen_enable_attention_dp_flag}
    
    echo "  Generated files in: ${output_dir}"
    echo ""
}

echo "Generating 1k/1k MTP=OFF configurations..."
echo "=========================================="
echo ""

# Config 1: TEP TP=8, Batch=128, MaxTokens=128
generate_config "config1_tep_tp8_b128_t128" "tep" 1 4 8 128 128 0.9 0

# Config 2: DEP TP=32, Batch=32, MaxTokens=32
generate_config "config2_dep_tp32_b32_t32" "dep" 1 1 32 32 32 0.7 0

# Config 3: DEP TP=16, Batch=64, MaxTokens=64
generate_config "config3_dep_tp16_b64_t64" "dep" 1 1 16 64 64 0.75 0

# Config 4: DEP TP=16, Batch=256, MaxTokens=256
generate_config "config4_dep_tp16_b256_t256" "dep" 2 1 16 256 256 0.75 0

# Config 5: DEP TP=8, Batch=512, MaxTokens=512
generate_config "config5_dep_tp8_b512_t512" "dep" 1 1 8 512 512 0.8 0

echo "=========================================="
echo "All configs generated in: ${OUTPUT_BASE_DIR}"
echo ""
echo "To view a config:"
echo "  cat ${OUTPUT_BASE_DIR}/config1_tep_tp8_b128_t128/prefill_config.yaml"
echo "  cat ${OUTPUT_BASE_DIR}/config1_tep_tp8_b128_t128/decode_config.yaml"
