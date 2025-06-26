#!/bin/bash

# Helper script that allows you to run sglang servers for prefill, decode, and mini_lb. You can use this from inside of the dynamo-deepep container
# Commands are from https://github.com/sgl-project/sglang/issues/6017
# Last updated: 6/25/2025

# Update these values to match your cluster configuration
HEAD_PREFILL_NODE_IP="10.52.48.42"
HEAD_DECODE_NODE_IP="10.52.48.107"
PREFILL_DIST_INIT_ADDR="${HEAD_PREFILL_NODE_IP}:5757"
DECODE_DIST_INIT_ADDR="${HEAD_DECODE_NODE_IP}:5757"
NUM_PREFILL_NODES=4
NUM_DECODE_NODES=9
PREFILL_TP_SIZE=32
PREFILL_DP_SIZE=32
DECODE_TP_SIZE=72
DECODE_DP_SIZE=72

# ./run_sgl_servers.sh --server-type {prefill, decode, mini_lb} [--node-rank N]

# Example usage 
# After updating the values above, on my head prefill node I would run:
#     ./run_sgl_servers.sh --server-type prefill --node-rank 0 &
#     ./run_sgl_servers.sh --server-type mini_lb
#
# On every other prefill node I would run:
#     ./run_sgl_servers.sh --server-type prefill --node-rank 1  # (or 2, 3, etc.)
#
# On other decode nodes I would run:
#     ./run_sgl_servers.sh --server-type decode --node-rank 1   # (or 2, 3, etc.)

SERVER_TYPE=""
NODE_RANK=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --server-type)
      SERVER_TYPE="$2"
      shift 2
      ;;
    --node-rank)
      NODE_RANK="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate server type
if [[ -z "$SERVER_TYPE" ]]; then
  echo "Error: --server-type is required"
  echo "Usage: $0 --server-type {prefill, decode, mini_lb} [--node-rank N]"
  exit 1
fi

# Validate node rank for prefill and decode
if [[ "$SERVER_TYPE" == "prefill" || "$SERVER_TYPE" == "decode" ]] && [[ -z "$NODE_RANK" ]]; then
  echo "Error: --node-rank is required for prefill and decode servers"
  echo "Usage: $0 --server-type {prefill, decode} --node-rank N"
  exit 1
fi

case $SERVER_TYPE in
  mini_lb)
    echo "Starting mini_lb server..."
    python3 -m sglang.srt.disaggregation.mini_lb \
      --prefill "http://${HEAD_PREFILL_NODE_IP}:30000" \
      --decode  "http://${HEAD_DECODE_NODE_IP}:30000"
    ;;
  
  prefill)
    echo "Starting prefill server..."
    MC_TE_METRIC=true \
    SGLANG_TBO_DEBUG=1 \
    python3 -m sglang.launch_server \
      --model-path /model/ \
      --disaggregation-transfer-backend nixl \
      --disaggregation-mode prefill \
      --dist-init-addr ${PREFILL_DIST_INIT_ADDR} \
      --nnodes ${NUM_PREFILL_NODES} \
      --node-rank ${NODE_RANK} \
      --tp-size ${PREFILL_TP_SIZE} \
      --dp-size ${PREFILL_DP_SIZE} \
      --enable-dp-attention \
      --decode-log-interval 1 \
      --enable-deepep-moe \
      --page-size 1 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --moe-dense-tp-size 1 \
      --enable-dp-lm-head \
      --disable-radix-cache \
      --watchdog-timeout 1000000 \
      --enable-two-batch-overlap \
      --deepep-mode normal \
      --mem-fraction-static 0.85 \
      --chunked-prefill-size 524288 \
      --max-running-requests 8192 \
      --max-total-tokens 131072 \
      --context-length 8192 \
      --init-expert-location /configs/prefill_in4096.json \
      --ep-num-redundant-experts 32 \
      --ep-dispatch-algorithm dynamic \
      --eplb-algorithm deepseek \
      --deepep-config /configs/deepep.json
    ;;
  
  decode)
    echo "Starting decode server..."
    MC_TE_METRIC=true \
    SGLANG_TBO_DEBUG=1 \
    python3 -m sglang.launch_server \
      --model-path /model/ \
      --disaggregation-transfer-backend nixl \
      --disaggregation-mode decode \
      --dist-init-addr ${DECODE_DIST_INIT_ADDR} \
      --nnodes ${NUM_DECODE_NODES} \
      --node-rank ${NODE_RANK} \
      --tp-size ${DECODE_TP_SIZE} \
      --dp-size ${DECODE_DP_SIZE} \
      --enable-dp-attention \
      --decode-log-interval 1 \
      --enable-deepep-moe \
      --page-size 1 \
      --host 0.0.0.0 \
      --trust-remote-code \
      --moe-dense-tp-size 1 \
      --enable-dp-lm-head \
      --disable-radix-cache \
      --watchdog-timeout 1000000 \
      --enable-two-batch-overlap \
      --deepep-mode low_latency \
      --mem-fraction-static 0.835 \
      --max-running-requests 18432 \
      --context-length 4500 \
      --ep-num-redundant-experts 32 \
      --cuda-graph-bs 256
    ;;
  
  *)
    echo "Error: Invalid server type '$SERVER_TYPE'"
    echo "Valid options: prefill, decode, mini_lb"
    exit 1
    ;;
esac
