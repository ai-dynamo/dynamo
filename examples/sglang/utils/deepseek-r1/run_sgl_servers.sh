#!/bin/bash

# Helper script that allows you to run sglang servers for prefill, decode, and mini_lb. You can use this from inside of the dynamo-deepep container
# Commands are from https://github.com/sgl-project/sglang/issues/6017
# Last updated: 6/25/2025

# Updating values
# Update `dist_init_addr`, `node-rank`, `HEAD_PREFILL_NODE_IP`, `HEAD_DECODE_NODE_IP` values to match your cluster
# ./run_sgl_servers.sh --server-type {prefill, decode, mini_lb

# Example usage
# After updating the values above, on my head prefill node I would run:
#     ./run_sgl_servers.sh --server-type prefill &
#     ./run_sgl_servers.sh --server-type mini_lb
#
# On every other prefill node I would run:
#     ./run_sgl_servers.sh --server-type prefill
#
# On other decode nodes I would run:
#     ./run_sgl_servers.sh --server-type decode

SERVER_TYPE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --server-type)
      SERVER_TYPE="$2"
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
  echo "Usage: $0 --server-type {prefill, decode, mini_lb}"
  exit 1
fi

case $SERVER_TYPE in
  mini_lb)
    echo "Starting mini_lb server..."
    python3 -m sglang.srt.disaggregation.mini_lb \
      --prefill "http://HEAD_PREFILL_NODE_IP:30000" \
      --decode  "http://HEAD_DECODE_NODE_IP:30000"
    ;;

  prefill)
    echo "Starting prefill server..."
    MC_TE_METRIC=true \
    SGLANG_TBO_DEBUG=1 \
    python3 -m sglang.launch_server \
      --model-path /model/ \
      --disaggregation-transfer-backend nixl \
      --disaggregation-mode prefill \
      --dist-init-addr 10.52.48.42:5757 \
      --nnodes 4 \
      --node-rank 2 \
      --tp-size 32 \
      --dp-size 32 \
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
      --dist-init-addr 10.52.48.107:5757 \
      --nnodes 9 \
      --node-rank 2 \
      --tp-size 72 \
      --dp-size 72 \
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
