python3 -m sglang.srt.disaggregation.mini_lb \
  --prefill "http://10.52.48.42:30000" \
  --decode  "http://10.52.48.107:30000"

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


MC_TE_METRIC=true \
SGLANG_TBO_DEBUG=1 \
python3 -m sglang.launch_server \
  --model-path /model/ \
  --disaggregation-transfer-backend nixl \
  --disaggregation-mode decode \
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
