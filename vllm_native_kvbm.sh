#!/usr/bin/env bash
# Reproduce: KVBM multimodal failure with vanilla vLLM (not dynamo.vllm)
# Proves the bug is in KVBM's _create_slot, not in dynamo.vllm.
#
# Run inside container (Context C).
set -euo pipefail

MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
PORT=8000

# ── Start vanilla vLLM with KVBM connector ──
echo "Starting vanilla vLLM serve with DynamoConnector..."
DYN_KVBM_CPU_CACHE_GB=20 \
vllm serve $MODEL \
  --port $PORT \
  --enforce-eager \
  --max-model-len 16384 \
  --kv-transfer-config '{
    "kv_connector": "DynamoConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "kvbm.vllm_integration.connector"
  }' &
VLLM_PID=$!

# ── Wait for ready ──
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
  if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
    echo "Ready after ${i}s"
    break
  fi
  sleep 1
done

# ── Text-only request (should succeed) ──
echo ""
echo "=== Text-only request (should succeed) ==="
echo ""
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.0
  }' | python -m json.tool

# ── Multimodal request (should fail with ValueError) ──
echo ""
echo "=== Multimodal request (expect ValueError in worker logs) ==="
echo ""
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {
          "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
        }}
      ]
    }],
    "max_tokens": 50,
    "temperature": 0.0
  }' | python -m json.tool || echo "Request failed (expected)"

# ── Cleanup ──
echo ""
echo "=== Cleanup ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "Done."
