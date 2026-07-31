<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Custom encoder benchmark services

The encoder-only service exposes the same OpenAI chat-completions request shape used
by aiperf while running only `AsyncVisionEncoder`. It loads the Qwen2.5-VL benchmark
vision tower, preprocesses exactly one inline image per request, executes the custom
encoder, and returns the dummy assistant content `ok`. It does not start a language
model or generate tokens.

## Start the service

The defaults match the performance-only Qwen2.5 benchmark configuration. The full
2048-wide vision output is computed and truncated to 1536 columns; this is not a
trained projection and makes no quality claim.

```bash
python -m examples.custom_encoder.benchmark.encoder_only_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

The server becomes reachable only after the encoder is loaded and warmed. Check it
with `curl --fail http://localhost:8000/health`.

Override the backend with `--custom-encoder-class module.ClassName`. Encoder tuning
continues to use the backend's existing `DYN_QWEN2_VL_*` variables. The service also
accepts `DYN_HTTP_HOST`, `DYN_HTTP_PORT`, `DYN_ENCODER_ONLY_MODEL`,
`DYN_CUSTOM_ENCODER_CLASS`, `DYN_CUSTOM_ENCODER_MAX_QUEUE_DELAY_US`, and
`DYN_HTTP_MAX_REQUEST_SIZE_MIB` (64 MiB by default, before base64 decoding).

## Run aiperf

Generate the existing deterministic single-image workload, then point aiperf at the
encoder-only endpoint:

```bash
export WORKLOAD_DIR=/dynamo-tmp/logs/encoder-only/workload
python -m examples.custom_encoder.benchmark.safeguard_proxy_workload generate \
  --output-dir "$WORKLOAD_DIR" \
  --image-size 500 \
  --unique-images 1

aiperf profile \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --input-file "$WORKLOAD_DIR/image_custom_1000_isl644.jsonl" \
  --custom-dataset-type single_turn \
  --concurrency 4 \
  --conversation-num 100 \
  --warmup-request-count 10 \
  --extra-inputs "max_tokens:1" \
  --extra-inputs "stream:true" \
  --streaming \
  --artifact-dir /dynamo-tmp/logs/encoder-only/aiperf \
  --ui none \
  --no-server-metrics
```

Do not enable `--use-server-token-count`: the dummy response deliberately reports
zero prompt tokens. Compare request latency, TTFT, and request throughput with the
combined custom-encoder-plus-decoder service. ITL and generated-token throughput are
not meaningful because the encoder-only service emits one dummy content chunk.
