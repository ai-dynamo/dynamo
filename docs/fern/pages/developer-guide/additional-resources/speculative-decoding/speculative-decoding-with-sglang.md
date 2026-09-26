---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Speculative Decoding with SGLang
---

Using Speculative Decoding with the SGLang backend.

For cross-backend documentation, see [Speculative Decoding Overview](overview.md).

## Prerequisites

- SGLang container or a local SGLang install. See the [SGLang Backend](../../knowledge-base/modular-components/backends/sglang/overview.md#installation) guide.
- One GPU for aggregated serving, two for disaggregated serving

## Quick Start: Qwen3-8B + EAGLE3

This guide deploys **Qwen/Qwen3-8B** with the **Tengyunw/qwen3_8b_eagle3** EAGLE3 draft model.

### Step 1: Start Infrastructure Services

```bash
docker compose -f dev/docker-compose.yml up -d
```

### Step 2: Run Aggregated Speculative Decoding

```bash
cd examples/backends/sglang
./launch/agg_spec_decoding.sh
```

### Step 3: Test the Deployment

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Write a poem about why Sakura trees are beautiful."}
    ],
    "max_tokens": 250
  }'
```

### Step 4: Check the Speculative Decoding Metrics

The worker exposes `sglang:spec_*` metrics on its system port:

```bash
curl -s localhost:8081/metrics | grep '^sglang:spec_'
```

See the [SGLang metrics documentation](https://docs.sglang.io/docs/references/production_metrics) for metric definitions.

## Disaggregated Serving with KV-Aware Routing

Pass the same speculative decoding flags to the prefill and decode workers, and start the frontend with `--router-mode kv`:

```bash
python3 -m dynamo.frontend --router-mode kv &

SPEC_ARGS=(
  --model-path Qwen/Qwen3-8B
  --page-size 16
  --enable-metrics
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --disaggregation-transfer-backend nixl
  --host 0.0.0.0
)

DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.sglang "${SPEC_ARGS[@]}" \
  --disaggregation-mode prefill \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' &

DYN_SYSTEM_PORT=8082 CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.sglang "${SPEC_ARGS[@]}" \
  --disaggregation-mode decode \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5558"}' &
```

Once the workers are ready, send the request from Step 3. Speculation runs on the decode worker, so read its metrics on port 8082.

## Limitations

- Only EAGLE3 is validated end to end

## See Also

| Document | Path |
|----------|------|
| Speculative Decoding Overview | [Overview](overview.md) |
| SGLang Backend Guide | [SGLang](../../knowledge-base/modular-components/backends/sglang/overview.md) |
| SGLang Speculative Decoding | [SGLang docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding) |
