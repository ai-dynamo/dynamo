---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Enable SGLang Hierarchical Cache (HiCache)"
---

This guide shows how to enable SGLang's Hierarchical Cache (HiCache) inside Dynamo.

## 1) Start the SGLang worker with HiCache enabled

```bash
python -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend nixl \
  --log-level debug \
  --skip-tokenizer-init
```

- **--enable-hierarchical-cache**: Enables hierarchical KV cache/offload
- **--hicache-ratio**: The ratio of the size of host KV cache memory pool to the size of device pool. Lower this number if your machine has less CPU memory.
- **--hicache-write-policy**: Write policy (e.g., `write_through` for synchronous host writes)
- **--hicache-storage-backend**: Host storage backend for HiCache (e.g., `nixl`). NIXL selects the concrete store automatically; see [PR #8488](https://github.com/sgl-project/sglang/pull/8488)


Then, start the frontend:
```bash
python -m dynamo.frontend --http-port 8000
```

## 2) Send a single request

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"
      }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

## 3) (Optional) Benchmarking

Run the perf script:
```bash
bash -x $DYNAMO_ROOT/benchmarks/llm/perf.sh \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallelism 1 \
  --data-parallelism 1 \
  --concurrency "2,4,8" \
  --input-sequence-length 2048 \
  --output-sequence-length 256
```
