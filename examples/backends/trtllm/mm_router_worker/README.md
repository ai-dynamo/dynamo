<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MM Router Worker

Multimodal-aware KV cache routing worker for TRT-LLM backends.

## Overview

This worker sits between the Dynamo frontend and TRT-LLM workers, providing MM-aware KV cache routing:

1. **Receives** OpenAI-format requests from the frontend
2. **Downloads** images and computes `mm_hash` (for routing decision only)
3. **Builds** multimodal routing metadata (`mm_routing_info`)
4. **Uses** KvRouter to select and route to the best TRT-LLM worker
5. **Streams** responses back to the frontend

## Architecture

```
Frontend (standard)      MM Router Worker (this)        TRT-LLM Worker (standard)
┌──────────────┐        ┌─────────────────────┐        ┌───────────────────┐
│              │───────>│ 1. Download images  │───────>│ python -m         │
│  round-robin │        │ 2. Compute mm_hash  │        │ dynamo.trtllm     │
│  to mm_router│<───────│ 3. Build routing    │<───────│ --modality mm     │
└──────────────┘        │ 4. KvRouter route   │        │ (processes images)│
                        └─────────────────────┘        └───────────────────┘
                                  │
                                  │ Subscribe KV events
                                  v
                            ┌──────────┐
                            │   NATS   │
                            └──────────┘
```

**Note**: Images are downloaded twice - once in MM Router (for mm_hash computation) and once in TRT-LLM worker (for actual processing). This simplifies the design by avoiding tensor serialization.

## Usage

### Quick Start

```bash
# Start all services
./launch.sh
```

### Manual Start

```bash
# 1. Start etcd and NATS
docker compose -f deploy/docker-compose.yml up -d

# 2. Start TRT-LLM worker(s)
python -m dynamo.trtllm \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --namespace default \
    --component trtllm \
    --endpoint generate \
    --modality multimodal \
    --publish-events-and-metrics &

# 3. Start MM Router Worker
python -m examples.backends.trtllm.mm_router_worker \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --model-type qwen2_vl \
    --namespace default \
    --component mm_router \
    --endpoint generate \
    --downstream-component trtllm \
    --downstream-endpoint generate &

# 4. Start Frontend
python -m dynamo.frontend \
    --http-port 8000 \
    --router-mode round-robin
```

### Test Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 100
  }'
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2-VL-2B-Instruct` | Model path or HuggingFace ID |
| `--model-type` | `qwen2_vl` | TRT-LLM model type for multimodal loader |
| `--block-size` | `32` | KV cache block size |
| `--namespace` | `default` | Dynamo namespace |
| `--component` | `mm_router` | This worker's component name |
| `--endpoint` | `generate` | This worker's endpoint name |
| `--downstream-component` | `trtllm` | TRT-LLM workers' component name |
| `--downstream-endpoint` | `generate` | TRT-LLM workers' endpoint name |

## How It Works

### MM Hash Computation

The worker uses TRT-LLM's `apply_mm_hashes()` function to compute a hash of each image's tensor representation. This hash is included in the block hash computation, ensuring that:

- Same image = Same mm_hash = Same block hashes = Cache hit
- Different image = Different mm_hash = Different block hashes = No false cache hit

### KV-Aware Routing

The worker uses `KvRouter.generate(...)` with explicit multimodal routing hints.
When a request comes in:

1. Build routing tokens (`routing_token_ids`) for the request
2. Build `block_mm_infos` with per-block image `mm_hash` metadata
3. Pass both as `mm_routing_info` to `KvRouter.generate(...)`
4. KvRouter computes overlap internally and routes to the best worker

### Block MM Info Structure

For each block that contains image tokens, we build `block_mm_infos`:

```python
block_mm_infos = [
    None,  # Block 0: no image
    {"mm_objects": [{"mm_hash": 12345, "offsets": [[32, 128]]}]},  # Block 1: has image
    {"mm_objects": [{"mm_hash": 12345, "offsets": [[32, 128]]}]},  # Block 2: same image
    None,  # Block 3: no image
]
```

This is included in `mm_routing_info` so KvRouter can compute MM-aware overlap.

## Files

| File | Description |
|------|-------------|
| `mm_router_worker.py` | Main worker with `@dynamo_worker()` |
| `handler.py` | `MMRouterHandler` - routing logic |
| `mm_processor.py` | MM processing utilities |
| `__main__.py` | Entry point |
| `launch.sh` | Launch script |

## Dependencies

- `tensorrt_llm >= 1.2.0rc6` - For `apply_mm_hashes()` and `default_multimodal_input_loader()`. Earlier versions may not include multimodal hash support in KV events.
- `transformers` - For `AutoProcessor`
- `dynamo` - For runtime and KvRouter

## Known Limitations

- **Qwen2-VL specific**: The `_compute_tokens_per_image()` logic in `mm_processor.py` currently only supports `qwen2_vl` model type. Supporting other multimodal models requires adding their visual token computation logic.
