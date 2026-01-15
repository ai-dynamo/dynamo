# MM Router Worker

Multimodal-aware KV cache routing worker for TRT-LLM backends.

## Overview

This worker sits between the Dynamo frontend and TRT-LLM workers, providing MM-aware KV cache routing:

1. **Receives** OpenAI-format requests from the frontend
2. **Downloads** images and computes `mm_hash` (for routing decision only)
3. **Queries** KvIndexer to find the worker with best KV cache overlap
4. **Routes** the original request to the selected TRT-LLM worker
5. **Streams** responses back to the frontend

## Architecture

```
Frontend (standard)      MM Router Worker (this)        TRT-LLM Worker (standard)
┌──────────────┐        ┌─────────────────────┐        ┌───────────────────┐
│  HTTP 入口   │───────>│ 1. Download images  │───────>│ python -m         │
│  round-robin │        │ 2. Compute mm_hash  │ direct │ dynamo.trtllm     │
│  to mm_router│<───────│ 3. Find best worker │<───────│ --modality mm     │
└──────────────┘        │ 4. Forward request  │        │ (processes images)│
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

The worker uses TRT-LLM's `apply_mm_hashes()` function to compute a BLAKE3 hash of each image's tensor representation. This hash is included in the block hash computation, ensuring that:

- Same image = Same mm_hash = Same block hashes = Cache hit
- Different image = Different mm_hash = Different block hashes = No false cache hit

### KV-Aware Routing

The worker queries `KvIndexer` which tracks each TRT-LLM worker's KV cache state via NATS events. When a request comes in:

1. Compute block hashes (including mm_hash for image blocks)
2. Query `indexer.find_matches(block_hashes)` to get overlap scores
3. Select the worker with the highest overlap score
4. Route using `client.direct(request, worker_id)`

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

This is passed to `compute_block_hash_for_seq_py()` to compute MM-aware block hashes.

## Files

| File | Description |
|------|-------------|
| `mm_router_worker.py` | Main worker with `@dynamo_worker()` |
| `handler.py` | `MMRouterHandler` - routing logic |
| `mm_processor.py` | MM processing utilities |
| `__main__.py` | Entry point |
| `launch.sh` | Launch script |

## Dependencies

- `tensorrt_llm` - For `apply_mm_hashes()` and `default_multimodal_input_loader()`
- `transformers` - For `AutoProcessor`
- `dynamo` - For runtime, KvIndexer, and compute_block_hash_for_seq_py
