# TRT-LLM Multimodal KV Router Example

This example demonstrates how to use Dynamo's standard TRT-LLM workers with a custom frontend that computes `mm_hash` for multimodal-aware KV routing.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    NATS Server (JetStream)                        │
│                    (KV events + service discovery)                │
└───────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  Custom API   │      │  KvIndexer    │      │  Standard     │
│  (api.py)     │      │  (in api.py)  │      │  TRT-LLM      │
│               │      │               │      │  Workers      │
│ - HTTP Server │      │ - RadixTree   │◄─────│               │
│ - mm_hash     │      │ - NATS sub    │      │ dynamo.trtllm │
│   computation │──────│               │      │ KvEventPublish│
│ - Routing     │      │               │      │               │
│ - Client      │──────│               │──────│               │
│   .direct()   │      │               │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
        │                                              │
        │           Client.direct(request, worker_id)  │
        └──────────────────────────────────────────────┘
```

### Components

- **Custom api.py**: HTTP server + mm_hash computation + routing logic
- **Standard workers**: `python -m dynamo.trtllm` (no custom worker code)
- **Dynamo Client.direct()**: Route to specific worker based on KV cache overlap
- **KvIndexer**: Subscribe to KV events from workers via NATS

### Key Design: MM-Aware Routing

The custom `api.py` computes `mm_hash` **before** routing, which ensures multimodal content (images) is included in block hash computation. This fixes a fundamental architectural mismatch in the standard Dynamo frontend where:

- Worker publishes: `block_hash = H(tokens + mm_hash)`
- Standard frontend routes with: `block_hash = H(tokens)` only (no mm_hash!)

By computing `mm_hash` in the custom frontend before routing, we ensure proper cache matching for multimodal requests.

## Requirements

- NVIDIA GPU with sufficient memory (H100 80GB can fit 2x Qwen2-VL-2B)
- Docker (for NATS server)
- TensorRT-LLM >= 1.2.0rc6
- Dynamo runtime

## Quick Start

```bash
# Default: 2 workers with Qwen2-VL-2B-Instruct on GPU 0
./launch.sh

# Custom configuration
NUM_WORKERS=3 MODEL=Qwen/Qwen2.5-VL-7B GPU_ID=1 ./launch.sh
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | Model name or path |
| `MODEL_TYPE` | `qwen2_vl` | Model type for TRT-LLM |
| `NUM_WORKERS` | `2` | Number of TRT-LLM workers |
| `BLOCK_SIZE` | `32` | KV cache block size |
| `GPU_ID` | `0` | GPU to use |
| `API_PORT` | `8000` | HTTP API port |
| `NAMESPACE` | `default` | Dynamo namespace |
| `COMPONENT` | `trtllm` | Dynamo component name |
| `ENDPOINT` | `generate` | Dynamo endpoint name |
| `DYNAMO_HOME` | (auto-detected) | Dynamo installation directory |

## API Endpoints

### Chat Completions (OpenAI-compatible)

```bash
# Text-only request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "stream": true
  }'

# Multimodal request with image
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
    "max_tokens": 100,
    "stream": true
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

## How MM-Aware Routing Works

1. **Request arrives**: User sends a chat completion request with optional images
2. **Process multimodal input**: `api.py` loads images and computes visual token expansion
3. **Compute mm_hash**: For each image, compute a hash of the image data using TRT-LLM's `apply_mm_hashes()`
4. **Build block_mm_infos**: Map mm_hash to the token ranges where each image appears
5. **Compute block hashes**: Use `compute_block_hash_for_seq_py(tokens, block_size, block_mm_infos)` - this includes mm_hash in the block hash!
6. **Query KvIndexer**: Find which worker has the highest overlap with computed block hashes
7. **Route with Client.direct()**: Send request directly to the best worker

## Comparison with Standalone Example

| Aspect | Standalone (`router_standalone_trtllm`) | This Example |
|--------|----------------------------------------|--------------|
| HTTP Server | Custom FastAPI | Custom FastAPI |
| Router | Custom KvRouter | **KvIndexer from Dynamo** |
| KV Events | ZMQ direct | **NATS JetStream** |
| Service Discovery | Sequential ports | **Dynamo runtime** |
| Worker Registration | None | **Automatic** |
| Request Routing | Custom direct call | **`Client.direct()`** |
| Worker | Custom TrtllmWorkers class | **Standard `python -m dynamo.trtllm`** |
| MM Hash Computation | In api.py | In api.py |

## Troubleshooting

### NATS not running

```bash
# Start NATS manually
docker compose -f $DYNAMO_HOME/deploy/docker-compose.yml up -d nats
```

### Workers not appearing

Check worker logs for registration errors:
```bash
# View worker logs
docker logs -f $(docker ps -q --filter name=nats)
```

### Out of GPU memory

Reduce `NUM_WORKERS` or use a smaller model:
```bash
NUM_WORKERS=1 ./launch.sh
```

### KvIndexer creation failed

This usually means workers haven't registered their KV event publishers. Wait longer or check worker startup logs.

## Files

- `api.py` - Custom frontend with mm_hash computation and Dynamo Client.direct() routing
- `launch.sh` - Launch script for workers and API
- `README.md` - This documentation
