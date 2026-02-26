---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM Multimodal
---

# vLLM Multimodal

This document provides a comprehensive guide for multimodal inference using vLLM backend in Dynamo.

> [!IMPORTANT]
> **Security Requirement**: All multimodal workers require the `--enable-multimodal` flag to be explicitly set at startup. This is a security feature to prevent unintended processing of multimodal data from untrusted sources. Workers will fail at startup if multimodal flags (e.g., `--multimodal-worker`, `--multimodal-processor`) are used without `--enable-multimodal`.
> This flag is analogous to `--enable-mm-embeds` in vllm serve but also extends it to all multimodal content (url, embeddings, b64).

## Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | Yes | Yes | Full support for all image models |
| **Image** | Data URL (Base64) | Yes | Yes | Inline base64-encoded images |
| **Video** | HTTP/HTTPS URL | Yes | Yes | Frame extraction and processing |
| **Audio** | HTTP/HTTPS URL | Yes | Yes | Experimental - requires audio dependencies |

### Supported URL Formats

| Format | Example | Description |
|--------|---------|-------------|
| **HTTP/HTTPS** | `http://example.com/image.jpg` | Remote media files |
| **Data URL** | `data:image/jpeg;base64,/9j/4AAQ...` | Base64-encoded inline data |

## Deployment Patterns

vLLM supports all multimodal deployment patterns. See [Architecture Patterns](README.md#architecture-patterns) for detailed explanations.

| Pattern | Supported | Launch Script | Notes |
|---------|-----------|---------------|-------|
| EPD (Simple Aggregated) | ✅ | `agg_multimodal.sh` | Easiest setup |
| E/PD (Encode Separate) | ✅ | `agg_multimodal_epd.sh` | Separate encode worker |
| E/P/D (Full Disaggregation) | ✅ | `disagg_multimodal_epd.sh` | All stages separate |
| EP/D (Traditional Disaggregated) | ✅ | `disagg_multimodal_llama.sh` | For Llama 4 models |

### Component Flags

| Component | Flag | Purpose |
|-----------|------|---------|
| Processor | `--multimodal-processor` | HTTP entry, tokenization |
| Encode Worker | `--multimodal-encode-worker` | Media encoding |
| PD Worker | `--multimodal-worker` | Prefill + Decode |
| Prefill Worker | `--multimodal-worker --disaggregation-mode prefill` | Prefill only |
| Decode Worker | `--multimodal-decode-worker` | Decode only |

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the [latest release](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

## Image Serving

### E/PD Serving (Encode Separate)

**Components:**

- workers: [EncodeWorkerHandler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py) for encoding and [MultimodalPDWorkerHandler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/multimodal_handlers/worker_handler.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the EncodeWorkerHandler.
- frontend: HTTP endpoint to handle incoming requests.

**Workflow:**

The EncodeWorkerHandler encodes the image and passes the embeddings to the MultimodalPDWorkerHandler via NATS and RDMA. The work complete event is sent via NATS, while the embeddings tensor is transferred via RDMA through the NIXL interface.

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> encode_worker
  encode_worker --> processor
  encode_worker --embeddings--> pd_worker
  pd_worker --> encode_worker
```

> **Note:** Aggregated serving supports LLaVA 1.5 7B and Qwen2.5-VL-7B-Instruct. Disaggregated serving is currently only confirmed for LLaVA.

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
# Serve a LLaVA 1.5 7B model:
bash launch/agg_multimodal_epd.sh --model llava-hf/llava-1.5-7b-hf
# Serve a Qwen2.5-VL model:
bash launch/agg_multimodal_epd.sh --model Qwen/Qwen2.5-VL-7B-Instruct
```

**Client:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llava-hf/llava-1.5-7b-hf",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 300,
      "temperature": 0.0,
      "stream": false
    }'
```

### E/P/D Serving (Full Disaggregation)

**Components:**

- workers: [EncodeWorkerHandler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py) for encoding, [MultimodalDecodeWorkerHandler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/multimodal_handlers/worker_handler.py) for decoding, and [MultimodalPDWorkerHandler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/multimodal_handlers/worker_handler.py) for prefilling.
- processor: Tokenizes the prompt and passes it to the EncodeWorkerHandler.
- frontend: HTTP endpoint to handle incoming requests.

**Workflow:**

For the LLaVA model, embeddings are only required during the prefill stage. The EncodeWorkerHandler is connected directly to the prefill worker, encoding the image and passing embeddings via NATS and RDMA. The prefill worker performs the prefilling step and forwards the KV cache to the decode worker.

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> encode_worker
  encode_worker --> processor
  encode_worker --embeddings--> prefill_worker
  prefill_worker --> encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_multimodal_epd.sh --model llava-hf/llava-1.5-7b-hf
```

> [!NOTE] Disaggregation is currently only confirmed to work with LLaVA. Qwen2.5-VL is not confirmed to be supported.

## Llama 4 Serving

The Llama 4 model family is natively multimodal. Unlike LLaVA, they do not directly consume image embeddings as input (see the [vLLM support matrix](https://docs.vllm.ai/en/latest/models/supported_models.html#text-generation_1)). Therefore, the encoder worker is not used and encoding is done alongside prefill.

Example model: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` on H100x8.

### Llama 4 Aggregated Serving

**Workflow:**

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> pd_worker
  pd_worker --> processor
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg_multimodal.sh --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
```

**Client:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
              }
            }
          ]
        }
      ],
      "max_tokens": 300,
      "temperature": 0.0,
      "stream": false
    }'
```

### Llama 4 Disaggregated Serving

**Workflow:**

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --image_url--> prefill_worker
  prefill_worker --> processor
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_multimodal_llama.sh --head-node

# On a separate node with NATS_SERVER and ETCD_ENDPOINTS pointing to head node:
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_multimodal_llama.sh
```

## Video Serving

### Video Aggregated Serving

**Components:**

- workers: [VideoEncodeWorker](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal/components/video_encode_worker.py) for decoding video into frames, and [VllmPDWorker](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal/components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the VideoEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

**Workflow:**

The VideoEncodeWorker decodes the video into frames. Unlike the image pipeline which generates embeddings, this pipeline passes raw frames directly to the VllmPDWorker via NATS and RDMA.

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --video_url--> video_encode_worker
  video_encode_worker --> processor
  video_encode_worker --frames--> pd_worker
  pd_worker --> video_encode_worker
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/video_agg.sh
```

**Client:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Describe the video in detail"
            },
            {
              "type": "video_url",
              "video_url": {
                "url": "https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
              }
            }
          ]
        }
      ],
      "max_tokens": 300,
      "stream": false
    }' | jq
```

### Video Disaggregated Serving

**Workflow:**

For the LLaVA-NeXT-Video-7B model, frames are only required during the prefill stage. The VideoEncodeWorker is connected directly to the prefill worker, decoding the video into frames and passing them via RDMA.

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --video_url--> video_encode_worker
  video_encode_worker --> processor
  video_encode_worker --frames--> prefill_worker
  prefill_worker --> video_encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/multimodal
bash launch/video_disagg.sh
```

## Audio Serving

### Audio Aggregated Serving

**Components:**

- workers: [AudioEncodeWorker](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal/components/audio_encode_worker.py) for decoding audio into embeddings, and [VllmPDWorker](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal/components/worker.py) for prefilling and decoding.
- processor: Tokenizes the prompt and passes it to the AudioEncodeWorker.
- frontend: HTTP endpoint to handle incoming requests.

**Workflow:**

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --audio_url--> audio_encode_worker
  audio_encode_worker --> processor
  audio_encode_worker --embeddings--> pd_worker
  pd_worker --> audio_encode_worker
```

**Launch:**

```bash
pip install 'vllm[audio]' accelerate # multimodal audio models dependency
cd $DYNAMO_HOME/examples/multimodal
bash launch/audio_agg.sh
```

**Client:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen/Qwen2-Audio-7B-Instruct",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is recited in the audio?"
            },
            {
              "type": "audio_url",
              "audio_url": {
                "url": "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav"
              }
            }
          ]
        }
      ],
      "max_tokens": 6000,
      "temperature": 0.8,
      "stream": false
    }' | jq
```

### Audio Disaggregated Serving

**Workflow:**

For the Qwen2-Audio model, audio embeddings are only required during the prefill stage. The AudioEncodeWorker is connected directly to the prefill worker.

```mermaid
flowchart LR
  HTTP --> processor
  processor --> HTTP
  processor --audio_url--> audio_encode_worker
  audio_encode_worker --> processor
  audio_encode_worker --embeddings--> prefill_worker
  prefill_worker --> audio_encode_worker
  prefill_worker --> decode_worker
  decode_worker --> prefill_worker
```

**Launch:**

```bash
pip install 'vllm[audio]' accelerate # multimodal audio models dependency
cd $DYNAMO_HOME/examples/multimodal
bash launch/audio_disagg.sh
```

## Embedding Cache

Dynamo supports embedding cache in both aggregated and disaggregated settings:

| Setting | Implementation | Launch Script |
|---------|---------------|---------------|
| **Aggregated** | `ec_both` via upstream vLLM (main or v0.17.0+) | `vllm_serve_embedding_cache.sh` |
| **Disaggregated encoder** | Dynamo-managed cache in the worker layer on top of vLLM engine | `disagg_multimodal_e_pd.sh` |

### Aggregated Worker

A single vLLM instance acts as both **producer** (encodes and saves embeddings to CPU cache) and **consumer** (loads cached embeddings back to GPU). Repeated images skip encoding entirely.

```mermaid
---
title: Embedding Cache — Aggregated Encoder (e.g. aggregated EP or EPD node)
---
flowchart LR
  req[Multimodal Request] --> gpu{GPU Encoder Cache<br/>hit?}
  gpu -- yes --> skip[Use cached GPU embedding<br/>no encoder, no connector]
  gpu -- no --> cpu{CPU Embedding Cache<br/>hit?}
  cpu -- yes --> load[Load: CPU → GPU<br/>skip encoder]
  cpu -- no --> encode[Run Encoder]
  encode -- save: GPU → CPU --> store[(CPU Embedding Cache<br/>LRU)]
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/vllm_serve_embedding_cache.sh --multimodal-embedding-cache-capacity-gb 10
```

This configures `vllm serve` with `ec_role=ec_both` and the `DynamoMultimodalEmbeddingCacheConnector` automatically. The capacity parameter controls the CPU-side LRU cache size in GB (0 = disabled).

### Disaggregated Encoder (Embedding Cache in Prefill Worker)

In the disaggregated setting, the Prefill Worker (P) owns a CPU-side LRU embedding cache (`EmbeddingCacheManager`). On each request P checks the cache first — on a hit, the Encode Worker is skipped entirely. On a miss, P routes to the Encode Worker (E), receives embeddings via NIXL, saves them to the cache, and then feeds the embeddings along with the request into the vLLM Instance for prefill.

```mermaid
---
title: Embedding Cache — Disaggregated Encoder
---
flowchart LR
    req[Request] --> cpu_check{"CPU cache hit?<br/>(EmbeddingCacheManager)"}

    subgraph P ["Prefill Worker (P)"]
        cpu_check -. hit .-> use[Use cached embedding]
        use --> vllm[vLLM Instance]
    end

    cpu_check -- miss --> E["Encode Worker (E)"]
    E -- "embeddings via NIXL" --> save["Save to cache"]
    save --> vllm
```

**Launch:**

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/disagg_multimodal_e_pd.sh --multimodal-embedding-cache-capacity-gb 10
```

**Client:** Same as [E/PD Serving](#epd-serving-encode-separate)

## NIXL Usage

| Use Case | Script | NIXL Used? | Data Transfer |
|----------|--------|------------|---------------|
| EPD (Simple Aggregated) | `agg_multimodal.sh` | No | All in one worker |
| E/PD (Encode Separate) | `agg_multimodal_epd.sh` | Yes | Encoder → PD (embeddings) |
| E/P/D (Full Disaggregation) | `disagg_multimodal_epd.sh` | Yes | Encoder → Prefill (embeddings), Prefill → Decode (KV cache) |
| EP/D (Llama 4) | `disagg_multimodal_llama.sh` | Yes | Prefill → Decode (KV cache) |
| EC Both (Local Node) | `vllm_serve_embedding_cache.sh` | No | ECConnector via CPU Embedding Cache |

## ModelInput Types and Registration

Dynamo's Rust SDK supports two input types that determine how the HTTP frontend preprocesses requests:

| ModelInput Type | Preprocessing | Use Case |
|-----------------|---------------|----------|
| `ModelInput.Text` | None (raw text passed through) | Components that tokenize themselves |
| `ModelInput.Tokens` | Rust SDK would tokenize (but bypassed in multimodal) | Components expecting pre-tokenized input |

**Registration Pattern:**

```python
# Processor - Entry point from HTTP frontend
await register_model(
    ModelInput.Text,        # Frontend sends raw text
    ModelType.Chat,
    generate_endpoint,
    model_name,
    ...
)

# Workers - Internal components
await register_model(
    ModelInput.Tokens,      # Expect pre-tokenized input
    ModelType.Chat,         # or ModelType.Prefill for prefill workers
    generate_endpoint,
    model_name,
    ...
)
```

## LoRA Adapters on Multimodal Workers

Multimodal workers support dynamic loading and unloading of LoRA adapters at runtime via the management API. This enables serving fine-tuned multimodal models alongside the base model.

### Loading a LoRA Adapter

Load an adapter on a running multimodal worker via the `load_lora` endpoint:

```bash
# For components workers (URI-based, requires DYN_LORA_ENABLED=true)
curl -X POST http://<worker-host>:<port>/load_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my-vlm-adapter",
    "source": {"uri": "s3://my-bucket/adapters/my-vlm-adapter"}
  }'

# For example workers (path-based)
curl -X POST http://<worker-host>:<port>/load_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my-vlm-adapter",
    "lora_path": "/path/to/adapter"
  }'
```

### Sending Requests with a LoRA

Set the `model` field in the request to the LoRA adapter name:

```bash
curl -X POST http://<frontend-host>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-vlm-adapter",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]}
    ]
  }'
```

Requests without a LoRA name (or with the base model name) will use the base model.

### Unloading a LoRA Adapter

```bash
curl -X POST http://<worker-host>:<port>/unload_lora \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my-vlm-adapter"}'
```

### Listing Loaded Adapters

```bash
curl -X POST http://<worker-host>:<port>/list_loras
```

### Disaggregated Mode

In disaggregated (prefill/decode) deployments, the **same LoRA adapter must be loaded on both the prefill and decode workers**. The LoRA identity (`model` field) is automatically propagated from the prefill worker to the decode worker in the forwarded request.

```bash
# Load on prefill worker
curl -X POST http://<prefill-worker>/load_lora \
  -d '{"lora_name": "my-adapter", "source": {"uri": "s3://bucket/adapter"}}'

# Load on decode worker (same adapter)
curl -X POST http://<decode-worker>/load_lora \
  -d '{"lora_name": "my-adapter", "source": {"uri": "s3://bucket/adapter"}}'
```

If a LoRA is loaded on the prefill worker but not on the decode worker, the decode worker will fall back to the base model for that request.

## Known Limitations

- **Disaggregated flows require Python Processor** - All multimodal disaggregation requires the Python Processor component (`ModelInput.Text`).

## Supported Models

The following models have been tested with Dynamo's vLLM multimodal backend:

- **Qwen2.5-VL** - `Qwen/Qwen2.5-VL-7B-Instruct`
- **Qwen3-VL** - `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- **LLaVA 1.5** - `llava-hf/llava-1.5-7b-hf`
- **Llama 4 Maverick** - `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- **LLaVA Next Video** - `llava-hf/LLaVA-NeXT-Video-7B-hf`
- **Qwen2-Audio** - `Qwen/Qwen2-Audio-7B-Instruct`

For a complete list of multimodal models supported by vLLM, see [vLLM Supported Multimodal Models](https://docs.vllm.ai/en/latest/models/supported_models/#list-of-multimodal-language-models). Models listed there should work with Simple Aggregated Mode but may not be explicitly tested.

## Key Files

| File | Description |
|------|-------------|
| `components/src/dynamo/vllm/main.py` | Worker initialization and setup |
| `components/src/dynamo/vllm/args.py` | Command-line argument parsing |
| `components/src/dynamo/vllm/multimodal_handlers/processor_handler.py` | Processor implementation |
| `components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py` | Encode worker implementations (custom and vLLM-native) |
| `components/src/dynamo/vllm/multimodal_handlers/worker_handler.py` | PD/Prefill/Decode worker implementation |
