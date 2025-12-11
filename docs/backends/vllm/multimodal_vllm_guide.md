<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# vLLM Multimodal Guide

This document provides a comprehensive guide for multimodal inference using vLLM backend in Dynamo. For more details on the multimodal examples, see [Multimodal Examples Documentation](./multimodal.md).

## Multimodal Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | Yes | Yes | Full support for all image models |
| **Image** | Data URL (Base64) | Yes | Yes | Inline base64-encoded images |
| **Video** | HTTP/HTTPS URL | Yes | Yes | Frame extraction and processing |
| **Audio** | HTTP/HTTPS URL | Yes | Yes | Experimental - requires audio dependencies |

## Architecture Comparison

vLLM multimodal supports three deployment patterns:

```text
SIMPLE AGGREGATED ([examples/backends/vllm/launch/agg_multimodal.sh](../../../examples/backends/vllm/launch/agg_multimodal.sh)):
  Client → Frontend (Rust processor) → Worker [image load, encode, P+D] → Response
  • 2 components • --connector none • Easiest setup

EPD AGGREGATED ([examples/backends/vllm/launch/agg_multimodal_epd.sh](../../../examples/backends/vllm/launch/agg_multimodal_epd.sh)):
  Client → Frontend → Processor → Encoder [NIXL] → PD Worker → Response
  • 4 components • --multimodal-processor • Custom templates, NIXL

DISAGGREGATED ([examples/backends/vllm/launch/disagg_multimodal_epd.sh](../../../examples/backends/vllm/launch/disagg_multimodal_epd.sh)):
  Client → Frontend → Processor → Encoder [NIXL] → Prefill [NIXL] → Decode → Response
  • 5 components • Separate P/D workers • Multi-node, max optimization
```

## Input Format Details

### Supported URL Formats

| Format | Example | Description | Support |
|--------|---------|-------------|---------|
| **HTTP/HTTPS** | `http://example.com/image.jpg` | Remote media files | ✅ |
| **Data URL** | `data:image/jpeg;base64,/9j/4AAQ...` | Base64-encoded inline data | ✅ |

## Simple Aggregated Mode (PD)

In simple aggregated mode, encoding, prefill, and decode happen within the same worker.

### Architecture

```text
HTTP Frontend with Rust processor
    ↓
Worker (Python - ModelInput.Tokens)
    ↓ encode + prefill + decode
Response
```

## EPD Aggregated Mode (PD)

In EPD aggregated mode, encoding happens in a separate worker and prefill and decode happen within the same pipeline.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Processor (Python - ModelInput.Text)
    ↓ tokenizes, extracts media URL
Encode Worker (Python - not registered)
    ↓ downloads media, generates embeddings, NIXL transfer
PD Worker (Python - ModelInput.Tokens)
    ↓ prefill + decode
Response
```

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Processor | `--multimodal-processor` | Text | Yes | HTTP entry, tokenization |
| Encode Worker | `--multimodal-encode-worker` | N/A | No | Media encoding |
| PD Worker | `--multimodal-worker` | Tokens | Yes | Prefill + Decode |

## EPD Disaggregated Mode (E->P->D)

In EPD disaggregated mode, encoding, prefill, and decode are handled by separate workers.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Processor (Python - ModelInput.Text)
    ↓ tokenizes, extracts media URL
Encode Worker (Python - not registered)
    ↓ downloads media, generates embeddings, NIXL transfer
Prefill Worker (Python - ModelInput.Tokens)
    ↓ prefill only, KV cache NIXL transfer
Decode Worker (Python - ModelInput.Tokens)
    ↓ decode only, token generation
Response
```

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Processor | `--multimodal-processor` | Text | Yes | HTTP entry, tokenization |
| Encode Worker | `--multimodal-encode-worker` | N/A | No | Media encoding |
| Prefill Worker | `--multimodal-worker --is-prefill-worker` | Tokens | Yes | Prefill only |
| Decode Worker | `--multimodal-decode-worker` | Tokens | Yes | Decode only |

## Traditional Disagg (EP->D)

Llama 4 models don't support pre-computed embeddings, so they use a combined Encode+Prefill worker.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Processor (Python - ModelInput.Text)
    ↓ tokenizes, extracts media URL
Encode+Prefill Worker (Python - ModelInput.Tokens)
    ↓ downloads media, encodes inline, prefill, KV cache NIXL transfer
Decode Worker (Python - ModelInput.Tokens)
    ↓ decode only, token generation
Response
```

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Processor | `--multimodal-processor` | Text | Yes | HTTP entry, tokenization |
| Encode+Prefill | `--multimodal-encode-prefill-worker --is-prefill-worker` | Tokens | Yes | Encode + Prefill |
| Decode Worker | `--multimodal-decode-worker` | Tokens | Yes | Decode only |

### Launch Script

Example: [`examples/backends/vllm/launch/disagg_multimodal_llama.sh`](../../../examples/backends/vllm/launch/disagg_multimodal_llama.sh)

## ModelInput Types and Registration

### Understanding ModelInput

Dynamo's Rust SDK supports two input types that determine how the HTTP frontend preprocesses requests:

| ModelInput Type | Preprocessing | Use Case |
|-----------------|---------------|----------|
| `ModelInput.Text` | None (raw text passed through) | Components that tokenize themselves |
| `ModelInput.Tokens` | Rust SDK would tokenize (but bypassed in multimodal) | Components expecting pre-tokenized input |

### Component Registration Pattern

```python
# Processor - Entry point from HTTP frontend
await register_llm(
    ModelInput.Text,        # Frontend sends raw text
    ModelType.Chat,
    generate_endpoint,
    model_name,
    ...
)

# Workers - Internal components
await register_llm(
    ModelInput.Tokens,      # Expect pre-tokenized input
    ModelType.Chat,         # or ModelType.Prefill for prefill workers
    generate_endpoint,
    model_name,
    ...
)
```

## **NIXL USE**

| Use Case | Script | NIXL Used? | Data Transfer |
|----------|--------|------------|---------------|
| Simple Aggregated | [`examples/backends/vllm/launch/agg_multimodal.sh`](../../../examples/backends/vllm/launch/agg_multimodal.sh) | ❌ No | All in one worker |
| E->PD Aggregated | [`examples/backends/vllm/launch/agg_multimodal_epd.sh`](../../../examples/backends/vllm/launch/agg_multimodal_epd.sh) | ✅ Yes | Encoder → PD (embeddings) |
| E->P->D Disaggregated | [`examples/backends/vllm/launch/disagg_multimodal_epd.sh`](../../../examples/backends/vllm/launch/disagg_multimodal_epd.sh) | ✅ Yes | Encoder → Prefill (embeddings)<br>Prefill → Decode (KV cache) |
| EP->D Disaggregated (Llama 4) | [`examples/backends/vllm/launch/disagg_multimodal_llama.sh`](../../../examples/backends/vllm/launch/disagg_multimodal_llama.sh) | ✅ Yes | Prefill → Decode (KV cache) |


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
| `components/src/dynamo/vllm/multimodal_handlers/encode_worker_handler.py` | Encode worker implementation |
| `components/src/dynamo/vllm/multimodal_handlers/worker_handler.py` | PD/Prefill/Decode worker implementation |

