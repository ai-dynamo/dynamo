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

# TRT-LLM Multimodal Guide

This document provides a comprehensive guide for multimodal inference using TensorRT-LLM backend in Dynamo. For more details on the multimodal examples, see [Multimodal Examples Documentation](./multimodal_support.md).

## Multimodal Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | Yes | Yes | Full support for all image models |
| **Image** | Pre-computed Embeddings (.pt, .pth, .bin) | Yes | Yes | Direct embedding files |
| **Video** | HTTP/HTTPS URL | ❌ No | ❌ No | Not implemented |
| **Audio** | HTTP/HTTPS URL | ❌ No | ❌ No | Not implemented |

## Architecture Comparison

TRT-LLM multimodal supports three deployment patterns:

```text
SIMPLE AGGREGATED (agg.sh):
  Client → Frontend (Rust) → Worker [image load, encode, P+D] → Response
  • 2 components • worker flag `--modality multimodal` • Easiest setup

DISAGGREGATED P->D (disagg_multimodal.sh):
  Client → Frontend → Prefill [image load, encode] → Decode → Response
  • 3 components • worker flag `--disaggregation-mode prefill/decode` • Multi-GPU, KV transfer

EPD DISAGGREGATED - WIP:
  Client → Frontend → Encode [MultimodalEncoder] → Prefill [via params] → Decode → Response
  • 4 components • worker flag `--disaggregation-mode encode/prefill/decode` • WIP PR #4668
```

## Input Format Details

### Supported URL Formats

| Format | Example | Description | Support |
|--------|---------|-------------|---------|
| **HTTP/HTTPS** | `http://example.com/image.jpg` | Remote media files | ✅ |
| **Pre-computed Embeddings** | `/path/to/embedding.pt` | Local embedding files (.pt, .pth, .bin) | ✅ |

## Simple Aggregated Mode (PD)

In aggregated mode, all processing (image loading, encoding, prefill, decode) happens within a single worker.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
TRT-LLM Worker (Python - ModelInput.Tokens)
    ↓ downloads media, encodes, prefill + decode
Response
```

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Worker | `--modality multimodal` | Tokens | Yes | Complete inference pipeline |

### Launch Script

Example: [`examples/backends/trtllm/launch/agg.sh`](../../../examples/backends/trtllm/launch/agg.sh)

## Disaggregated Mode (P->D)

In disaggregated mode, prefill and decode are handled by separate workers. The prefill worker handles image loading and encoding internally.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Prefill Worker (Python - ModelInput.Tokens)
    ↓ downloads media, encodes, prefill, KV cache transfer
Decode Worker (Python - ModelInput.Tokens)
    ↓ decode only, token generation
Response
```

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Prefill Worker | `--disaggregation-mode prefill` | Tokens | Yes | Image processing + Prefill |
| Decode Worker | `--disaggregation-mode decode` | Tokens | Yes | Decode only |

### Launch Script

Example: [`examples/backends/trtllm/launch/disagg_multimodal.sh`](../../../examples/backends/trtllm/launch/disagg_multimodal.sh)

## Pre-computed Embeddings

TRT-LLM supports providing pre-computed embeddings, bypassing image-to-embedding processing.

### Supported File Types

- `.pt` - PyTorch tensor files
- `.pth` - PyTorch checkpoint files
- `.bin` - Binary tensor files

### Embedding File Formats

TRT-LLM supports two formats for embedding files:

#### 1. Simple Tensor Format

- Direct tensor saved as `.pt` file
- Example: `llava_next_mm_embed_seashore.pt`
- Contains only the embedding tensor

```python
# Example: Simple tensor format
embedding_tensor = torch.rand(1, 576, 4096)  # [batch, seq_len, hidden_dim]
torch.save(embedding_tensor, "embedding.pt")
```

#### 2. Dictionary Format with Auxiliary Data

- Dictionary containing multiple keys
- Used by models like Llama-4 that require additional metadata
- Must contain `mm_embeddings` key with the main tensor
- Can include auxiliary data like special tokens, offsets, etc.

```python
# Example: Dictionary format (Llama-4 style)
embedding_dict = {
    "mm_embeddings": torch.rand(1, 576, 4096),
    "special_tokens": [128256, 128257],
    "image_token_offsets": [[0, 576]],
    # ... other model-specific metadata
}
torch.save(embedding_dict, "llama4_embedding.pt")
```

**How They're Used:**
- **Simple tensors**: Loaded directly and passed to `mm_embeddings` parameter
- **Dictionary format**: `mm_embeddings` key extracted as main tensor, other keys preserved as auxiliary data and transferred separately

### Launch Script

Example: [`examples/backends/trtllm/launch/epd_disagg.sh`](../../../examples/backends/trtllm/launch/epd_disagg.sh)

### Security Considerations

For EPD mode with local embedding files:

- `--allowed-local-media-path` - Specify secure directory for embedding files (default: `/tmp`)
- `--max-file-size-mb` - Limit max file size to prevent DoS attacks (default: `50MB`)

## EPD Disaggregated Mode (E->P->D) - WIP

**Status:** Work In Progress (WIP PR #4668) - Full EPD flow with MultimodalEncoder

In EPD mode, encoding, prefill, and decode are handled by separate workers. The encode worker uses TensorRT-LLM's `MultimodalEncoder` to process images and transfer embeddings via disaggregated parameters.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Encode Worker (Python - NOT registered, uses MultimodalEncoder)
    ↓ downloads image, encodes with vision model, transfers via disaggregated_params
Prefill Worker (Python - ModelInput.Tokens)
    ↓ receives embeddings via disaggregated_params, prefill only, KV cache transfer
Decode Worker (Python - ModelInput.Tokens)
    ↓ decode only, token generation
Response
```

**Note (WIP):** The encode worker uses `MultimodalEncoder` from TensorRT-LLM to actually encode images, not just load pre-computed embeddings. This is a significant change from the legacy NIXL-based embedding transfer.

### Components

| Component | Flag | ModelInput | Registered | Purpose |
|-----------|------|-----------|------------|---------|
| Encode Worker | `--disaggregation-mode encode` | N/A | No | Image encoding with MultimodalEncoder |
| Prefill Worker | `--disaggregation-mode prefill --encode-endpoint` | Tokens | Yes | Prefill only |
| Decode Worker | `--disaggregation-mode decode` | Tokens | Yes | Decode only |


## ModelInput Types and Registration

### Understanding ModelInput

TRT-LLM workers register with Dynamo using:

| ModelInput Type | Preprocessing | Use Case |
|-----------------|---------------|----------|
| `ModelInput.Tokens` | Rust SDK tokenizes text (bypassed for multimodal) | All TRT-LLM workers |

### Component Registration Pattern

```python
# TRT-LLM Worker - Register with Tokens
await register_llm(
    ModelInput.Tokens,      # Rust does minimal preprocessing
    model_type,             # ModelType.Chat or ModelType.Prefill
    generate_endpoint,
    model_name,
    ...
)
```

## Inter-Component Communication

| Transfer Stage | Message      | NIXL Transfer |
|----------------|--------------|---------------|
| **Frontend → Prefill** | Request with image URL or embedding path | No |
| **Encode → Prefill (pre-computed embeddings)** | NIXL metadata (pre-computed embeddings) | Yes (Embeddings tensor) |
| **Encode → Prefill (Image URL) (WIP)** | Disaggregated params with multimodal handles | No (Handles via params) |
| **Prefill → Decode** | Disaggregated params | Configurable (KV cache: NIXL default, UCX optional) |


## **NIXL USE**

| Use Case | Script | NIXL Used? | Data Transfer |
|----------|--------|------------|---------------|
| Simple Aggregated | [`examples/backends/trtllm/launch/agg.sh`](../../../examples/backends/trtllm/launch/agg.sh) | ❌ No | All in one worker |
| P->D Disaggregated | [`examples/backends/trtllm/launch/disagg_multimodal.sh`](../../../examples/backends/trtllm/launch/disagg_multimodal.sh) | ⚙️ Optional | Prefill → Decode (KV cache via UCX or NIXL) |
| E->P->D Disaggregated (pre-computed embeddings) | [`examples/backends/trtllm/launch/epd_disagg.sh`](../../../examples/backends/trtllm/launch/epd_disagg.sh) | ✅ Yes | Encoder → Prefill (pre-computed embeddings via NIXL) |
| E->P->D Disaggregated (WIP) | X | ❌ No | Encoder → Prefill (multimodal handles via disaggregated_params)<br>Prefill → Decode (KV cache via UCX/NIXL) |

**Note:** NIXL for KV cache transfer is currently beta and only supported on AMD64 (x86_64) architecture.


## Key Files

| File | Description |
|------|-------------|
| `components/src/dynamo/trtllm/main.py` | Worker initialization and setup |
| `components/src/dynamo/trtllm/utils/trtllm_utils.py` | Command-line argument parsing |
| `components/src/dynamo/trtllm/multimodal_processor.py` | Multimodal request processing |
| `components/src/dynamo/trtllm/request_handlers/handlers.py` | Request handler factory |
| `components/src/dynamo/trtllm/request_handlers/handler_base.py` | Base handler and disaggregation modes |

## Known Limitations

- **No Data URL support** - Only HTTP/HTTPS URLs supported; `data:image/...` base64 URLs not supported
- **No video support** - No video encoder implementation
- **No audio support** - No audio encoder implementation
- **No Rust preprocessing** - All preprocessing happens in Python workers
- **E->P->D mode is WIP** - Full EPD with image URLs under development

## Supported Models

Multimodal models listed in [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md) are supported by Dynamo.

Common examples:
- Llama 4 Vision models (Maverick, Scout)
- Qwen2-VL models
- Other vision-language models with TRT-LLM support

