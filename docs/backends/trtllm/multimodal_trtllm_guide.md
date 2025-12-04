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

This document provides a comprehensive guide for multimodal inference using TensorRT-LLM backend in Dynamo.

## Multimodal Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | Yes | Yes | Full support for all image models |
| **Image** | Pre-computed Embeddings (.pt, .pth, .bin) | Yes | Yes | Direct embedding files |

## Architecture Comparison

TRT-LLM multimodal supports three deployment patterns:

```
SIMPLE AGGREGATED (agg.sh):
  Client → Frontend (Rust) → Worker [image load, encode, P+D] → Response
  • 2 components • worker flag `--modality multimodal` • Easiest setup

DISAGGREGATED P->D (disagg_multimodal.sh):
  Client → Frontend → Prefill [image load, encode] → Decode → Response
  • 3 components • worker flag `--disaggregation-mode prefill/decode` • Multi-GPU, KV transfer

EPD DISAGGREGATED - WIP (epd_disagg.sh):
  Client → Frontend → Encode [MultimodalEncoder] → Prefill [via params] → Decode → Response
  • 4 components • worker flag `--disaggregation-mode encode/prefill/decode` • WIP PR #3818
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

```
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

Example: `[examples/backends/trtllm/launch/agg.sh](../../../examples/backends/trtllm/launch/agg.sh)`

## Disaggregated Mode (P->D)

In disaggregated mode, prefill and decode are handled by separate workers. The prefill worker handles image loading and encoding internally.

### Architecture

```
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

Example: `[examples/backends/trtllm/launch/disagg_multimodal.sh](../../../examples/backends/trtllm/launch/disagg_multimodal.sh)`

## EPD Disaggregated Mode (E->P->D) - WIP

**Status:** Work In Progress (WIP PR #3818) - Full EPD flow with MultimodalEncoder

In EPD mode, encoding, prefill, and decode are handled by separate workers. The encode worker uses TensorRT-LLM's `MultimodalEncoder` to process images and transfer embeddings via disaggregated parameters.

### Architecture

```
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

### Launch Script

Example: `[examples/backends/trtllm/launch/epd_disagg.sh](../../../examples/backends/trtllm/launch/epd_disagg.sh)`

**Note (WIP):** The default model in the WIP PR is `llava-hf/llava-v1.6-mistral-7b-hf`.

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

### NATS-Based Messaging

TRT-LLM components communicate using NATS messaging:

| Transfer Stage | NATS Message | NIXL Transfer |
|----------------|--------------|---------------|
| **Frontend → Prefill** | Request with image URL or embedding path | No |
| **Encode → Prefill (Precomputed Embeddings)** | NIXL metadata (pre-computed embeddings) | Yes (Embeddings tensor) |
| **Encode → Prefill (Image URL) (WIP)** | Disaggregated params with multimodal handles | No (Handles via params) |
| **Prefill → Decode** | Disaggregated params | Yes/No (KV cache - UCX or NIXL) |


## **NIXL USE**

| Use Case | Script | NIXL Used? | Data Transfer |
|----------|--------|------------|---------------|
| Simple Aggregated | `[examples/backends/trtllm/launch/agg.sh](../../../examples/backends/trtllm/launch/agg.sh)` | ❌ No | All in one worker |
| P->D Disaggregated | `[examples/backends/trtllm/launch/disagg_multimodal.sh](../../../examples/backends/trtllm/launch/disagg_multimodal.sh)` | ⚙️ Optional | Prefill → Decode (KV cache via UCX or NIXL) |
| E->P->D Disaggregated (Precomputed Embeddings) | `[examples/backends/trtllm/launch/epd_disagg.sh](../../../examples/backends/trtllm/launch/epd_disagg.sh)` | ✅ Yes | Encoder → Prefill (pre-computed embeddings via NIXL) |
| E->P->D Disaggregated (WIP) | `examples/backends/trtllm/launch/url_epd_disagg.sh` | ❌ No | Encoder → Prefill (multimodal handles via disaggregated_params)<br>Prefill → Decode (KV cache via UCX/NIXL) |

**Note:** NIXL for KV cache transfer is currently beta and only supported on AMD64 (x86_64) architecture.

## **GAPS and Known Limitations**

### 1. No Base64 Data URL Support

**Current State:**
- TRT-LLM does NOT support base64-encoded `data:image/...` URLs
- Use HTTP/HTTPS URLs or pre-computed embedding files instead

### 2. E->P->D Mode is WIP

**Current State (WIP PR #3818):**
- EPD mode (E->P->D) is under active development
- Uses `MultimodalEncoder` from TensorRT-LLM for actual image encoding (not just pre-computed embeddings)
- Embeddings transferred via `disaggregated_params` (includes `multimodal_embedding_handles` and `multimodal_hashes`)
- Encode worker does not register with frontend; accessed via `--encode-endpoint`


### 3. NIXL KV Cache Transfer Beta

## Pre-computed Embeddings (Legacy)

TRT-LLM supports providing pre-computed embeddings, bypassing image-to-embedding processing. This is the **Embeddings URL** approach for EPD mode.

### Supported File Types

- `.pt` - PyTorch tensor files
- `.pth` - PyTorch checkpoint files
- `.bin` - Binary tensor files

### Embedding File Formats

TRT-LLM supports two formats for embedding files:

**1. Simple Tensor Format**
- Direct tensor saved as `.pt` file
- Example: `llava_next_mm_embed_seashore.pt`
- Contains only the embedding tensor

```python
# Example: Simple tensor format
embedding_tensor = torch.rand(1, 576, 4096)  # [batch, seq_len, hidden_dim]
torch.save(embedding_tensor, "embedding.pt")
```

**2. Dictionary Format with Auxiliary Data**
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

### Security Considerations

For EPD mode with local embedding files:

- `--allowed-local-media-path` - Specify secure directory for embedding files (default: `/tmp`)
- `--max-file-size-mb` - Limit max file size to prevent DoS attacks (default: `50MB`)

## Full EPD with Image URLs (WIP)

**Status:** Work In Progress (PR #3818)

The WIP full EPD flow allows sending image URLs directly to the encode worker, which uses `MultimodalEncoder` to encode them.

### How It Works (WIP)

1. **Client** sends image URL in request
2. **Frontend** routes to **Prefill Worker**
3. **Prefill Worker** calls **Encode Worker** with image URL
4. **Encode Worker**:
   - Downloads image using `default_multimodal_input_loader`
   - Encodes with `MultimodalEncoder.generate()`
   - Returns `ep_disaggregated_params` containing:
     - `multimodal_embedding_handles` - GPU memory handles for embeddings
     - `multimodal_hashes` - Hashes for embedding verification
     - `processed_prompt` - Prompt with `<image>` placeholders
     - `prompt_token_ids` - Pre-tokenized prompt
5. **Prefill Worker** receives embeddings via disaggregated params, performs prefill
6. **Decode Worker** continues generation

## Key Files

| File | Description |
|------|-------------|
| `components/src/dynamo/trtllm/main.py` | Worker initialization and setup |
| `components/src/dynamo/trtllm/utils/trtllm_utils.py` | Command-line argument parsing |
| `components/src/dynamo/trtllm/multimodal_processor.py` | Multimodal request processing |
| `components/src/dynamo/trtllm/request_handlers/handlers.py` | Request handler factory |
| `components/src/dynamo/trtllm/request_handlers/handler_base.py` | Base handler and disaggregation modes |

## **GAPS and Known Limitations**

### 1. All Processing Happens in Python Workers

**Current State:**
- TRT-LLM multimodal workers register with `ModelInput.Tokens`
- However, **all multimodal preprocessing happens in Python workers**, not in Rust frontend
- Rust frontend only validates URLs and tokenizes text-only prompts
- Python workers handle:
  - Image downloading
  - Image decoding (pixel-level)
  - Vision encoding
  - Multimodal prompt processing (adding `<image>` tokens)
  - Tokenization of multimodal prompts

**Why This Is a Gap:**
- No reuse of Rust preprocessing/postprocessing logic for multimodal requests
- Inconsistent with text-only flows where Rust handles tokenization
- Limits optimization opportunities in the frontend

### 2. TRT-LLM Requires Text Prompts, Not Tokens (Current)

**Current State:**
- TRT-LLM's `MultimodalEncoder` and `LLM.generate_async()` expect **text prompts**, not pre-tokenized input
- This differs from vLLM which can accept `TokensPrompt` directly
- Forces Python workers to handle tokenization, even though workers register as `ModelInput.Tokens`

**Ideal State:**
- TRT-LLM should accept **pre-tokenized input** (token IDs)
- Rust frontend could tokenize multimodal prompts (with `<image>` placeholders)
- Python workers would only handle vision encoding

**In Progress:**
- TRT-LLM team is working on accepting tokens instead of text prompts
- This would enable Rust preprocessing/postprocessing reuse for multimodal requests
- Would align TRT-LLM with vLLM's architecture where workers truly consume tokens

### 3. Multimodal Processor Uses `ModelInput.Text` Semantics

**Current State:**
- `MultimodalRequestProcessor` in TRT-LLM workers expects OpenAI format messages with raw text
- Workers effectively operate as `ModelInput.Text` despite registering as `ModelInput.Tokens`
- This is a workaround until TRT-LLM accepts tokenized input

**Impact:**
- Architectural inconsistency between registration and actual behavior
- Cannot leverage Rust SDK's tokenization capabilities
- Additional complexity in Python worker code

### 4. No Audio/Video Support in Dynamo TRT-LLM Backend

**Current State:**
- TensorRT-LLM engine natively supports audio and video modalities
- Dynamo's TRT-LLM backend does **not yet** expose these capabilities
- Only image modality is currently supported: `--modality multimodal` (images only)

**Why:**
- Dynamo backend implementation has not been extended to handle audio/video
- `MultimodalRequestProcessor` only extracts `image_url` from messages
- No handlers for `audio_url` or `video_url` content types

**What's Missing:**
- Audio content type processing (`"type": "audio_url"`)
- Video content type processing (`"type": "video_url"`)
- Integration with TensorRT-LLM's audio/video input loaders
- Model-specific audio/video preprocessing

**In Progress:**
- Backend extension to support audio and video is planned
- Will follow similar patterns to image support once implemented

## Supported Models

Multimodal models listed in [TensorRT-LLM supported models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/inputs/utils.py#L221) are supported by Dynamo.

Common examples:
- Llama 4 Vision models (Maverick, Scout)
- Qwen2-VL models
- Other vision-language models with TRT-LLM support

