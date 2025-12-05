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

# SGLang Multimodal Guide

This document provides a comprehensive guide for multimodal inference using SGLang backend in Dynamo.

## Multimodal Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | ✅ Yes | ✅ Yes | Vision encoder generates embeddings |
| **Image** | Data URL (Base64) | ❌ No | ❌ No | Not supported |
| **Video** | HTTP/HTTPS URL | ❌ No | ❌ No | Protocol accepts, but encode worker doesn't process |
| **Audio** | HTTP/HTTPS URL | ❌ No | ❌ No | Not implemented |

## Architecture Comparison

SGLang multimodal supports two deployment patterns:

```text
AGGREGATED (E->PD):
  Client → Frontend (Rust) → Processor → Encoder [NIXL] → PD Worker → Response
  • 3 components • Vision encoder in Python • NIXL embeddings transfer

DISAGGREGATED (E->P->D):
  Client → Frontend → Processor → Encoder [NIXL] → Prefill [bootstrap] → Decode → Response
  • 4 components • Vision encoder in Python • KV cache transfer via bootstrap mechanism
```

## Aggregated Mode (E->PD)

In aggregated mode, encoding happens in a separate worker, but prefill and decode share the same engine.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Processor (Python - ModelInput.Text - REGISTERED)
    ↓ tokenizes with chat template, extracts image URL
Encode Worker (Python - NOT registered)
    ↓ downloads image, runs vision encoder, generates embeddings, NIXL transfer
PD Worker (Python - NOT registered)
    ↓ receives embeddings via NIXL, prefill + decode
Response → Processor → Frontend
```

### Components

| Component | Flag | ModelInput | Registered | Has SGLang Engine? | Purpose |
|-----------|------|-----------|------------|-------------------|---------|
| Processor | `--multimodal-processor` | Text | ✅ Yes | ❌ No | HTTP entry, OpenAI→SGLang conversion |
| Encode Worker | `--multimodal-encode-worker` | N/A | ❌ No | ❌ No | Vision encoder, embeddings generation |
| PD Worker | `--multimodal-worker` | N/A | ❌ No | ✅ Yes | Prefill + Decode with embeddings |

### Key Characteristics

- **Vision Encoder in Python**: Encode worker loads vision model (AutoModel) and image processor (AutoImageProcessor)
- **Token Expansion**: Single `<|image_pad|>` token replaced with N tokens based on embedding shape
- **NIXL Transfer**: Embeddings transferred from Encoder → PD Worker using NIXL
- **No Rust Processing**: All tokenization and image handling happens in Python

## Disaggregated Mode (E->P->D)

In disaggregated mode, encoding, prefill, and decode are handled by separate workers using SGLang's bootstrap coordination.

### Architecture

```text
HTTP Frontend (Rust)
    ↓
Processor (Python - ModelInput.Text - REGISTERED)
    ↓ tokenizes with chat template, extracts image URL
Encode Worker (Python - NOT registered)
    ↓ downloads image, runs vision encoder, generates embeddings, NIXL transfer
Prefill Worker (Python - NOT registered)
    ↓ receives embeddings via NIXL, prefill only, returns bootstrap info
Decode Worker (Python - NOT registered)
    ↓ uses bootstrap info, decode only, token generation
Response → Processor → Frontend
```

### Components

| Component | Flag | ModelInput | Registered | Has SGLang Engine? | Purpose |
|-----------|------|-----------|------------|-------------------|---------|
| Processor | `--multimodal-processor` | Text | ✅ Yes | ❌ No | HTTP entry, OpenAI→SGLang conversion |
| Encode Worker | `--multimodal-encode-worker` | N/A | ❌ No | ❌ No | Vision encoder, embeddings generation |
| Decode Worker | `--multimodal-worker --serving-mode=decode` | N/A | ❌ No | ✅ Yes | **Entry point for disaggregation**, calls Prefill |
| Prefill Worker | `--multimodal-worker --serving-mode=prefill` | N/A | ❌ No | ✅ Yes | Called by Decode, bootstrap coordination |

### Bootstrap Coordination

SGLang disaggregation uses a bootstrap mechanism for P->D coordination:

**Request Flow (Important):**
```text
Client → Frontend → Processor → Encode → DECODE Worker → Prefill Worker
                                               ↑
                                    Entry point for disaggregation!
```

**Bootstrap Process:**
1. **Decode Worker** receives request from Encode Worker
2. **Decode Worker** calls Prefill Worker via NATS to request bootstrap info
3. **Prefill Worker** generates `{host, port, room}` and returns immediately
4. **Both workers** connect to same "room" using bootstrap coordinates
5. **SGLang internally** transfers KV cache state via bootstrap connection (not NIXL)

**Key Difference from vLLM:**
- vLLM: Frontend → Prefill → Decode (Prefill is entry point)
- SGLang: Frontend → Processor → Encode → **Decode → Prefill** (Decode is entry point)

## ModelInput Types and Registration

**Only the Processor registers with Dynamo Rust.**

### Registration Pattern

```python
# ONLY Processor registers with Dynamo Rust
await register_llm_with_readiness_gate(
    None,                   # No engine for processor
    generate_endpoint,
    server_args,
    dynamo_args,
    input_type=ModelInput.Text,  # Receives raw OpenAI format
    readiness_gate=ready_event,
)

# Workers do NOT register - they are internal components
# They communicate via NATS clients created in main.py
```

### Component Initialization

```python
# Encode Worker - connects to downstream PD worker
pd_worker_client = (
    await runtime.namespace(dynamo_args.namespace)
    .component("backend")
    .endpoint("generate")
    .client()
)

# PD Worker (Decode mode) - connects to upstream Prefill worker
prefill_client = (
    await runtime.namespace(dynamo_args.namespace)
    .component("prefill")
    .endpoint("generate")
    .client()
)
```

## Inter-Component Communication

### Control Flow (NATS)

All component-to-component communication happens via NATS:

**Aggregated Mode (E→PD):**
```text
Processor → Encode Worker → PD Worker
  (NATS)        (NATS + NIXL embeddings)
```

**Disaggregated Mode (E→P→D):**
```text
Processor → Encode Worker → DECODE Worker → Prefill Worker
  (NATS)        (NATS)            (NATS)
                             ↓
                    Decode requests bootstrap
                             ↓
                    Prefill returns {host, port, room}
                             ↓
                    Both connect via bootstrap
                             ↓
                    SGLang internal KV cache transfer
```

**Detailed Message Flow:**

```text
Processor → Encode Worker:
  - NATS round_robin with SglangMultimodalRequest
  - Contains: tokenized input_ids, image URL, sampling params

Encode Worker → Decode/PD Worker:
  - NATS round_robin to "backend" component
  - Contains: expanded token_ids, NIXL metadata, embeddings shape
  - NIXL transfer: embeddings tensor

Decode Worker → Prefill Worker (disagg only):
  - NATS call to "prefill" component
  - Decode requests bootstrap coordinates
  - Prefill returns: {bootstrap_host, bootstrap_port, bootstrap_room}

Prefill ↔ Decode (via bootstrap):
  - SGLang internal connection (not NATS)
  - KV cache state shared via bootstrap mechanism
```

### Data Transfer (NIXL)

NIXL is used only for embedding transfer:

```python
Encode Worker:
  descriptor = connect.Descriptor(precomputed_embeddings)
  with connector.create_readable(descriptor) as readable:
      request.serialized_request = readable.metadata()
      # Send request with NIXL metadata
      await pd_worker_client.round_robin(request)
      await readable.wait_for_completion()

PD Worker:
  embeddings = torch.empty(request.embeddings_shape, dtype=torch.float16)
  descriptor = connect.Descriptor(embeddings)
  read_op = await connector.begin_read(request.serialized_request, descriptor)
  await read_op.wait_for_completion()
```

## Vision Encoding Details

### Encode Worker Components

The encode worker loads and runs the vision model in Python:

```python
# Vision components loaded in encode worker
self.image_processor = AutoImageProcessor.from_pretrained(
    model_path, trust_remote_code=True
)
self.vision_model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
```

### Token Expansion Process

1. Processor inserts single image token (e.g., `<|image_pad|>`)
2. Encode worker generates embeddings: `shape = (batch, num_patches, hidden_dim)`
3. Encode worker replaces single token with `num_patches` tokens
4. Downstream worker receives expanded token sequence

Example:
```python
# Before: ["Hello", "<|image_pad|>", "world"]
# After:  ["Hello", "<|image_pad|>", "<|image_pad|>", ...(576 tokens), "world"]
```

## Chat Template Processing

SGLang uses its own chat template system:

```python
from sglang.srt.parser.conversation import chat_templates

conv = chat_templates["qwen2-vl"].copy()
conv.append_message(conv.roles[0], f"{conv.image_token} Describe this image")
processed = tokenizer(text=conv.get_prompt(), return_tensors="pt")
```

Supported templates: `qwen2-vl`, `llama-3`, `vicuna`, etc.

## NIXL USE

| Use Case | NIXL Used? | Data Transfer | Notes |
|----------|------------|---------------|-------|
| E→PD Aggregated | ✅ Yes | Encoder → PD (embeddings) | Vision encoder separate |
| E→P→D Disaggregated | ✅ Yes | Encoder → Prefill (embeddings) | KV cache via SGLang bootstrap |

**Key Difference:** SGLang P→D uses bootstrap mechanism, not NIXL for KV cache like vLLM.

## GAPS and Known Limitations

### 1. No Base64 (Data URL) Support

**Current State:**
- Only HTTP/HTTPS URLs supported for images
- Data URLs (`data:image/jpeg;base64,...`) are **not supported**
- vLLM and TRT-LLM support data URLs, SGLang does not

**Impact:**
- Cannot send embedded images in requests
- Requires external image hosting for all images

### 2. No Pre-computed Embeddings Support

**Current State:**
- No support for pre-computed embeddings (`.pt`, `.pth`, `.bin` files)
- Vision encoder must run for every request
- Cannot bypass encoding like TRT-LLM legacy flow

**Impact:**
- Higher latency for repeated images
- Cannot optimize by pre-computing embeddings offline

### 3. Only Processor Registers with Rust

**Current State:**
- Only the Processor component registers with Dynamo Rust using `ModelInput.Text`
- All workers are internal and do not register
- Different from vLLM/TRT-LLM where workers also register

**Implications:**
- Frontend always routes to Processor (cannot route directly to workers)
- No token-based entry point (no `ModelInput.Tokens` registration for workers)
- More complex multi-component setup required for all multimodal requests

### 4. All Processing Happens in Python Workers

**Current State:**
- No Rust-based image decoding or preprocessing
- No Rust tokenization (all tokenization in Python Processor)
- Frontend only handles HTTP routing

**Impact:**
- Cannot leverage Rust performance for preprocessing
- All multimodal logic in Python components
- Similar limitation to TRT-LLM

### 5. No Video/Audio Model Support

**Current State:**
- **Video models are NOT supported** - Encode worker only implements image loading and processing
- **Audio models are NOT supported** - No audio encoder implementation
- Only **image modality** is production-ready
- Protocol accepts `video_url` and Processor can forward it, but Encode Worker **only processes `image_url`**

**Why:**
```python
# encode_worker_handler.py only checks for image_url
if not request.multimodal_input.image_url:
    raise ValueError("image_url is required for the encode worker.")
```

**Impact:**
- Cannot run video models like `LLaVA-NeXT-Video-7B-hf`
- Cannot run audio models like `Qwen2-Audio-7B-Instruct`
- Use **vLLM backend** for video/audio support (has full implementation)

**Workaround:**
- For video models: Use vLLM ([`examples/multimodal/launch/video_agg.sh`](../../../examples/multimodal/launch/video_agg.sh))
- For audio models: Use vLLM ([`examples/multimodal/launch/audio_agg.sh`](../../../examples/multimodal/launch/audio_agg.sh))
- Or implement custom video/audio encode worker for SGLang

### 6. Bootstrap Coordination and Routing Complexity

**Current State:**
- Disaggregated mode requires bootstrap coordination between P and D workers
- Uses host/port/room mechanism from SGLang
- **Decode Worker is the entry point** (not Prefill like vLLM)
- Request path: `Encode → Decode → Prefill` (Decode calls Prefill)

**Architectural Pattern:**
```text
Encode Worker → pd_worker_client → DECODE Worker
                                         ↓
                                    prefill_client → PREFILL Worker
```

**Impact:**
- More complex P→D coordination than vLLM
- Requires network connectivity between P and D workers
- Different debugging model than vLLM

**Routing Implications:**

**Cannot Route Directly to Prefill:**
- Prefill Worker does NOT register with Dynamo
- Frontend cannot route requests to Prefill directly
- All disaggregated requests MUST go through Decode Worker first
- Decode Worker initiates bootstrap coordination with Prefill

**Load Balancing Constraints:**
- Cannot distribute load directly to Prefill workers
- Must load balance at Decode Worker level
- Decode Worker becomes bottleneck for prefill requests
- Different from vLLM where frontend can route to prefill workers directly

**Multi-Instance Limitations:**
- If you scale Prefill workers, Decode must discover them
- Cannot use frontend routing to select specific Prefill worker
- Decode Worker uses `prefill_client.generate()` (round-robin to any prefill)
- Less control over prefill worker selection compared to vLLM

### 7. Manual Token Expansion in Encode Worker

**Current State:**
- Encode worker **manually** expands image tokens from 1 → N based on embedding shape
- Token expansion happens in Python code, not handled by SGLang engine
- Hard-coded logic specific to model architecture

**Code Location:**
```python
# encode_worker_handler.py:144-157
# Find single image token in sequence
image_token_id_index = request.request.token_ids.index(self.image_token_id)

# Get number of patches from embedding shape
num_image_tokens = precomputed_embeddings.shape[1]  # e.g., 576 patches

# Replace 1 token with N tokens
request.request.token_ids = (
    request.request.token_ids[:image_token_id_index]
    + [self.image_token_id] * num_image_tokens
    + request.request.token_ids[image_token_id_index + 1:]
)
```

**Why This Is Error-Prone:**

1. **Model-Specific Logic Required:**
   - Different models have different patch sizes and embedding dimensions
   - Number of tokens depends on: image resolution, patch size, pooling strategy
   - Must update code for each new model architecture
   - Example: Qwen2-VL uses 576 patches, but other models may use different counts

2. **Assumes Shape[1] is Patch Count:**
   - Hard-coded assumption: `num_image_tokens = precomputed_embeddings.shape[1]`
   - Works for: `(batch, patches, hidden_dim)` format
   - Breaks for: Different embedding formats (e.g., pooled, multi-scale)
   - No validation that shape[1] is actually the patch dimension

3. **Single Image Token Assumption:**
   - Assumes exactly one image token in sequence
   - Fails for: Multiple images, video frames, complex layouts
   - `token_ids.index()` throws error if token not found or multiple tokens

4. **No Dynamic Resolution Support:**
   - Fixed expansion based on embedding shape
   - Cannot handle dynamic image resolutions without code changes
   - Models with resolution-dependent patch counts need special handling

5. **Tight Coupling with Chat Template:**
   - Must know exact image token string from chat template
   - Hard-coded token extraction logic (lines 72-87)
   - Different templates may use different token formats

**Impact:**
- **Maintenance burden**: Must update encode worker for each new model
- **Error-prone**: Easy to miscalculate token counts for new architectures
- **No abstraction**: Token expansion logic embedded in handler, not engine
- **Limited flexibility**: Cannot easily support models with variable patch counts
- **Debugging difficulty**: Token count mismatches hard to diagnose

**Comparison with vLLM:**
- vLLM handles token expansion **internally in the engine**
- vLLM workers just pass image data, engine figures out tokens
- More robust and less prone to manual errors

**Workaround:**
- Carefully study each new model's architecture
- Test token expansion with known inputs
- Add extensive logging for token count validation


## Supported Models

SGLang multimodal **only supports image-based vision-language models**:

### ✅ Supported (Images Only)
- **Qwen2-VL** / **Qwen2.5-VL** (primary support)
- Models with `AutoImageProcessor` and vision tower
- Models compatible with SGLang's image embedding format


## Key Files

| File | Description |
|------|-------------|
| `components/src/dynamo/sglang/main.py` | Component initialization, only Processor registers |
| `components/src/dynamo/sglang/request_handlers/multimodal/processor_handler.py` | Processor implementation, OpenAI→SGLang |
| `components/src/dynamo/sglang/request_handlers/multimodal/encode_worker_handler.py` | Vision encoder, embeddings generation |
| `components/src/dynamo/sglang/request_handlers/multimodal/worker_handler.py` | PD/Prefill/Decode workers, NIXL read |
| `components/src/dynamo/sglang/multimodal_utils/multimodal_chat_processor.py` | Chat template processing |
| `components/src/dynamo/sglang/protocol.py` | Request/response data structures |
| `components/src/dynamo/sglang/register.py` | Registration logic (only called for Processor) |

