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

This document provides a comprehensive guide for multimodal inference using SGLang backend in Dynamo. For more details on the multimodal examples, see [Multimodal Examples Documentation](./multimodal_epd.md).

## Multimodal Support Matrix

| Modality | Input Format | Aggregated | Disaggregated | Notes |
|----------|--------------|------------|---------------|-------|
| **Image** | HTTP/HTTPS URL | ✅ Yes | ✅ Yes | Vision encoder generates embeddings |
| **Image** | Data URL (Base64) | ❌ No | ❌ No | Not supported |
| **Video** | HTTP/HTTPS URL | ❌ No | ❌ No | Not implemented |
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
  with await connector.create_readable(descriptor) as readable:
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

## Known Limitations

- **No Data URL support** - Only HTTP/HTTPS URLs supported; `data:image/...` base64 URLs not supported
- **No pre-computed embeddings** - Cannot use `.pt`, `.pth`, `.bin` embedding files; vision encoder runs for every request
- **No video support** - No video encoder implementation
- **No audio support** - No audio encoder implementation
- **Only Processor registers with Dynamo** - Workers are internal components, frontend routes to Processor only
- **Disaggregated routing** - Decode Worker is the entry point (calls Prefill), cannot route directly to Prefill workers
- **Limited model generalization** - Token expansion logic is model-specific; adding new models may require implementation updates

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

