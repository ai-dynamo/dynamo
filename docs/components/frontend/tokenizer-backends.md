---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tokenizer Backends
---

The Dynamo Frontend supports multiple tokenizer backends for BPE-based models. The backend controls how input text is tokenized before being sent to the inference engine.

## Tokenizer Backends

#### `default` HuggingFace Tokenizers

The default backend uses the [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) library (Rust). 
It supports features in `tokenizer.json` files (normalizers, pre-tokenizers, post-processors, decoders, added tokens with special-token flags, and byte-fallback).

#### `fasttokens` High-Performance BPE Encoding

The `fasttokens` backend uses the [`fastokens`](https://github.com/Atero-ai/fastokens) crate, a purpose-built BPE encoder optimized for throughput. 
It is a _hybrid_ backend: encoding uses `fastokens` while decoding falls back to HuggingFace so that incremental detokenization, byte-fallback, and special-token handling work correctly.

Use this backend when tokenization is a measurable bottleneck, for example on high-concurrency prefill-heavy workloads.

#### Compatibility notes:

- Works with standard BPE `tokenizer.json` files (Qwen, LLaMA, GPT-family, Mistral, DeepSeek, etc.).
- If `fastokens` cannot load a particular tokenizer file, the frontend logs a warning and transparently falls back to HuggingFace; requests are never dropped.
- Has no effect on TikToken-format tokenizers (`.model` / `.tiktoken` files), which always use the TikToken backend.

## Configuration

Set the backend with a CLI flag or environment variable. The CLI flag takes precedence.

| CLI Argument | Env Var | Valid values | Default |
|---|---|---|---|
| `--dyn-tokenizer-backend` | `DYN_TOKENIZER_BACKEND` | `default`, `fasttokens` | `default` |

**Examples:**

```bash
# CLI flag
python -m dynamo.frontend --dyn-tokenizer-backend fasttokens

# Environment variable
export DYN_TOKENIZER_BACKEND=fasttokens
python -m dynamo.frontend
```

## Dynamo Frontend Behavior

When `DYN_TOKENIZER_BACKEND=fasttokens` is set:

1. The frontend passes the environment variable to the Rust runtime.
2. When building the tokenizer for a model, `ModelDeploymentCard::tokenizer()` attempts to load `fastokens::Tokenizer` from the same `tokenizer.json` file.
3. If loading succeeds, a hybrid `FastTokenizer` is created that encodes with `fastokens` and decodes with HuggingFace.
4. If loading fails (unsupported tokenizer features, missing file, etc.), the frontend logs a warning and falls back to the standard HuggingFace backend; no operator intervention is needed.
