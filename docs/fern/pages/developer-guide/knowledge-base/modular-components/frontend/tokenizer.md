---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tokenizer
subtitle: Selects the HuggingFace, fastokens, or Baseten tokenizer backend for BPE models served through the Dynamo Frontend.
---

The Dynamo Frontend supports multiple tokenizer backends for BPE-based `tokenizer.json` models. `BPE` is the underlying tokenization algorithm, not a backend-specific feature: the default HuggingFace, `fastokens`, and `basetenkenizer` paths can all serve supported BPE models. The backend choice controls which implementation performs tokenization before requests are sent to the inference engine.

## Tokenizer Backends

#### `default` HuggingFace Tokenizers

The default backend uses the [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) library (Rust).
It supports features in `tokenizer.json` files (normalizers, pre-tokenizers, post-processors, decoders, added tokens with special-token flags, and byte-fallback).

#### `fastokens` High-Performance Encoder

The `fastokens` backend uses the [`fastokens`](https://github.com/Atero-ai/fastokens) crate, a purpose-built encoder optimized for throughput on supported BPE `tokenizer.json` models.
It is a _hybrid_ backend: encoding uses `fastokens` while decoding falls back to HuggingFace so that incremental detokenization, byte-fallback, and special-token handling work correctly.
It supports segmented encoding so renderers can distinguish trusted control tokens from ordinary content.

Use this backend when tokenization is a measurable bottleneck, for example on high-concurrency prefill-heavy workloads.

#### `basetenkenizer` Native Encoder and Decoder

The `basetenkenizer` backend uses the Baseten Tokenizer implementation exposed by `dynamo-tokenizers`, a high-performance Rust BPE implementation for inference. It performs both encoding and decoding natively and supports segmented encoding for renderers that must preserve trusted control-token boundaries.

Use this backend for supported `tokenizer.json` models when you need Baseten Tokenizer behavior, including token-compatible Kimi tokenizer artifacts.

#### Compatibility notes:

- Works with standard BPE `tokenizer.json` files (Qwen, LLaMA, GPT-family, Mistral, DeepSeek, etc.).
- If `fastokens` or `basetenkenizer` cannot load a particular tokenizer file, the frontend logs a warning and transparently falls back to HuggingFace; requests are never dropped.
- Special tokens declared only in a sibling `tokenizer_config.json` are preserved for Baseten encoding and decoding and for Dynamo's L1 prefix-cache boundaries.
- Has no effect on TikToken-format tokenizers (`.model` / `.tiktoken` files), which always use the TikToken backend.

## Configuration

Set the backend with a CLI flag or environment variable. The CLI flag takes precedence.

| CLI Argument | Env Var | Valid values | Default |
|---|---|---|---|
| `--tokenizer` | `DYN_TOKENIZER` | `default`, `fastokens`, `basetenkenizer` | `default` |

**Examples:**

```bash
# CLI flag
python -m dynamo.frontend --tokenizer fastokens

# Environment variable
export DYN_TOKENIZER=fastokens
python -m dynamo.frontend

# Baseten Tokenizer
python -m dynamo.frontend --tokenizer basetenkenizer
```

## Dynamo Frontend Behavior

When a non-default backend is selected:

1. The frontend resolves `--tokenizer` / `DYN_TOKENIZER` and passes the selected backend to the Rust runtime.
2. `ModelDeploymentCard::tokenizer()` loads the HuggingFace tokenizer first for fallback behavior and L1 cache special-token metadata.
3. Dynamo constructs `FastTokenizer` for `fastokens` or `BasetenTokenizer` for `basetenkenizer` from the same `tokenizer.json` file.
4. If construction fails because the tokenizer uses unsupported features, Dynamo logs a warning and falls back to HuggingFace.
5. When the L1 prefix cache is enabled, Dynamo wraps the selected backend with the same special-token boundary metadata and cache metrics used by the default path.
