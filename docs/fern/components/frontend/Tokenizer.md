---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tokenizer
subtitle: Selects between the default HuggingFace, fastokens, and Baseten tokenizer backends for BPE models served through the Dynamo Frontend.
---

The Dynamo Frontend supports multiple tokenizer backends for BPE-based `tokenizer.json` models. `BPE` is the underlying tokenization algorithm, not a backend-specific feature: the default HuggingFace path, the `fastokens` path, and the `basetenkenizer` path can all serve these models. The backend choice controls which implementation performs tokenization before requests are sent to the inference engine.

## Tokenizer Backends

#### `default` HuggingFace Tokenizers

The default backend uses the [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) library (Rust).
It supports features in `tokenizer.json` files (normalizers, pre-tokenizers, post-processors, decoders, added tokens with special-token flags, and byte-fallback).

#### `fastokens` High-Performance Encoder

The `fastokens` backend uses the [`fastokens`](https://github.com/Atero-ai/fastokens) crate, a purpose-built encoder optimized for throughput on supported BPE `tokenizer.json` models.
It is a _hybrid_ backend: encoding uses `fastokens` while decoding falls back to HuggingFace so that incremental detokenization, byte-fallback, and special-token handling work correctly.

Use this backend when tokenization is a measurable bottleneck, for example on high-concurrency prefill-heavy workloads.

#### `basetenkenizer` Baseten Tokenizer

The `basetenkenizer` backend uses the Baseten tokenizer from the [`dynamo-tokenizers`](https://crates.io/crates/dynamo-tokenizers) crate.
Unlike `fastokens`, it merges the special tokens declared in `tokenizer_config.json` before encoding, and it supports segmented encoding, which preserves the trust boundary between renderer-produced control tokens and user content.

It accepts a narrower set of `tokenizer.json` files than the default backend. Files using an unsupported normalizer or model type are rejected at load time and fall back to HuggingFace in the same way as `fastokens`.

#### Compatibility notes:

- Works with standard BPE `tokenizer.json` files (Qwen, LLaMA, GPT-family, Mistral, DeepSeek, etc.).
- If `fastokens` cannot load a particular tokenizer file, the frontend logs a warning and transparently falls back to HuggingFace; requests are never dropped.
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
```

## Dynamo Frontend Behavior

When `DYN_TOKENIZER=fastokens` is set:

1. The frontend passes the environment variable to the Rust runtime.
2. When building the tokenizer for a model, `ModelDeploymentCard::tokenizer()` attempts to load `fastokens::Tokenizer` from the same `tokenizer.json` file.
3. If loading succeeds, a hybrid `FastTokenizer` is created that encodes with `fastokens` and decodes with HuggingFace.
4. If loading fails (unsupported tokenizer features, missing file, etc.), the frontend logs a warning and falls back to the standard HuggingFace backend; no operator intervention is needed.
