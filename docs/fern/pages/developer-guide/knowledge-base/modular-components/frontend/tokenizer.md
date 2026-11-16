---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Tokenizer
subtitle: Selects the HuggingFace, fastokens, or Baseten tokenizer backend for BPE models served through the Dynamo Frontend.
---

The Dynamo Frontend supports multiple tokenizer backends for BPE-based `tokenizer.json` models. `BPE` is the underlying tokenization algorithm, not a backend-specific feature: the default HuggingFace, `fastokens`, and `basetenkenizer` paths can all serve supported BPE models. The backend choice controls which implementation performs tokenization before requests are sent to the inference engine.

## Tokenizer Backends

#### `default` HuggingFace Tokenizers

The default backend uses the [HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) library (Rust), at its 1.0 release candidate.
It supports features in `tokenizer.json` files (normalizers, pre-tokenizers, post-processors, decoders, added tokens with special-token flags, and byte-fallback), except the configurations listed under [Compatibility notes](#compatibility-notes).

#### `fastokens` High-Performance Encoder

The `fastokens` backend uses the [`fastokens`](https://github.com/Atero-ai/fastokens) crate, a purpose-built encoder optimized for throughput on supported BPE `tokenizer.json` models.
It is a _hybrid_ backend: encoding uses `fastokens` while decoding uses the HuggingFace backend, so that incremental detokenization, byte-fallback, and special-token handling match the default backend.
It supports segmented encoding so renderers can distinguish trusted control tokens from ordinary content.

Use this backend when tokenization is a measurable bottleneck, for example on high-concurrency prefill-heavy workloads.

#### `basetenkenizer` Native Encoder and Decoder

The `basetenkenizer` backend uses the Baseten Tokenizer implementation exposed by `dynamo-tokenizers`, a high-performance Rust BPE implementation for inference. It performs both encoding and decoding natively and supports segmented encoding for renderers that must preserve trusted control-token boundaries.

Use this backend for supported `tokenizer.json` models when you need Baseten Tokenizer behavior, including token-compatible Kimi tokenizer artifacts.

The frontend selects the tokenizer backend as follows. When both `tokenizer.json` and a TikToken artifact are present, `tokenizer.json` takes precedence.

```mermaid
flowchart TD
    A["Frontend resolves<br/>--tokenizer or DYN_TOKENIZER"] --> B{"Available tokenizer artifact?"}
    B -->|"tiktoken.model / *.tiktoken"| C["TikToken backend<br/>backend flag has no effect"]
    B -->|"tokenizer.json"| D{"Requested backend?"}
    B -->|"none"| K["Model load fails<br/>(dynamic discovery retries)"]
    D -->|"default"| L{"HuggingFace<br/>loads?"}
    D -->|"fastokens"| F{"fastokens and its<br/>HuggingFace decoder load?"}
    D -->|"basetenkenizer"| G{"Baseten loads?"}
    F -->|"yes"| H["fastokens encode<br/>HuggingFace decode"]
    G -->|"yes"| I["Baseten encode<br/>Baseten decode"]
    F -->|"no"| J{"Fallback allowed?"}
    G -->|"no"| J
    J -->|"yes, the default"| L
    J -->|"no"| K
    L -->|"yes"| E["HuggingFace<br/>encode + decode"]
    L -->|"no"| K

    style A fill:#d6e9f8,stroke:#7fb3d5
    style B fill:#fdf3d0,stroke:#e0c56e
    style C fill:#e3e0f3,stroke:#a99fd4
    style D fill:#fdf3d0,stroke:#e0c56e
    style E fill:#d9f0dc,stroke:#8fc79a
    style F fill:#fdf3d0,stroke:#e0c56e
    style G fill:#fdf3d0,stroke:#e0c56e
    style H fill:#d9f0dc,stroke:#8fc79a
    style I fill:#d9f0dc,stroke:#8fc79a
    style J fill:#fdf3d0,stroke:#e0c56e
    style K fill:#f8d7da,stroke:#e08b93
    style L fill:#fdf3d0,stroke:#e0c56e
```

#### Compatibility notes:

- Works with standard BPE `tokenizer.json` files (Qwen, LLaMA, GPT-family, Mistral, DeepSeek, etc.).
- If `fastokens` or `basetenkenizer` cannot load a particular tokenizer file, the frontend logs a warning and transparently falls back to HuggingFace by default. Use `--no-tokenizer-fallback` to reject incompatible tokenizers during model initialization.
- The HuggingFace backend rejects a `tokenizer.json` that it cannot serve with the token IDs and text the file describes, and model initialization fails. `fastokens` decodes through the same backend, so it rejects the same files. The rejected configurations are:
  - an added token with `normalized: true` whose normalized form differs from its content, for example an uppercase token under a lowercase normalizer;
  - a byte-level BPE model whose decoder is anything other than `ByteLevel` alone;
  - a byte-fallback BPE vocabulary that lacks any of the 256 `<0xNN>` byte tokens;
  - a `ByteLevel` pre-tokenizer with `add_prefix_space: true`.
- A `truncation` or `padding` block in `tokenizer.json` is ignored by the HuggingFace and `fastokens` backends. Prompts are never clipped or padded by the tokenizer.
- Special tokens declared only in a sibling `tokenizer_config.json` are merged into the
  HuggingFace and Baseten paths, and into Dynamo's L1 prefix-cache boundaries. The FastTokenizer
  encoder loads `tokenizer.json` alone and cannot receive that merge, so a model that declares a
  special token only in `tokenizer_config.json` encodes it as ordinary text under `fastokens` and
  produces different token IDs than the other two backends.
- Multimodal KV routing is disabled while `fastokens` is active, because image placeholders such as
  `<|image_pad|>` are frequently declared only in `tokenizer_config.json`. Requests still complete,
  and per-image token metrics remain available when the model is supported by the image-token
  counter; routing falls back to text-prefix overlap. See
  [Multimodal KV Routing](../../../../use-cases/image-understanding/vlm-performance-optimization/multimodal-kv-routing.md).
- Has no effect on TikToken-format tokenizers (`.model` / `.tiktoken` files), which always use the TikToken backend.
- Dedicated vLLM embedding workers let vLLM tokenize raw text by default. When [`--embedding-frontend-tokenization`](../../../../reference/backends/vllm-configuration.mdx) is enabled, raw-text requests use a request-specific Dynamo tokenizer. The request's `add_special_tokens` value overrides `DYN_EMBEDDING_TOKENIZATION_ADD_SPECIAL_TOKENS`; the default is `true`. For `tokenizer.json` models, `true` selects HuggingFace while `false` follows the configured backend selection shown above. TikToken artifacts always use TikToken, and token-ID inputs bypass frontend tokenization.

## Configuration

Set the backend with a CLI flag or environment variable. The CLI flag takes precedence.

> [!WARNING]
> Automatic tokenizer fallback is deprecated and will be disabled by default in a future release.
> Set `--no-tokenizer-fallback` or `DYN_TOKENIZER_FALLBACK=false` to adopt the future behavior now.

| CLI Argument | Env Var | Valid values | Default |
|---|---|---|---|
| `--tokenizer` | `DYN_TOKENIZER` | `default`, `fastokens`, `basetenkenizer` | `default` |
| `--tokenizer-fallback` / `--no-tokenizer-fallback` | `DYN_TOKENIZER_FALLBACK` | `true`/`false`, `1`/`0`, `on`/`off`, `yes`/`no` | `true` |

**Examples:**

```bash
# CLI flag
python -m dynamo.frontend --tokenizer fastokens

# Environment variable
export DYN_TOKENIZER=fastokens
python -m dynamo.frontend

# Baseten Tokenizer
python -m dynamo.frontend --tokenizer basetenkenizer

# Require Baseten Tokenizer instead of falling back to HuggingFace
python -m dynamo.frontend --tokenizer basetenkenizer --no-tokenizer-fallback
```

## Dynamo Frontend Behavior

When a non-default backend is selected:

1. The frontend resolves `--tokenizer` / `DYN_TOKENIZER` and passes the selected backend to the Rust runtime.
2. `ModelDeploymentCard::tokenizer()` constructs `FastTokenizer` for `fastokens` or `BasetenTokenizer` for `basetenkenizer` from `tokenizer.json`. `FastTokenizer` also loads the HuggingFace backend from the same file, to decode.
3. If construction fails because the tokenizer uses unsupported features, Dynamo logs a warning and falls back to HuggingFace. With `--no-tokenizer-fallback`, model initialization fails and reports the backend loading error instead. In dynamic mode, discovery retries the load while the frontend continues running.
4. When the L1 prefix cache is enabled, Dynamo wraps the selected backend with the special-token boundaries that backend reports and the cache metrics used by the default path.
