---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang Processing Modes
subtitle: Choose Dynamo or SGLang for preprocessing and postprocessing
---

Dynamo's SGLang backend can choose preprocessing and postprocessing ownership independently on the worker:

- `--preprocessor dynamo|sglang` controls chat template rendering and tokenization.
- `--postprocessor dynamo|sglang` controls detokenization, tool-call parsing, and reasoning parsing.

Use SGLang postprocessing when SGLang already supports a model's tool-call or reasoning format and Dynamo's Rust parser does not. This makes Dynamo's `/v1/chat/completions` behavior match `sglang.launch_server` for those parser paths.

## Modes

| Preprocessor | Postprocessor | Model I/O | Use case |
|---|---|---|---|
| `dynamo` | `dynamo` | `Tokens/Tokens` | Default Rust path. Enables Dynamo tokenization and postprocessing. |
| `dynamo` | `sglang` | `Tokens/Text` | Dynamo tokenizes for KV routing, SGLang handles detokenization plus tool and reasoning parsing. |
| `sglang` | `sglang` | `Text/Text` | Worker-side passthrough, closest to `sglang.launch_server`. |
| `sglang` | `dynamo` | `Text/Tokens` | Backend tokenizes, Dynamo postprocesses token output. Useful only for specialized fallback cases. |

## Quick Start

Keep Dynamo preprocessing for KV routing, but delegate output parsing to SGLang:

```bash
python -m dynamo.frontend --router-mode kv

CUDA_VISIBLE_DEVICES=0 python -m dynamo.sglang \
  --model-path Qwen/Qwen3-14B-FP8 \
  --served-model-name Qwen/Qwen3-14B-FP8 \
  --preprocessor dynamo \
  --postprocessor sglang \
  --tool-call-parser qwen25 \
  --reasoning-parser qwen3 \
  --tp 1 --trust-remote-code \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
```

For full SGLang-native preprocessing and postprocessing:

```bash
python -m dynamo.frontend --router-mode round-robin

CUDA_VISIBLE_DEVICES=0 python -m dynamo.sglang \
  --model-path Qwen/Qwen3-14B-FP8 \
  --served-model-name Qwen/Qwen3-14B-FP8 \
  --preprocessor sglang \
  --postprocessor sglang \
  --tool-call-parser qwen25 \
  --reasoning-parser qwen3 \
  --tp 1 --trust-remote-code
```

## Parser Flags

When `--postprocessor sglang`, use SGLang's parser names with SGLang's native flags:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --postprocessor sglang \
  --tool-call-parser <sglang-parser> \
  --reasoning-parser <sglang-parser>
```

When `--postprocessor dynamo`, use Dynamo's parser names with the `--dyn-*` flags:

```bash
python -m dynamo.sglang \
  --model-path <model> \
  --postprocessor dynamo \
  --dyn-tool-call-parser <dynamo-parser> \
  --dyn-reasoning-parser <dynamo-parser>
```

The parser name sets can differ between Dynamo and SGLang.

## Migration from `--use-sglang-tokenizer`

`--use-sglang-tokenizer` is deprecated. It remains as a compatibility alias for:

```bash
--preprocessor sglang --postprocessor sglang
```

Replace old launches like this:

```diff
- python -m dynamo.sglang --model-path <model> --use-sglang-tokenizer
+ python -m dynamo.sglang --model-path <model> --preprocessor sglang --postprocessor sglang
```

`--custom-jinja-template` requires `--preprocessor dynamo`. Use SGLang's `--chat-template` flag when delegating preprocessing to SGLang.

## See Also

- [Tool Calling](../../agents/tool-calling.md)
- [Reasoning](../../agents/reasoning.md)
- [Reference Guide](sglang-reference-guide.md)
- [Agentic Workloads](agents.md)
