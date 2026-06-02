---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parser Configuration
subtitle: How --dyn-chat-processor, --dyn-tool-call-parser, and --dyn-reasoning-parser fit together
---

Dynamo turns a model's raw tool-call and reasoning markup into structured `tool_calls` and `reasoning_content`. Two independent choices control how that parsing happens. This page is the single source of truth for **which flags combine and which combinations don't make sense**. For the parser *names* themselves, follow the per-stage links at the bottom.

## The two choices

**1. Who parses — `--dyn-chat-processor`** (a *frontend* flag; default `dynamo`):

- `dynamo` (default) — Dynamo's framework-agnostic Rust parser. Works on every backend (vLLM, SGLang, TRT-LLM) and with disaggregated serving.
- `vllm` / `sglang` — delegate parsing to that engine's own Python parser ("engine fallback"). Use only when Dynamo does not ship a parser for your model.

**2. Which parser** — the flag name *and where it goes* depend on choice 1:

| Chat processor | Parser flag(s) and where they go | Parses with | Disaggregated serving | Backends |
|---|---|---|---|---|
| `dynamo` (default) | `--dyn-tool-call-parser <name>` and/or `--dyn-reasoning-parser <name>` — on the **worker** | Dynamo Rust frontend | Supported | vLLM, SGLang, TRT-LLM |
| `vllm` | `--tool-call-parser <name>` and/or `--reasoning-parser <name>` — on the **frontend** | vLLM Python | Not yet | vLLM |
| `sglang` | `--tool-call-parser <name>` and/or `--reasoning-parser <name>` — on the **frontend** | SGLang Python | Not yet | SGLang |

## The pairing rule

- The **`--dyn-*` parser flags pair with the `dynamo` chat processor** and go on the **worker**: `--dyn-tool-call-parser`, `--dyn-reasoning-parser`.
- The **bare `--tool-call-parser` / `--reasoning-parser` flags pair with `vllm` / `sglang`** and go on the **frontend**.

Tool calling and reasoning are independent — set one, the other, or both — but always from the same family as your chat processor. You never mix the two families.

## What does NOT make sense

| Combination | Why it's wrong |
|---|---|
| `--dyn-chat-processor dynamo` + `--tool-call-parser` / `--reasoning-parser` | The bare flags drive the engine-fallback path; the default Dynamo path uses the `--dyn-` flags. Use `--dyn-tool-call-parser` / `--dyn-reasoning-parser`. |
| `--dyn-chat-processor vllm`/`sglang` + `--dyn-tool-call-parser` / `--dyn-reasoning-parser` | The `--dyn-` flags only drive Dynamo's native parser; an engine processor reads its own `--tool-call-parser` / `--reasoning-parser`. |
| `--dyn-chat-processor vllm`/`sglang` + disaggregated serving | Engine-fallback parsing does not support disaggregated serving yet. Use the default `dynamo` processor. |
| `--dyn-chat-processor vllm`/`sglang` on TRT-LLM | TRT-LLM engine fallback is a work in progress. Use the default `dynamo` processor. |
| Reusing a parser name across families | The registries differ — e.g. Dynamo `deepseek_v3` vs vLLM/SGLang `deepseekv3`, Dynamo `nemotron3` vs vLLM `nemotron_v3`. Use the name from the registry that matches your chat processor. |

## Examples

Default (Dynamo-native) — the common case; the chat processor defaults to `dynamo`, so no frontend flag is needed:

```bash
# Worker selects the Dynamo parsers; frontend is launched plainly.
python -m dynamo.trtllm --model-path /model --served-model-name openai/gpt-oss-120b \
  --dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss
python -m dynamo.frontend
```

Engine fallback (vLLM) — only when Dynamo lacks a parser for your model:

```bash
python -m dynamo.vllm --model <model>
python -m dynamo.frontend --dyn-chat-processor vllm --tool-call-parser hermes --reasoning-parser deepseek_r1
```

## Parser names and per-stage details

- Tool calling: [Tool Call Parsing (Dynamo)](tool-calling/dynamo.md) (native parser names) and [Tool Call Parsing (Engine Fallback)](tool-calling/engine-fallback.md).
- Reasoning: [Reasoning Parsing (Dynamo)](reasoning/dynamo.md) (native parser names) and [Reasoning Parsing (Engine Fallback)](reasoning/engine-fallback.md).
- Engine processors: [vLLM Chat Processor](backends/vllm/vllm-chat-processor.md) and [SGLang Chat Processor](backends/sglang/sglang-chat-processor.md).
- Every frontend flag: [Frontend Configuration Reference](components/frontend/configuration.md).
