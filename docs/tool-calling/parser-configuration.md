---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parser Configuration
subtitle: How --dyn-chat-processor, --dyn-tool-call-parser, and --dyn-reasoning-parser fit together
---

Dynamo turns a model's raw tool-call and reasoning markup into structured `tool_calls` and `reasoning_content`. Two independent choices control how that parsing happens. This page is the single source of truth for **which flags combine and which combinations don't make sense**. For the parser *names* themselves, follow the per-stage links at the bottom.

## The choices

**1. Who parses — `--dyn-chat-processor`** (a *frontend* flag; default `dynamo`):

- `dynamo` (default) — Dynamo's framework-agnostic Rust parser. Works on every backend (vLLM, SGLang, TRT-LLM) and with disaggregated serving.
- `vllm` / `sglang` — delegate parsing to that engine's own Python parser ("engine fallback"). Use only when Dynamo does not ship a parser for your model.

**2. Which parser** — the flag name *and where it goes* depend on choice 1:

| Parser Implementation | Parser flag(s) and where they go | Parses with | Disaggregated serving | Backends |
|---|---|---|---|---|
| `dynamo` (default) | `--dyn-tool-call-parser <name>` and/or `--dyn-reasoning-parser <name>` — on the **worker** | Dynamo Rust frontend | Supported | vLLM, SGLang, TRT-LLM |
| `vllm` | `--tool-call-parser <name>` and/or `--reasoning-parser <name>` — on the **frontend** | vLLM Python | Supported | vLLM |
| `sglang` | `--tool-call-parser <name>` and/or `--reasoning-parser <name>` — on the **frontend** | SGLang Python | Supported | SGLang |

## The pairing rule

- The **`--dyn-*` parser flags pair with the `dynamo` chat processor** and go on the **worker**: `--dyn-tool-call-parser`, `--dyn-reasoning-parser`.
- The **bare `--tool-call-parser` / `--reasoning-parser` flags pair with `vllm` / `sglang`** and normally go on the **frontend**.

> [!TIP]
> For guided decoding and structured outputs, the engine itself needs to be
> aware of which tokens are reasoning or not so that it can only start enforcing the
> grammar after reasoning is complete. For example, the model should be able to
> reason freely before it needs to start outputting structured JSON content to the user.
> Similarly, for reasoning/thinking related settings like a thinking token budget, the engine
> needs to be aware of when reasoning is done independent of the frontend parsing the
> reasoning tokens. Therefore, both the `--reasoning-parser` (engine reasoning awareness)
> and the `--dyn-reasoning-parser` (frontend response parsing) flags should be set when using
> `--dyn-chat-processor dynamo` (default).
> Native and Dynamo tool-call parsers remain mutually exclusive.

## What does NOT make sense

| Combination | Why it's wrong |
|---|---|
| `--dyn-chat-processor dynamo` + `--tool-call-parser` | Native and Dynamo tool-call parsers both construct tool calls. Use `--dyn-tool-call-parser`. |
| `--dyn-chat-processor dynamo` + `--reasoning-parser` without `--dyn-reasoning-parser` | The native parser can gate structured output but does not populate Dynamo's `reasoning_content`. Configure both reasoning parsers. |
| `--dyn-chat-processor vllm`/`sglang` + `--dyn-tool-call-parser` / `--dyn-reasoning-parser` | The `--dyn-` flags only drive Dynamo's native parser; an engine processor reads its own `--tool-call-parser` / `--reasoning-parser`. |
| `--dyn-chat-processor vllm`/`sglang` on TRT-LLM | TRT-LLM engine fallback is a work in progress. Use the default `dynamo` processor. |
| Assuming parser names match across registries | Names can differ — e.g. Dynamo `deepseek_v3` vs vLLM/SGLang `deepseekv3`, Dynamo `nemotron3` vs vLLM `nemotron_v3`. Resolve each flag against its own registry. |

## Examples

Default (Dynamo-native) — the common case. The same `--dyn-*` flags work on every backend; pick one worker. The chat processor defaults to `dynamo`, so the frontend flag is optional:

```bash
# Frontend — chat processor defaults to `dynamo`, so these two are identical:
python -m dynamo.frontend
python -m dynamo.frontend --dyn-chat-processor dynamo

# Workers select the Dynamo parsers. vLLM and SGLang also select their native
# reasoner when reasoning can precede structured output:
python -m dynamo.vllm   --model Qwen/Qwen3-0.6B \
  --dyn-tool-call-parser hermes --reasoning-parser qwen3 --dyn-reasoning-parser qwen3
python -m dynamo.sglang --model Qwen/Qwen3-0.6B \
  --dyn-tool-call-parser hermes --reasoning-parser qwen3 --dyn-reasoning-parser qwen3
python -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
  --dyn-tool-call-parser hermes --dyn-reasoning-parser qwen3
```

Engine fallback — only when Dynamo lacks a parser for your model. Supported on vLLM and SGLang (not TRT-LLM); the parser flags go on the **frontend** and use the engine's own parser names:

```bash
# vLLM chat processor — frontend carries the parser flags, then launch the worker:
python -m dynamo.frontend --dyn-chat-processor vllm   --tool-call-parser hermes  --reasoning-parser qwen3
python -m dynamo.vllm   --model Qwen/Qwen3-0.6B

# SGLang chat processor
python -m dynamo.frontend --dyn-chat-processor sglang --tool-call-parser qwen25  --reasoning-parser qwen3
python -m dynamo.sglang --model Qwen/Qwen3-0.6B
```

## Parser names and per-stage details

- Tool calling: [Tool Call Parsing (Dynamo)](README.md) (native parser names).
- Reasoning: [Reasoning Parsing (Dynamo)](../reasoning/README.md) (native parser names).
- Engine fallback (vLLM / SGLang): [Parser Engine Fallback](engine-fallback.md).
- Engine processors: [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) and [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md).
- Every frontend flag: [Frontend Configuration Reference](../components/frontend/configuration.md).
