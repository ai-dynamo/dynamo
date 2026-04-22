---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Chat Processor Options
subtitle: Choose the right preprocessing pipeline for tool calling, reasoning, and tokenization
---

Dynamo splits work between a **frontend** process (HTTP server, tokenization,
routing, parsing) and one or more **worker** processes (the engine running the
model). Several CLI flags control which code path handles chat template
rendering, tool-call parsing, and reasoning-content separation. This page
explains the available configurations, when to use each, and how they interact
with KV cache routing.

For the list of individual parser names, see
[Tool Calling](tool-calling.md) and [Reasoning](reasoning.md).

## Configurations

There are five supported configurations. Each is set at startup -- Dynamo does
not switch between them per request.

| | Frontend flags | Worker flags | KV routing | Notes |
|---|---|---|---|---|
| **A** Dynamo-native (default) | `--dyn-chat-processor dynamo` | `--dyn-tool-call-parser <name>` `--dyn-reasoning-parser <name>` | Yes | Rust preprocessor. Lowest latency. |
| **B** vLLM chat processor | `--dyn-chat-processor vllm` `--tool-call-parser <name>` `--reasoning-parser <name>` | *(none)* | Yes | Delegates to vLLM's Python preprocessor. |
| **C** SGLang chat processor | `--dyn-chat-processor sglang` `--tool-call-parser <name>` `--reasoning-parser <name>` | *(none)* | Yes | Delegates to SGLang's Python preprocessor. See [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md). |
| **D** vLLM tokenizer delegation | `--router-mode round-robin` | `--use-vllm-tokenizer` | No | Engine-side tokenization. Day-0 model fallback. |
| **E** SGLang tokenizer delegation | `--router-mode round-robin` | `--use-sglang-tokenizer` | No | **Deprecated** -- use option C instead. |

> [!NOTE]
> Although `dynamo` is the default for `--dyn-chat-processor`, specifying it
> explicitly in launch scripts makes the choice visible in logs and support
> diagnostics.

## Flag reference

### `--dyn-chat-processor {dynamo | vllm | sglang}`

Frontend flag (default `dynamo`). Selects the chat processor that renders
templates, tokenizes, and dispatches parsing.

- `dynamo` -- Rust preprocessor. Parser names come from Dynamo's registry
  (see [Tool Calling](tool-calling.md) and [Reasoning](reasoning.md)).
- `vllm` -- vLLM's Python preprocessor. Parser names come from vLLM's
  registry, which may differ from Dynamo's.
- `sglang` -- SGLang's Python preprocessor. Parser names come from SGLang's
  registry. See [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md).

### `--dyn-tool-call-parser <name>` / `--dyn-reasoning-parser <name>`

Worker flags. Names from Dynamo's parser registry. Only effective under
`--dyn-chat-processor dynamo` (option A); silently ignored under other chat
processors.

The flags are declared on the worker CLI, but the parser runs on the frontend --
the name propagates via model metadata. For supported names, see
[Tool Calling](tool-calling.md) and [Reasoning](reasoning.md).

### `--tool-call-parser <name>` / `--reasoning-parser <name>`

Frontend flags (no `--dyn-` prefix). Names from the upstream engine's registry.
Only accepted when paired with the matching chat processor:

- Under `--dyn-chat-processor vllm`: accepted. Use vLLM parser names.
- Under `--dyn-chat-processor sglang`: accepted. Use SGLang parser names.
- Under `--dyn-chat-processor dynamo`: **rejected at startup** with
  `Unknown arguments specified: ...`. Use the `--dyn-*` worker flags instead.

Upstream parser names are pinned to the engine version shipped in the Dynamo
container. They may differ from Dynamo's names for the same model (e.g.,
SGLang uses `deepseekv3` where Dynamo uses `deepseek_v3`).

### `--use-vllm-tokenizer` / `--use-sglang-tokenizer`

Worker flags (boolean). Hand tokenization to the engine instead of the
frontend. The flag must match the engine on the worker.

`--use-sglang-tokenizer` is deprecated. New SGLang deployments should use
`--dyn-chat-processor sglang` (option C) instead. See
[Migration from --use-sglang-tokenizer](../backends/sglang/sglang-chat-processor.md#migration-from---use-sglang-tokenizer).

## Which option should I pick?

1. **Does Dynamo have a parser for your model?** Check the per-model tables in
   [Tool Calling](tool-calling.md) and [Reasoning](reasoning.md). If yes, use
   **option A**. This is the default path: Rust parsing on the frontend,
   KV-routable, lowest latency.

2. **Does the upstream engine have a parser but Dynamo doesn't?** Use
   **option B** (vLLM) or **option C** (SGLang). Still KV-routable.

3. **Is the tokenizer itself the problem** (day-0 model, custom special tokens,
   rope variants)? Use **option D**. KV routing is off; pair with
   `--router-mode round-robin`.

4. **SGLang + day-0 model?** Use **option C** with the appropriate upstream
   parser name. Do not use option E (deprecated).

## Invalid and silently broken combinations

### Rejected at startup

- **`--dyn-chat-processor dynamo` with `--tool-call-parser <name>`** (or
  `--reasoning-parser`). The un-prefixed flags are not recognized under the
  Dynamo chat processor. Use `--dyn-tool-call-parser` on the worker instead.

- **`--tool-call-parser` and `--dyn-tool-call-parser` together** on the same
  SGLang worker. SGLang rejects this: `Cannot use both --tool-call-parser and
  --dyn-tool-call-parser`. Pick one namespace.

- **`--use-vllm-tokenizer` on an SGLang worker** (and vice versa). The flag
  must match the engine.

### Silently broken (no startup error, wrong results)

- **Tokenizer delegation + `--router-mode kv`** -- Options D/E with `kv`
  routing produces prefix-hash mismatches and silent cache misses.

- **`--dyn-tool-call-parser` + `--use-vllm-tokenizer`** on the same vLLM
  worker. The worker bypasses Dynamo's preprocessor while the frontend-side
  parser is still wired up, producing mismatched token streams. No
  mutual-exclusivity check exists today.

## Routing compatibility

`--router-mode kv` needs frontend tokenization to compute prefix-hash routing
keys. Options A, B, and C keep the tokenizer on the frontend and are
KV-routable. Options D and E move tokenization to the worker and are **not**
KV-routable -- pair them with `round-robin` or `random`.

| Option | `kv` routing | `round-robin` / `random` |
|--------|:---:|:---:|
| A (Dynamo-native) | Yes | Yes |
| B (vLLM processor) | Yes | Yes |
| C (SGLang processor) | Yes | Yes |
| D (vLLM tokenizer delegation) | **No** | Yes |
| E (SGLang tokenizer delegation) | **No** | Yes |

## Why each flag exists

- **Frontend tokenization** is required for KV cache routing. The frontend
  needs token IDs to compute prefix-hash routing keys before the request
  reaches a worker. Parser flags on the Rust-native path (option A) co-locate
  with tokenization on the frontend for this reason.

- **Backend tokenization** is a fallback for when frontend tokenization can't
  or shouldn't run: unsupported model, day-0 support, tokenizer edge cases
  (custom special tokens, rope variants). The engine owns the tokenizer in
  this mode, so KV routing drops out.

- **Chat-processor swap** (options B/C) is the middle ground: tokenization
  stays on the frontend (KV-routable), but parsing delegates to the upstream
  engine's Python implementation. This covers models where Dynamo's Rust
  parser hasn't been written yet.

## Parser names by model

For the full list of supported parser names, which models they cover, and
upstream name divergences (relevant for options B and C):

- [Tool Calling](tool-calling.md) -- supported tool call parsers with model
  mappings and upstream name differences
- [Reasoning](reasoning.md) -- supported reasoning parsers with model mappings
  and force-reasoning behavior

## Canonical launch examples

```bash
# A -- Dynamo-native (default).
python -m dynamo.vllm \
  --dyn-tool-call-parser kimi_k2 \
  --dyn-reasoning-parser kimi_k25
python -m dynamo.frontend --dyn-chat-processor dynamo

# B -- vLLM chat-processor (upstream parser names on the frontend).
python -m dynamo.vllm ...
python -m dynamo.frontend \
  --dyn-chat-processor vllm \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1

# C -- SGLang chat-processor.
python -m dynamo.sglang ...
python -m dynamo.frontend \
  --dyn-chat-processor sglang \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k25

# D -- vLLM tokenizer delegation (no KV routing).
python -m dynamo.vllm --use-vllm-tokenizer ...
python -m dynamo.frontend --router-mode round-robin
```

## See Also

- [Tool Calling](tool-calling.md) -- Supported tool call parser names, request examples
- [Reasoning](reasoning.md) -- Supported reasoning parser names, common pairings
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- Option C details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
