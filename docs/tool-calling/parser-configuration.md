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
- The **bare `--tool-call-parser` / `--reasoning-parser` flags pair with `vllm` / `sglang`** and go on the **frontend**.

Tool calling and reasoning are independent — set one, the other, or both — but always from the same family as your chat processor. You never mix the two families.

## What does NOT make sense

| Combination | Why it's wrong |
|---|---|
| `--dyn-chat-processor dynamo` + `--tool-call-parser` / `--reasoning-parser` | The bare flags drive the engine-fallback path; the default Dynamo path uses the `--dyn-` flags. Use `--dyn-tool-call-parser` / `--dyn-reasoning-parser`. |
| `--dyn-chat-processor vllm`/`sglang` + `--dyn-tool-call-parser` / `--dyn-reasoning-parser` | The `--dyn-` flags only drive Dynamo's native parser; an engine processor reads its own `--tool-call-parser` / `--reasoning-parser`. |
| `--dyn-chat-processor vllm`/`sglang` on TRT-LLM | TRT-LLM engine fallback is a work in progress. Use the default `dynamo` processor. |
| Reusing a parser name across families | The registries differ — e.g. Dynamo `deepseek_v3` vs vLLM/SGLang `deepseekv3`, Dynamo `nemotron3` vs vLLM `nemotron_v3`. Use the name from the registry that matches your chat processor. |

## Examples

**Default (Dynamo-native) — the common case.** The chat processor defaults to `dynamo`, so the Frontend needs no extra flags; the parsers go on the **worker** as `--dyn-*` args. The same worker flags work on vLLM, SGLang, or TRT-LLM — pick one worker:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-parsers
spec:
  components:
  - name: Frontend            # chat processor defaults to `dynamo` — no args needed
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
  - name: VllmWorker          # worker selects the Dynamo parsers
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-0.6B
          - --dyn-tool-call-parser
          - hermes
          - --dyn-reasoning-parser
          - qwen3
```

To set the chat processor explicitly, add `DYN_CHAT_PROCESSOR: dynamo` to the Frontend `env:` (equivalent to the default).

**Engine fallback — only when Dynamo lacks a parser for your model.** Supported on vLLM and SGLang (not TRT-LLM). The parser flags move to the **Frontend** `args:` and use the engine's own parser names; the worker carries no parser flags. Set `--dyn-chat-processor vllm` (or `sglang`) on the Frontend to match the worker's backend:

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-engine-fallback
spec:
  components:
  - name: Frontend            # engine fallback: parser flags live here
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          command:
          - python3
          - -m
          - dynamo.frontend
          args:
          - --dyn-chat-processor
          - vllm
          - --tool-call-parser
          - hermes
          - --reasoning-parser
          - qwen3
  - name: VllmWorker          # no parser flags on the worker
    type: worker
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          envFrom:
          - secretRef:
              name: hf-token-secret
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-0.6B
```

For SGLang engine fallback, set `--dyn-chat-processor sglang` on the Frontend (with the SGLang parser names, e.g. `qwen25`) and run a `dynamo.sglang` worker.

## Parser names and per-stage details

- Tool calling: [Tool Call Parsing (Dynamo)](README.md) (native parser names).
- Reasoning: [Reasoning Parsing (Dynamo)](../reasoning/README.md) (native parser names).
- Engine fallback (vLLM / SGLang): [Parser Engine Fallback](engine-fallback.md).
- Engine processors: [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) and [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md).
- Every frontend flag: [Frontend Configuration Reference](../components/frontend/configuration.md).
