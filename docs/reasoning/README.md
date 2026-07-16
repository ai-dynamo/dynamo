---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning Parsing (Dynamo)
subtitle: Configure Dynamo's built-in reasoning parsers for models that emit thinking content
---

Some models emit reasoning or thinking content separately from their final response. Dynamo can split that output into `reasoning_content` and normal assistant content by configuring `--dyn-reasoning-parser` on the backend worker.

This page covers parser names for the default Dynamo-native path. If Dynamo
does not list a parser for your model, see
[Parser Engine Fallback](../tool-calling/engine-fallback.md). For how
`--dyn-reasoning-parser` combines with `--dyn-chat-processor` and
`--dyn-tool-call-parser` (and which combinations are invalid), see
[Parser Configuration](../tool-calling/parser-configuration.md).

## Prerequisites

To enable reasoning parsing, launch the backend worker with:

- `--dyn-reasoning-parser`: select the reasoning parser from the supported list below

For vLLM structured output, or SGLang required/named tool choice, also configure
the engine's native `--reasoning-parser`. It controls when the grammar starts;
Dynamo's parser populates `reasoning_content`. Parser names can differ between
registries.

```bash
# <backend> can be sglang, trtllm, vllm, etc. based on your installation
python -m dynamo.<backend> --help
```

> [!TIP]
> Some models need both a reasoning parser and a tool call parser. For supported tool call parser names, see [Tool Call Parsing (Dynamo)](../tool-calling/README.md).

## Deployment-Level Thinking Default

Set `--dyn-default-thinking-mode` on a backend worker to choose the thinking
mode used when a request does not provide one. The worker publishes the value
through model runtime metadata, and the frontend applies it while rendering the
chat template. The option works with vLLM, SGLang, and TensorRT-LLM workers.

| CLI argument | Environment variable | Values | Default |
|---|---|---|---|
| `--dyn-default-thinking-mode` | `DYN_DEFAULT_THINKING_MODE` | `enabled`, `disabled` | Unset |

For example, make thinking disabled by default for a Qwen3 deployment:

```bash
python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --dyn-default-thinking-mode disabled
```

You can configure the same value through the environment:

```bash
DYN_DEFAULT_THINKING_MODE=disabled \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

Use the same setting on every worker that serves the model. When the option is
unset, Dynamo preserves the model or chat template's native default.

Thinking controls use this precedence order:

1. An explicit request control
2. `--dyn-default-thinking-mode` or `DYN_DEFAULT_THINKING_MODE`
3. The model or chat template's native default

Explicit request controls include root-level `thinking`, top-level
`reasoning_effort`, and the `thinking`, `enable_thinking`, `thinking_mode`, or
`reasoning_effort` fields in `chat_template_args` or
`chat_template_kwargs`.

The deployment option intentionally accepts only `enabled` and `disabled`.
Models that support adaptive thinking can still select it per request:

```json
{
  "model": "MiniMaxAI/MiniMax-M3",
  "messages": [{"role": "user", "content": "Explain this result."}],
  "thinking": {"type": "adaptive"}
}
```

Dynamo normalizes this request to `thinking_mode=adaptive`. Because adaptive is
an explicit request control, it takes precedence over an `enabled` or
`disabled` deployment default.

> [!NOTE]
> This option controls chat-template rendering. It does not force the inference
> engine or model to follow the requested mode. Models that ignore their
> template's thinking control may still emit reasoning.

## Supported Reasoning Parsers

The table below lists the currently supported reasoning parsers in Dynamo's registry. The
**Upstream name** column shows where the vLLM or SGLang parser name differs
from Dynamo's. This is relevant for engine fallback and when configuring the
native structured-output reasoning gate. A blank upstream
column means the same name works everywhere. `Dynamo-only` means no upstream
parser exists for this format.

Parsers marked **force-reasoning** emit reasoning content from token one
without requiring an explicit opening tag (`<think>`, etc.). All others
require the opening tag to be present in the model output.

| Parser Name | Models | Upstream name | Force-reasoning | Notes |
|---|---|---|---|---|
| `kimi_k25` | Kimi K2.5 / Kimi K2.6 format-compatible thinking models | Dynamo-only | Yes | `<think>...</think>` with force-reasoning |
| `kimi` | Kimi K2 Instruct / Thinking with Unicode delimiters | Dynamo-only | No | `◁think▷...◁/think▷` |
| `minimax_m2` | MiniMax M2 / M2.5 / M2.7 | vLLM: `minimax_m2` | Yes | `<think>...</think>` with force-reasoning |
| `minimax_m3` | MiniMax M3 | vLLM: `minimax_m3` | No | `<mm:think>...</mm:think>`; recovers a prompt-prefilled opener |
| `minimax_append_think` | MiniMax M2 / M2.1 | Dynamo-only | No | Deprecated. Legacy pass-through with an implicit `<think>` opener; use `minimax_m2` for MiniMax M2 tool-calling deployments |
| `deepseek_v4` | DeepSeek V4 Pro / Flash | vLLM: `deepseek_v4`; SGLang: `deepseek-v4` | No | `<think>...</think>`. Aliases: `deepseek-v4`, `deepseekv4` |
| `deepseek_r1` | DeepSeek R1, DeepSeek V3.1, DeepSeek V3.2 | | Yes | Pass explicitly for V3.1/V3.2 (no alias) |
| `qwen3` | Qwen3.5, QwQ-32B, Qwen3-Think, Qwen3-Coder | | No | `<think>...</think>` |
| `glm45` | GLM-4.5, GLM-4.7 | Dynamo-only | No | Alias for `nemotron_deci`. `<think>...</think>` |
| `nemotron3` | Nemotron-3 / Mini | vLLM: `nemotron_v3` | Yes | Alias for `deepseek_r1`. Also accepts `nemotron_v3` |
| `nemotron_deci` | Nemotron-Super / -Ultra / -Deci, Llama-Nemotron | Dynamo-only | No | `<think>...</think>` |
| `nemotron_nano` | Nemotron-Nano | Dynamo-only | Yes | Alias for `deepseek_r1` |
| `gemma4` | Google Gemma 4 (thinking models) | vLLM: `gemma4` | No | `<\|channel>thought\n...<channel\|>` with `thought\n` role label stripped. Aliases: `gemma-4` |
| `gpt_oss` | gpt-oss-20b / -120b | Dynamo-only | No | Harmony channel reasoning format |
| `mistral` | Magistral | | Yes | `[THINK]...[/THINK]` |
| `granite` | IBM Granite 3.x / Granite 3.2 language models | | No | `Here's my thought process:` / `Here's my response:` |
| `step3` | Step-3 / Step-3-Reasoning | Dynamo-only | Yes | `<think>...</think>` |
| `basic` | Generic CoT models | Dynamo-only | No | Plain `<think>...</think>` |

## Model-Specific Limitations

> [!WARNING]
> Kimi K2.7 may ignore `chat_template_kwargs.thinking=false` and continue to
> generate reasoning. Dynamo can separate emitted reasoning when a compatible
> parser is configured, but it cannot force the model to disable reasoning.
> Treat the request flag as best-effort for Kimi K2.7.

## Common Parser Pairings

Some models need both parsers configured together. Common pairings include:

- `openai/gpt-oss-*`: `--dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss`
- `deepseek-ai/DeepSeek-V4-*`: `--dyn-tool-call-parser deepseek_v4 --dyn-reasoning-parser deepseek_v4`
- `zai-org/GLM-4.7`: `--dyn-tool-call-parser glm47 --dyn-reasoning-parser glm45`
- `moonshotai/Kimi-K2.5*` / Kimi K2.6 format-compatible outputs: `--dyn-tool-call-parser kimi_k2 --dyn-reasoning-parser kimi_k25`
- `google/gemma-4-*` thinking models: `--dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4 --custom-jinja-template examples/chat_templates/gemma4_tool.jinja`
- `Qwen/Qwen3.5*`: `--dyn-tool-call-parser qwen3_coder --dyn-reasoning-parser qwen3`
- MiniMax M2 style outputs: `--dyn-tool-call-parser minimax_m2 --dyn-reasoning-parser minimax_m2`
- MiniMax M3 style outputs: `--dyn-tool-call-parser minimax_m3 --dyn-reasoning-parser minimax_m3`

> [!WARNING]
> `minimax_append_think` is deprecated for MiniMax M2 tool-calling deployments.
> Use `--dyn-reasoning-parser minimax_m2` with `--dyn-tool-call-parser minimax_m2`
> so Dynamo can separate reasoning and pass MiniMax XML tool calls to the tool
> parser.

## Tool Calling Interplay

Reasoning parsing happens before tool call parsing. If a model emits both reasoning content and tool calls, configure both parsers so Dynamo can first separate reasoning text and then parse tool calls from the remaining assistant output.

## Examples

### Launch Dynamo Frontend and Backend

```bash
# launch backend worker (or dynamo.vllm)
python -m dynamo.sglang --model Qwen/Qwen3.5-4B --dyn-tool-call-parser qwen3_coder --reasoning-parser qwen3 --dyn-reasoning-parser qwen3

# launch frontend worker
python -m dynamo.frontend
```

### Reasoning Request Example

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "If a train leaves at 3pm going 60 mph and another leaves at 4pm going 80 mph, when does the second catch up?"}]
  }'
```

Dynamo splits the model output so the chain-of-thought lands in
`reasoning_content` and the user-facing answer stays in `content`:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "The first train has a 1-hour head start at 60 mph, so it is 60 miles ahead at 4pm. The second train closes the gap at 80 - 60 = 20 mph. 60 / 20 = 3 hours after 4pm.",
        "content": "The second train catches up at 7pm."
      },
      "finish_reason": "stop"
    }
  ]
}
```
