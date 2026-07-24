---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reasoning Parsing (Dynamo)
subtitle: Configure Dynamo's built-in reasoning parsers for models that emit thinking content
---

Some models emit reasoning separately from their final response. Dynamo can split that output into `reasoning_content` and normal assistant `content` by configuring `--dyn-reasoning-parser` on the backend worker. This page covers Dynamo-native parsing; if your model is not listed, use [engine fallback](../tool-calling/chat-processors.mdx), which also explains valid combinations of `--dyn-reasoning-parser`, `--dyn-chat-processor`, and `--dyn-tool-call-parser`.

## Configure Reasoning Parsing

<Steps>
  <Step title="Launch Dynamo backend">
    Select `--dyn-reasoning-parser` from the supported list below. The backend can be SGLang, TensorRT-LLM, vLLM, or another installed backend.

    For vLLM structured output or SGLang required/named tool choice, also configure the engine's native `--reasoning-parser`. The native parser controls when the grammar starts; Dynamo's parser populates `reasoning_content`. Parser names can differ between registries.

    To inspect the available worker flags, run:

    ```bash
    python -m dynamo.<backend> --help
    ```

    The following example starts an SGLang worker for Qwen3.5:

    ```bash
    python -m dynamo.sglang \
      --model Qwen/Qwen3.5-4B \
      --dyn-tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --dyn-reasoning-parser qwen3
    ```

    <details>
    <summary>Some models need both parsers configured together. Common pairings include:</summary>

    - `openai/gpt-oss-*`: `--dyn-tool-call-parser harmony --dyn-reasoning-parser gpt_oss`
    - `deepseek-ai/DeepSeek-V4-*`: `--dyn-tool-call-parser deepseek_v4 --dyn-reasoning-parser deepseek_v4`
    - `zai-org/GLM-4.7`: `--dyn-tool-call-parser glm47 --dyn-reasoning-parser glm45`
    - `moonshotai/Kimi-K2.5*` / Kimi K2.6 format-compatible outputs: `--dyn-tool-call-parser kimi_k2 --dyn-reasoning-parser kimi_k25`
    - `google/gemma-4-*` thinking models: `--dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4 --custom-jinja-template examples/chat_templates/gemma4_tool.jinja`
    - `Qwen/Qwen3.5*`: `--dyn-tool-call-parser qwen3_coder --dyn-reasoning-parser qwen3`
    - MiniMax M2 style outputs: `--dyn-tool-call-parser minimax_m2 --dyn-reasoning-parser minimax_m2`
    - MiniMax M3 style outputs: `--dyn-tool-call-parser minimax_m3 --dyn-reasoning-parser minimax_m3`

    <Warning>
    `minimax_append_think` is deprecated for MiniMax M2 tool-calling deployments. Use `--dyn-reasoning-parser minimax_m2` with `--dyn-tool-call-parser minimax_m2` so Dynamo can separate reasoning and pass MiniMax XML tool calls to the tool parser.
    </Warning>

    </details>

    Reasoning parsing happens before tool call parsing. See [Tool Call Parsing (Dynamo)](../tool-calling/README.mdx) for supported `--dyn-tool-call-parser` values.
  </Step>

  <Step title="Start the Frontend">
    Start the Dynamo frontend in another terminal:

    ```bash
    python -m dynamo.frontend
    ```
  </Step>

  <Step title="Wait for Readiness">
    Wait for the frontend health endpoint to return a successful response:

    ```bash
    curl --fail http://localhost:8000/health
    ```
  </Step>

  <Step title="Send a Reasoning Request">
    Send an OpenAI-compatible chat completion request:

    ```bash
    curl -s http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [
          {
            "role": "user",
            "content": "If a train leaves at 3pm going 60 mph and another leaves at 4pm going 80 mph, when does the second catch up?"
          }
        ]
      }'
    ```

    Dynamo places the chain of thought in `reasoning_content` and the user-facing answer in `content`.

    <Accordion title="View an example response">
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
    </Accordion>
  </Step>
</Steps>

## Supported Reasoning Parsers

Choose a model family, then expand the matching model option to see its parser name and configuration details.

The **Upstream name** column shows where the vLLM or SGLang parser name differs from Dynamo's. This is relevant for engine fallback and when configuring the native structured-output reasoning gate. A blank upstream column means the same name works everywhere. `Dynamo-only` means no upstream parser exists for this format.

Parsers marked **Force-reasoning: Yes** emit reasoning content from token one without requiring an explicit opening tag such as `<think>`. All others require the opening tag in the model output.

<AccordionGroup>
  <Accordion title="Kimi">
    <AccordionGroup>
      <Accordion title="Kimi K2.5 / Kimi K2.6 format-compatible thinking models">
        **Parser:** `kimi_k25`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** `<think>...</think>` with force-reasoning
      </Accordion>

      <Accordion title="Kimi K2 Instruct / Thinking with Unicode delimiters">
        **Parser:** `kimi`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `◁think▷...◁/think▷`
      </Accordion>
    </AccordionGroup>

    <Warning>
    Kimi K2.7 may ignore `chat_template_kwargs.thinking=false` and continue to generate reasoning. Dynamo can separate emitted reasoning when a compatible parser is configured, but it cannot force the model to disable reasoning. Treat the request flag as best-effort for Kimi K2.7.
    </Warning>
  </Accordion>

  <Accordion title="MiniMax">
    <AccordionGroup>
      <Accordion title="MiniMax M2 / M2.5 / M2.7">
        **Parser:** `minimax_m2`

        **Upstream name:** vLLM: `minimax_m2`

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** `<think>...</think>` with force-reasoning
      </Accordion>

      <Accordion title="MiniMax M3">
        **Parser:** `minimax_m3`

        **Upstream name:** vLLM: `minimax_m3`

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `<mm:think>...</mm:think>`; recovers a prompt-prefilled opener
      </Accordion>

      <Accordion title="MiniMax M2 / M2.1 (legacy)">
        **Parser:** `minimax_append_think` <Badge intent="error" minimal>Deprecated</Badge>

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** Legacy pass-through with an implicit `<think>` opener. Use `minimax_m2` for MiniMax M2 tool-calling deployments.
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="DeepSeek">
    <AccordionGroup>
      <Accordion title="DeepSeek V4 Pro / Flash">
        **Parser:** `deepseek_v4`

        **Upstream name:** vLLM: `deepseek_v4`; SGLang: `deepseek-v4`

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `<think>...</think>`. Aliases: `deepseek-v4`, `deepseekv4`
      </Accordion>

      <Accordion title="DeepSeek R1, DeepSeek V3.1, and DeepSeek V3.2">
        **Parser:** `deepseek_r1`

        **Upstream name:** Same name across Dynamo, vLLM, and SGLang

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** Pass explicitly for V3.1/V3.2 (no alias)
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Qwen and QwQ">
    <AccordionGroup>
      <Accordion title="Qwen3.5, QwQ-32B, Qwen3-Think, and Qwen3-Coder">
        **Parser:** `qwen3`

        **Upstream name:** Same name across Dynamo, vLLM, and SGLang

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `<think>...</think>`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="GLM">
    <AccordionGroup>
      <Accordion title="GLM-4.5 and GLM-4.7">
        **Parser:** `glm45`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** Alias for `nemotron_deci`. `<think>...</think>`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Nemotron">
    <AccordionGroup>
      <Accordion title="Nemotron-3 / Mini">
        **Parser:** `nemotron3`

        **Upstream name:** vLLM: `nemotron_v3`

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** Alias for `deepseek_r1`. Also accepts `nemotron_v3`
      </Accordion>

      <Accordion title="Nemotron-Super / -Ultra / -Deci and Llama-Nemotron">
        **Parser:** `nemotron_deci`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `<think>...</think>`
      </Accordion>

      <Accordion title="Nemotron-Nano">
        **Parser:** `nemotron_nano`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** Alias for `deepseek_r1`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Gemma">
    <AccordionGroup>
      <Accordion title="Google Gemma 4 thinking models">
        **Parser:** `gemma4`

        **Upstream name:** vLLM: `gemma4`

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `<\|channel>thought\n...<channel\|>` with `thought\n` role label stripped. Aliases: `gemma-4`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="gpt-oss">
    <AccordionGroup>
      <Accordion title="gpt-oss-20b and gpt-oss-120b">
        **Parser:** `gpt_oss`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** Harmony channel reasoning format
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Mistral">
    <AccordionGroup>
      <Accordion title="Magistral">
        **Parser:** `mistral`

        **Upstream name:** Same name across Dynamo, vLLM, and SGLang

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** `[THINK]...[/THINK]`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Granite">
    <AccordionGroup>
      <Accordion title="IBM Granite 3.x / Granite 3.2 language models">
        **Parser:** `granite`

        **Upstream name:** Same name across Dynamo, vLLM, and SGLang

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** `Here's my thought process:` / `Here's my response:`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Step">
    <AccordionGroup>
      <Accordion title="Step-3 / Step-3-Reasoning">
        **Parser:** `step3`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="success" minimal>Yes</Badge>

        **Notes:** `<think>...</think>`
      </Accordion>
    </AccordionGroup>
  </Accordion>

  <Accordion title="Generic">
    <AccordionGroup>
      <Accordion title="Generic chain-of-thought models">
        **Parser:** `basic`

        **Upstream name:** <Badge intent="note" minimal>Dynamo-only</Badge>

        **Force-reasoning:** <Badge intent="note" minimal>No</Badge>

        **Notes:** Plain `<think>...</think>`
      </Accordion>
    </AccordionGroup>
  </Accordion>
</AccordionGroup>
