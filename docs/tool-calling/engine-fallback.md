---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parser Engine Fallback
subtitle: Use upstream vLLM or SGLang tool-call and reasoning parsers when Dynamo does not ship one
---

When Dynamo's registry does not list a tool-call or reasoning parser for your model, fall back to the upstream engine's parser via a **chat-processor swap**, which keeps frontend tokenization and KV routing.

For the Dynamo-native default path, see [Tool Call Parsing (Dynamo)](README.md) and [Reasoning Parsing (Dynamo)](../reasoning/README.md).

> [!IMPORTANT]
> How `--dyn-chat-processor` combines with the parser flags — and which combinations are invalid (engine fallback supports disaggregated serving on vLLM and SGLang; TRT-LLM engine fallback is a work in progress) — is documented once in [Parser Configuration](parser-configuration.md). Read that first; this page covers only the engine-fallback specifics.

## Configuration

Engine fallback runs parsing in the engine's own Python frontend. Select it with `--dyn-chat-processor vllm` or `sglang`, then name the parser with the engine's **frontend** flags:

- `--tool-call-parser <name>` — the engine's tool-call parser
- `--reasoning-parser <name>` — the engine's reasoning parser

These are distinct from the Dynamo-native `--dyn-tool-call-parser` / `--dyn-reasoning-parser` (which go on the worker). The accepted names come from the engine's registry and may differ from Dynamo's — e.g. vLLM `nemotron_v3` vs Dynamo `nemotron3`, SGLang `deepseekv3` vs Dynamo `deepseek_v3`.

## Examples

Engine fallback puts the parser flags on the **Frontend** and leaves the worker clean. Set `--dyn-chat-processor` to match the worker's backend (`vllm` or `sglang`), and use the engine's own parser names.

**vLLM chat processor:**

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-vllm-fallback
spec:
  components:
  - name: Frontend            # frontend carries the engine-fallback parser flags
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

**SGLang chat processor** — swap the Frontend to `--dyn-chat-processor sglang` with the SGLang parser name (`qwen25`) and run a `dynamo.sglang` worker:

```yaml
  - name: Frontend
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
          - sglang
          - --tool-call-parser
          - qwen25
          - --reasoning-parser
          - qwen3
  - name: SGLangWorker
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
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3-0.6B
          - --served-model-name
          - Qwen/Qwen3-0.6B
```

> [!TIP]
> If a tool call or reasoning split comes back wrong, add `"logprobs": true` to a single repro request and share the response. See [Troubleshooting Tool Calls](troubleshooting.md) for what to capture.

## See Also

- [Parser Configuration](parser-configuration.md) -- how the chat-processor and parser flags combine, and which combinations are invalid (start here)
- [Tool Call Parsing (Dynamo)](README.md) -- Dynamo-native tool-call parser names
- [Reasoning Parsing (Dynamo)](../reasoning/README.md) -- Dynamo-native reasoning parser names
- [vLLM Chat Processor](../backends/vllm/vllm-chat-processor.md) -- vLLM chat-processor details
- [SGLang Chat Processor](../backends/sglang/sglang-chat-processor.md) -- SGLang chat-processor details
- [Frontend Configuration Reference](../components/frontend/configuration.md) -- Full CLI flag reference
