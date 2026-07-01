---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Set up KV-Aware Routing
subtitle: Enable the KV router in a DynamoGraphDeployment so requests land on the worker most likely to have the prompt's prefix cached.
---

KV-aware routing sends each request to the worker most likely to already hold the prompt's KV cache prefix, improving Time To First Token (TTFT) and throughput over round-robin. Turning it on in a DynamoGraphDeployment (DGD) takes two parts: switch the **Frontend** into KV mode, and have the **workers** publish KV cache events so the router knows what each worker has cached.

This is a [how-to](dgd-guide.md) for an existing deployment. For the routing cost model and concepts, see [Routing Concepts](../components/router/router-concepts.md); for the full flag and env reference, see the [Router Guide](../components/router/router-guide.md).

## Part 1: Put the Frontend in KV mode

Set `--router-mode kv` on the Frontend container, or the equivalent `DYN_ROUTER_MODE=kv` environment variable:

```yaml
spec:
  components:
  - name: Frontend
    type: frontend
    podTemplate:
      spec:
        containers:
        - name: main
          command:
          - python3
          - -m
          - dynamo.frontend
          args:
          - --router-mode
          - kv
```

That alone gives you cache-aware routing using load signals. To make routing decisions from real cache contents, add Part 2.

## Part 2: Publish KV events from the workers

For the router to track which blocks each worker holds, workers must publish KV cache events. On a vLLM worker, add `--kv-events-config`:

```yaml
  - name: VllmPrefillWorker
    type: prefill
    podTemplate:
      spec:
        containers:
        - name: main
          command:
          - python3
          - -m
          - dynamo.vllm
          args:
          - --model
          - Qwen/Qwen3-32B
          - --kv-events-config
          - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

This is the half that the [Router Guide](../components/router/router-guide.md) does not show. Without it, the router falls back to load-only decisions even in `kv` mode.

The Frontend and worker snippets above are drawn from the [disagg-kv-router recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml), where six prefill workers publish KV events and the Frontend routes across them.

## Tuning knobs

Set these as Frontend `args` (or the `DYN_*` env equivalents) to bias the router's cost model:

| Argument | Env | Effect |
|---|---|---|
| `--router-reset-states` | — | Clear cached routing state on Frontend restart |
| `--kv-overlap-score-weight` | `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` | Weight given to prefix-cache overlap when scoring workers |
| `--router-prefill-load-scale` | `DYN_ROUTER_PREFILL_LOAD_SCALE` | Weight given to prompt-side load |
| `--no-router-kv-events` | `DYN_ROUTER_USE_KV_EVENTS=false` | Disable event tracking and route on load only |

## Routing with disaggregated serving

In a disaggregated graph, the router operates over prefill and decode workers separately. The prefill workers publish KV events (Part 2) and the router selects among them; the internal prefill router activates automatically. See [Router with Disaggregated Serving](../components/router/router-disaggregated-serving.md).

## Related pages

- [Router Guide](../components/router/router-guide.md) — deployment modes, full CLI and env reference.
- [Routing Concepts](../components/router/router-concepts.md) — cost model and worker selection.
- [Router with Disaggregated Serving](../components/router/router-disaggregated-serving.md) — prefill/decode routing.
