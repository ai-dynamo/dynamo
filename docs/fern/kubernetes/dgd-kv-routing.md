---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Set up KV-Aware Routing
subtitle: Enable the KV router in a DynamoGraphDeployment so requests land on the worker most likely to have the prompt's prefix cached.
---

KV-aware routing sends each request to the worker most likely to already hold the prompt's KV cache prefix, improving Time To First Token (TTFT) and throughput over round-robin. Turning it on in a DynamoGraphDeployment (DGD) takes two steps: switch the **Frontend** into KV mode, and have the **workers** publish KV cache events so the router knows what each worker has cached.

This is a [how-to](dgd-guide.md) for an existing deployment. For the routing cost model and concepts, see [Routing Concepts](../components/router/router-concepts.md); for the full flag and env reference, see the [Router Guide](../components/router/router-guide.md).

<Steps toc={true} tocDepth={2}>

<Step title="Put the Frontend in KV mode">

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

That alone gives you cache-aware routing using load signals. To make routing decisions from real cache contents, complete the next step.

</Step>

<Step title="Publish KV events from the workers">

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

</Step>

</Steps>

## Tuning knobs

The KV router scores each worker as `prefill_load_scale * adjusted_prefill_blocks + decode_blocks`, where cache-overlap credit subtracts from the prefill load. Two knobs shift that balance; set them as Frontend `args` (or the `DYN_*` env equivalents). Start with the defaults and adjust only if you have a measured TTFT or ITL problem.

For the flag/env/default reference, see the [Frontend Configuration Reference](../components/frontend/frontend-config-reference.mdx#router); for the full cost-model detail and every related flag, see [Configuration and Tuning](../components/router/router-configuration.md#tuning-guidelines).

### Cache-overlap credit

`--router-kv-overlap-score-credit` (env `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT`) is the primary cache-reuse knob. It credits device-local prefix overlap against a worker's prefill load, biasing requests toward workers that already hold the prompt's prefix.

- **Range:** `0.0` to `1.0`. **Default:** `1.0`.
- **Raise toward `1.0`** to prioritize cache reuse and lower TTFT — the router more aggressively co-locates requests that share a prefix.
- **Lower toward `0.0`** to spread load more evenly and lower ITL, at the cost of more redundant prefills. `0.0` ignores prefix caches entirely and skips building the local indexer (equivalent to load-only routing).

Most deployments should leave this at `1.0`. Lower it only when cache-rich workers are getting overloaded while others sit idle.

### Prompt-side load weight

`--router-prefill-load-scale` (env `DYN_ROUTER_PREFILL_LOAD_SCALE`) scales the prompt-side prefill load after overlap credit is applied, setting how much prompt work counts relative to decode-side block load.

- **Minimum:** `0.0` (ignore prompt-side load). **Default:** `1.0`. No hard maximum — values above `1.0` weight prefill more heavily.
- **Raise above `1.0`** when long prompts are saturating workers and you want the router to steer new requests away from workers already doing heavy prefill.
- **Lower below `1.0`** when decode-side pressure dominates and you want routing driven mainly by active decode blocks.

### Route on load only

`--no-router-kv-events` (env `DYN_ROUTER_USE_KV_EVENTS=false`) disables event tracking; the router predicts cache state from its own routing decisions with TTL-based expiration instead of consuming real KV events. Use it only when you are not confident the backend emits KV events correctly. `--router-reset-states` clears cached routing state on Frontend restart.

## Routing with disaggregated serving

In a disaggregated graph, the router operates over prefill and decode workers separately. The prefill workers publish KV events (the second step above) and the router selects among them; the internal prefill router activates automatically. See [Router with Disaggregated Serving](../components/router/router-disaggregated-serving.md).

## Related pages

- [Router Guide](../components/router/router-guide.md) — deployment modes, full CLI and env reference.
- [Routing Concepts](../components/router/router-concepts.md) — cost model and worker selection.
- [Router with Disaggregated Serving](../components/router/router-disaggregated-serving.md) — prefill/decode routing.
