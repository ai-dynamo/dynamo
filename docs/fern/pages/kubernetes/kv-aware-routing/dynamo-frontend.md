---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Using the Dynamo Frontend
sidebar-title: Using the Dynamo Frontend
subtitle: Configure the Dynamo Frontend to select the worker most likely to have the prompt prefix cached.
---

In this topology, the Dynamo Frontend receives each request and selects the worker most likely to already hold the prompt's KV cache prefix. Use it when clients send requests directly to a Dynamo Frontend Service. If a Kubernetes Gateway receives requests first, use [GAIE with Dynamo](gateway-api.mdx) instead.

Turning it on in a DynamoGraphDeployment (DGD) takes two steps: switch the **Frontend** into KV mode, and have the **workers** publish KV cache events so the router knows what each worker has cached. This is a [how-to](../model-deployment/deploy-with-dgd.md) for an existing deployment. For the routing cost model and concepts, see [Routing Concepts](../../developer-guide/knowledge-base/modular-components/router/routing-concepts.md); for the full flag and env reference, see the [Router Guide](../../developer-guide/knowledge-base/modular-components/router/router-guide.md).

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

This configuration enables the router. It does not enable worker KV event publication. Complete the
next step to route requests by actual cache contents. If you do not publish events, add
`--no-router-kv-events` for approximate cache prediction. The router does not switch modes
automatically.

</Step>

<Step title="Publish KV events from the workers">

Workers must publish KV cache events so that the router can track their cache blocks. Add the
backend configuration to each aggregated worker. For a standard disaggregated deployment, add it
to each prefill worker.

For vLLM:

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
          - --stream-interval
          - "20"
          - --kv-events-config
          - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
```

For SGLang:

```yaml
  - name: SglangPrefillWorker
    type: prefill
    podTemplate:
      spec:
        containers:
        - name: main
          command:
          - python3
          - -m
          - dynamo.sglang
          args:
          - --model-path
          - Qwen/Qwen3-32B
          - --stream-interval
          - "20"
          - --kv-events-config
          - '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
```

Engine event publication requires `--kv-events-config`. The worker-local indexer does not
activate the engine publisher. The engine publishes raw events to the Dynamo worker. The worker
updates its worker-local indexer before it relays normalized updates to the router. This relay uses
Dynamo's event plane. For workers that share a pod or network namespace, use a unique endpoint
port.

The backend JSON schemas differ. vLLM requires `"enable_kv_cache_events":true`. It can infer the
ZMQ publisher and default endpoint. SGLang requires `"publisher":"zmq"` and an explicit
`"endpoint"`. The `"topic"` field is optional. Do not copy the vLLM-only
`{"enable_kv_cache_events":true}` form into an SGLang worker.

`--stream-interval 20` is a starting value for host efficiency. KV routing does not require it. The
value reduces host-side engine output processing and Dynamo bridge crossings. The tradeoff is
coarser stream updates. See the [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/reference-guide.md#recommended-stream-interval)
and [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md#recommended-stream-interval)
reference guides.

The vLLM snippets above come from the [disagg-kv-router recipe](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml).
In this recipe, six prefill workers publish KV events and the Frontend routes across them.

</Step>

</Steps>

## Tuning Knobs

The KV router scores each worker as `prefill_load_scale * adjusted_prefill_blocks + decode_blocks`, where cache-overlap credit subtracts from the prefill load. Two knobs shift that balance; set them as Frontend `args` (or the `DYN_*` env equivalents). Start with the defaults and adjust only if you have a measured TTFT or ITL problem.

For the flag/env/default reference, see the [Frontend Configuration Reference](../../reference/components/frontend-configuration.mdx#router); for the full cost-model detail and every related flag, see [Configuration and Tuning](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md#tuning-guidelines).

### Cache-Overlap Credit

`--router-kv-overlap-score-credit` (env `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT`) is the primary cache-reuse knob. It credits device-local prefix overlap against a worker's prefill load, biasing requests toward workers that already hold the prompt's prefix.

- **Range:** `0.0` to `1.0`. **Default:** `1.0`.
- **Raise toward `1.0`** to prioritize cache reuse and lower TTFT — the router more aggressively co-locates requests that share a prefix.
- **Lower toward `0.0`** to spread load more evenly and lower ITL, at the cost of more redundant prefills. `0.0` ignores prefix caches entirely and skips building the local indexer (equivalent to load-only routing).

Most deployments should leave this at `1.0`. Lower it only when cache-rich workers are getting overloaded while others sit idle.

### Prompt-Side Load Weight

`--router-prefill-load-scale` (env `DYN_ROUTER_PREFILL_LOAD_SCALE`) scales the prompt-side prefill load after overlap credit is applied, setting how much prompt work counts relative to decode-side block load.

- **Minimum:** `0.0` (ignore prompt-side load). **Default:** `1.0`. No hard maximum — values above `1.0` weight prefill more heavily.
- **Raise above `1.0`** when long prompts are saturating workers and you want the router to steer new requests away from workers already doing heavy prefill.
- **Lower below `1.0`** when decode-side pressure dominates and you want routing driven mainly by active decode blocks.

### Route on Load Only

`--no-router-kv-events` (env `DYN_ROUTER_USE_KV_EVENTS=false`) disables event tracking; the router predicts cache state from its own routing decisions with TTL-based expiration instead of consuming real KV events. Use it only when you are not confident the backend emits KV events correctly.

## Routing with Disaggregated Serving

In a disaggregated graph, the router operates over prefill and decode workers separately. The prefill workers publish KV events (the second step above) and the router selects among them; the internal prefill router activates automatically. See [Router with Disaggregated Serving](../../developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md).

## Related Pages

- [KV-Aware Routing on Kubernetes](overview.md) — compare the Frontend and GAIE topologies.
- [Using GAIE with Dynamo](gateway-api.mdx) — place endpoint selection in the Dynamo EPP.
- [Router Guide](../../developer-guide/knowledge-base/modular-components/router/router-guide.md) — deployment modes, full CLI and env reference.
- [Routing Concepts](../../developer-guide/knowledge-base/modular-components/router/routing-concepts.md) — cost model and worker selection.
- [Router with Disaggregated Serving](../../developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md) — prefill/decode routing.
