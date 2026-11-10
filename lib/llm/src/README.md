<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend, model discovery, and shared routing

The integrated frontend accepts API requests, preprocesses them, selects routes, dispatches backend work, and tracks response lifetime. Runtime EPP reuses parts of this library but currently assembles its own preprocessing and routing path.

## Shared selection implementation

The frontend reaches selection through:

```text
RoutingHost → KvRouter → EmbeddedSelection
                         → SelectionService.core().run_selection(...)
```

The standalone selection HTTP handlers reach the same `SelectionCore` through service methods. `EmbeddedSelection` is the runtime adapter around that service: it connects discovered workers, cache ingress, load reporting, and request ownership. It is an in-process use of the shared selection implementation, not a separate selection algorithm or a complete copy of the standalone deployment.

Source: [embedded service owner](kv_router/embedded.rs) and [core call](kv_router/embedded.rs). See the [selection API and ownership guide](../../kv-router/src/services/selection/README.md) for mode semantics.

## What ModelManager does in runtime EPP

EPP constructs its own local instance:

```rust
let model_manager = Arc::new(ModelManager::new());
```

It uses that instance to build the decode (D) worker pool's managed KV router and passes a clone to `PrefillRouter`. In this path the manager:

- Constructs KV routers and prepares their selection policies.
- Owns reusable runtime-config watches keyed by endpoint.
- Provides KV-source membership watches when the policy/configuration needs them.
- Provides optional shared-cache and other configured routing support.

It is therefore not an unused placeholder. However, EPP does not populate the frontend's full model/worker-set engine catalog through this setup. It constructs routers directly and creates its preprocessor separately. It also does not perform the frontend handler's normal model-alias lookup on each request.

Sources: [EPP manager construction](../../../deploy/inference-gateway/ext-proc/src/epp.rs), [router construction](discovery/model_manager.rs), and [watcher registry](discovery/model_manager.rs).

## Discovery remains live after endpoint construction

An `Endpoint` identifies a logical namespace/component/endpoint. `endpoint.client()` creates a client backed by discovery. Runtime-config watching joins two live inputs: available worker instances and discovered model runtime configs. It publishes configs for workers present in both sets.

A worker joining the same pool becomes visible when the necessary discovery/config information arrives. A departing worker is removed when discovery reports that removal. This is asynchronous observation, not instantaneous failure detection. The same client can follow pool membership changes without rebuilding the router.

EPP's initial discovery of which prefill pool to use is one-shot. It does not automatically retarget that pool when a rolling deployment changes the namespace. See [prefill activation](kv_router/prefill_router/README.md) for that boundary.

Sources: [client construction](../../runtime/src/component.rs), [shared discovery source](../../runtime/src/component/client.rs), and [membership/config join](discovery/runtime_configs.rs).

## Tokenization in runtime EPP

Runtime EPP uses `OpenAIPreprocessor` from the Dynamo LLM library, initialized from a discovered model deployment card and its assets. Chat requests go through template application and tokenization. Completion requests can supply text or token IDs. This occurs in the EPP process; it is not an RPC to a separate Dynamo frontend tokenizer.

EPP can inject compatible precomputed tokens into the forwarded request. Its tokenizer path still does not imply full parity with the frontend's preprocessing, multimodal routing metadata, model selection, or all routing hints.

Sources: [EPP tokenization path](../../../deploy/inference-gateway/ext-proc/src/epp.rs) and [preprocessor construction](../../../deploy/inference-gateway/ext-proc/src/epp.rs).

## Proposed frontend reuse through an EPP interface

The proposed architecture is one Dynamo process with OpenAI and EPP protocol adapters sharing preprocessing and routing. It does not require EPP to make another HTTP call back into that process.

```text
OpenAI adapter ─┐
                ├→ shared preprocessing and routing → execution + response stream
EPP adapter ────┘                                 └→ schedule result + booking
```

The chat-completions handler already enters the frontend engine pipeline, so an EPP adapter could reuse that logic internally. Calling the handler unchanged is not yet a complete schedule-only implementation: it returns an API response and owns execution-oriented context, while EPP must return worker destinations and retain reservations until later Envoy events.

`query_instance_id` demonstrates routing information returned through annotations. On the KV path it books nothing and dispatches nothing, but it still enters admission in the current implementation. On built-in routing paths the annotation is not a universal no-execution guarantee. It cannot serve as a complete schedule-only flag without defining mode-specific behavior and booking ownership.

Sources: [chat-completions handler](http/service/openai.rs), [query-mode contract and routing](kv_router/routing_host.rs), and [KV query behavior](kv_router/routing_host.rs).

## What reuse must preserve

A shared scheduling pipeline needs an explicit result contract: selected prefill (P) and decode (D) worker identities and ranks, endpoint resolution, backend routing metadata, and a reservation owner. The EPP adapter must continue handling candidate subsets and authoritative header/body mutations.

It also needs a lifecycle bridge. The frontend's request guard observes actual generated output, touches leases, updates output blocks and approximate cache state, and advances classifier state. EPP currently receives coarser HTTP events and uses request-ID-based decode bookkeeping. Reusing preprocessing alone does not restore those dispatch/stream effects.

Feature coverage to carry through includes conditional disaggregation, session/agent affinity, expected output length, LoRA and multimodal routing inputs, KV-transfer hints, migration/retry behavior, and structured failure mapping. Some are request inputs; others happen during dispatch or response handling. A single schedule-only flag must account for both categories.

This is a design direction, not an implemented claim of feature parity. See the [runtime EPP guide](../../../deploy/inference-gateway/ext-proc/README.md) for observed behavior and specific gaps.
