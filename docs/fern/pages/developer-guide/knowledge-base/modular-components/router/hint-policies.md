---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Send KV Hints from a Worker-Selection Policy
subtitle: Attach KV actions, such as a fetch, to the worker a policy books
---

> [!WARNING]
> **Experimental.** The hint API, its configuration, and its actions can change.

A KV hint is an engine-directed KV action, such as a [KV fetch](topology-aware-kv-transfer.md#kv-fetch-hints), that the router sends with the worker it books for a request. The [worker-selection policy](custom-worker-selection.mdx) that picked the worker is the one that sends hints: it attaches a hint emitter to itself, and the router calls the emitter once per booking. Hints never change which worker is selected.

## Configure the Built-In Hints

Dynamo's default policies configure their hints through their own `parameters`. `kvcr` sends `kv_fetch` by default; `dynamo-default-cost-fn` sends nothing unless told to.

```yaml
worker_selection:
  prefill:
    type: kvcr
    parameters:
      shared_cache_multiplier: 0.5
      enable_hints: [kv_fetch]        # the hints to send; replaces the default list; [] sends none
      hint_parameters:                # optional, keyed by hint name
        kv_fetch:
          min_additional_blocks: 2
  decode: dynamo-default-cost-fn      # no hints
```

- `enable_hints` names the hints the policy sends. Omit it for the policy's defaults; `[]` sends none. A name no default policy knows fails startup.
- `hint_parameters` holds each enabled hint's parameters under its name. A key for a hint that is not enabled fails startup.
- `kvcr` with `shared_cache_multiplier` above `0` credits the booked worker for the KV it would copy, so it must send `kv_fetch`; disabling it fails startup.
- The router sends one hint per booking, only to workers that advertise `router_hint`, and only on the KV path, which is the one that books. It numbers the actions `a1`, `a2`, and so on, up to 32.

| Hint | Behavior |
|---|---|
| `kv_fetch` | Sends the booked worker a `kv.fetch` hint naming a peer that holds KV it lacks. A fetch the policy's score credited always goes out; a best-effort fetch goes out when it adds at least `min_additional_blocks` (default `1`). On by default under `kvcr`. |

## Send Hints from a Custom Policy

Implement `HintEmitter` from `dynamo_kv_router::plugins::prelude` and attach it to the policy with `WorkerSelectionPolicy::with_hints`. The router calls `emit` once per booking with a `HintContext`, which carries the request ID, the booked worker, the prompt length, the block size, and the session context. It returns the actions for that worker, in the engine's format. An error fails the booking, which the router releases; return an empty list for a hint that is merely unavailable.

```rust
struct HarnessPrefetch { max_blocks: usize }

impl HintEmitter for HarnessPrefetch {
    fn emit(&self, context: &HintContext<'_>) -> Result<Vec<KvHintAction>, HintError> {
        Ok(vec![KvHintAction::unnumbered("harness.prefetch", "1", payload_for(context, self.max_blocks))])
    }
}

registry.register_worker_selection(
    "my_policy",
    Arc::new(|parameters| {
        let config: MyParameters = parameters.deserialize()?;
        Ok(Arc::new(move |selection, role, _partition| {
            WorkerSelectionPolicy::new(selection.clone(), role.as_str(), scorers(&config), picker(&config))
                .with_hints(Arc::new(HarnessPrefetch { max_blocks: config.max_blocks }))
        }))
    }),
)?;
```

Because the policy owns its hints, anything the policy decided while scoring is available to the emitter through the policy's own state, and the policy parses whatever configuration its hints need from its `parameters`. Link the policy into Dynamo as for any [custom worker-selection policy](custom-worker-selection.mdx#link-the-policy-into-dynamo).

### Fetching the Booked Worker's Copy

An emitter that declares `WorkerInputs::KV_COPY` from `required_worker_inputs` gets the booked worker's copy source: the peer the router found holding more of the request's KV than the worker does, when there is one, through `HintContext::kv_copy`, and where its blocks are through `HintContext::kv_copy_locations`. `KvHintAction::kv_fetch` builds the `kv.fetch@1.0` action from those locations, so a custom policy that wants the same fetch `kvcr` sends calls it rather than spelling the wire format itself. `dynamo_router_policies::kv_fetch::KvFetch` is that emitter, ready to attach.

A policy whose scorer reads `KV_COPY` commits the booked worker to the copy its score credited, so its emitter must declare `KV_COPY` as well; the router rejects the policy at startup otherwise.
