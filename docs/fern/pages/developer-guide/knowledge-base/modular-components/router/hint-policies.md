---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Write a Hint Policy
subtitle: Attach engine-directed KV actions to the worker a request is routed to
---

> [!WARNING]
> **Experimental.** The hint-policy API, its configuration, and its actions can change.

A hint policy adds engine-directed KV actions, such as a [KV fetch hint](topology-aware-kv-transfer.md#kv-fetch-hints), to the worker the router books for a request. It never changes which worker is selected; the [worker-selection policy](custom-worker-selection.mdx) does that.

## Configure Hints for a Role

A role's `worker_selection` entry lists the hint policies it runs, in order, under `hints`. Each item has a `type` and optional `parameters`. Add a `name` only when a role lists the same type twice.

```yaml
worker_selection:
  decode:
    type: dynamo-default-cost-fn
    hints:
      - type: harness_prefetch
        parameters:
          max_blocks: 64
  prefill:
    type: dynamo-default-cost-fn
    hints: []
```

- A role can list any linked hint policy, and `hints: []` runs none.
- A role whose entry has no `hints`, including a bare entry such as `prefill: dynamo-default-cost-fn`, runs the hint policies registered to run by default under its worker-selection policy.
- Hints run only on the KV path. A role that lists `hints` under a non-KV `--router-mode` fails at startup.
- A role whose worker-selection policy commits KV copies, such as `kvcr` with `shared_cache_multiplier` above `0`, must run exactly one hint policy that sends them, or startup fails.
- The router sends one hint per booking, only to workers that advertise `router_hint`. It numbers the actions `a1`, `a2`, and so on, up to 32. A policy that fails drops only its own actions.

## Implement and Register a Policy

Implement `HintPolicy` from `dynamo_kv_router::plugins::hint_policy`. The router calls `build_hints` once per booking with a `HintContext`, which carries the request ID, the booked worker, the prompt length, the block size, and the session context. It returns the actions for that worker, in the engine's format. A policy that sends committed KV copies declares `Commitments::KV_COPY` and reads the copy from `HintContext::committed_kv_copy`.

Register the policy with a provider that parses its parameters:

```rust
registry.register_hint_policy(
    "harness_prefetch",
    Arc::new(|parameters| {
        let config: PrefetchConfig = parameters.deserialize()?;
        Ok(HintPolicyConstructor::shared(HarnessPrefetch::new(config)))
    }),
    [],
)?;
```

The last argument lists the worker-selection policies the hint runs under by default, such as `HintHost::worker_selection("my_policy")` or `HintHost::AnyWorkerSelection`. An empty list means the hint runs only where a role lists it. Link the catalog into Dynamo as for a [custom worker-selection policy](custom-worker-selection.mdx#link-the-policy-into-dynamo).
