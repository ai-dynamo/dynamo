---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Worker Selection
subtitle: Link native Rust scorers and pickers into a Dynamo frontend or EPP
---

**Experimental.** Implement custom worker selection in a normal Rust crate and link it into a custom frontend or Endpoint Picker Provider (EPP) image. Dynamo continues to own worker discovery, queueing, eligibility, validation, accounting, reservations, and metrics.

This is a compile-time extension point. It does not load policies dynamically or through a C ABI.

## Selection Boundary

Custom code replaces only scoring and picking:

```text
Frontend or EPP
  -> SelectionService
  -> policy-class queue and scheduler
  -> host-owned worker eligibility
  -> WorkerScorer(s)
  -> WorkerPicker
  -> host validation, accounting, and reservation
```

Hard filters run before the custom policy. They enforce request allowlists, exact worker and data-parallel rank pins, required taints, and busy-threshold overload state. See [Router Filtering](../knowledge-base/modular-components/router/worker-filtering.md) for the complete eligibility behavior.

## Implement a Policy

Implement one or more `WorkerScorer` traits and one `WorkerPicker` trait. Scorers contribute finite, lower-is-better costs for every eligible row. Dynamo sums those contributions before calling the picker. The picker returns one row index, which Dynamo validates before it books the request.

The following policy selects the worker with the fewest active requests:

```rust
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerCandidate, WorkerInputView, WorkerInputs,
    WorkerPicker, WorkerScorer, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};

struct ActiveRequestScorer;

impl WorkerScorer for ActiveRequestScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        candidate: &WorkerCandidate,
    ) -> Result<f64, WorkerSelectionPolicyError> {
        let load = candidate
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load inputs unavailable"))?;
        Ok(load.active_requests() as f64)
    }
}

struct MinimumCostPicker;

impl WorkerPicker for MinimumCostPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        input
            .candidates()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.cost().total_cmp(&right.cost()))
            .map(|(row, _)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

fn active_request_policy(
    config: &KvRouterConfig,
    worker_type: &'static str,
    _partition: RoutingPartitionRef<'_>,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config.clone(),
        worker_type,
        vec![Box::new(ActiveRequestScorer)],
        Box::new(MinimumCostPicker),
    )
}
```

The factory runs when Dynamo constructs a decode or prefill worker set, not for each request. It receives the model and routing group through `RoutingPartitionRef`. Policy state belongs to that scheduler queue actor and is called serially.

## Request Worker Inputs

Return only the signal groups that the scorer or picker reads. Dynamo creates the union requested by the composed policy and skips unused per-worker calculations.

| Input | Available Signals |
| --- | --- |
| `WorkerInputs::CACHE` | Effective, device, host, disk, and shared-cache overlap |
| `WorkerInputs::LOAD` | Raw prefill blocks, active prefill tokens, decode cost blocks, and active requests |
| `WorkerInputs::ROUTING` | Preferred-taint cost multiplier |
| `WorkerInputs::NONE` | Worker identity and accumulated scorer cost only |

A scorer reads its requested values from `WorkerCandidate`. A picker receives index-aligned columns through `WorkerInputView`. Candidate row order is unspecified; inspect each candidate's worker identity or signals instead of relying on position.

## Link a Custom Frontend

Keep the policy in the crate's library target. Add a binary target that constructs the normal `DistributedRuntime` and `EngineConfig`, then inject the policy factory into Dynamo's complete HTTP frontend:

```rust
HttpFrontend::default()
    .worker_selection_policy_factory(active_request_policy)
    .run(distributed_runtime, engine_config)
    .await?;
```

The binary reuses Dynamo's model watcher, tokenizer, request pipeline, worker registry, scheduler, HTTP routes, validation, accounting, and metrics. Only the factory differs from the default frontend.

Build and run that binary in the custom image. `python3 -m dynamo.frontend` starts the frontend compiled into the installed Dynamo Python extension and cannot discover an external statically linked Rust crate. Without an injected factory, it uses the concrete default worker selector.

## Link a Custom EPP

Build a `SelectionService` with the same factory and pass it to the standard EPP runner:

```rust
use dynamo_kv_router::services::selection::SelectionServiceBuilder;

let service = SelectionServiceBuilder::new(kv_router_config)
    .worker_selection_policy_factory(active_request_policy)
    .build()
    .await?;

dynamo_ext_proc::run_with_selection_service(service).await?;
```

The runner uses the stock TLS, health, metrics, discovery, and readiness bootstrap. Supplying a prebuilt service selects standalone mode and transfers ownership of the service so its workers, peer membership, and background tasks share the EPP lifecycle. The standard EPP binary calls the same runner without a prebuilt service.

## Policy Contract

- Return finite scorer contributions. Dynamo rejects non-finite contributions and accumulated costs.
- Return an index into `WorkerInputView::candidates()`. Dynamo rejects an out-of-range index before accounting or reservation.
- Treat row order as unspecified.
- Request only the signal groups the policy reads.
- Use host eligibility for hard exclusion. Scorers bias costs; the picker chooses among eligible rows.
- Keep blocking I/O out of `score` and `pick`; both execute in the scheduler queue actor.
- Do not panic in `score` or `pick`. Return `WorkerSelectionPolicyError`; a panic stops the scheduler queue actor.

For Dynamo's default cost model, see [Routing Concepts](../knowledge-base/modular-components/router/routing-concepts.md). For the embedded service lifecycle and reservation API, see [Standalone Selection Service](../knowledge-base/modular-components/router/standalone-selection.md).
