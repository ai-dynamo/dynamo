<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA mocker routing E2E plan

## Goal

Exercise LoRA-aware frontend routing and placement without GPUs, first as a local-process
pytest and then in a disposable Minikube deployment. The primary bounded case uses `N=8`
mocker workers, `K=4` advertised LoRA slots per worker, and at most `L=32` distinct adapters,
so `L <= N * K`.

The first milestone validates routing targets. It does not claim to measure real GPU adapter
loads, unloads, or eviction latency. Those require a later mock residency feedback loop.

## Validation completed

The first implementation was validated at two levels with `L=2`, `N=2`, and `K=1`, so
`L=N*K`:

- Local process tests passed for both HRW and min-cost flow. Each adapter converged to one
  replica, repeated requests stayed on one worker, the two adapters used distinct workers, and
  min-cost-flow overflow stayed at zero.
- A local planner image containing the change and the repository's offline mock-Llama tokenizer
  was deployed to a disposable `dynamo-lora-e2e` Minikube profile. Direct etcd, mocker, and
  frontend Deployments exercised the same discovery and TCP request-plane paths without relying
  on an external registry or operator image.
- Min-cost flow kept `mock-lora-a` and `mock-lora-b` stable on distinct workers with zero
  overflow. HRW also kept a stable one-to-one placement, with the adapter-to-worker mapping
  reversed from min-cost flow. Final metrics reported both adapters active, replica factor one,
  zero current churn, and zero overflow.
- All three pods were healthy with zero restarts. The named Minikube profile was deleted after
  evidence collection, and the pre-existing kubectl context remained unchanged.

The first container attempt used the TinyLlama fixture, which has no chat template and therefore
cannot be registered by the HTTP frontend. The final smoke test used
`mock-llama-3.1-8b-instruct`, whose tokenizer metadata includes the required template.

## Gaps addressed by this change

Before this change, the mocker already supported multiple CPU-only workers and returned selected
worker IDs through `nvext`, but it could not participate in LoRA placement end to end:

1. `dynamo.mocker` does not expose a LoRA slot-capacity argument.
2. The Python `ModelRuntimeConfig` binding does not expose `max_gpu_lora_count`, even though the
   Rust model card and discovery watcher support capacity-only base cards.
3. Mocker publishes only a base-model card. The frontend therefore has no synthetic LoRA model
   to accept through `/v1/chat/completions`, and the LoRA state tracker has no initial residency.
4. Mocker does not update LoRA model cards after a lazy load or eviction. Static initial cards
   can validate routing-target selection, but not the complete residency feedback loop.

## Milestone 1: deterministic mocker model cards

Add the minimum production-shaped surface instead of test-only discovery injection.

### CLI and validation

Add these mocker inputs:

- `--max-gpu-lora-count K`: positive per-worker resident-slot capacity.
- `--initial-lora-placement PATH`: JSON file containing one adapter-name list per in-process
  worker, for example `{"workers":[["lora-a"],["lora-b"],[],[]]}`.

Validate at startup, before creating runtimes:

- the placement has exactly `--num-workers` entries;
- adapter names are non-empty and unique within a worker;
- every worker has at most `K` initial adapters;
- the total distinct adapter count is at most `num_workers * K` for the bounded test;
- placement requires `--max-gpu-lora-count` and a base `--model-path`.

The JSON file is preferable to a large inline argument: it is readable in pytest artifacts and
can be mounted as a Kubernetes ConfigMap without shell escaping.

### Registration path

1. Add a Python getter/setter for `ModelRuntimeConfig.max_gpu_lora_count` and cover its `None`
   and positive-value round trips.
2. Set that property in `build_runtime_config` for every base mocker worker. Discovery then seeds
   worker capacity without inventing a phantom adapter.
3. In `launch_workers`, after the base engine has registered, obtain the worker endpoint from its
   `DistributedRuntime` and call `register_model` once for each adapter assigned to that worker.
   Use the base model as `base_model_path`, `ignore_weights=True`, the adapter name as
   `lora_name`, and the same `max_gpu_lora_count` on each adapter card.
4. Keep all registrations attached to the worker's runtime so shutdown removes both base and
   adapter cards through the normal discovery lifecycle.

This gives each worker a realistic capacity-only base card plus zero or more loaded-adapter
cards. The mock engine may ignore adapter math, but preprocessing, discovery, the controller,
the filter, and request dispatch use the same path as a real backend.

## Milestone 2: local-process pytest

Extend `tests/router/mocker_process.py` to pass the new arguments and place reusable assertions
in `tests/router/common.py`, keeping `test_router_e2e_with_mockers.py` thin.

For each algorithm (`hrw`, `min_cost_flow`):

1. Start eight aggregated mocker workers with four slots each and a seeded placement containing
   no more than 32 adapters.
2. Start the frontend with `DYN_LORA_ENABLED=true`, a one-second allocation timestep, comparison
   cooldown zero, and the selected allocation algorithm.
3. Wait until `/v1/models` contains every synthetic adapter and the frontend metrics expose the
   expected adapter set.
4. Send a fixed, seeded request schedule. Include
   `"nvext":{"extra_fields":["worker_id"]}` and record the decode worker for every response.
5. Scrape `dynamo_frontend_lora_replica_factor`, `lora_is_active`, estimated load, MCF churn, and
   overflow metrics after controller ticks.

Required assertions:

- every request is served by a live worker;
- a known-loaded worker is preferred before the first allocation tick;
- repeated HRW runs produce identical worker sequences;
- repeated MCF runs produce identical worker sequences;
- steady demand reaches zero routing-target churn after convergence;
- no MCF overflow occurs in the bounded equal-demand case where all `L = N * K` adapters request
  exactly one replica; skewed multi-replica demand reports overflow separately rather than
  assuming the distinct-adapter bound is sufficient;
- replica factors never exceed the number of workers;
- removing one mocker process removes its worker IDs from subsequent responses;
- base-model requests remain routable across all workers.

Do not assert that MCF always beats HRW for every workload. Report the observed difference and
assert only properties guaranteed by each algorithm. The separate seeded simulation supplies
the comparative churn evidence.

## Milestone 3: disposable Minikube smoke test

Use an explicit profile and kube context for every command. The developer's default kubectl
context may point at a shared cluster and must never be used implicitly.

Suggested flow:

```bash
minikube start -p dynamo-lora-e2e --driver=docker --cpus=6 --memory=12288

make -C deploy/operator docker-build-planner REGISTRY=local TAG=lora-e2e
make -C deploy/operator docker-build-operator REGISTRY=local TAG=lora-e2e
minikube -p dynamo-lora-e2e image load local/dynamo-planner:lora-e2e
minikube -p dynamo-lora-e2e image load local/dynamo-operator:lora-e2e

helm dependency build deploy/helm/charts/platform
helm upgrade --install dynamo-platform deploy/helm/charts/platform \
  --kube-context=dynamo-lora-e2e \
  --namespace=dynamo-system --create-namespace \
  --set global.nats.install=true \
  --set global.etcd.install=false \
  --set global.grove.install=false \
  --set global.kai-scheduler.install=false \
  --set dynamo-operator.controllerManager.manager.image.repository=local/dynamo-operator \
  --set dynamo-operator.controllerManager.manager.image.tag=lora-e2e \
  --set dynamo-operator.controllerManager.manager.image.pullPolicy=Never
```

Apply a `v1beta1` `DynamoGraphDeployment` with one frontend and one CPU-only mocker pod. The
mocker pod launches eight in-process workers, reads the placement file from a ConfigMap, and uses
the local planner image with `imagePullPolicy: Never`. The frontend uses the same image and LoRA
environment variables as the local pytest. Port-forward the frontend with
`kubectl --context=dynamo-lora-e2e`, replay the same request schedule, and reuse the same response
and metric assertions.

Always collect the DGD, pods, events, frontend metrics, and pod logs before cleanup. Finally run
`minikube delete -p dynamo-lora-e2e`; do not alter the caller's default kubectl context.

## Milestone 4: residency feedback (separate change)

Static initial model cards cannot prove that a route-target addition caused a backend load. Add
that only after Milestones 1-3 are stable:

1. Teach the annotated mock engine to emit a small internal event when it serves a LoRA request.
2. Maintain a per-worker, capacity-bounded LRU adapter set.
3. Publish a LoRA model card on a simulated load and unregister the evicted adapter card.
4. Add configurable load/unload latency and counters.
5. Assert both routing-target churn and observed mock residency churn, using distinct names and
   metrics.

This phase enables closed-loop lazy-load and eviction tests. It should not be folded into the
first implementation because it changes mock engine behavior and needs its own concurrency and
shutdown review.

## Acceptance gates

- Unit tests cover CLI validation, runtime-config capacity, and placement parsing.
- Local mocker LoRA E2E passes for HRW and MCF without a GPU.
- The bounded case explicitly reports `L`, `N`, `K`, and verifies `L <= N * K`.
- Two identical seeded runs produce identical response worker sequences and metric snapshots.
- Minikube smoke test passes from a clean named profile and leaves the prior kubectl context
  unchanged.
- Logs and reports consistently distinguish routing targets from mock residency.
