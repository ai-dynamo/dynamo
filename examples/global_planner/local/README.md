<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Router — local bare-process example (mocker, CPU-only)

A runnable, GPU-free way to see the **Global Router** route across multiple pools as plain
local processes — no Kubernetes, no model download for inference (the [mocker](../../../components/src/dynamo/mocker)
simulates the engine; only a tokenizer is fetched). The sibling YAMLs in this directory cover the
Kubernetes path; this is the local equivalent that the other backends provide under `launch/`.

## Topology

```
client --> frontend (ns: grdemo) --> global router (ns: grdemo)
                                       |-- agg pool 0: local router + mocker  (ns: agg_pool_0)
                                       +-- agg pool 1: local router + mocker  (ns: agg_pool_1)
```

Two **aggregated** pools (each pool does prefill+decode). The global router selects a pool from the
request's SLA target using the 2D grid in [`agg_config.json`](./agg_config.json): with
`ttft_resolution: 2` and `agg_pool_mapping: [[0],[1]]`, a **tight** TTFT target routes to **pool 0**
and a **relaxed** one to **pool 1**.

## Prerequisites

- `etcd` (`:2379`) and `NATS` (`:4222`) running — e.g. `docker compose -f deploy/docker-compose.yml up -d`.
- The `dynamo` Python package available (`pip install "ai-dynamo[...]"`, or an editable source install on `PYTHONPATH`).

## Run

```bash
cd examples/global_planner/local
./launch.sh                 # starts mockers, per-pool routers, global router, frontend on :8000
# in another shell:
./client.sh 8000            # sends a default request + a tight-TTFT (pool 0) + a relaxed-TTFT (pool 1)
```

`launch.sh` honors `MODEL`, `PORT`, and `DYN_LOG` env vars. Stop everything with:

```bash
pkill -f 'dynamo.(mocker|router|global_router|frontend)'
```

Which pool served a request is shown in `logs/global_router.log`.

## How it maps to the config

| Piece | Command | Why |
|---|---|---|
| Pool worker | `dynamo.mocker --endpoint dyn://agg_pool_N.backend.generate --disaggregation-mode agg` | Registers under the pool's namespace. `backend` is where agg workers register; the pool's local router must point at the same endpoint. |
| Pool router | `dynamo.router --endpoint agg_pool_N.backend.generate` | The "local router" the global router forwards to for that pool. |
| Global router | `dynamo.global_router --config agg_config.json --model-name <HF id> --namespace grdemo` | `--model-name` **must be a real HuggingFace id** — the global router fetches its model card (a fake name returns `401`). |
| Frontend | `dynamo.frontend --namespace grdemo` | Public entrypoint; runs in the global router's namespace. |

## Gotchas (the same ones documented in the [global_router README](../../../components/src/dynamo/global_router/README.md#common-pitfalls))

- **Distinct `--model-name` for the pool workers** (`qwen-pool-demo` here). Discovery is global across
  namespaces, so if the pool workers registered the public model name, the frontend would discover them
  directly and bypass the global router (you'd see a fast `500 bootstrap_info is required`).
- **Bring pool routers up before the global router** — it errors at init if a pool's local router isn't registered yet.
- **agg vs disagg**: this example uses `mode: "agg"`. For disaggregated pools (separate prefill/decode),
  use a `mode: "disagg"` config; note the SGLang/TRT-LLM bootstrap paths are backend-dependent (see the global_router README).
- **This demo routes by config grid, not measured performance.** The two mocker pools are identical, so
  pool selection here is purely the `agg_pool_mapping` × SLA grid — it demonstrates the *control plane*,
  not a perf/cost win. A real benefit requires pools tuned differently (e.g. TP1 vs TP4), ISL-diverse
  traffic, and an SLA in the band where the cheaper pool meets short prompts but not long ones; with
  identical pools or a uniform workload, routing changes nothing observable.

## Note on verification

The config and process wiring here were validated against the `main` global-router config loader/validator
and the documented module interfaces; the control-plane topology (global router + per-pool local router +
namespace-scoped frontend + distinct served-model-name) was exercised end-to-end during testing. Run it on a
clean `dynamo` install — a mismatched `dynamo` Python/`_core` pair will surface as `KvRouterConfig` keyword
errors (see the version-skew notes in the troubleshooting docs).
