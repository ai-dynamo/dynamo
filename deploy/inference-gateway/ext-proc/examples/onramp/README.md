<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router-only on-ramp (raw vLLM + Dynamo EPP + selection service)

This example puts the Dynamo EPP in front of an existing vanilla `vllm serve`
fleet behind a GAIE gateway, with **no Dynamo operator, no etcd/NATS, and no
Dynamo runtime**. You replace only your endpoint picker (EPP); your workers stay
stock vLLM.

KV-aware selection is provided by the runtime-free
[selection core](../../../../../docs/components/router/standalone-selection.md).
The EPP reaches it one of two ways:

- **embedded** (this `agg.yaml`): the EPP and the selection core run in **one
  process** — no separate selector Deployment. Single-replica; for evaluation.
- **http**: the EPP is a thin client of one or more `dynamo.select_service`
  replicas. Replicated/production. See [Replicated (http) mode](#replicated-http-mode).

## How it works (embedded)

```text
                          ┌───────────── Dynamo EPP (router-only, embedded selector) ─────────────┐
client → Gateway/Envoy ──▶│ tokenize → in-process select (Ready subset) → set routing headers     │
                          │   ▲ reads InferencePool (selector + target port)                       │
                          │   └ in-process SelectionCore subscribes to each pod's KV-event socket  │
                          └───────────────────────────────────────────────────────────────┬──────┘
                                                                                            │ endpoint + dp_rank
        raw vLLM pods ◀───────────────────────────── Envoy forwards HTTP ◀─────────────────┘
```

1. The EPP reads the `InferencePool` it backs (read-only) to learn the pod
   selector and HTTP target port — the same object the gateway routes to — and
   watches those pods, Ready-filtered.
2. For each Ready pod, the EPP registers a worker into its **in-process**
   selection core with the pod's endpoint, block size, and KV-event socket.
3. The selection core subscribes directly to each pod's KV-cache event socket
   and maintains the KV index, scheduler, and load accounting.
4. On each request the EPP tokenizes locally (for routing only), selects a
   worker constrained to the currently-Ready pods, and returns the worker's
   endpoint and `dp_rank` to Envoy as routing headers. The worker re-tokenizes
   the forwarded request.

This is the **aggregated, query-only** flow. It does not use reservations.
Load-aware accuracy (reservations + prefill/free lifecycle) and disaggregated
serving are follow-ups.

## Prerequisites

- A GAIE-compatible `Gateway` named `inference-gateway` in your namespace.
- A secret named `hf-token-secret` with key `HF_TOKEN`.

The default EPP image supports both selector modes; `embedded` vs `http` is a
runtime choice via `DYN_EPP_SELECTOR_MODE`, so no special build is needed. If you
want a lean HTTP-only image (no libzmq / no embedded selection service), build
with:

```bash
docker build --build-arg EPP_CARGO_FLAGS="--no-default-features --features selector-http" ...
```

## Run

```bash
kubectl apply -n <ns> -f agg.yaml
GW=$(kubectl get gateway inference-gateway -o jsonpath='{.status.addresses[0].value}')
curl http://$GW/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role":"user","content":"hello"}]
}'
```

> [!NOTE]
> The `model` field MUST match `DYN_MODEL_NAME` in the EPP deployment, and the
> EPP's `DYN_KV_CACHE_BLOCK_SIZE` MUST equal the vLLM `--block-size`.

## EPP environment contract (router-only mode)

| Env | Meaning | Required? |
|---|---|---|
| `DYN_EPP_MODE=router-only` | Select router-only (selector) mode | required |
| `DYN_EPP_SELECTOR_MODE` | `embedded` (single image) or `http` (replicated) | optional; default `http` |
| `DYN_EPP_POOL_NAME` | Name of the `InferencePool` this EPP backs (its selector + target port drive pod discovery) | required |
| `DYN_EPP_POOL_NAMESPACE` | Namespace of the `InferencePool` | optional; default `POD_NAMESPACE` |
| `DYN_MODEL_NAME` | Model id (no model card in this mode) | required |
| `DYN_KV_CACHE_BLOCK_SIZE` | MUST equal vLLM `--block-size` | required |
| `DYN_EPP_KV_EVENT_PORT` | vLLM KV-events PUB port (not in the pool spec) | optional; default 5557 |
| `DYN_EPP_KV_EVENT_TOPIC` | vLLM KV-events topic | optional; default `""` |
| `DYN_EPP_REPLAY_PORT` | ZMQ REQ port for live-stream gap replay | optional |
| `DYN_DATA_PARALLEL_SIZE` | Engine DP size (V1 supports 1) | optional; default 1 |
| `DYN_TOTAL_KV_BLOCKS` | Per-worker total KV blocks hint | optional |
| `DYN_MAX_NUM_BATCHED_TOKENS` | Per-worker max batched tokens | optional |
| `DYN_EPP_SELECTOR_THREADS` | KV indexer threads (embedded mode) | optional; default 4 |
| `DYN_EPP_SELECTOR_SERVICE` | Name of the selection-service `Service`; the EPP watches its EndpointSlices to discover replicas | required for `http` mode |
| `DYN_EPP_SELECTOR_SERVICE_NAMESPACE` | Namespace of the selection-service `Service` | optional; default `POD_NAMESPACE` |
| `DYN_EPP_SELECTOR_HTTP_PORT` | HTTP port each selector replica serves on | optional; default 8092 |
| `DYN_EPP_SELECTOR_REPLICA_SYNC_PORT` | ZMQ replica-sync PUB port each selector replica binds (used to wire the peer mesh) | optional; default 9092 |
| `POD_NAMESPACE` | EPP's own namespace (downward API) | required |

## Replicated (http) mode

Embedded mode is single-replica only: the in-process selector has no
cross-replica index/load synchronization. For a replicated deployment, run a
separate `dynamo.select_service` `Deployment` (one or more replicas) fronted by a
`Service`, and switch the EPP to http mode at runtime (no rebuild — the default
image supports both). Point the EPP at the **Service name**, not a static URL:

```yaml
- name: DYN_EPP_SELECTOR_MODE
  value: "http"
- name: DYN_EPP_SELECTOR_SERVICE
  value: "dynamo-selector"            # a headless/ClusterIP Service in this namespace
# Optional overrides (defaults shown):
# - name: DYN_EPP_SELECTOR_HTTP_PORT
#   value: "8092"
# - name: DYN_EPP_SELECTOR_REPLICA_SYNC_PORT
#   value: "9092"
```

The EPP watches the Service's EndpointSlices and keeps its replica set in sync as
replicas come and go: each new (or restarted) replica is bootstrapped — its
catalog reconciled from scratch — before it receives selection traffic, and
replicas that disappear are dropped. It reconciles the catalog on every live
replica independently (`GET /workers` → diff → `POST`/`DELETE /workers`) and
routes selections only to replicas reporting `GET /ready`.

The EPP also wires the selectors' **replica-sync peer mesh** dynamically as the
replica set changes (`POST /replica_sync/register_peer` /
`.../deregister_peer`), so active-load and admission events propagate across the
fleet without static `--replica-sync-peers` wiring. Each selector replica only
needs its own replica-sync port bound:

```bash
python -m dynamo.select_service \
  --port 8092 \
  --replica-sync-port 9092
```

`--indexer-peers` remains useful at startup so a fresh replica can recover KV
index state from an existing peer before the EPP finishes bootstrapping it.
Replica synchronization is best-effort; see the
[selection service docs](../../../../../docs/components/router/standalone-selection.md)
for the consistency invariants.

### Additional RBAC for http mode

HTTP mode discovers selector replicas from EndpointSlices, so the EPP's `Role`
needs read access to them (in addition to the `pods` / `inferencepools` rules in
`agg.yaml`):

```yaml
- apiGroups: ["discovery.k8s.io"]
  resources: ["endpointslices"]
  verbs: ["get", "list", "watch"]
```

## Limitations (V1)

- Aggregated serving only (no disaggregated prefill/decode).
- Query-only selection; no reservations, so active-load accuracy is approximate
  (and, in http mode, cross-replica load is only best-effort synchronized).
  Load-accurate routing is a follow-up.
- Embedded mode is single-replica; use http mode to replicate.
- `worker_id` is a stable hash of the pod name. A pod restart yields a new
  `worker_id`; sticky routing is preserved separately via `stable_routing_id`.
