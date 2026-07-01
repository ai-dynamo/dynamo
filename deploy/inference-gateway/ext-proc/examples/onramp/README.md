<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone on-ramp (raw vLLM + Dynamo EPP + in-process selector)

This example puts the Dynamo EPP in front of an existing vanilla `vllm serve`
fleet behind a GAIE gateway, with **no Dynamo operator, no etcd/NATS, and no
Dynamo runtime**. You replace only your endpoint picker (EPP); your workers stay
stock vLLM.

KV-aware selection is provided by the runtime-free
[selection service](../../../../../docs/components/router/standalone-selection.md),
which the EPP runs **in-process**: the EPP and the selection service are compiled
into one binary, so there is no separate selector Deployment and no HTTP hop. The
EPP can run single-replica, or **replicated** with cross-replica active-load sync
between EPP pods (see [Replicated mode](#replicated-mode)).

## How it works

```text
                          ┌───────────── Dynamo EPP (standalone, in-process selector) ────────────┐
client → Gateway/Envoy ──▶│ tokenize → in-process select (Ready subset) → set routing headers      │
                          │   ▲ reads InferencePool (selector + target port)                        │
                          │   └ in-process selector subscribes to each pod's KV-event socket        │
                          └───────────────────────────────────────────────────────────────┬───────┘
                                                                                            │ endpoint + dp_rank
        raw vLLM pods ◀───────────────────────────── Envoy forwards HTTP ◀─────────────────┘
```

1. The EPP reads the `InferencePool` it backs (read-only) to learn the pod
   selector and HTTP target port — the same object the gateway routes to — and
   watches those pods, Ready-filtered.
2. For each Ready pod, the EPP registers a worker into its **in-process**
   selector with the pod's endpoint, block size, and KV-event socket.
3. The selector subscribes directly to each pod's KV-cache event socket and
   maintains the KV index, scheduler, and load accounting.
4. On each request the EPP sends the unchanged body to the configured vLLM
   `/v1/chat/completions/render` endpoint, uses the returned tokens to select a
   currently-Ready worker (booking its load via select-and-reserve), and returns
   the worker's endpoint and `dp_rank` to Envoy as routing headers. The selected
   worker processes the original forwarded request.

This is the **aggregated** flow. Disaggregated prefill/decode is a follow-up.

## Prerequisites

- A GAIE-compatible `Gateway` named `inference-gateway`. It need not live in
  this namespace — the agentgateway controller, for example, installs it in
  `agentgateway-system`. The `vllm-qwen-route` `parentRefs` in `agg.yaml` must
  name the Gateway's namespace and listener (`namespace` + `sectionName`), or the
  route silently fails to attach (its `status.parents` stays empty). It ships
  pointing at `agentgateway-system`/`http`; edit it to match your Gateway (or
  drop `namespace` if the Gateway is in this namespace). The Gateway's listener
  must also allow routes from this namespace.
- A secret named `hf-token-secret` with key `HF_TOKEN` for the vLLM worker.

## Run

```bash
kubectl apply -n <ns> -f agg.yaml
# The Gateway may live in another namespace (e.g. agentgateway-system):
GW=$(kubectl get gateway inference-gateway -n agentgateway-system -o jsonpath='{.status.addresses[0].value}')
curl http://$GW/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role":"user","content":"hello"}]
}'
```

> [!NOTE]
> The `model` field MUST match `DYN_MODEL_NAME` in the EPP deployment, and the
> EPP's `DYN_KV_CACHE_BLOCK_SIZE` MUST equal the vLLM `--block-size`.

## EPP environment contract (standalone mode)

| Env | Meaning | Required? |
|---|---|---|
| `DYN_EPP_MODE=standalone` | Select standalone (selector) mode | required |
| `DYN_EPP_INFERENCE_POOL_NAME` | Name of the `InferencePool` this EPP backs (its selector + target port drive pod discovery) | required |
| `DYN_MODEL_NAME` | Model id (no model card in this mode) | required |
| `DYN_EPP_VLLM_RENDER_URL` | Base URL of the vLLM `/v1/chat/completions/render` service | required |
| `DYN_KV_CACHE_BLOCK_SIZE` | MUST equal vLLM `--block-size` | required |
| `DYN_EPP_KV_EVENT_PORT` | vLLM KV-events PUB port (not in the pool spec) | optional; default 5557 |
| `DYN_EPP_KV_EVENT_REPLAY_PORT` | ZMQ REQ port for live-stream gap replay | optional |
| `DYN_EPP_TOKENIZATION_TIMEOUT_MS` | Deadline for the vLLM render request | optional; default 5000 |
| `DYN_EPP_TOTAL_KV_BLOCKS` | Per-worker total KV blocks hint | optional |
| `DYN_EPP_MAX_NUM_BATCHED_TOKENS` | Per-worker max batched tokens | optional |
| `DYN_EPP_SELECTION_INDEXER_THREADS` | KV indexer thread pool | optional; default 4 |
| `DYN_EPP_PEER_SERVICE` | The EPP's own Service. Set = replicated; the Service must expose named port `replica-agg`. Unset = single local replica | optional; unset by default |
| `POD_NAMESPACE` | The namespace the EPP, its `InferencePool`, worker pods, and sibling replicas all live in (downward API) | required |
| `POD_IP` | EPP's own pod IP (downward API); needed only for replication, so a replica excludes itself from its peer set | required when `DYN_EPP_PEER_SERVICE` is set |

## Replicated mode

You can run **more than one** EPP replica. Each replica still runs its own
in-process selector, but the replicas discover each other and synchronize active
load so they don't each under-count load and herd onto the same worker. This is
what `agg.yaml` deploys (`replicas: 2`); set `replicas: 1` and drop
`DYN_EPP_PEER_SERVICE` for a single local replica.

```yaml
spec:
  replicas: 2
  # ...
        env:
          # Watch THIS Deployment's own Service to find sibling replicas:
          - name: DYN_EPP_PEER_SERVICE
            value: "dynamo-epp"
          # Own pod IP, so a replica excludes itself from its peer set:
          - name: POD_IP
            valueFrom:
              fieldRef:
                fieldPath: status.podIP
        ports:
          - name: replica-agg
            containerPort: 9092
---
apiVersion: v1
kind: Service
spec:
  ports:
    - name: replica-agg
      port: 9092
      targetPort: replica-agg
```

How it works:

1. Each replica builds its in-process selector with **replica-sync enabled**
   (resolving and binding the Service's named `replica-agg` port).
2. It watches its **own** `Service`'s EndpointSlices (the `DYN_EPP_PEER_SERVICE`
   above) and registers/deregisters every other replica as a replica-sync peer
   as pods come and go — excluding itself via `POD_IP`. Peers connect pod-to-pod
   on the resolved `replica-agg` endpoint port.
3. Admission (`select_and_reserve`), `prefill_complete`, and `free` are broadcast
   over ZMQ, so every replica's selector converges on the same active-load view.
   This is the same consistency model as the standalone selector — output-block
   growth stays local by design; admission/prefill/free are synchronized.

> [!NOTE]
> **No cross-peer KV-index warm-up.** A freshly started replica starts
> subscribing to worker KV events *forward* and relies on live sync + per-worker
> replay to catch up; it does not pull a peer's index snapshot. Expect a brief
> cold window on a new replica.

The EPP's `Role` needs `discovery.k8s.io/endpointslices` read access to watch its
own Service (already included in `agg.yaml`).

Replica synchronization is best-effort; see the
[selection service docs](../../../../../docs/components/router/standalone-selection.md)
for the consistency invariants.

## Limitations (V1)

- Aggregated serving only (no disaggregated prefill/decode).
- Cross-replica sync covers active load (admission/prefill/free), not the KV
  index — a new replica re-warms its index from live traffic + replay.
- `worker_id` is a stable hash of the pod name. A pod restart yields a new
  `worker_id`; sticky routing is preserved separately via `stable_routing_id`.
