<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime-Free Router Example: External SGLang

This smoke test deploys a standalone selection service and one aggregated
SGLang worker. It covers catalog registration, selector readiness, `/select`,
and KV event ingestion through the `SelectionTopologyController` and SGLang
metadata probes. It does not exercise Gateway or Endpoint Picker Protocol (EPP)
request-path routing.

## Prerequisites

- A Kubernetes cluster with GPU nodes.
- NVIDIA Dynamo operator built from a revision that includes the
  `SelectionTopologyController`.
- A Dynamo image that includes `python -m dynamo.select_service`.
- An SGLang image that exposes `/v1/models`, `/model_info`, and `/server_info`.
- Optional `hf-token-secret` with key `HF_TOKEN` for gated models. The default
  `Qwen/Qwen3-0.6B` model does not require it.

For development builds, point the Kustomize `images` entries at images built
from the same source revision. The selector image must include
`python -m dynamo.select_service`. Use a local Kustomize overlay for image
overrides or SGLang launch flags that your GPU or SGLang image requires.

```bash
export NAMESPACE=default
```

## Deploy

For a gated model, create the optional token secret before applying the
manifests. Skip this step for the default Qwen model:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n "${NAMESPACE}"
```

Before applying, replace the `my-registry/dynamo-select-service:my-tag`
placeholder in `kustomization.yaml` with a Dynamo image that includes
`python -m dynamo.select_service`.

Apply the manifests and wait for the Pods:

```bash
kubectl apply -n "${NAMESPACE}" -k examples/runtime-free-router/sglang
kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/part-of=runtime-free-router-sglang
```

Wait for the worker EndpointSlice to have a ready address:

```bash
kubectl get endpointslice -n "${NAMESPACE}" \
  -l kubernetes.io/service-name=runtime-free-sglang-worker
```

Port-forward the selection service:

```bash
kubectl port-forward -n "${NAMESPACE}" service/runtime-free-selector 8092:8092
```

In another terminal, check the worker catalog and selector readiness:

```bash
curl -s 'http://127.0.0.1:8092/workers?model_name=Qwen/Qwen3-0.6B&tenant_id=default'
curl -i http://127.0.0.1:8092/ready
```

Send a minimal selection request:

```bash
curl -s http://127.0.0.1:8092/select \
  -H 'content-type: application/json' \
  -d '{
    "selection_id": "manual-smoke",
    "model_name": "Qwen/Qwen3-0.6B",
    "tenant_id": "default",
    "block_hashes": [11, 12, 13, 14],
    "sequence_hashes": [11, 12, 13, 14],
    "isl_tokens": 4
  }'
```

The response includes the selected `worker_id`, `dp_rank`, and SGLang worker
`endpoint`.

## Verify KV Event Ingestion

The `/workers`, `/ready`, and `/select` checks cover catalog and selection. To
verify KV event ingestion, send one real SGLang request and inspect the selector
dump.

Port-forward the SGLang worker in another terminal:

```bash
kubectl port-forward -n "${NAMESPACE}" service/runtime-free-sglang-worker 30000:30000
```

Send a prompt long enough to create at least one KV block:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain why prefix cache routing helps repeated long prompts. Include three concrete reasons and keep the answer concise."
      }
    ],
    "max_tokens": 8,
    "stream": false
  }'
```

Poll the selector indexer snapshot:

```bash
curl -s http://127.0.0.1:8092/dump | grep -F '"Qwen/Qwen3-0.6B:default"'
```

The dump entry should contain a non-empty `events` array. That means the
selector received and applied KV events from the SGLang ZMQ publisher. Use
`/dump` for smoke validation, not production monitoring.

## Notes

- The selector Pod uses `/health` for Kubernetes readiness. Do not use `/ready`
  as the Pod readiness probe because `/ready` waits for a schedulable worker.
- Keep the worker annotation pointed at a Kubernetes Service URL such as
  `http://runtime-free-selector:8092`; raw Pod IP and host URLs are not covered
  by this example.
- When checking `/workers`, use entries with `lifecycle` set to `schedulable`;
  other entries are not active endpoints.
- The SGLang worker startup/readiness probes check `/v1/models`,
  `/model_info`, and `/server_info` before the Pod enters EndpointSlices.
- Catalog registration does not prove KV event delivery; use `/dump` to confirm
  that the selector can reach and ingest `kv_events_endpoints`.
- The worker Service carries the `external-sglang` selection annotations.
- Multi-node SGLang groups and `--dp-size` greater than `1` are not covered by
  this example.

## Cleanup

```bash
kubectl delete -n "${NAMESPACE}" -k examples/runtime-free-router/sglang
```
