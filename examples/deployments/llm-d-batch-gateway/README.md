<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# llm-d Batch Gateway with Dynamo

Use this example to deploy llm-d Batch Gateway in front of a dedicated NVIDIA Dynamo
worker pool. The steps submit an OpenAI Batch job, run its inference requests on
Dynamo, and retrieve the results through the Batch API.

This is an experimental validation example, not a supported production recipe.

## Before You Start

You need:

- A Kubernetes cluster with the Dynamo platform installed.
- One available GPU.
- `kubectl`, Helm 3, and Python 3.9 or newer.
- A `model-cache` PVC for the model worker.
- An `hf-token-secret` secret in the target namespace.
- A default storage class that supports `ReadWriteMany` persistent volumes.

The example uses the following versions:

| Component | Version |
| --- | --- |
| Dynamo runtime images | `1.3.0` |
| llm-d Batch Gateway chart and images | `0.3.0` |
| llm-d Async chart and image | `v0.9.0` |
| Valkey | `8.0.10-alpine` |
| Model | `Qwen/Qwen3-0.6B` |

Update the chart version and all three Batch Gateway image tags together, then
rerun the complete example. Update the Dynamo frontend and worker images
together and rerun both a direct chat completion and the batch example.

## Recreate the Example

### 1. Deploy the Dedicated Dynamo Backend

Set a namespace and apply the dedicated backend:

```bash
cd examples/deployments/llm-d-batch-gateway
export NAMESPACE=your-namespace

kubectl apply -n "${NAMESPACE}" -f dynamo.yaml
kubectl wait -n "${NAMESPACE}" \
  --for=condition=Ready \
  dynamographdeployment/qwen3-0-6b-batch \
  --timeout=900s
```

### 2. Verify Inference Before Adding the Batch Layer

Verify the backend directly before adding the batch layer. Keep the port
forward running while you send the request:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  service/qwen3-0-6b-batch-frontend 8000:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

### 3. Deploy Batch Metadata and File Storage

Apply the validation-only Valkey service and shared file PVC. Valkey provides
the Redis-compatible metadata service expected by llm-d Batch Gateway. The
v0.3.0 chart treats `global.secretName` as a reference to a pre-existing Secret;
it does not create or own that Secret. `batch-infra.yaml` creates the referenced
Secret, and the cleanup order below removes it after the Helm release:

```bash
kubectl apply -n "${NAMESPACE}" -f batch-infra.yaml
kubectl rollout status -n "${NAMESPACE}" \
  statefulset/batch-gateway-valkey \
  --timeout=180s
```

### 4. Deploy the synchronous control

Install the pinned upstream chart in its default synchronous dispatch mode. The
values file routes exactly one model to the dedicated Dynamo frontend:

```bash
helm upgrade --install batch-gateway \
  oci://ghcr.io/llm-d/charts/batch-gateway \
  --version 0.3.0 \
  --namespace "${NAMESPACE}" \
  --values batch-gateway-values.yaml
```

After Helm reports a successful deployment, wait for both components:

```bash
kubectl rollout status -n "${NAMESPACE}" \
  deployment/batch-gateway-apiserver \
  --timeout=180s
kubectl rollout status -n "${NAMESPACE}" \
  deployment/batch-gateway-processor \
  --timeout=180s
```

### 5. Run the synchronous batch lifecycle

Forward the Batch API in one terminal:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  service/batch-gateway-apiserver 8001:8000
```

Run the example client in another terminal:

```bash
python3 run_example.py --base-url http://127.0.0.1:8001
```

The example client performs three jobs:

1. It uploads `batch-input.jsonl`, waits for two successful requests, and
   retrieves the output file.
2. It submits an unmapped model and retrieves the resulting error file.
3. It submits a longer batch, cancels it, and waits for `cancelled`.

The command exits with an error if a job reaches an unexpected terminal state
or returns unexpected request counts or output identifiers.

### 6. Switch to llm-d Async dispatch

Install the pinned llm-d Async chart. The Async Processor reads requests from a
Redis sorted set, sends each request to the same Dynamo frontend, and writes
results to the configured Redis list:

```bash
helm upgrade --install async-dispatch \
  oci://ghcr.io/llm-d/charts/llm-d-async \
  --version v0.9.0 \
  --namespace "${NAMESPACE}" \
  --values llm-d-async-values.yaml

kubectl rollout status -n "${NAMESPACE}" \
  deployment/async-dispatch-llm-d-async \
  --timeout=180s
```

Switch the Batch Processor to Async dispatch. The second values file is an
overlay; keep the base values file first:

```bash
helm upgrade --install batch-gateway \
  oci://ghcr.io/llm-d/charts/batch-gateway \
  --version 0.3.0 \
  --namespace "${NAMESPACE}" \
  --values batch-gateway-values.yaml \
  --values batch-gateway-async-values.yaml

kubectl rollout status -n "${NAMESPACE}" \
  deployment/batch-gateway-processor \
  --timeout=180s
```

Keep the Batch API port-forward running and repeat the lifecycle:

```bash
python3 run_example.py --base-url http://127.0.0.1:8001
```

The same client validates both modes. In Async mode, Batch Gateway continues to
own files, job state, cancellation, and output assembly. llm-d Async owns the
request and result queues and sends ordinary OpenAI-compatible requests to the
Dynamo frontend. Dynamo remains responsible for routing each request to a
worker.

## What This Example Covers

- Uploading a JSONL file and creating, polling, and cancelling Batch jobs.
- Retrieving successful output and request-level error files.
- Running real `/v1/chat/completions` inference through a Dynamo frontend and
  one-GPU vLLM worker.
- Running the same lifecycle with synchronous dispatch and with requests routed
  through llm-d Async Redis queues.
- Isolating batch traffic in a `DynamoGraphDeployment` that does not share a
  frontend, router, or worker with an online pool.
- Pinning the Dynamo and llm-d versions used by the example.
- Reproducing the lifecycle with the standalone example client.

This workflow was validated with one B200 GPU, Dynamo `1.3.0`, llm-d Batch
Gateway `0.3.0`, llm-d Async `v0.9.0`, and Qwen3-0.6B.

## What This Example Does Not Cover

- Cancelling requests that are already running on Dynamo. Cancellation was
  validated before the requests were dispatched.
- Propagating a Dynamo-generated backend error into the Batch error file. The
  example client uses an llm-d unmapped-model error.
- Fault-injecting authentication forwarding, request timeouts, retries, worker
  restarts, or processor recovery.
- Expiration and garbage collection. The example disables the garbage
  collector while keeping the `24h` Batch completion window.
- TLS, ingress, production authentication, metadata-store authentication, HA,
  or multi-replica Batch Gateway components.
- Shared online and offline capacity or Planner-controlled dispatch budgets.
- Durable request redelivery after an llm-d Async failure, multi-replica Batch
  Processor result routing, or resuming a batch after Processor pod or node
  loss. Track these limitations in
  [llm-d Async issue 404](https://github.com/llm-d/llm-d-async/issues/404),
  [Batch Gateway issue 644](https://github.com/llm-d/llm-d-batch-gateway/issues/644),
  and [Batch Gateway issue 645](https://github.com/llm-d/llm-d-batch-gateway/issues/645).
- Completions, embeddings, multimodal inputs, Parquet, or object storage.
- Performance benchmarking, CI coverage, an upgrade matrix, or a supported
  deployment recipe.

The checked-in worker manifest uses the standard `nvidia.com/gpu` resource. The
validation cluster required a temporary Kubernetes Dynamic Resource Allocation
(DRA) override, so the checked-in GPU allocation was server-side validated but
was not exercised unchanged.

## Configuration Boundaries

| Area | Behavior in this example |
| --- | --- |
| Authentication | `X-MaaS-Username` selects the llm-d tenant. `Authorization` is passed to Dynamo, but the example does not deploy an authentication boundary. |
| Model | `Qwen/Qwen3-0.6B` maps directly to `qwen3-0-6b-batch-frontend`. |
| Request timeout | The processor allows five minutes and up to three retries per inference request. |
| Cancellation | llm-d stops queued dispatch and assembles the terminal result. Requests already accepted by Dynamo can still finish. |
| Output and errors | llm-d stores and assembles Batch files. Dynamo returns ordinary OpenAI-compatible responses. |
| Dispatch | Sync mode calls the Dynamo frontend directly. Async mode uses Redis request/result queues and a constant-open gate before calling the same frontend. |
| Capacity | The processor can reach only the dedicated Dynamo frontend and worker in this graph. |

llm-d owns the Batch API, file and job state, queueing, cancellation, recovery,
and output assembly. Dynamo owns inference execution and dedicated worker
capacity. Dynamo's disabled Batch route skeleton is not enabled or used by this
example.

## Cleanup

Delete only resources created by this example:

```bash
helm uninstall async-dispatch -n "${NAMESPACE}" --ignore-not-found
helm uninstall batch-gateway -n "${NAMESPACE}"
kubectl delete -n "${NAMESPACE}" -f batch-infra.yaml
kubectl delete -n "${NAMESPACE}" -f dynamo.yaml
```

The StatefulSet PVC may remain after deletion. Inspect it before removing it:

```bash
kubectl get pvc -n "${NAMESPACE}" \
  -l app.kubernetes.io/name=batch-gateway-valkey
```

After confirming the PVC belongs to this example, remove it:

```bash
kubectl delete pvc data-batch-gateway-valkey-0 -n "${NAMESPACE}"
```

## Further Reading

- [llm-d Batch Gateway](https://github.com/llm-d/llm-d-batch-gateway)
- [Dynamo Kubernetes quickstart](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
- [Dynamo vLLM deployment examples](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/README.md)
