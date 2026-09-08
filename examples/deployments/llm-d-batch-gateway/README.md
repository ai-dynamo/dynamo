<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Offline Batch Inference with Dynamo

Use this example to submit OpenAI Batch jobs to a dedicated NVIDIA Dynamo worker pool. The batch layer manages the job lifecycle and sends each inference request through the Dynamo frontend and router.

> [!WARNING]
> This is an experimental validation example, not a supported production recipe.

## Before You Start

You need:

- A Kubernetes cluster with the Dynamo platform installed.
- One available GPU.
- `kubectl`, Helm 3, and Python 3.9 or newer.
- A `model-cache` PVC for the model worker.
- An `hf-token-secret` secret in the target namespace.
- A default storage class that supports `ReadWriteMany` persistent volumes.
- A Prometheus server installed at the service address used in `llm-d-async-values.yaml` and configured to scrape Dynamo frontend pods. The [Dynamo observability installation guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/installation/observability.md) creates this service and the required PodMonitor setup.

The example uses the following versions:

| Component | Version |
| --- | --- |
| Dynamo runtime images | `1.5.0` |
| Batch Gateway chart and images | `0.3.0` |
| Async Processor chart and image | `v0.9.0` |
| Valkey | `8.0.10-alpine` |
| Model | `Qwen/Qwen3-0.6B` |

The readiness-gated Async path requires a Dynamo build that exposes `dynamo_frontend_model_ready`. If the `1.5.0` runtime image has not been published yet, use frontend and worker images built from a Dynamo revision that contains that metric. Older runtime images do not expose the series, so the fail-closed gate remains at zero.

Update the chart version and all three Batch Gateway image tags together, then
rerun the complete example. Update the Dynamo frontend and worker images
together and rerun both a direct chat completion and the batch example.

## Recreate the Example

### 1. Deploy the Dedicated Dynamo Backend

Set a namespace and apply the dedicated backend:

```bash
cd examples/deployments/llm-d-batch-gateway
export NAMESPACE=dynamo-batch-example

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

Verify that the frontend exposes the same model readiness decision used by request routing:

```bash
curl --fail --silent http://127.0.0.1:8000/metrics \
  | grep 'dynamo_frontend_model_ready{model="Qwen/Qwen3-0.6B"} 1'
```

### 3. Deploy Batch Metadata and File Storage

Apply the validation-only Valkey service and shared file PVC. Valkey provides
the Redis-compatible metadata service expected by Batch Gateway. The
v0.3.0 chart treats `global.secretName` as a reference to a pre-existing Secret;
it does not create or own that Secret. `batch-infra.yaml` creates the referenced
Secret, and the cleanup order below removes it after the Helm release:

```bash
kubectl apply -n "${NAMESPACE}" -f batch-infra.yaml
kubectl rollout status -n "${NAMESPACE}" \
  statefulset/batch-gateway-valkey \
  --timeout=180s
```

### 4. Deploy the Synchronous Control

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

### 5. Run the Synchronous Batch Lifecycle

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

### 6. Switch to Asynchronous Dispatch

Install the pinned Async Processor chart. The processor reads requests from a Redis sorted set, sends each request to the same Dynamo frontend, and writes results to the configured Redis list. Its Prometheus-query gate reads `dynamo_frontend_model_ready` and fails closed with a zero dispatch budget if Prometheus or the metric is unavailable. A registered model with no complete serving topology also reports zero, so queued requests remain in Redis instead of being sent to a frontend that cannot route them. The example uses a one-second metric cache; with the Dynamo PodMonitor's default five-second scrape interval, dispatch normally opens within roughly six seconds of the first serving unit registering:

```bash
helm upgrade --install async-dispatch \
  oci://ghcr.io/llm-d/charts/llm-d-async \
  --version v0.9.0 \
  --namespace "${NAMESPACE}" \
  --values llm-d-async-values.yaml \
  --set-string ap.redis.gateParams.query="min(dynamo_frontend_model_ready{model=\"Qwen/Qwen3-0.6B\"\\,namespace=\"${NAMESPACE}\"})"

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

The same client validates both modes. In asynchronous mode, Batch Gateway continues to own files, job state, cancellation, and output assembly. The Async Processor owns the request and result queues and sends ordinary OpenAI-compatible requests to the Dynamo frontend. Dynamo remains responsible for routing each request to a worker.

### 7. Validate Dispatch Across Zero Capacity

This check proves that Async preserves a submitted batch while Dynamo has no routable worker, then begins dispatch after the first serving unit registers. It does not wait for every desired worker replica to become ready.

Forward the Async Processor metrics in a third terminal. Keep this port-forward running for the rest of the check:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  deployment/async-dispatch-llm-d-async 9090:9090
```

Scale the worker component to zero:

```bash
kubectl patch -n "${NAMESPACE}" \
  dynamographdeployment/qwen3-0-6b-batch \
  --type=json \
  --patch='[{"op":"replace","path":"/spec/components/1/replicas","value":0}]'
```

Wait for the gate to close. This loop fails if the metrics request fails and exits successfully only after the dispatch budget is exactly `0`. The source-availability metric is diagnostic: `1` means Async read an explicit readiness value of zero, and `0` means the series was absent and Async used its fail-closed fallback:

```bash
for attempt in $(seq 1 60); do
  metrics="$(curl --fail --silent http://127.0.0.1:9090/metrics)" || exit 1
  if printf '%s\n' "${metrics}" | awk '$1 ~ /^llm_d_async_async_dispatch_budget{/ && $2 == 0 { found=1 } END { exit !found }'; then
    break
  fi
  if [ "${attempt}" -eq 60 ]; then
    echo "dispatch budget did not reach zero" >&2
    exit 1
  fi
  sleep 2
done

printf '%s\n' "${metrics}" \
  | grep '^llm_d_async_async_gate_metric_source_available{'
```

With the Batch API port-forward still running, start a successful batch in another terminal. The client remains in a non-terminal state while the dispatch budget is zero:

```bash
python3 run_example.py \
  --base-url http://127.0.0.1:8001 \
  --success-only
```

Restore one worker. The pending client should complete after the worker registers and Prometheus observes the readiness value of `1`:

```bash
kubectl patch -n "${NAMESPACE}" \
  dynamographdeployment/qwen3-0-6b-batch \
  --type=json \
  --patch='[{"op":"replace","path":"/spec/components/1/replicas","value":1}]'

kubectl wait -n "${NAMESPACE}" \
  --for=condition=Ready \
  dynamographdeployment/qwen3-0-6b-batch \
  --timeout=900s
```

## What This Example Covers

- Uploading a JSONL file and creating, polling, and cancelling Batch jobs.
- Retrieving successful output and request-level error files.
- Running real `/v1/chat/completions` inference through a Dynamo frontend and
  one-GPU vLLM worker.
- Running the same lifecycle with synchronous dispatch and with requests routed
  through Redis request and result queues.
- Holding queued requests while Dynamo has no routable worker and resuming dispatch after the first complete serving unit registers.
- Isolating batch traffic in a `DynamoGraphDeployment` that does not share a
  frontend, router, or worker with an online pool.
- Pinning the Dynamo, Batch Gateway, and Async Processor versions used by the example.
- Reproducing the lifecycle with the standalone example client.

This workflow was validated on one H200 GPU with a Dynamo `1.5.0` CI image, Batch Gateway `0.3.0`, Async Processor `v0.9.0`, and Qwen3-0.6B.

## What This Example Does Not Cover

- Cancelling requests that are already running on Dynamo. Cancellation was
  validated before the requests were dispatched.
- Propagating a Dynamo-generated backend error into the Batch error file. The
  example client uses a Batch Gateway unmapped-model error.
- Fault-injecting authentication forwarding, request timeouts, retries, worker
  restarts, or processor recovery.
- Expiration and garbage collection. The example disables the garbage
  collector while keeping the `24h` Batch completion window.
- TLS, ingress, production authentication, metadata-store authentication, HA,
  or multi-replica Batch Gateway components.
- Shared online and offline capacity or Planner-controlled dispatch budgets.
- Durable request redelivery after an Async Processor failure, multi-replica Batch
  Processor result routing, or resuming a batch after Processor pod or node
  loss. Track these limitations in
  [Async Processor issue 404](https://github.com/llm-d/llm-d-async/issues/404),
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
| Authentication | `X-MaaS-Username` selects the Batch API tenant. `Authorization` is passed to Dynamo, but the example does not deploy an authentication boundary. |
| Model | `Qwen/Qwen3-0.6B` maps directly to `qwen3-0-6b-batch-frontend`. |
| Request timeout | The processor allows five minutes and up to three retries per inference request. |
| Cancellation | Batch Gateway stops queued dispatch and assembles the terminal result. Requests already accepted by Dynamo can still finish. |
| Output and errors | Batch Gateway stores and assembles Batch files. Dynamo returns ordinary OpenAI-compatible responses. |
| Dispatch | Sync mode calls the Dynamo frontend directly. Async mode uses Redis request/result queues and opens its fail-closed Prometheus-query gate only while the model is routable through the Dynamo frontend. |
| Capacity | The processor can reach only the dedicated Dynamo frontend and worker in this graph. |

Batch Gateway owns the Batch API, file and job state, queueing, cancellation, recovery, and output assembly. Dynamo owns inference execution and dedicated worker capacity. Dynamo's disabled Batch route skeleton is not enabled or used by this example.

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

- [Batch Gateway upstream repository](https://github.com/llm-d/llm-d-batch-gateway)
- [Dynamo Kubernetes quickstart](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/getting-started/quickstart.mdx)
- [Dynamo vLLM deployment examples](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/README.md)
