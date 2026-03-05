---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploying Your First Model
---

# Deploying Your First Model

End-to-end tutorial for deploying `Qwen/Qwen3-0.6B` on Kubernetes using Dynamo's recommended
`DynamoGraphDeploymentRequest` (DGDR) workflow — from zero to your first inference response.

> [!NOTE]
> This guide assumes you have already completed the
> [platform installation](installation-guide.md) and that the Dynamo operator and CRDs are
> running in your cluster.

## What is a DynamoGraphDeploymentRequest?

A `DynamoGraphDeploymentRequest` (DGDR) is Dynamo's **deploy-by-intent** API. You describe what
you want to run and your performance targets; Dynamo's profiler determines the optimal
configuration automatically, then creates the live deployment for you.

| | DGDR (this guide) | DGD (manual) |
|---|---|---|
| **You provide** | Model + optional SLA targets | Full deployment spec |
| **Profiling** | Automated | You bring your own config |
| **Best for** | Getting started, SLA-driven deployments | Fine-grained control |

For a deeper comparison, see [Understanding Dynamo's Custom Resources](README.md#understanding-dynamos-custom-resources).

## Prerequisites

Before starting, confirm:

- Platform installed: `kubectl get pods -n ${NAMESPACE}` shows operator pods `Running`
- CRDs present: `kubectl get crd | grep dynamo` shows `dynamographdeploymentrequests.nvidia.com`
- NGC credentials configured (for pulling runtime images)
- `kubectl` and `helm` available in your shell

Set these variables once — they are referenced throughout the guide:

```bash
export NAMESPACE=dynamo-system      # namespace where the platform is installed
export RELEASE_VERSION=1.x.x       # match the installed platform version (e.g. 1.0.0)
export HF_TOKEN=<your-hf-token>    # HuggingFace token
```

> [!TIP]
> `Qwen/Qwen3-0.6B` is a public model. A HuggingFace token is not strictly required to download
> it, but is recommended to avoid rate limiting.

## Step 1: Configure Namespace and Secrets

```bash
# Create the namespace (idempotent — safe to run even if it already exists)
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Create the HuggingFace token secret for model download
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n ${NAMESPACE}
```

Verify the secret was created:

```bash
kubectl get secret hf-token-secret -n ${NAMESPACE}
```

## Step 2: Create the DynamoGraphDeploymentRequest

Save the following as `qwen3-first-model.yaml`, replacing `RELEASE_VERSION` with your version
(e.g. `1.0.0`):

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-first-model
spec:
  # Model to profile and deploy
  model: Qwen/Qwen3-0.6B

  # Profiling job container image — must match your installed platform version
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:RELEASE_VERSION"
```

Apply it:

```bash
kubectl apply -f qwen3-first-model.yaml -n ${NAMESPACE}
```

### Field reference

| Field | Required | Default | Purpose |
|---|---|---|---|
| `model` | Yes | — | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `image` | No | — | Container image for the profiling job (`dynamo-frontend`) |
| `backend` | No | `auto` | Inference engine (`auto`, `vllm`, `sglang`, `trtllm`) |
| `searchStrategy` | No | `rapid` | Profiling depth — `rapid` (~30s, AIC simulation) or `thorough` (2–4h, real GPUs) |
| `autoApply` | No | `true` | Automatically create and start the deployment after profiling |
| `sla` | No | — | Target latency (TTFT, ITL in ms) for profiler optimization |
| `workload` | No | — | Expected traffic shape (ISL, OSL, request rate) |
| `hardware` | No | auto-detected | GPU SKU and count override; required when GPU discovery is disabled |
| `hardware.maxGpus` | No | — | Cap on the number of GPUs the profiler may use when auto-discovering the cluster GPU count. Set this to avoid the profiler recommending a deployment that consumes all available cluster GPUs. |

For the full spec reference, see the [DGDR API Reference](api-reference.md) and
[Profiler Guide](../components/profiler/profiler-guide.md).

> [!IMPORTANT]
> If you are using a **namespace-scoped operator** with GPU discovery disabled, you must also
> provide explicit hardware info or the DGDR will be rejected at admission:
>
> ```yaml
> spec:
>   ...
>   hardware:
>     numGpusPerNode: 1
>     gpuSku: "H100-SXM5-80GB"
>     vramMb: 81920
> ```
>
> See the [installation guide](installation-guide.md#gpu-discovery-for-dynamographdeploymentrequests-with-namespace-scoped-operators)
> for details.

## Step 3: Monitor Profiling Progress

Watch the DGDR status in real time:

```bash
kubectl get dynamographdeploymentrequest qwen3-first-model -n ${NAMESPACE} -w
```

The `PHASE` column progresses through:

| Phase | What is happening |
|---|---|
| `Pending` (condition: `DiscoveringHardware`) | Spec validated; operator is discovering GPU hardware and preparing the profiling job |
| `Profiling` | Profiling job is running (AIC simulation or real-GPU sweep) |
| `Deploying` | Profiling complete; creating the `DynamoGraphDeployment` |
| `Deployed` | DGD is running and healthy |
| `Failed` | Unrecoverable error — check events for details |

> [!TIP]
> `Deployed` is the success terminal state when `autoApply: true` (the default).
> If you set `autoApply: false`, the phase stops at `Ready` — profiling is complete and the
> generated DGD spec is stored in `.status`, but no deployment is created automatically.

For a full status summary and events:

```bash
kubectl describe dynamographdeploymentrequest qwen3-first-model -n ${NAMESPACE}
```

To follow the profiling job logs:

```bash
# Find the profiling pod
kubectl get pods -n ${NAMESPACE} -l nvidia.com/dgdr-name=qwen3-first-model

# Stream its logs
kubectl logs -f <profiling-pod-name> -n ${NAMESPACE}
```

> [!TIP]
> With `searchStrategy: rapid`, profiling typically completes in under 15 minutes on a single GPU.

## Step 4: Verify the Deployment

Once the DGDR reaches `Ready`, the `DynamoGraphDeployment` has been created automatically.
Check that everything is running:

```bash
# See the auto-created DGD
kubectl get dynamographdeployment -n ${NAMESPACE}

# Confirm all pods are Running
kubectl get pods -n ${NAMESPACE}
```

Wait until pods are ready:

```bash
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-deployment=qwen3-first-model \
  -n ${NAMESPACE} \
  --timeout=600s
```

Find the frontend service name:

```bash
kubectl get svc -n ${NAMESPACE} | grep frontend
```

## Step 5: Send Your First Request

Port-forward to the frontend and send an inference request:

```bash
# Start port-forward (replace <frontend-service-name> with the name from Step 4)
kubectl port-forward svc/<frontend-service-name> 8000:8000 -n ${NAMESPACE} &

# Confirm the model is available
curl http://localhost:8000/v1/models

# Send a chat completion request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is NVIDIA Dynamo?"}],
    "max_tokens": 200
  }'
```

A successful response looks like:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "Qwen/Qwen3-0.6B",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "NVIDIA Dynamo is a high-performance inference framework..."
    }
  }]
}
```

Your first model is now live.

## Cleanup

To remove the deployment and profiling artifacts:

```bash
kubectl delete dynamographdeploymentrequest qwen3-first-model -n ${NAMESPACE}
```

> [!NOTE]
> Deleting a DGDR does **not** delete the `DynamoGraphDeployment` it created. The DGD persists
> independently so it can continue serving traffic.

## Troubleshooting

**DGDR stuck in `Pending`**

```bash
kubectl describe dynamographdeploymentrequest qwen3-first-model -n ${NAMESPACE}
# Check the Events section at the bottom
```

Common causes: no available GPU nodes, image pull failure (check NGC credentials or image tag),
missing `hardware` config for a namespace-scoped operator.

**Profiling job fails**

```bash
kubectl get pods -n ${NAMESPACE} -l nvidia.com/dgdr-name=qwen3-first-model
kubectl logs <profiling-pod-name> -n ${NAMESPACE}
# If the pod has already exited:
kubectl logs <profiling-pod-name> -n ${NAMESPACE} --previous
```

**Pods not starting after profiling**

```bash
kubectl describe pod <pod-name> -n ${NAMESPACE}
# Look for ImagePullBackOff, OOMKilled, or Insufficient resources
```

**Model not responding after port-forward**

```bash
# Check frontend is ready
kubectl get pods -n ${NAMESPACE} | grep frontend

# Check frontend logs
kubectl logs <frontend-pod-name> -n ${NAMESPACE}
```

## Next Steps

- **Tune for production SLAs**: Add `sla` (TTFT, ITL) and `workload` (ISL, OSL) targets to
  your DGDR so the profiler optimizes for your specific traffic. See the
  [Profiler Guide](../components/profiler/profiler-guide.md) for the full configuration
  reference and picking modes.
- **Scale the deployment**: [Autoscaling guide](autoscaling.md)
- **SLA-aware autoscaling**: Enable the Planner via `features.planner` in the DGDR —
  see the [Planner Guide](../components/planner/planner-guide.md).
- **Inspect the generated config**: Set `autoApply: false` and extract the DGD spec with
  `kubectl get dgdr <name> -o jsonpath='{.status.profilingResults.selectedConfig}'`
  before deploying.
- **Direct control**: [Creating Deployments](deployment/create-deployment.md) — write your own
  `DynamoGraphDeployment` spec for full customization.
- **Monitor performance**: [Observability](observability/metrics.md)
- **Try specific backends**: [vLLM](../backends/vllm/README.md),
  [SGLang](../backends/sglang/README.md), [TensorRT-LLM](../backends/trtllm/README.md)
