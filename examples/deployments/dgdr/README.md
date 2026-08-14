<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DynamoGraphDeploymentRequest Examples

These manifests demonstrate common ways to generate a `DynamoGraphDeployment` (DGD) with a
`DynamoGraphDeploymentRequest` (DGDR).

## Examples

| File | Description |
|---|---|
| `rapid.yaml` | Generate and deploy a DGD with fast AIConfigurator-backed profiling. |
| `thorough.yaml` | Benchmark candidate configurations on real GPUs before generating the DGD. |
| `planner.yaml` | Generate a disaggregated DGD with SLA-driven Planner autoscaling. |
| `moe-sglang.yaml` | Profile a multinode Mixture-of-Experts model with SGLang. |
| `mocker.yaml` | Generate a simulated deployment for testing without model-serving GPUs. |
| `review-before-deploy.yaml` | Stop after generation so you can inspect the DGD before applying it. |
| `generated-dgd-override.yaml` | Merge KV-aware routing configuration into the generated DGD. |
| `model-cache.yaml` | Mount model weights from a pre-populated shared PVC. |
| `profiling-artifacts.yaml` | Persist detailed profiling output to an existing PVC. |

## Prerequisites

- Install the Dynamo Kubernetes platform and operator.
- Set `NAMESPACE` to the namespace where Dynamo is installed.
- Create `hf-token-secret` in that namespace when the selected model requires authentication.
- For namespace-restricted operator installations, set `hardware.gpuSku`, `hardware.vramMb`, and
  `hardware.numGpusPerNode` explicitly because the operator cannot discover cluster nodes.

Release installations select the matching profiler image automatically when `spec.image` is omitted.
For a local operator build without a known release version, set `spec.image` to a compatible
`dynamo-planner` image.

## Apply an Example

```bash
export NAMESPACE=dynamo-cloud
kubectl apply -f rapid.yaml -n ${NAMESPACE}
```

Watch the request and inspect the generated DGD:

```bash
kubectl get dgdr qwen-rapid -n ${NAMESPACE} -w
kubectl get dgd -n ${NAMESPACE}
```

For an example with `autoApply: false`, extract the generated DGD after the request reaches `Ready`:

```bash
kubectl get dgdr qwen-review -n ${NAMESPACE} \
  -o jsonpath='{.status.profilingResults.selectedConfig}' > generated-dgd.yaml
```

## Example-Specific Setup

- Replace `<commit-hash>` in `model-cache.yaml` and create the referenced `model-cache` PVC before
  applying the manifest.
- Create the `dynamo-pvc` PVC before applying `profiling-artifacts.yaml`. The helper at
  [`deploy/utils/setup_benchmarking_resources.sh`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/utils/setup_benchmarking_resources.sh)
  can create the benchmarking resources.
- `moe-sglang.yaml` is a large multinode example. Adjust its GPU budget, model cache, SLA, and
  scheduling configuration for your cluster before applying it.

See the [DGDR walkthrough](../../../docs/fern/pages/kubernetes/auto-deployment/auto-deploy-with-dgdr.md)
and [DGDR reference](../../../docs/fern/kubernetes/dgdr-reference.mdx)
for field descriptions and lifecycle details.
