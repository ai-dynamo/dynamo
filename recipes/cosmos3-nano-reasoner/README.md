<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano Reasoner

This experimental recipe serves Cosmos3 Nano Reasoner with the released
`nvcr.io/nim/nvidia/cosmos3:2.1.0` NIM image, pinned by digest, and NVIDIA
Dynamo.

## Available configuration

| Configuration | GPUs | Precision | Routing |
| --- | --- | --- | --- |
| `vllm/agg-h200-round-robin` | 8x H200 | FP8 | Round robin, no affinity |
| `vllm/agg-h200-session-affinity` | 8x H200 | FP8 | Round robin with 300-second session affinity |

The workers use the NIM entrypoint for profile selection and launch
`dynamo.vllm` internally. This recipe has been adapted from a deployment
validated on H200. Its AIPerf job runs a deterministic video concurrency
sweep against the aggregated frontend.

## Prerequisites

- A Kubernetes cluster with the Dynamo platform and the
  `nvidia.com/v1beta1` `DynamoGraphDeployment` API.
- Eight available H200 GPUs.
- A ReadWriteMany storage class.
- An NGC API key with access to the Cosmos3 NIM image and model artifacts.

Create the namespace and model-download credential:

```bash
export NAMESPACE="<your-namespace>"
kubectl create namespace "$NAMESPACE"
kubectl -n "$NAMESPACE" create secret generic ngc-api-secret \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

The benchmark dataset is private. Create a read-only Hugging Face token Secret:

```bash
kubectl -n "$NAMESPACE" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN"
```

If the cluster requires registry authentication, add its image-pull Secret
through the cluster-owned Kustomization. Do not add credentials to the
portable `deploy.yaml`. A cluster that requires the NGC API key at worker
runtime must inject `ngc-api-secret` into the Worker through the same private
Kustomization.

The commands below assume the recipe directory as the working directory:

```bash
cd recipes/cosmos3-nano-reasoner
```

## Populate the model cache

First check whether the platform already created the namespace-local cache:

```bash
kubectl -n "$NAMESPACE" get pvc shared-model-cache
```

If the PVC exists, use it and do not apply `model-cache/model-cache.yaml`.
PVC storage class and access mode fields are immutable, and applying the
portable placeholder over a platform-owned claim will fail.

If the PVC does not exist, set `storageClassName` in
`model-cache/model-cache.yaml` to a ReadWriteMany-capable storage class and
create it. Then populate the cache:

```bash
kubectl -n "$NAMESPACE" apply -f model-cache/model-cache.yaml # only when absent
kubectl -n "$NAMESPACE" apply -f model-cache/model-download.yaml
kubectl -n "$NAMESPACE" wait --for=condition=Complete \
  job/cosmos3-nano-reasoner-model-download --timeout=3600s
```

The download Job and serving workers must use the same image, model size,
precision, and cache.

## Deploy and check

Qualify the portable manifest through the target cluster's Kustomization
before applying it. Choose one routing profile:

```bash
export PROFILE=round-robin # or: session-affinity
export DGD="cosmos3-nano-reasoner-agg-h200-${PROFILE}"

kubectl -n "$NAMESPACE" apply -f "vllm/agg-h200-${PROFILE}/deploy.yaml"
kubectl -n "$NAMESPACE" wait --for=condition=Ready \
  "dynamographdeployment/${DGD}" --timeout=1200s
kubectl -n "$NAMESPACE" get pods \
  -l "nvidia.com/dynamo-graph-deployment-name=${DGD}"
```

Forward the generated frontend Service and discover the served model:

```bash
kubectl -n "$NAMESPACE" port-forward \
  "service/${DGD}-frontend" 8000:8000
curl -fsS http://localhost:8000/v1/models
```

## Run the video benchmark

The benchmark downloads `jkiczka/cosmos3-benchmark-videos` to the shared PVC,
serves the videos inside the cluster, and runs AIPerf at concurrency levels
1, 2, 4, 8, and 16. No Hugging Face download is included in measured request
latency. Apply the benchmark profile matching the deployed target:

```bash
kubectl -n "$NAMESPACE" apply -f "perf/profiles/${PROFILE}.yaml"
kubectl -n "$NAMESPACE" delete job cosmos3-nano-reasoner-perf \
  --ignore-not-found
kubectl -n "$NAMESPACE" apply -f perf/perf.yaml
kubectl -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-perf
```

Artifacts are written to
`/shared-model-cache/perf/cosmos3-nano-reasoner/<profile>/<epoch>_<job-name>/`.
The Job records the pinned dataset commit, workload settings, and AIPerf
exports. For session affinity, every request carries
`X-Dynamo-Session-ID: 2fps-30s:<video-filename>`; pure round robin sends no
session-affinity header. See [`perf/README.md`](perf/README.md) for artifact
collection and comparison guidance.

## Run the video smoke test

The smoke test downloads one real dataset video, serves it inside the cluster,
sends a non-streaming multimodal chat completion, and fails unless the model
returns a non-empty answer:

```bash
kubectl -n "$NAMESPACE" apply -f "perf/profiles/${PROFILE}.yaml"
kubectl -n "$NAMESPACE" delete job cosmos3-nano-reasoner-smoke \
  --ignore-not-found
kubectl -n "$NAMESPACE" apply -f perf/smoke.yaml
kubectl -n "$NAMESPACE" logs -f job/cosmos3-nano-reasoner-smoke
kubectl -n "$NAMESPACE" wait --for=condition=Complete \
  job/cosmos3-nano-reasoner-smoke --timeout=1800s
```

