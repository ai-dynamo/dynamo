<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning vLLM Recipes

Dynamo vLLM recipes for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`. Every hardware family has
aggregate configurations with MTP, DFlash, and DSpark speculative decoding.
Disaggregated 1-prefill/1-decode configurations are provided for selected
hardware and speculative decoding combinations.

## Configurations

| Hardware | Aggregate recipes | Disaggregated recipes | P/D transport |
|---|---|---|---|
| H100 | `agg-h100-{mtp,dflash,dspark}` | `disagg-h100-{mtp,dflash,dspark}` | AWS EFA / Libfabric |
| H200 | `agg-h200-{mtp,dflash,dspark}` | `disagg-h200-{mtp,dflash,dspark}` | UCX over IB/RDMA |
| B200 | `agg-b200-{mtp,dflash,dspark}` | `disagg-b200-dspark` | UCX over IB/RDMA |
| GB200 | `agg-gb200-{mtp,dflash,dspark}` | `disagg-gb200-{dflash,dspark}` | UCX/NIXL, cluster-specific fabric resources |

Each recipe identifier corresponds to `identifier/deploy.yaml` in this directory.

### KV-aware aggregate DSpark variants

For users who want the highest-performing aggregate DSpark shape with KV-aware routing ready to deploy, the repo also includes event-backed KV-router variants. These keep the vLLM aggregate DSpark worker configuration, run four aggregate replicas, and publish KV-cache events to the frontend KV router. Start benchmark concurrency at 4x the matching non-KV aggregate recommendation.

| Recipe | Worker replicas | Base concurrency | Suggested KV-router concurrency |
|---|---:|---:|---:|
| `agg-h100-dspark-kv-router` | 4 | 20 | 80 |
| `agg-h200-dspark-kv-router` | 4 | 20 | 80 |
| `agg-b200-dspark-kv-router` | 4 | 24 | 96 |

These recipes are provided for workloads that can benefit from distributing
reusable KV state across several workers, for example when a single worker sees
enough distinct long-prefix families to create KV-cache pressure. For the current
validation trace, KV-aware routing was not promoted over the single-worker
aggregate DSpark recipe: even H100, the lowest-VRAM SKU tested here, peaked at
only 38.1% GPU KV-cache usage in the one-worker full-trace run. The single
worker therefore had enough cache headroom to keep useful prefixes resident
without meaningful eviction pressure, while adding replicas split work across
more GPUs.

## Model configuration

| Recipe type | Target model | Draft model |
|---|---|---|
| MTP | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | Embedded MTP |
| DFlash | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` |
| DSpark | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` |

## Prerequisites

1. Dynamo Platform and the `DynamoGraphDeployment` CRD installed on the target
   Kubernetes cluster.
2. The shared model cache created from `../model-cache/model-cache.yaml` and
   populated with `../model-cache/model-download.yaml`.
3. A Hugging Face secret named `hf-token-secret` and an image-pull secret when
   required by the cluster.
4. For H100 disaggregated recipes, AWS EFA userspace must already be available
   in the runtime image or installed through your cluster's approved bootstrap
   flow. These manifests request `vpc.amazonaws.com/efa` and select the NIXL
   LIBFABRIC backend, but they do not install EFA packages during pod startup.
   Use `check-efa-userspace.sh` to validate a prepared image or running pod. If
   you use its install mode, set `EFA_INSTALLER_VERSION` and
   `EFA_INSTALLER_SHA256` to pin and verify the installer archive.

```bash
kubectl exec -i -n "${NAMESPACE}" <h100-worker-pod> -- \
  bash -s -- check < check-efa-userspace.sh
```

## Quick start

```bash
export NAMESPACE=your-namespace
GPU=h100        # h100, h200, b200, or gb200
MODE=agg        # agg or disagg
SPECDEC=dflash  # choose a valid speculative decoding mode from the table

kubectl apply -n "${NAMESPACE}" -f "${MODE}-${GPU}-${SPECDEC}/deploy.yaml"
kubectl get dgd -n "${NAMESPACE}" "vllm-${MODE}-${GPU}-${SPECDEC}" -w
```

Port-forward the frontend service:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  "svc/vllm-${MODE}-${GPU}-${SPECDEC}-frontend" 8000:8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

## Synthetic acceptance configuration

Each manifest defines two draft configurations:

- `speculative-config` is the standard serving configuration.
- `speculative-config-synthetic` enables synthetic rejection sampling for
  benchmark reproduction.

Select the synthetic configuration by changing the `SPECULATIVE_CONFIG`
ConfigMap key.

| Family | Aggregate draft tokens / expected AL | Disaggregated draft tokens / expected AL |
|---|---:|---:|
| MTP H100/H200 | 7 / 3.687 | 5 / 3.421 |
| MTP B200/GB200 | 3 / 2.874 | Not provided |
| DFlash H100/H200/B200/GB200 | 5 / 3.18 | H100/H200: 3 / 2.73; GB200: 7 / 3.41 |
| DSpark | 7 / 3.69 | 1 / 1.83 |

## Notes

- GB200 recipes use ARM-node scheduling.
- B200 DFlash recipes disable the FlashInfer FP8 ScaledMM linear kernel.
- Disaggregated recipes require matching speculative-decoding settings on the
  prefill and decode workers for compatible NIXL cache metadata.
- H100 disaggregated recipes require the EFA userspace stack before deployment.
- Replace `your-image-pull-secret` and any cluster-specific networking resource
  names with values available in the target cluster.
