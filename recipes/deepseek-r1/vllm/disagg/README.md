<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## DeepSeek-R1 with vLLM — Disaggregated Serving

This directory contains recipes for deploying DeepSeek-R1 using vLLM in a disaggregated prefill/decode setup on different GPU architectures.

| Recipe | GPUs | Nodes | Architecture | Model Precision | Manifest |
|--------|------|-------|-------------|-----------------|----------|
| Hopper | 32 (16 prefill + 16 decode) | 4× H100/H200 | DP16 + EP | BF16/FP8 | `deploy_hopper_16gpu.yaml` |
| GB200  | 16 (8 prefill + 8 decode)   | 4× GB200     | DP8 + EP  | FP4 (NVFP4) | `deploy_gb200_16gpu.yaml` |

---

### Prerequisites

Follow the Kubernetes deployment guide to install the Dynamo platform and prerequisites (CRDs/operator, etc.):
- `docs/pages/kubernetes/README.md`

### 1) Set namespace

```bash
export NAMESPACE=dynamo-system
kubectl create namespace ${NAMESPACE} || true
```

### 2) Apply Hugging Face secret

Edit your HF token into the provided secret and apply:

```bash
# Option A: Apply YAML (edit the file to set your token)
kubectl apply -f ../../hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}

# Option B: Create directly
# kubectl create secret generic hf-token-secret \
#   --from-literal=HF_TOKEN="<your-hf-token>" \
#   -n ${NAMESPACE}
```

### 3) Provision model cache and download models

Update `storageClassName` in `recipes/deepseek-r1/model-cache/model-cache.yaml` to match your cluster, then apply:

```bash
# PVC for model cache
# Ensure storageClassName in model-cache.yaml matches an available StorageClass on your cluster
kubectl apply -f ../../../deepseek-r1/model-cache/model-cache.yaml -n ${NAMESPACE}

# Download DeepSeek-R1 weights into the cache
kubectl apply -f ../../../deepseek-r1/model-cache/model-download.yaml -n ${NAMESPACE}

# Wait for download job to finish
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s
```

This will populate:
- `/model-cache/deepseek-r1` (BF16 weights, used by Hopper recipe)
- `/model-cache/deepseek-r1-fp4` (NVFP4 weights, used by GB200 recipe)

---

### Hopper Deployment (32× H100/H200)

Deploys across 4 Hopper nodes (32 GPUs total: 16 for prefill, 16 for decode) using 16-way Data-Expert Parallel with BF16/FP8 precision.

**Requirements**: 4 nodes × 8 H100/H200 GPUs, InfiniBand with IBGDA enabled.

```bash
kubectl apply -f ./deploy_hopper_16gpu.yaml -n ${NAMESPACE}
```

**Key settings**:
- `--data-parallel-size 16` across 16 GPUs per role
- `VLLM_MOE_DP_CHUNK_SIZE=384` (tuned for 16× H200)
- InfiniBand RDMA: 8 IB devices per node
- Shared memory: 80Gi

---

### GB200 Deployment (16× GB200)

Deploys across 4 GB200 nodes (16 GPUs total: 8 for prefill, 8 for decode) using 8-way Data-Expert Parallel with NVFP4 precision for native Blackwell FP4 tensor core support.

**Requirements**: 4 nodes × 4 GB200 GPUs with NVLink-C2C connectivity.

```bash
kubectl apply -f ./deploy_gb200_16gpu.yaml -n ${NAMESPACE}
```

**Key settings**:
- Uses FP4-quantized model (`/model-cache/deepseek-r1-fp4`) for native Blackwell FP4 tensor cores
- `--all2all-backend flashinfer_all2allv` — MNNVL-native all2all backend (~21% throughput gain over AllGather-ReduceScatter); used for both prefill and decode (unlike Hopper which uses separate DeepEP backends)
- `--data-parallel-size 8` across 8 GPUs per role
- `--block-size 1` (required for DeepSeek MLA architecture with FP4)
- `VLLM_USE_FLASHINFER_MOE_FP4=1` — activates FlashInfer's TRT-LLM Gen kernels for FP4
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` — optimized Cutlass backend for FP4/FP8 MoE
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=0` — disabled (GB200 has sufficient memory)
- `VLLM_ENABLE_MOE_DP_CHUNK=0` on prefill (disable chunking for throughput)
- `VLLM_MOE_DP_CHUNK_SIZE=512` on decode (tuned for 8× GB200)
- `NCCL_MNNVL_ENABLE=1` — multi-node NVLink
- `NCCL_CUMEM_ENABLE=1` — CUDA unified memory for NVLink-C2C
- Shared memory: 800Gi

**Performance** (from [vLLM blog](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)):
- Prefill: ~26.2K tokens per GPU second (TPGS)
- Decode: ~10.1K TPGS
- 3-5× improvement over H200

---

### Test the deployment

Port-forward and send a test request (works for both Hopper and GB200):

```bash
# For Hopper:
kubectl port-forward svc/vllm-dsr1-frontend 8000:8000 -n ${NAMESPACE} &

# For GB200:
kubectl port-forward svc/vllm-dsr1-gb200-frontend 8000:8000 -n ${NAMESPACE} &
```

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dummy' \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role":"user","content":"Say hello!"}],
    "max_tokens": 64
  }'
```

---

### Notes

- For more details on expert parallel and advanced deployment configurations, refer to [vLLM Expert Parallel Deployment Documentation](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/).
- If your cluster/network requires specific interfaces, adjust environment variables (e.g., `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`) in the manifest accordingly.
- If your storage class differs, update `storageClassName` before applying the PVC.
- **Hopper multi-node**: IBGDA (InfiniBand GPU Direct Async) must be enabled. See [configure_system_drivers.sh](https://github.com/vllm-project/vllm/blob/v0.11.2/tools/ep_kernels/configure_system_drivers.sh) — requires system reboot.
- **GB200 multi-node**: Ensure NVLink-C2C connectivity between nodes. No IBGDA configuration is needed — GB200 uses NVLink-C2C for GPU-to-GPU communication.
- `VLLM_MOE_DP_CHUNK_SIZE` can be tuned further. The Hopper value (384) was chosen as the largest that fits on 16× H200s; the GB200 value (512) was chosen for 8× GB200 with FP4. This value should be greater than per-rank concurrency.
- GB200's NVLink-C2C connection between CPU and GPU makes weight offloading particularly effective. Consider adding `--cpu-offload-gb` for additional KV cache capacity if needed.
