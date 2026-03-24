<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

### DeepSeek-R1 with vLLM — Disaggregated on 32x Hopper

This recipe deploys DeepSeek-R1 using vLLM in a disaggregated prefill/decode setup across four Hopper nodes (32 GPUs total: 16 for prefill, 16 for decode).

- Model cache PVC + download job: `recipes/deepseek-r1/model-cache/`
- Deployment manifest: `recipes/deepseek-r1/vllm/disagg/deploy_hopper_16gpu.yaml`

### 0) Prerequisites: Install the platform

Follow the Kubernetes deployment guide to install the Dynamo platform and prerequisites (CRDs/operator, etc.):
- `docs/kubernetes/README.md`

Ensure you have a GPU-enabled cluster with sufficient capacity (32x H100/H200 "Hopper" across 4 nodes), and that the NVIDIA GPU Operator is healthy.

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
- `/model-cache/deepseek-r1`
- `/model-cache/deepseek-r1-fp4`

### 4) Deploy vLLM (Disaggregated, 16-way Data-Expert Parallel)

Apply the multi-node disaggregated deployment:

```bash
kubectl apply -f ./deploy_hopper_16gpu.yaml -n ${NAMESPACE}
```

The manifest runs separate prefill and decode workers across multiple nodes, each mounting the shared model cache, with settings tuned for Hopper GPUs.

Test the deployment locally by port-forwarding and sending a request:

```bash
# Port-forward the frontend Service to localhost:8000 (replace <frontend-svc> with the actual Service name)
kubectl port-forward svc/vllm-dsr1-frontend 8000:8000 -n ${NAMESPACE} &
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



### Notes
- For more details on expert parallel and advanced deployment configurations, refer to [vLLM Expert Parallel Deployment Documentation](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/).
- If your cluster/network requires specific interfaces, adjust environment variables (e.g., `NCCL_SOCKET_IFNAME`) in the manifest accordingly.
- If your storage class differs, update `storageClassName` before applying the PVC.
- **If you want to run multinode deployments, IBGDA (InfiniBand GPU Direct Async) must be enabled on your nodes.** To enable IBGDA, you can follow this configuration script: [configure_system_drivers.sh](https://github.com/vllm-project/vllm/blob/v0.11.2/tools/ep_kernels/configure_system_drivers.sh). The script configures NVIDIA driver parameters and requires a system reboot to take effect.
- `VLLM_MOE_DP_CHUNK_SIZE` can be tuned further. The value 384 was chosen to be largest possible that still can be deployed on 16 H200s. This value should be greater than per rank concurrency.


