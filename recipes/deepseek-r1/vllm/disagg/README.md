<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

### DeepSeek-R1 with vLLM — Disaggregated on 8x Hopper

This recipe deploys DeepSeek-R1 using vLLM in a disaggregated prefill/decode setup on a single Hopper node with 8 GPUs.

- Model cache PVC + download job: `recipes/deepseek-r1/model-cache/`
- Deployment manifest: `recipes/deepseek-r1/vllm/disagg/deploy_hopper_8gpu.yaml`

### 0) Prerequisites: Install the platform

Follow the Kubernetes deployment guide to install the Dynamo platform and prerequisites (CRDs/operator, etc.):
- `docs/kubernetes/README.md`

Ensure you have a GPU-enabled cluster with sufficient capacity (8x H100/H200 “Hopper”), and that the NVIDIA GPU Operator is healthy.

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

### 4) Deploy vLLM (Disaggregated, Prefill DEP16, Decode DEP16)

Apply the single-node disaggregated deployment:

```bash
kubectl apply -f ./deploy_hopper_16gpu.yaml -n ${NAMESPACE}
```

The manifest runs separate prefill and decode workers, each mounting the shared model cache, with settings tuned for Hopper.

Test the deployment locally by port-forwarding and sending a request:

```bash
# Port-forward the frontend Service to localhost:8000 (replace <frontend-svc> with the actual Service name)
kubectl port-forward svc/test3-vllm-dsr1-frontend 8000:8000 -n ${NAMESPACE} &
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
- If your cluster/network requires specific interfaces, adjust environment variables (e.g., `NCCL_SOCKET_IFNAME`) in the manifest accordingly.
- If your storage class differs, update `storageClassName` before applying the PVC.
- **If you want to run multinode deployments, IBGDA (InfiniBand GPU Direct Async) must be enabled on your nodes.** To enable IBGDA, you can follow this configuration script: [configure_system_drivers.sh](https://github.com/vllm-project/vllm/blob/v0.11.2/tools/ep_kernels/configure_system_drivers.sh). The script configures NVIDIA driver parameters and requires a system reboot to take effect.


