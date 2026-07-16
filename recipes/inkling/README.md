# TML Inkling Recipes

Recipes for **thinkingmachines/Inkling-NVFP4**.

## Configurations

Dynamo + SGLang deployment profile:

|                          | B200                                  |
|--------------------------|---------------------------------------|
| **GPU** (per worker)     | 8x B200                               |
| **Mode**                 | aggregated                            |
| **Framework**            | SGLang (sglang-inkling custom build)  |
| **Precision**            | NVFP4 (`modelopt_fp4`)                |
| **Parallelism**          | TP8                                   |
| **Attention backend**    | FA4                                   |
| **FP4 GEMM backend**     | FLASHINFER_TRTLLM                     |
| **MoE runner backend**   | FLASHINFER_TRTLLM_ROUTED              |
| **AllReduce backend**    | Torch symmetric memory                |
| **Mamba radix cache**    | extra_buffer strategy                 |
| **Speculative decoding** | EAGLE multi-layer (8 steps, topk 1, 9 draft tokens, rejection sampling) |

## Supported features

- Modalities: Text + Image + Audio input
- Reasoning (`inkling` reasoning parser)
- Tool calling (`inkling` tool-call parser)

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **Image pull secret** with access to `nvcr.io/nvstaging/nim` (staging registry):
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret docker-registry nvcr-imagepullsecret \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password="your-ngc-api-key" \
     -n ${NAMESPACE}
   ```
3. **HuggingFace token** (the model is public, but a token avoids rate limits):
   ```bash
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create namespace

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Create Storage

> **Note:** Edit `model-cache/model-cache.yaml` first and update `storageClassName` to match your cluster (`kubectl get storageclass`). On clusters that already provide a shared RWX PVC (e.g. `shared-model-cache` on the Dynamo dev clusters), skip this step and replace the `model-cache` claim name in the manifests with the shared PVC name.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

The checkpoint is ~592 GB.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/inkling-model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

```bash
kubectl apply -f sglang/agg-b200/deploy.yaml -n ${NAMESPACE}
```

### 5. Smoke test

```bash
kubectl port-forward svc/tml-inkling-sglang-agg-frontend 8000:8000 -n ${NAMESPACE} &

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingmachines/Inkling-NVFP4",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 64
  }'
```

## Known issues

1. `deploy.yaml` overrides `DYN_FORWARDPASS_METRIC_PORT` to an empty value — required to
   make MTP/EAGLE speculative decoding work with SGLang: the forward-pass metrics reporter
   (auto-enabled by the operator-injected port) crashes the scheduler on speculative
   batches (`batch.seq_lens_cpu` is `None`). Only per-forward-pass telemetry is lost.
   Remove the override once the image carries a fix.
2. `reasoning_content` is not populated in responses — reasoning text is returned inline
   in `content`. The `inkling` reasoning parser is registered in SGLang but the split is
   not engaged on the Dynamo frontend path.
