---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LoRA Adapter Walkthrough
subtitle: Serve text and vision-language LoRA adapters on Kubernetes from Hugging Face Hub or MinIO.
---

This walkthrough deploys LoRA-enabled vLLM inference on Kubernetes from the example manifests in
[`examples/backends/vllm/deploy/lora`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/lora).
Load adapters directly from Hugging Face Hub, or copy them into S3-compatible storage backed by MinIO
and load them from there. The same steps serve a text model (`Qwen/Qwen3-0.6B`) and a vision-language
model (`Qwen/Qwen3-VL-2B-Instruct`), on NVIDIA GPUs or on Intel XPU with Kubernetes Dynamic Resource
Allocation (DRA).

For backend support, the S3 credential provider chain, how a `DynamoModel` loads adapters, and
KV-aware LoRA routing, see [LoRA Adapters](lora-adapters.md).

> [!NOTE]
> The DynamoGraphDeployment (DGD) manifests use `nvidia.com/v1beta1`. `DynamoModel` does not have a
> `v1beta1` API and continues to use `nvidia.com/v1alpha1`.

## Prerequisites

- A Kubernetes cluster with GPU support
- Helm 3.x
- `kubectl` configured to access your cluster
- The Dynamo Kubernetes Platform installed; see the [Installation Guide](../installation/install-dynamo.md)
- A Hugging Face token for downloading private base models and LoRA adapters
- Optional: `yq` to replace the container image in the manifests, `jq` to format responses, and the
  AWS CLI to inspect the MinIO bucket

## Example Manifests

Paths are relative to
[`examples/backends/vllm/deploy`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy).

| Manifest | Description |
|---|---|
| [`lora/agg_lora.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/agg_lora.yaml) | DGD for text vLLM with LoRA support, loading adapters from MinIO |
| [`lora/agg_lora_hf.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/agg_lora_hf.yaml) | DGD and `DynamoModel` for direct Hugging Face Hub loading |
| [`lora/minio-secret.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/minio-secret.yaml) | Kubernetes Secret for MinIO credentials |
| [`lora/sync-lora-job.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/sync-lora-job.yaml) | Job to download a LoRA from Hugging Face Hub and upload it to MinIO |
| [`lora/lora-model.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/lora-model.yaml) | `DynamoModel` for registering the text LoRA adapter |
| [`lora/multimodal/agg_qwen_lora.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/multimodal/agg_qwen_lora.yaml) | DGD for multimodal vLLM with LoRA support |
| [`lora/multimodal/minio-secret.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/multimodal/minio-secret.yaml) | Kubernetes Secret for MinIO credentials |
| [`lora/multimodal/sync-lora-job.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/multimodal/sync-lora-job.yaml) | Job to download a vision LoRA from Hugging Face Hub and upload it to MinIO |
| [`lora/multimodal/lora-model.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/lora/multimodal/lora-model.yaml) | `DynamoModel` for registering the vision LoRA adapter |
| [`xpu/agg_lora_xpu_dra.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/xpu/agg_lora_xpu_dra.yaml) | Text generation with dynamic LoRA adapters on Intel XPU with DRA |
| [`xpu/agg_qwen_lora_xpu_dra.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/xpu/agg_qwen_lora_xpu_dra.yaml) | Vision-language generation with dynamic LoRA adapters on Intel XPU with DRA |

The Intel XPU templates have no supporting manifests of their own. `agg_lora_xpu_dra.yaml` uses
`lora/minio-secret.yaml`, `lora/sync-lora-job.yaml`, and `lora/lora-model.yaml`;
`agg_qwen_lora_xpu_dra.yaml` uses the copies in `lora/multimodal/`.

Each variant runs from its own working directory and creates a DGD with a fixed name. The commands
on this page use that name as `DEPLOYMENT_NAME`:

| Variant | Working directory | DGD manifest | `DEPLOYMENT_NAME` |
|---|---|---|---|
| Text, Hugging Face Hub | `examples/backends/vllm/deploy/lora` | `agg_lora_hf.yaml` | `vllm-agg-lora-hf` |
| Text, MinIO | `examples/backends/vllm/deploy/lora` | `agg_lora.yaml` | `vllm-agg-lora` |
| Text, MinIO, Intel XPU | `examples/backends/vllm/deploy/lora` | `../xpu/agg_lora_xpu_dra.yaml` | `vllm-agg-lora-xpu-dra` |
| Vision-language, MinIO | `examples/backends/vllm/deploy/lora/multimodal` | `agg_qwen_lora.yaml` | `agg-qwen-multimodal-lora` |
| Vision-language, MinIO, Intel XPU | `examples/backends/vllm/deploy/lora/multimodal` | `../../xpu/agg_qwen_lora_xpu_dra.yaml` | `agg-qwen-multimodal-lora-xpu-dra` |

## Deploy from Hugging Face Hub

`agg_lora_hf.yaml` loads the text adapter directly from Hugging Face Hub through an `hf://` source,
without MinIO. Its `DynamoModel`, `codelion-recovery-lora-hf`, registers
`codelion/Qwen3-0.6B-accuracy-recovery-lora` from `hf://codelion/Qwen3-0.6B-accuracy-recovery-lora`
on the `Qwen/Qwen3-0.6B` worker.

From the repository root, change to the text example directory:

```bash
cd examples/backends/vllm/deploy/lora
```

Update the two `image` fields in `agg_lora_hf.yaml` to use your vLLM runtime image, then apply the
deployment and `DynamoModel`:

```bash
export NAMESPACE=dynamo
kubectl apply -f agg_lora_hf.yaml -n ${NAMESPACE}
kubectl get dynamomodel codelion-recovery-lora-hf -n ${NAMESPACE} -w
```

The example repository is public. For a private base model or adapter, create `hf-token-secret`
before applying the manifest:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

The worker stores both the base model and adapter in the cache rooted at `HF_HOME`
(`/home/dynamo/.cache/huggingface` in the manifest). To send requests, follow
[Run Inference](#run-inference) with `DEPLOYMENT_NAME=vllm-agg-lora-hf`.

## Deploy with MinIO

The MinIO workflow downloads an adapter from Hugging Face Hub with a Kubernetes Job, copies it to
MinIO, and loads it from the S3-compatible endpoint. The text and vision-language examples share these
steps; [Vision-Language Differences](#vision-language-differences) lists what changes for the
vision-language model.

### Set Environment Variables

From the repository root, change to the example directory for your variant. For the text model:

```bash
cd examples/backends/vllm/deploy/lora
```

For the vision-language model:

```bash
cd examples/backends/vllm/deploy/lora/multimodal
```

Set the namespace and your Hugging Face token:

```bash
export NAMESPACE=dynamo  # Your Dynamo namespace
export HF_TOKEN=your_hf_token  # Your Hugging Face token
```

### Create Secrets

Create the Hugging Face token Secret:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Create the MinIO credentials Secret. The example uses the default MinIO credentials; change them to
point to your own S3-compatible storage.

```bash
kubectl apply -f minio-secret.yaml -n ${NAMESPACE}
```

### Install MinIO

Add the MinIO Helm repository:

```bash
helm repo add minio https://charts.min.io/
helm repo update
```

Deploy MinIO:

```bash
helm install minio minio/minio \
  --namespace ${NAMESPACE} \
  --set rootUser=minioadmin \
  --set rootPassword=minioadmin \
  --set mode=standalone \
  --set replicas=1 \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set resources.requests.memory=512Mi \
  --set service.type=ClusterIP \
  --set consoleService.type=ClusterIP
```

Verify the installation:

```bash
kubectl get pods -n ${NAMESPACE} | grep minio
kubectl get svc -n ${NAMESPACE} | grep minio
```

Expected output:

```text
minio-xxxx-xxxx   1/1     Running   0          1m
```

Optionally, port-forward the MinIO console:

```bash
kubectl port-forward svc/minio-console -n ${NAMESPACE} 9001:9001 9000:9000
```

Open `http://localhost:9001` in your browser and sign in with username `minioadmin` and password
`minioadmin`.

### Upload the Adapter to MinIO

Apply the Job that downloads the LoRA adapter from Hugging Face Hub and uploads it to MinIO:

```bash
kubectl apply -f sync-lora-job.yaml -n ${NAMESPACE}
```

The text Job syncs `codelion/Qwen3-0.6B-accuracy-recovery-lora`. The vision-language Job syncs
`Chhagan005/Chhagan-DocVL-Qwen3`. Both upload to `s3://my-loras/<adapter-name>`.

Monitor the Job and wait for it to complete successfully:

```bash
# Watch job progress
kubectl get jobs -n ${NAMESPACE} -w

# Check job logs
kubectl logs job/sync-hf-lora-to-minio -n ${NAMESPACE} -f
```

Optionally, verify the upload:

```bash
# Port-forward MinIO API
kubectl port-forward svc/minio -n ${NAMESPACE} 9000:9000 &

# Check uploaded files
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
aws s3 ls s3://my-loras/ --recursive
```

To upload a different LoRA adapter, edit `sync-lora-job.yaml` and change the `MODEL_NAME`
environment variable:

```yaml
env:
- name: MODEL_NAME
  value: your-org/your-lora-adapter
```

For the vision-language example, set it to a vision LoRA adapter, such as
`your-org/your-vision-lora-adapter`.

### Deploy the LoRA-Enabled Graph

Replace the image in the DGD manifest with your container image. For the text model, with `yq`:

```bash
# Using yq to update the image
export FRAMEWORK_RUNTIME_IMAGE=your-registry/your-image:tag
yq '.spec.components[].podTemplate.spec.containers[] |= (if .name == "main" then .image = env(FRAMEWORK_RUNTIME_IMAGE) else . end)' agg_lora.yaml > agg_lora_updated.yaml
```

For the vision-language model:

```bash
# Using yq to update the image
export FRAMEWORK_RUNTIME_IMAGE=your-registry/your-image:tag
yq '.spec.components[].podTemplate.spec.containers[] |= (if .name == "main" then .image = env(FRAMEWORK_RUNTIME_IMAGE) else . end)' agg_qwen_lora.yaml > agg_qwen_lora_updated.yaml
```

Deploy the text graph:

```bash
export DEPLOYMENT_NAME=vllm-agg-lora
kubectl apply -f agg_lora_updated.yaml -n ${NAMESPACE}
```

Or deploy the vision-language graph:

```bash
export DEPLOYMENT_NAME=agg-qwen-multimodal-lora
kubectl apply -f agg_qwen_lora_updated.yaml -n ${NAMESPACE}
```

To deploy on Intel XPU instead, skip this step and follow
[Deploy on Intel XPU with DRA](#deploy-on-intel-xpu-with-dra).

### Deploy on Intel XPU with DRA

To deploy on Intel XPU, apply the XPU manifest instead of the updated manifest from
[Deploy the LoRA-Enabled Graph](#deploy-the-lora-enabled-graph). Before applying it, complete the
steps from [Set Environment Variables](#set-environment-variables) through
[Upload the Adapter to MinIO](#upload-the-adapter-to-minio) to create `hf-token-secret` and
`minio-secret`, deploy MinIO, and upload the adapter.

Use Kubernetes v1.34 or newer with DRA API v1 and the
[Intel resource drivers for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes)
installed. In the XPU manifest, replace the Frontend `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`
image and the worker `nvcr.io/nvidia/ai-dynamo/vllm-runtime-xpu:my-tag` image with images available to
the cluster. See [Deploy on Intel GPUs with DRA](../model-deployment/deploy-on-intel-gpus.mdx) for the
XPU image build commands and additional requirements.

For the text model, from `examples/backends/vllm/deploy/lora`:

```bash
export DEPLOYMENT_NAME=vllm-agg-lora-xpu-dra
kubectl apply -f ../xpu/agg_lora_xpu_dra.yaml -n ${NAMESPACE}
```

For the vision-language model, from `examples/backends/vllm/deploy/lora/multimodal`:

```bash
export DEPLOYMENT_NAME=agg-qwen-multimodal-lora-xpu-dra
kubectl apply -f ../../xpu/agg_qwen_lora_xpu_dra.yaml -n ${NAMESPACE}
```

### Verify the Deployment

```bash
# Check pods
kubectl get pods -n ${NAMESPACE}

# Watch worker logs
kubectl logs -f -n "${NAMESPACE}" \
  -l "nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT_NAME},nvidia.com/dynamo-component=worker"
```

Wait for the worker to show `Application startup complete`.

### Register the Adapter with DynamoModel

Apply `lora-model.yaml` to register the LoRA adapter declaratively through the `DynamoModel` custom
resource:

```bash
kubectl apply -f lora-model.yaml -n ${NAMESPACE}
```

| Variant | `DynamoModel` | `modelName` (adapter identity) | `baseModelName` | `source.uri` (MinIO location) |
|---|---|---|---|---|
| Text | `codelion-recovery-lora` | `codelion/Qwen3-0.6B-accuracy-recovery-lora` | `Qwen/Qwen3-0.6B` | `s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora` |
| Vision-language | `chhagan-docvl-qwen` | `Chhagan005/Chhagan-DocVL-Qwen3` | `Qwen/Qwen3-VL-2B-Instruct` | `s3://my-loras/Chhagan005/Chhagan-DocVL-Qwen3` |

### Run Inference

Port-forward the frontend and list the available models:

```bash
# Port-forward the frontend
kubectl port-forward svc/${DEPLOYMENT_NAME}-frontend -n ${NAMESPACE} 8000:8000 &

# List available models
curl http://localhost:8000/v1/models | jq .
```

For the text model, set `model` to the adapter name:

```bash
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
    "messages": [{"role": "user", "content": "What is deep learning?"}],
    "max_tokens": 300,
    "temperature": 0.0
  }' | jq .
```

For the vision-language model, see [Vision-Language Requests](#vision-language-requests).

## Vision-Language Differences

The `lora/multimodal` example serves the `Qwen/Qwen3-VL-2B-Instruct` vision-language model with
dynamic LoRA adapters loaded from MinIO. It uses the aggregated single-worker architecture, in which
the Rust `OpenAIPreprocessor` in the Frontend handles image URLs directly. Follow
[Deploy with MinIO](#deploy-with-minio) from `examples/backends/vllm/deploy/lora/multimodal`; this
section lists what differs from the text example.

### Model and Adapter

| Setting | Text | Vision-language |
|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-VL-2B-Instruct` |
| Default adapter | `codelion/Qwen3-0.6B-accuracy-recovery-lora` | `Chhagan005/Chhagan-DocVL-Qwen3`, a document-understanding LoRA for Qwen3-VL-2B |

### Manifest Settings

`agg_qwen_lora.yaml` sets `DYN_REQUEST_PLANE=tcp` on both the Frontend and the worker, using the TCP
request plane to avoid the NATS 1 MB message limit for multimodal requests. The worker caches adapters
in `DYN_LORA_PATH=/tmp/dynamo_loras_multimodal`. See
[Environment Variables](#environment-variables) for the full list.

The worker passes these vLLM arguments in addition to `--enable-lora` and `--max-lora-rank 64`:

| Argument | Description |
|---|---|
| `--enable-multimodal` | Enable multimodal (vision) support |
| `--gpu-memory-utilization` | Fraction of GPU memory to use (default `0.85`) |
| `--max-model-len` | Maximum sequence length (default `8192`) |
| `--max-num-batched-tokens` | Maximum batched tokens (default `8192`) |

Qwen VL models use dynamic image resolution, so large images can exhaust GPU memory. To lower memory
use, see [Out of Memory During Vision-Language Inference](#out-of-memory-during-vision-language-inference).

### Vision-Language Requests

After you register the adapter and port-forward the frontend as in [Run Inference](#run-inference),
send an image request with the LoRA adapter:

```bash
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "Chhagan005/Chhagan-DocVL-Qwen3",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image in detail"},
      {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
    ]}],
    "max_tokens": 300,
    "temperature": 0.0
  }' | jq .
```

To compare, send the same request to the base model:

```bash
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image in detail"},
      {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
    ]}],
    "max_tokens": 300,
    "temperature": 0.0
  }' | jq .
```

## Configuration Reference

### Environment Variables

The MinIO example manifests set these variables on the worker:

| Variable | Description | Text | Vision-language |
|---|---|---|---|
| `DYN_LORA_ENABLED` | Enable LoRA support | `true` | `true` |
| `DYN_LORA_PATH` | Local cache path for LoRA files | `/tmp/dynamo_loras_minio` | `/tmp/dynamo_loras_multimodal` |
| `DYN_SYSTEM_ENABLED` | Enable the system management API | `true` | `true` |
| `DYN_SYSTEM_PORT` | Port for the LoRA management API | `9090` | `9090` |
| `DYN_REQUEST_PLANE` | Transport plane; TCP for multimodal to avoid the NATS 1 MB limit | Not set | `tcp` |
| `AWS_ENDPOINT` | MinIO/S3 endpoint URL | `http://minio:9000` | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` | MinIO access key | From secret | From secret |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | From secret | From secret |
| `AWS_REGION` | AWS region (required for the S3 SDK) | `us-east-1` | `us-east-1` |
| `AWS_ALLOW_HTTP` | Allow HTTP connections | `true` | `true` |
| `BUCKET_NAME` | MinIO bucket name | `my-loras` | `my-loras` |

These variables control remote downloads and the direct Hugging Face Hub source:

| Variable | Description | Default |
|---|---|---|
| `LORA_DOWNLOAD_TIMEOUT_SECS` | Timeout for each remote LoRA file download | `3600` |
| `HF_HOME` | Hugging Face cache root used by direct `hf://` sources | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Override for the Hugging Face repository cache | `$HF_HOME/hub` |
| `HF_TOKEN` | Authentication token for private Hugging Face repositories | Not set |
| `HF_TOKEN_PATH` | Path to the stored Hugging Face token | `$HF_HOME/token` |
| `HF_HUB_OFFLINE` | Reuse cached snapshots without Hub API calls | `false` |

For the other AWS credential and endpoint variables and their precedence, see the worker environment
variables in [Serve a LoRA Adapter](lora-adapters.md#serve-a-lora-adapter).

### vLLM Arguments

The text manifests pass `--enable-lora`, `--max-lora-rank 64`, and `--enforce-eager`. For
`--enable-lora`, `--max-lora-rank`, and `--max-loras`, see the vLLM worker arguments in
[Serve a LoRA Adapter](lora-adapters.md#serve-a-lora-adapter). The vision-language arguments are in
[Manifest Settings](#manifest-settings).

## Cleanup

Run these commands from the working directory of your variant.

Remove the direct Hugging Face Hub deployment:

```bash
kubectl delete -f agg_lora_hf.yaml -n ${NAMESPACE}
```

Remove the vLLM deployment by deleting the manifest that you applied. For the text model:

```bash
# Standard v1beta1 deployment
kubectl delete -f agg_lora_updated.yaml -n ${NAMESPACE}

# Intel XPU deployment
kubectl delete -f ../xpu/agg_lora_xpu_dra.yaml -n ${NAMESPACE}
```

For the vision-language model:

```bash
# Standard v1beta1 deployment
kubectl delete -f agg_qwen_lora_updated.yaml -n ${NAMESPACE}

# Intel XPU deployment
kubectl delete -f ../../xpu/agg_qwen_lora_xpu_dra.yaml -n ${NAMESPACE}
```

Remove the `DynamoModel`:

```bash
kubectl delete -f lora-model.yaml -n ${NAMESPACE}
```

Remove the sync Job:

```bash
kubectl delete -f sync-lora-job.yaml -n ${NAMESPACE}
```

Remove MinIO:

```bash
helm uninstall minio -n ${NAMESPACE}
```

Remove the Secrets:

```bash
kubectl delete -f minio-secret.yaml -n ${NAMESPACE}
kubectl delete secret hf-token-secret -n ${NAMESPACE}
```

## Troubleshooting

Before running these commands, set `DEPLOYMENT_NAME` to the name of your variant from
[Example Manifests](#example-manifests): `vllm-agg-lora` or `vllm-agg-lora-xpu-dra` for the text
model, or `agg-qwen-multimodal-lora` or `agg-qwen-multimodal-lora-xpu-dra` for the vision-language
model. For adapter-name, rank, and base-model-response problems, also see
[Troubleshooting](lora-adapters.md#troubleshooting) in the LoRA Adapters guide.

### LoRA Fails to Load

1. Check MinIO connectivity from the worker:

   ```bash
   WORKER_POD=$(kubectl get pods -n "${NAMESPACE}" \
     -l "nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT_NAME},nvidia.com/dynamo-component=worker" \
     -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -it -n "${NAMESPACE}" "${WORKER_POD}" -- \
     curl http://minio:9000/minio/health/live
   ```

2. Verify that the LoRA exists in MinIO:

   ```bash
   kubectl port-forward svc/minio -n ${NAMESPACE} 9000:9000 &
   aws --endpoint-url=http://localhost:9000 s3 ls s3://my-loras/ --recursive
   ```

3. Check the worker logs:

   ```bash
   kubectl logs -n "${NAMESPACE}" \
     -l "nvidia.com/dynamo-graph-deployment-name=${DEPLOYMENT_NAME},nvidia.com/dynamo-component=worker"
   ```

4. For the vision-language model, verify adapter compatibility: the LoRA adapter must be trained for
   the same base model architecture (Qwen3-VL-2B), and `max-lora-rank` (default 64) must be greater
   than or equal to the adapter's rank.

### Sync Job Fails

1. Check the Job logs:

   ```bash
   kubectl logs job/sync-hf-lora-to-minio -n ${NAMESPACE}
   ```

2. Verify the Hugging Face token:

   ```bash
   kubectl get secret hf-token-secret -n ${NAMESPACE} -o yaml
   ```

3. Check that MinIO is accessible:

   ```bash
   kubectl get svc minio -n ${NAMESPACE}
   ```

### Out of Memory During Vision-Language Inference

Qwen VL models use dynamic resolution: a 2560px image can produce 5000+ tokens. To reduce memory use:

- Reduce `--max-model-len` in the `agg_qwen_lora.yaml` args.
- Add `--mm-processor-kwargs '{"max_pixels": 1003520}'` to cap image resolution.
- Lower `--gpu-memory-utilization` to `0.80`.

### MinIO Connection Refused

- Ensure the MinIO pods are running: `kubectl get pods -n ${NAMESPACE} | grep minio`
- Check the MinIO service: `kubectl get svc minio -n ${NAMESPACE}`
- Verify that the `AWS_ENDPOINT` URL matches the service name

## See Also

- [LoRA Adapters](lora-adapters.md): LoRA concepts, backend support, and KV-aware LoRA routing
- [CLI LoRA Adapter Walkthrough](../../cli/operations/lora-adapter-walkthrough.md): run the same
  text and vision-language adapters locally with launch scripts
- [Kubernetes Quickstart](../getting-started/quickstart.mdx): Dynamo Platform setup
- [Installation Guide](../installation/install-dynamo.md): Dynamo Platform installation
- [vLLM deployment examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy):
  other vLLM deployment patterns
