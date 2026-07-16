<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA Deployment on Kubernetes

This guide explains how to deploy LoRA-enabled vLLM inference on Kubernetes. Load adapters directly
from Hugging Face Hub or through S3-compatible storage backed by MinIO.

## Overview

The examples cover direct Hugging Face Hub downloads and an HF-to-MinIO synchronization workflow.

## Prerequisites

- Kubernetes cluster with GPU support
- Helm 3.x installed
- `kubectl` configured to access your cluster
- Dynamo Kubernetes Platform installed ([Installation Guide](../../../../../docs/kubernetes/installation-guide.md))
- Hugging Face token for downloading private base models and LoRA adapters

## Files in This Directory

| File | Description |
|------|-------------|
| `agg_lora.yaml` | DynamoGraphDeployment for vLLM with LoRA support |
| `agg_lora_hf.yaml` | DynamoGraphDeployment and DynamoModel for direct Hugging Face Hub loading |
| `agg_lora_xpu_dra.yaml` | Intel XPU DRA DynamoGraphDeployment for vLLM with LoRA support |
| `minio-secret.yaml` | Kubernetes secret for MinIO credentials |
| `sync-lora-job.yaml` | Job to download a LoRA from Hugging Face Hub and upload it to MinIO |
| `lora-model.yaml` | DynamoModel CRD for registering LoRA adapters |

## Deploy from Hugging Face Hub

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

The worker stores both the base model and adapter in the cache rooted at `HF_HOME`.

## MinIO Deployment

The remaining steps download an adapter from Hugging Face Hub, copy it to MinIO, and load it from
the S3-compatible endpoint.

### Step 1: Set Up Environment Variables

```bash
export NAMESPACE=dynamo  # Your Dynamo namespace
export HF_TOKEN=your_hf_token  # Your Hugging Face token
```

---

### Step 2: Create Secrets

#### Create Hugging Face Token Secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

#### Create MinIO Credentials Secret

in this example, we are using the default credentials for MinIO.
You can change the credentials to point to your own S3 compatible storage.

```bash
kubectl apply -f minio-secret.yaml -n ${NAMESPACE}
```

---

### Step 3: Install MinIO

#### Add MinIO Helm Repository

```bash
helm repo add minio https://charts.min.io/
helm repo update
```

#### Deploy MinIO

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

#### Verify MinIO Installation

```bash
kubectl get pods -n ${NAMESPACE} | grep minio
kubectl get svc -n ${NAMESPACE} | grep minio
```

Expected output:

```text
minio-xxxx-xxxx   1/1     Running   0          1m
```

#### Access the MinIO Console (Optional)

```bash
kubectl port-forward svc/minio-console -n ${NAMESPACE} 9001:9001 9000:9000
```

Open http://localhost:9001 in your browser:
- Username: `minioadmin`
- Password: `minioadmin`

---

### Step 4: Upload LoRA Adapters to MinIO

Use the provided Kubernetes Job to download a LoRA adapter from Hugging Face Hub and upload it to MinIO:

```bash
kubectl apply -f sync-lora-job.yaml -n ${NAMESPACE}
```

#### Monitor the Job

```bash
# Watch job progress
kubectl get jobs -n ${NAMESPACE} -w

# Check job logs
kubectl logs job/sync-hf-lora-to-minio -n ${NAMESPACE} -f
```

Wait for the job to complete successfully.

#### Verify the Upload (Optional)

```bash
# Port-forward MinIO API
kubectl port-forward svc/minio -n ${NAMESPACE} 9000:9000 &

# Check uploaded files
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000
aws s3 ls s3://my-loras/ --recursive
```

#### Customize the LoRA Adapter

To upload a different LoRA adapter, edit `sync-lora-job.yaml` and change the `MODEL_NAME` environment variable:

```yaml
env:
- name: MODEL_NAME
  value: your-org/your-lora-adapter
```

---

### Step 5: Deploy vLLM with LoRA Support

#### Update the Image

Edit `v1beta1/agg_lora.yaml` to use your container image:

```bash
# Using yq to update the image
export FRAMEWORK_RUNTIME_IMAGE=your-registry/your-image:tag
yq '.spec.components[].podTemplate.spec.containers[] |= (if .name == "main" then .image = env(FRAMEWORK_RUNTIME_IMAGE) else . end)' v1beta1/agg_lora.yaml > v1beta1/agg_lora_updated.yaml
```

#### Deploy the LoRA-Enabled vLLM Graph

```bash
kubectl apply -f agg_lora_updated.yaml -n ${NAMESPACE}
```

#### Verify the Deployment

```bash
# Check pods
kubectl get pods -n ${NAMESPACE}

# Watch worker logs
kubectl logs -f deployment/vllm-agg-lora-vllmdecode-worker -n ${NAMESPACE}
```

Wait for the worker to show "Application startup complete".


### Step 6: Apply the DynamoModel CRD

The `lora-model.yaml` file demonstrates how to register a LoRA adapter using the DynamoModel Custom Resource:

```bash
kubectl apply -f lora-model.yaml -n ${NAMESPACE}
```

This creates a declarative way to manage LoRA adapters in your cluster.

---

## Deploy on Intel XPU with DRA

To use `agg_lora_xpu_dra.yaml`, install Kubernetes DRA v1 and the Intel
resource drivers for Kubernetes with the `gpu.intel.com` DeviceClass. Build and
push the `vllm-runtime-xpu:my-tag` worker image before deployment. See the
[Intel XPU Deployment Examples](../xpu/README.md) for the image build commands.

```bash
kubectl apply -f agg_lora_xpu_dra.yaml -n ${NAMESPACE}
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ENDPOINT` | MinIO/S3 endpoint URL | `http://minio:9000` |
| `AWS_ACCESS_KEY_ID` | MinIO access key | From secret |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key | From secret |
| `AWS_REGION` | AWS region (required for S3 SDK) | `us-east-1` |
| `AWS_ALLOW_HTTP` | Allow HTTP connections | `true` |
| `DYN_LORA_ENABLED` | Enable LoRA support | `true` |
| `DYN_LORA_PATH` | Local cache path for LoRA files | `/tmp/dynamo_loras_minio` |
| `LORA_DOWNLOAD_TIMEOUT_SECS` | Timeout for each remote LoRA file download | `3600` |
| `BUCKET_NAME` | MinIO bucket name | `my-loras` |
| `HF_HOME` | Hugging Face cache root used by direct `hf://` sources | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Override for the Hugging Face repository cache | `$HF_HOME/hub` |
| `HF_TOKEN` | Authentication token for private Hugging Face repositories | Not set |
| `HF_TOKEN_PATH` | Path to the stored Hugging Face token | `$HF_HOME/token` |
| `HF_HUB_OFFLINE` | Reuse cached snapshots without Hub API calls | `false` |

### vLLM LoRA Arguments

| Argument | Description |
|----------|-------------|
| `--enable-lora` | Enable LoRA adapter support |
| `--max-lora-rank` | Maximum LoRA rank (must be >= your LoRA's rank) |
| `--max-loras` | Maximum number of LoRAs to load simultaneously |

---

## Cleanup

### Remove Direct Hugging Face Hub Deployment

```bash
kubectl delete -f agg_lora_hf.yaml -n ${NAMESPACE}
```

### Remove vLLM Deployment

```bash
kubectl delete -f agg_lora.yaml -n ${NAMESPACE}
```

### Remove Sync Job

```bash
kubectl delete -f sync-lora-job.yaml -n ${NAMESPACE}
```

### Remove MinIO

```bash
helm uninstall minio -n ${NAMESPACE}
```

### Remove Secrets

```bash
kubectl delete -f minio-secret.yaml -n ${NAMESPACE}
kubectl delete secret hf-token-secret -n ${NAMESPACE}
```

---

## Troubleshooting

### LoRA Fails to Load

1. **Check MinIO connectivity from worker**:
   ```bash
   kubectl exec -it deployment/vllm-agg-lora-vllmdecode-worker -n ${NAMESPACE} -- \
     curl http://minio:9000/minio/health/live
   ```

2. **Verify LoRA exists in MinIO**:
   ```bash
   kubectl port-forward svc/minio -n ${NAMESPACE} 9000:9000 &
   aws --endpoint-url=http://localhost:9000 s3 ls s3://my-loras/ --recursive
   ```

3. **Check worker logs**:
   ```bash
   kubectl logs deployment/vllm-agg-lora-vllmdecode-worker -n ${NAMESPACE}
   ```

### Sync Job Fails

1. **Check job logs**:
   ```bash
   kubectl logs job/sync-hf-lora-to-minio -n ${NAMESPACE}
   ```

2. **Verify Hugging Face token**:
   ```bash
   kubectl get secret hf-token-secret -n ${NAMESPACE} -o yaml
   ```

3. **Check MinIO is accessible**:
   ```bash
   kubectl get svc minio -n ${NAMESPACE}
   ```

### MinIO Connection Refused

- Ensure MinIO pods are running: `kubectl get pods -n ${NAMESPACE} | grep minio`
- Check MinIO service: `kubectl get svc minio -n ${NAMESPACE}`
- Verify the `AWS_ENDPOINT` URL matches the service name

## Further Reading

- [vLLM Deployment Guide](../README.md) - Other deployment patterns
- [Dynamo Kubernetes Guide](../../../../../docs/kubernetes/README.md) - Platform setup
- [Installation Guide](../../../../../docs/kubernetes/installation-guide.md) - Platform installation
