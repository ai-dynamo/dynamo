<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DynamoModel: Model Artifact Management

## Overview

`DynamoModel` is a Kubernetes Custom Resource Definition (CRD) that provides a high-level abstraction for managing model artifacts cached in PVCs within your cluster. It solves the critical problem of **model version drift** by ensuring all deployments and benchmarking jobs referencing the same `DynamoModel` use identical model artifacts.

### Problem Statement

Without `DynamoModel`, teams face several challenges:

1. **Version Drift**: Different jobs might download different versions of a model, leading to inconsistent results
2. **Manual PVC Management**: Teams must manually create PVCs, download models, and track versions
3. **Duplicate Downloads**: Multiple jobs download the same model repeatedly, wasting time and bandwidth
4. **No Version Pinning**: Difficult to ensure deployments and benchmarks use the exact same model artifact

### Solution

`DynamoModel` provides:

- **Version Pinning**: Pin deployments to specific model versions (SHA or tag)
- **Automated Downloads**: Automatically downloads and caches models in PVCs
- **Guaranteed Consistency**: All jobs referencing the same `DynamoModel` use identical artifacts
- **Flexible Sources**: Support for HuggingFace, S3, NGC, and custom sources
- **Simplified Management**: Declarative model management with Kubernetes-native tooling

## Key Features

### 1. Model Name and Version Pinning

```yaml
spec:
  name: meta-llama/Llama-3.3-70B-Instruct
  version: abcd12345  # Source SHA, avoids drift
```

Enables version pinning, avoiding drift/inconsistency in deployments versus benchmarking.

### 2. Flexible Source Management

```yaml
spec:
  sourceURL: hf://meta-llama/Llama-3.3-70B-Instruct
  # Or: s3://bucket/path/to/model
  # Or: ngc://nvidia/model
```

Supports multiple source types with automatic protocol detection.

### 3. Credential Injection

```yaml
spec:
  secretRef: llama-hf-secret
```

Securely inject credentials for private repositories.

### 4. Extensibility

```yaml
spec:
  downloaderRef: custom-downloader  # Optional
```

Plug in custom downloaders or workflows (e.g., MLFlow or internal tools).

## How It Works

### User Flow

1. **Create Model Definition:**
   ```bash
   kubectl apply -f components/models/qwen3-0.6b.yaml
   ```

2. **Watch Model Download:**
   ```bash
   kubectl get dynamomodel qwen3-0.6b -w
   ```

3. **Reference in Deployment:**
   ```yaml
   apiVersion: nvidia.com/v1alpha1
   kind: DynamoGraphDeployment
   spec:
     modelRef: qwen3-0.6b
     backendFramework: vllm
   ```

### Controller Flow

1. **DGD Controller** checks if `modelRef` is specified
2. Waits for `DynamoModel` to reach "Ready" state
3. Passes model name to Grove/Component pod generation
4. **Backend-specific logic** injects model arguments:
   - vLLM: `--model Qwen/Qwen3-0.6B`
   - SGLang/TRT-LLM: `--model-path Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B`
5. Sets `HF_HOME=/model-cache` so backends can resolve models
6. Auto-mounts model PVC to `/model-cache`

### Model Resolution

When `HF_HOME=/model-cache` is set and backends use the canonical model name:
```
/model-cache/
  models--Qwen--Qwen3-0.6B/
    snapshots/
      <hash>/
        config.json
        model-*.safetensors
```

Backends automatically resolve `Qwen/Qwen3-0.6B` → `/model-cache/models--Qwen--Qwen3-0.6B/snapshots/<hash>`

## Quick Start

### Step 1: Create a Model Secret (if needed)

For private models, create a secret with your credentials:

```bash
kubectl create secret generic llama-hf-secret \
  --from-literal=HF_TOKEN="your-huggingface-token" \
  -n your-namespace
```

### Step 2: Define a DynamoModel

Create a `DynamoModel` resource:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: llama-3-70b-instruct-v1
  namespace: your-namespace
spec:
  name: meta-llama/Llama-3.3-70B-Instruct
  version: abcd12345
  sourceURL: hf://meta-llama/Llama-3.3-70B-Instruct
  secretRef: llama-hf-secret
  pvc:
    create: true
    storageClass: your-storage-class
    size: 200Gi
    volumeAccessMode: ReadWriteMany
```

Apply it:

```bash
kubectl apply -f dynamomodel.yaml
```

### Step 3: Check Model Status

```bash
# Check status
kubectl get dynamomodel llama-3-70b-instruct-v1 -n your-namespace

# Watch download progress
kubectl get dynamomodel llama-3-70b-instruct-v1 -n your-namespace -w

# View detailed status
kubectl describe dynamomodel llama-3-70b-instruct-v1 -n your-namespace

# Check download job logs
kubectl logs job/llama-3-70b-instruct-v1-download -n your-namespace
```

### Step 4: Reference in DynamoGraphDeployment

Once the model is ready, reference it in your deployment:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
  namespace: your-namespace
spec:
  modelRef: llama-3-70b-instruct-v1  # Reference at top-level
  backendFramework: vllm
  services:
    VllmDecodeWorker:
      replicas: 2
      resources:
        limits:
          nvidia.com/gpu: "2"
      # Model arguments will be auto-injected by the controller
```

The controller will automatically:
1. Wait for the model to be ready
2. Inject the appropriate model arguments for your backend
3. Mount the model's PVC to `/model-cache`
4. Set `HF_HOME=/model-cache` for model resolution
5. Ensure all replicas use the same model artifact

## API Reference

### DynamoModelSpec

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Canonical model name (e.g., "meta-llama/Llama-3.3-70B-Instruct") |
| `version` | string | No | Version pin (SHA or tag) to prevent drift |
| `sourceURL` | string | Yes | Source location (hf://, s3://, ngc://) |
| `secretRef` | string | No | Reference to secret for credentials |
| `downloaderRef` | string | No | Reference to custom downloader |
| `pvc` | PVCSpec | Yes | PVC configuration for model storage |

### PVCSpec

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `create` | bool | No | true | Whether to create a new PVC |
| `name` | string | No | model name | Name of the PVC |
| `storageClass` | string | Yes* | - | Storage class for PVC creation |
| `size` | Quantity | Yes* | - | Size of the volume |
| `volumeAccessMode` | string | No | ReadWriteMany | Volume access mode |

\* Required when `create` is true

### DynamoModelStatus

| Field | Type | Description |
|-------|------|-------------|
| `state` | string | Lifecycle state: "Pending", "Downloading", "Ready", "Failed" |
| `conditions` | []Condition | Detailed status conditions |
| `pvcName` | string | Name of the created/used PVC |
| `downloadJobName` | string | Name of the download Job |
| `lastDownloadTime` | Time | Timestamp of last successful download |

## Supported Source Types

### HuggingFace

```yaml
sourceURL: hf://meta-llama/Llama-3.3-70B-Instruct
secretRef: hf-token-secret  # Optional for public models
```

Downloads using `huggingface-cli` with HF Transfer enabled for faster downloads.

### S3

```yaml
sourceURL: s3://my-bucket/models/llama-70b
secretRef: aws-credentials  # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

Downloads using AWS CLI.

### NGC (NVIDIA GPU Cloud)

```yaml
sourceURL: ngc://nvidia/llama-70b
secretRef: ngc-api-key
```

Downloads using NGC CLI.

### HTTP/HTTPS

```yaml
sourceURL: https://example.com/models/model.tar.gz
```

Generic HTTP download using wget.

## Advanced Usage

### Using Existing PVC

If you already have a PVC with a model:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: existing-model
spec:
  name: my-org/my-model
  sourceURL: hf://my-org/my-model
  pvc:
    create: false
    name: existing-model-pvc
```

### Custom Downloader

For specialized workflows:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: custom-model
spec:
  name: my-org/custom-model
  sourceURL: custom://my-internal-registry/model
  downloaderRef: mlflow-downloader
  pvc:
    create: true
    storageClass: fast-ssd
    size: 500Gi
```
