<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LoRA Adapters

LoRA (Low-Rank Adaptation) enables efficient fine-tuning and serving of specialized model variants without duplicating full model weights. Dynamo provides built-in support for dynamic LoRA adapter loading, caching, and inference routing.

## Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| vLLM | ✅ | Full support including KV-aware routing |
| SGLang | 🚧 | In progress |
| TensorRT-LLM | ❌ | Not yet supported |

See the [Feature Matrix](../../reference/feature-matrix.md) for full compatibility details.

## Overview

Dynamo's LoRA implementation provides:

- **Dynamic loading**: Load and unload LoRA adapters at runtime without restarting workers
- **Multiple sources**: Load from local filesystem (`file://`), S3-compatible storage (`s3://`), or Hugging Face Hub (`hf://`)
- **Automatic caching**: Downloaded adapters are cached locally to avoid repeated downloads
- **Discovery integration**: Loaded LoRAs are automatically registered and discoverable via `/v1/models`
- **KV-aware routing**: Route requests to workers with the appropriate LoRA loaded
- **Kubernetes native**: Declarative LoRA management via the `DynamoModel` CRD

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        LoRA Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Frontend   │────▶│    Router    │────▶│   Workers    │     │
│  │  /v1/models  │     │  LoRA-aware  │     │  LoRA-loaded │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                                                   │              │
│                                                   ▼              │
│                              ┌─────────────────────────────────┐ │
│                              │         LoRA Manager            │ │
│                              │  ┌───────────┐ ┌─────────────┐  │ │
│                              │  │ Downloader│ │    Cache    │  │ │
│                              │  └───────────┘ └─────────────┘  │ │
│                              └─────────────────────────────────┘ │
│                                         │                        │
│                     ┌───────────────────┼───────────────────┐   │
│                     ▼                   ▼                   ▼   │
│              ┌────────────┐      ┌────────────┐      ┌─────────┐│
│              │  file://   │      │   s3://    │      │  hf://  ││
│              │   Local    │      │  S3/MinIO  │      │(custom) ││
│              └────────────┘      └────────────┘      └─────────┘│
└─────────────────────────────────────────────────────────────────┘
```

The LoRA system consists of:

- **Rust Core** (`lib/llm/src/lora/`): High-performance downloading, caching, and validation
- **Python Manager** (`components/src/dynamo/common/lora/`): Extensible wrapper with custom source support
- **Worker Handlers** (`components/src/dynamo/vllm/handlers.py`): Load/unload API and inference integration

## Quick Start

### Prerequisites

- Dynamo installed with vLLM support
- For S3 sources: AWS credentials configured
- A LoRA adapter compatible with your base model

### Local Development

**1. Start Dynamo with LoRA support:**

```bash
# Start vLLM worker with LoRA flags
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager \
    --connector none \
    --enable-lora \
    --max-lora-rank 64
```

**2. Load a LoRA adapter:**

```bash
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my-lora",
    "source": {
      "uri": "file:///path/to/my-lora"
    }
  }'
```

**3. Run inference with the LoRA:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-lora",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### S3-Compatible Storage

For production deployments, store LoRA adapters in S3-compatible storage:

```bash
# Configure S3 credentials
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_ENDPOINT=http://minio:9000  # For MinIO
export AWS_REGION=us-east-1

# Load LoRA from S3
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "customer-support-lora",
    "source": {
      "uri": "s3://my-loras/customer-support-v1"
    }
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_LORA_ENABLED` | Enable LoRA adapter support | `false` |
| `DYN_LORA_PATH` | Local cache directory for downloaded LoRAs | `~/.cache/dynamo_loras` |
| `AWS_ACCESS_KEY_ID` | S3 access key (for `s3://` URIs) | - |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key (for `s3://` URIs) | - |
| `AWS_ENDPOINT` | Custom S3 endpoint (for MinIO, etc.) | - |
| `AWS_REGION` | AWS region | `us-east-1` |
| `AWS_ALLOW_HTTP` | Allow HTTP (non-TLS) connections | `false` |

### vLLM Arguments

| Argument | Description |
|----------|-------------|
| `--enable-lora` | Enable LoRA adapter support in vLLM |
| `--max-lora-rank` | Maximum LoRA rank (must be >= your LoRA's rank) |
| `--max-loras` | Maximum number of LoRAs to load simultaneously |

## Backend API Reference

### Load LoRA

Load a LoRA adapter from a source URI.

```text
POST /v1/loras
```

**Request:**
```json
{
  "lora_name": "string",
  "source": {
    "uri": "string"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "LoRA adapter 'my-lora' loaded successfully",
  "lora_name": "my-lora",
  "lora_id": 1207343256
}
```

### List LoRAs

List all loaded LoRA adapters.

```text
GET /v1/loras
```

**Response:**
```json
{
  "status": "success",
  "loras": {
    "my-lora": 1207343256,
    "another-lora": 987654321
  },
  "count": 2
}
```

### Unload LoRA

Unload a LoRA adapter from the worker.

```text
DELETE /v1/loras/{lora_name}
```

**Response:**
```json
{
  "status": "success",
  "message": "LoRA adapter 'my-lora' unloaded successfully",
  "lora_name": "my-lora",
  "lora_id": 1207343256
}
```

## Kubernetes Deployment

For Kubernetes deployments, use the `DynamoModel` Custom Resource to declaratively manage LoRA adapters.

### DynamoModel CRD

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: customer-support-lora
  namespace: dynamo-system
spec:
  modelName: customer-support-adapter-v1
  baseModelName: Qwen/Qwen3-0.6B  # Must match modelRef.name in DGD
  modelType: lora
  source:
    uri: s3://my-models-bucket/loras/customer-support/v1
```

### How It Works

When you create a `DynamoModel`:

1. **Discovers endpoints**: Finds all pods running your `baseModelName`
2. **Creates service**: Automatically creates a Kubernetes Service
3. **Loads LoRA**: Calls the LoRA load API on each endpoint
4. **Updates status**: Reports which endpoints are ready

### Verify Deployment

```bash
# Check LoRA status
kubectl get dynamomodel customer-support-lora

# Expected output:
# NAME                    TOTAL   READY   AGE
# customer-support-lora   2       2       30s
```

For complete Kubernetes deployment details, see:
- [Managing Models with DynamoModel](../../kubernetes/deployment/dynamomodel-guide.md)
- [Kubernetes LoRA Deployment Example](../../../examples/backends/vllm/deploy/lora/README.md)

## Examples

| Example | Description |
|---------|-------------|
| [Local LoRA with MinIO](../../../examples/backends/vllm/launch/lora/README.md) | Local development with S3-compatible storage |
| [Kubernetes LoRA Deployment](../../../examples/backends/vllm/deploy/lora/README.md) | Production deployment with DynamoModel CRD |

## Troubleshooting

### LoRA Fails to Load

**Check S3 connectivity:**
```bash
# Verify LoRA exists in S3
aws --endpoint-url=$AWS_ENDPOINT s3 ls s3://my-loras/ --recursive
```

**Check cache directory:**
```bash
ls -la ~/.cache/dynamo_loras/
```

**Check worker logs:**
```bash
# Look for LoRA-related messages
kubectl logs deployment/my-worker | grep -i lora
```

### Model Not Found After Loading

- Verify the LoRA name matches exactly (case-sensitive)
- Check if the LoRA is listed: `curl http://localhost:8081/v1/loras`
- Ensure discovery registration succeeded (check worker logs)

### Inference Returns Base Model Response

- Verify the `model` field in your request matches the `lora_name`
- Check that the LoRA is loaded on the worker handling your request
- For disaggregated serving, ensure both prefill and decode workers have the LoRA

## See Also

- [Feature Matrix](../../reference/feature-matrix.md) - Backend compatibility overview
- [vLLM Backend](../../backends/vllm/README.md) - vLLM-specific configuration
- [Dynamo Operator](../../kubernetes/dynamo_operator.md) - Kubernetes operator overview
- [KV-Aware Routing](../../components/router/router_guide.md) - LoRA-aware request routing
