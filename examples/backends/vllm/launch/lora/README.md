# S3-compatible storage backend LoRA Integration Guide

This guide explains how to set up and use LoRA (Low-Rank Adaptation) adapters with Dynamo using S3-compatible storage backend (e.g. MinIO, AWS S3, GCS, etc.).

## Overview

This example demonstrates how to:
1. Set up MinIO as a local S3-compatible storage
2. Download LoRA adapters from Hugging Face Hub
3. Upload LoRA adapters to MinIO
4. Load and use LoRA adapters with Dynamo
5. Run inference with LoRA-adapted models
6. Manage (load/unload) LoRA adapters

## Prerequisites

### Required Software
- Docker (for running MinIO)
- Python 3.8+
- AWS CLI: `pip install awscli`
- Hugging Face CLI: `pip install huggingface-hub`
- jq (optional, for pretty JSON output): `sudo apt install jq`

### Python Dependencies
Make sure you have Dynamo installed with vLLM support:
```bash
pip install dynamo vllm
```

## Quick Start

### Step 1: Setup MinIO and Upload LoRA

Run the setup script to start MinIO and download/upload a LoRA adapter from Hugging Face:

```bash
./setup_minio.sh
```

This script will:
- Start MinIO in a Docker container
- Download a LoRA adapter from Hugging Face Hub (default: `codelion/Qwen3-0.6B-accuracy-recovery-lora`)
- Upload the LoRA to MinIO at `s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora`

#### Script Options

The setup script supports different modes:

```bash
# Full setup (default) - start MinIO, download & upload LoRA
./setup_minio.sh

# Start MinIO only (without downloading/uploading)
./setup_minio.sh --start

# Stop MinIO
./setup_minio.sh --stop

# Show help
./setup_minio.sh --help
```

#### Customize the LoRA to Download

You can specify a different LoRA repository and name:

```bash
HF_LORA_REPO="username/lora-repo" \
LORA_NAME="my-lora" \
  ./setup_minio.sh
```

### Step 2: Launch Dynamo with LoRA Support

Start the Dynamo frontend and worker with LoRA support enabled:

```bash
./agg_lora.sh
```

This will:
- Set up AWS credentials for MinIO
- Start the Dynamo frontend on port 8000
- Start the Dynamo worker (vLLM) on port 8081 with LoRA support

Wait for the services to start (check the logs for "Application startup complete").

## Working with LoRAs

### 1. Check Available Models

List all available models (base model only at first):

```bash
curl http://localhost:8000/v1/models | jq .
```

### 2. Load a LoRA Adapter

Load a LoRA from S3-compatible storage backend (e.g. MinIO):

```bash
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
    "source": {
      "uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"
    }
  }' | jq .
```

Expected response:
```json
{
  "status": "success",
  "message": "LoRA adapter 'codelion/Qwen3-0.6B-accuracy-recovery-lora' loaded successfully",
  "lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
  "lora_id": 1207343256
}
```

### 3. List Loaded LoRAs

Check which LoRAs are currently loaded:

```bash
curl http://localhost:8081/v1/loras | jq .
```

### 4. Verify LoRA in Models List

After loading, the LoRA should appear in the models list:

```bash
curl http://localhost:8000/v1/models | jq .
```

You should see both the base model and the LoRA adapter listed.

### 5. Run Inference with LoRA

#### Using the LoRA-adapted model:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codelion/Qwen3-0.6B-accuracy-recovery-lora",
    "messages": [{
      "role": "user",
      "content": "What is good low risk investment strategy?"
    }],
    "max_tokens": 300,
    "temperature": 0.1
  }' | jq .
```

#### For comparison, using the base model:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{
      "role": "user",
      "content": "What is good low risk investment strategy?"
    }],
    "max_tokens": 300
  }' | jq .
```

### 6. Unload a LoRA

When you no longer need a LoRA, unload it to free up resources:

```bash
curl -X DELETE http://localhost:8081/v1/loras/codelion/Qwen3-0.6B-accuracy-recovery-lora | jq .
```

Expected response:
```json
{
  "status": "success",
  "message": "LoRA unloaded successfully"
}
```

After unloading, the LoRA will be removed from both `/v1/loras` and `/v1/models` endpoints.

## Configuration

### Environment Variables

The following environment variables can be configured:

```bash
# S3-compatible storage backend Configuration
export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1

# Dynamo LoRA Configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio
```

### MinIO Console

Access the MinIO web console at http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin`

## Troubleshooting

### MinIO won't start
- Check if ports 9000 and 9001 are already in use
- Ensure Docker is running
- Check Docker logs: `docker logs dynamo-minio`
- Try stopping any existing MinIO containers: `./setup_minio.sh --stop`
- Restart MinIO: `./setup_minio.sh --start`

### LoRA fails to load
- Verify the LoRA is uploaded to MinIO: `aws --endpoint-url=http://localhost:9000 s3 ls s3://my-loras/`
- Check AWS credentials are set correctly
- Ensure the LoRA files are compatible with the base model
- Check vLLM logs for detailed error messages

### Inference fails
- Verify the model name matches exactly (case-sensitive)
- Check if the LoRA is loaded: `curl http://localhost:8081/v1/loras`
- Ensure the base model supports the LoRA rank
- Check that max_lora_rank in the worker config is >= the LoRA rank

### Cache issues
- Check the cache directory: `ls -la /tmp/dynamo_loras_minio/`
- Clear the cache if needed: `rm -rf /tmp/dynamo_loras_minio/*`
- Ensure the cache directory is writable

## Advanced Usage

### Loading Multiple LoRAs

You can load multiple LoRA adapters simultaneously:

```bash
# Load first LoRA
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "lora1", "source": {"uri": "s3://my-loras/lora1"}}'

# Load second LoRA
curl -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "lora2", "source": {"uri": "s3://my-loras/lora2"}}'
```

### Using Different Base Models

To use a different base model, modify the `--model` parameter in `agg_lora.sh`:

```bash
python -m dynamo.vllm --model meta-llama/Llama-2-7b-hf --enable-lora --max-lora-rank 64
```

Ensure your LoRAs are compatible with the chosen base model.

## Cleanup

### Stop Services

Press `Ctrl+C` in the terminal running `agg_lora.sh` to stop Dynamo services.

### Stop MinIO

```bash
# Using the setup script (recommended)
./setup_minio.sh --stop

# Or manually with Docker
docker stop dynamo-minio
docker rm dynamo-minio
```

### Clean Up Data

```bash
# Remove MinIO data
rm -rf ~/dynamo_minio_data

# Remove LoRA cache
rm -rf /tmp/dynamo_loras_minio
```

## API Reference

### Load LoRA
- **Endpoint**: `POST /v1/loras`
- **Body**: `{"lora_name": "string", "source": {"uri": "string"}}`
- **Response**: `{"status": "success", "lora_id": int}`

### List LoRAs
- **Endpoint**: `GET /v1/loras`
- **Response**: Array of loaded LoRAs

### Unload LoRA
- **Endpoint**: `DELETE /v1/loras/{lora_name}`
- **Response**: `{"status": "success", "message": "string"}`

### List Models
- **Endpoint**: `GET /v1/models`
- **Response**: OpenAI-compatible models list

### Chat Completions
- **Endpoint**: `POST /v1/chat/completions`
- **Body**: OpenAI-compatible chat completion request
- **Response**: OpenAI-compatible chat completion response
