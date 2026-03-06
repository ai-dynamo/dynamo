# Multimodal LoRA Serving Guide

Serve vision-language models (VLMs) with dynamically loadable LoRA adapters using Dynamo's aggregated architecture.

## Prerequisites

- **GPU**: NVIDIA GPU with sufficient VRAM (8 GB+ for 2B models, 24 GB+ for 7B models)
- **Dynamo**: Installed with vLLM support (`pip install dynamo[vllm]`)
- **jq** (optional, for pretty JSON output): `sudo apt install jq`
- **hf CLI** (optional, for downloading adapters): `pip install huggingface-hub`

## Quick Start

### 1. Launch the server

```bash
cd examples/multimodal/launch/lora
./lora_agg.sh
```

This starts the frontend (port 8000) and vLLM worker (port 8081) with `Qwen/Qwen3-VL-2B-Instruct` as the base model.

Wait for both services to report ready in the logs (look for `Application startup complete`).

### 2. Verify the server is running

```bash
curl http://localhost:8000/v1/models | jq .
```

You should see the base model listed.

### 3. Download a LoRA adapter

Download a compatible vision LoRA to your local filesystem:

```bash
export HF_TOKEN=<your-huggingface-token>

hf download Chhagan005/Chhagan-DocVL-Qwen3 --local-dir /tmp/my-vlm-lora
```

### 4. Load the LoRA adapter

```bash
curl -s -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "my-vlm-lora",
    "source": {"uri": "file:///tmp/my-vlm-lora"}
  }' | jq .
```

Expected response:
```json
{
  "status": "success",
  "message": "LoRA adapter 'my-vlm-lora' loaded successfully",
  "lora_name": "my-vlm-lora",
  "lora_id": 1207343256
}
```

### 5. Run inference with the LoRA adapter

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-vlm-lora",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Describe this image in detail"},
      {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
    ]}],
    "max_tokens": 300,
    "temperature": 0.0
  }' | jq .
```

### 6. Compare with the base model

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
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

### 7. Unload the LoRA adapter

```bash
curl -X DELETE http://localhost:8081/v1/loras/my-vlm-lora | jq .
```

### 8. Stop the server

Press `Ctrl+C` in the terminal running `lora_agg.sh`. The trap handler will clean up child processes.

## Configuration

### Command-line options

```bash
./lora_agg.sh --model llava-hf/llava-1.5-7b-hf            # Use a different base model
./lora_agg.sh -- --enforce-eager                            # Pass extra vLLM args
./lora_agg.sh -- --mm-processor-kwargs '{"max_pixels": 1003520}'  # Cap image resolution
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DYN_MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` | Base VLM model |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |
| `DYN_SYSTEM_PORT` | `8081` | Worker system/admin port |
| `DYN_LORA_PATH` | `/tmp/dynamo_loras_multimodal` | Local LoRA adapter cache |
| `DYN_MAX_LORA_RANK` | `64` | Maximum LoRA rank supported |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

### base models supported by this script

| Model | Notes |
|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | Default. Good for single-GPU testing. |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Higher quality, needs 24 GB+ VRAM. |
| `llava-hf/llava-1.5-7b-hf` | LLaVA architecture, 4096 max context. |

## LoRA Management API

All management endpoints are served on the system port (default 8081).

### Load a LoRA

```
POST /v1/loras
```

```json
{
  "lora_name": "my-adapter",
  "source": {
    "uri": "file:///path/to/adapter"
  }
}
```

Supported URI schemes:
- `file://` — local filesystem path
- `s3://` — S3-compatible storage (requires `AWS_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

### List loaded LoRAs

```
GET /v1/loras
```

### Unload a LoRA

```
DELETE /v1/loras/{lora_name}
```

### List all models (base + LoRAs)

```
GET /v1/models        (on the frontend port, default 8000)
```

## Running the validation script

A validation script is provided to test the LoRA endpoints against a running server:

```bash
# Start the server in one terminal
./lora_agg.sh

# In another terminal, download a LoRA adapter
hf download Chhagan005/Chhagan-DocVL-Qwen3 --local-dir /tmp/my-vlm-lora

# Run the full test suite (with end-to-end LoRA load/infer/unload)
./validate_lora_agg.sh --lora-path /tmp/my-vlm-lora

# Or run only the error-handling and base-model tests (no adapter needed)
./validate_lora_agg.sh
```

The validation script covers:
- Frontend health and base model discovery
- LoRA load/unload error handling (missing fields, non-existent adapter)
- End-to-end LoRA lifecycle: load, verify in `/v1/models`, infer, unload (when `--lora-path` provided)
- Base model multimodal inference

## Troubleshooting

### Frontend fails to start
- Check if port 8000 is already in use: `lsof -i :8000`
- Set a different port: `DYN_HTTP_PORT=8001 ./lora_agg.sh`

### OOM during inference
- Reduce `--max-model-len` via extra args: `./lora_agg.sh -- --max-model-len 4096`
- Cap image resolution: `./lora_agg.sh -- --mm-processor-kwargs '{"max_pixels": 1003520}'`
- Lower GPU memory utilization: `./lora_agg.sh -- --gpu-memory-utilization 0.80`

### LoRA fails to load
- Verify the adapter path exists and contains `adapter_config.json` and `adapter_model.safetensors`
- Ensure the adapter is compatible with the base model architecture
- Check that `max-lora-rank` (default 64) is >= the adapter's rank
- Review worker logs for detailed error messages

### Inference returns errors after loading LoRA
- Verify the LoRA is loaded: `curl http://localhost:8081/v1/loras | jq .`
- Confirm the model name in the request matches the `lora_name` exactly (case-sensitive)
- Check that the adapter was trained for the same base model

### Cache issues
- Inspect the cache: `ls -la /tmp/dynamo_loras_multimodal/`
- Clear the cache: `rm -rf /tmp/dynamo_loras_multimodal/*`

## Cleanup

```bash
# Remove LoRA cache
rm -rf /tmp/dynamo_loras_multimodal

# Remove downloaded adapter
rm -rf /tmp/my-vlm-lora
```
