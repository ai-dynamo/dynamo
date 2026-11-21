---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LoRA Adapter Walkthrough
subtitle: Run LoRA adapters locally for text, vision-language, and vLLM-Omni image generation.
---

This walkthrough runs the local LoRA examples end to end. It stores a LoRA adapter in S3-compatible
storage (MinIO, AWS S3, GCS, and others) or loads it from Hugging Face Hub or a local path, loads it
into a running worker through the worker system API, runs inference against it, and unloads it. It
covers text models with vLLM and SGLang, KV-aware routing across two workers, vision-language models,
and vLLM-Omni image generation. For LoRA concepts, backend support, and the equivalent manual launch
commands, see [LoRA Adapters](lora-adapters.md).

| Example | Directory | Launch scripts |
|---|---|---|
| Text, vLLM | [`examples/backends/vllm/launch/lora`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora) | `agg_lora.sh`, `agg_lora_hf.sh`, `agg_lora_router.sh` |
| Text, SGLang | [`examples/backends/sglang/launch/lora`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/launch/lora) | `agg_lora.sh`, `agg_lora_router.sh` |
| Vision-language, vLLM | [`examples/backends/vllm/launch/lora/multimodal`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/multimodal) | `lora_agg.sh` |
| Image generation, vLLM-Omni | [`examples/backends/vllm/launch/lora/omni`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/omni) | `omni_lora_agg.sh` |
| Text, vLLM on Intel XPU | [`examples/backends/vllm/launch/lora/xpu`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/xpu) | `agg_lora_xpu.sh`, `agg_lora_router_xpu.sh` |

Run each command from the example directory named in its section. Every section starts with the
`cd` command from the repository root.

## Prerequisites

- Docker, for running MinIO
- Python 3.11+
- AWS CLI: `pip install awscli`
- Hugging Face CLI: `pip install huggingface-hub[cli]`
- `jq` (optional, for pretty JSON output): `sudo apt install jq`
- Dynamo installed with your chosen backend; see [Local Installation](../installation/install-dynamo.mdx)
- For the vision-language example, an NVIDIA GPU with sufficient VRAM: 8 GB+ for 2B models, 24 GB+ for
  7B models

The vision-language and vLLM-Omni examples load adapters from a local path and do not need Docker,
MinIO, or the AWS CLI.

## Set Up Local MinIO

The text examples load the adapter from MinIO, a local S3-compatible store. From the repository root,
change to the vLLM example directory and run the setup script:

```bash
cd examples/backends/vllm/launch/lora
./setup_minio.sh
```

The SGLang directory, `examples/backends/sglang/launch/lora`, has its own copy of `setup_minio.sh`
with the same options.

The script:

- Starts MinIO in a Docker container
- Downloads a LoRA adapter from Hugging Face Hub (default: `codelion/Qwen3-0.6B-accuracy-recovery-lora`)
- Uploads the LoRA to MinIO at `s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora`

### Script Options

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

### Customize the Adapter

To download a different LoRA, set the Hugging Face repository and the LoRA name:

```bash
HF_LORA_REPO="username/lora-repo" \
LORA_NAME="my-lora" \
  ./setup_minio.sh
```

### MinIO Console

Open the MinIO web console at `http://localhost:9001` and sign in with username `minioadmin` and
password `minioadmin`.

## Text LoRA with vLLM

All commands in this section run from the vLLM example directory:

```bash
cd examples/backends/vllm/launch/lora
```

### Launch with a Hugging Face Hub Adapter

To load the adapter directly from Hugging Face Hub, without MinIO, run:

```bash
./agg_lora_hf.sh
```

Set `HF_TOKEN` before running the script when the base model or adapter is private. Override
`HF_LORA_REPO`, `LORA_NAME`, or `LORA_URI` to use another adapter or revision.

### Launch with a MinIO Adapter

To use S3-compatible storage through MinIO, upload the adapter as described in
[Set Up Local MinIO](#set-up-local-minio), then start the frontend and a LoRA-enabled worker:

```bash
./setup_minio.sh    # Start MinIO, download & upload LoRA
./agg_lora.sh       # Launch vLLM frontend + worker with LoRA
```

`agg_lora.sh` sets up AWS credentials for MinIO, starts the Dynamo frontend on port 8000, and starts
the Dynamo worker with LoRA support and its system API on port 8081. For the variables it sets and the
equivalent manual commands, see the Local tab of [Serve a LoRA Adapter](lora-adapters.md#serve-a-lora-adapter).

Wait for the services to start; check the logs for `Application startup complete`.

### vLLM Launch Overrides

- The default `--max-lora-rank` is `64`, the same as SGLang.
- `agg_lora.sh` reads `MAX_MODEL_LEN`, `MAX_CONCURRENT_SEQS`, `DYN_HTTP_PORT`, and
  `DYN_SYSTEM_PORT1` from the environment. It sets the base model in the script; edit `MODEL` there
  to change it. `agg_lora_hf.sh` also reads `MODEL`.

### Load, Query, and Unload the Adapter

These requests use the default adapter from the MinIO setup, the frontend on port 8000, and the worker
system API on port 8081. They work the same way for the SGLang worker.

1. List the available models. At first, only the base model is listed.

   ```bash
   curl http://localhost:8000/v1/models | jq .
   ```

2. Load the LoRA from S3-compatible storage (MinIO):

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

3. List the LoRAs that are currently loaded:

   ```bash
   curl http://localhost:8081/v1/loras | jq .
   ```

4. Verify that the LoRA appears in the models list. Both the base model and the LoRA adapter are
   listed.

   ```bash
   curl http://localhost:8000/v1/models | jq .
   ```

5. Run inference with the LoRA-adapted model:

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

   For comparison, send the same prompt to the base model:

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

6. When you no longer need the LoRA, unload it to free resources:

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

   After unloading, the LoRA is removed from both the `/v1/loras` and `/v1/models` endpoints.

## Text LoRA with SGLang

**Experimental.** For what the SGLang LoRA path validates, see
[Serve a LoRA Adapter with SGLang](lora-adapters.md#serve-a-lora-adapter-with-sglang).

From the repository root, change to the SGLang example directory, upload the adapter, and launch the
frontend and a LoRA-enabled SGLang worker:

```bash
cd examples/backends/sglang/launch/lora
./setup_minio.sh    # Start MinIO, download & upload LoRA
./agg_lora.sh       # Launch SGLang frontend + worker with LoRA
```

Then run the requests in [Load, Query, and Unload the Adapter](#load-query-and-unload-the-adapter).
The SGLang script uses the same ports and adapter name.

### SGLang Launch Notes

- The launch script uses `--lora-target-modules all` and `--max-lora-rank 64` by default.
- Override settings with the `MODEL`, `LORA_NAME`, `DYN_SYSTEM_PORT`, and `DYN_HTTP_PORT` environment
  variables.
- SGLang LoRA loading goes through `engine.tokenizer_manager.load_lora_adapter()`.

## KV-Aware Routing with Two GPUs

Both backends provide `agg_lora_router.sh`, which launches two workers behind a KV-aware router. For
how the router accounts for adapter identity and which router modes support LoRA, see
[KV Cache-Aware LoRA Routing](lora-adapters.md#kv-cache-aware-lora-routing).

### vLLM Routing Example

Upload the adapter as described in [Set Up Local MinIO](#set-up-local-minio), then launch two vLLM
workers behind the KV-aware router:

```bash
cd examples/backends/vllm/launch/lora
./agg_lora_router.sh
```

Load the LoRA to both workers, on ports 8081 and 8082:

```bash
curl -s -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora", "source": {"uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"}}' | jq .

curl -s -X POST http://localhost:8082/v1/loras \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "codelion/Qwen3-0.6B-accuracy-recovery-lora", "source": {"uri": "s3://my-loras/codelion/Qwen3-0.6B-accuracy-recovery-lora"}}' | jq .
```

The router then uses KV cache affinity to improve cache hit rates.

### SGLang Routing Example

**Experimental.** Launch two SGLang workers behind the KV router (2 GPUs):

```bash
cd examples/backends/sglang/launch/lora
./agg_lora_router.sh
```

Load the LoRA to both workers, on ports 8081 and 8082, with the same requests as the vLLM example.

`dynamo.sglang` adds `"format": "dynamo"` to `--kv-events-config` when no format is set. Its events
carry the adapter name that LoRA-aware KV routing needs.

## Vision-Language LoRA with vLLM

Serve vision-language models (VLMs) with dynamically loadable LoRA adapters using Dynamo's aggregated
architecture. For the LoRA management API on multimodal workers in disaggregated deployments, see
[LoRA Adapters on Multimodal Workers](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#lora-adapters-on-multimodal-workers).

This example needs Dynamo installed with vLLM support (`ai-dynamo[vllm]`). The `hf` CLI
(`pip install huggingface-hub`) is optional, for downloading adapters.

### Launch the Vision-Language Server

```bash
cd examples/backends/vllm/launch/lora/multimodal
./lora_agg.sh
```

This starts the frontend (port 8000) and the vLLM worker (port 8081) with `Qwen/Qwen3-VL-2B-Instruct`
as the base model. Wait for both services to report ready in the logs (look for
`Application startup complete`).

Verify that the server is running. The base model is listed.

```bash
curl http://localhost:8000/v1/models | jq .
```

### Load and Query a Vision Adapter

1. Download a compatible vision LoRA to your local filesystem:

   ```bash
   export HF_TOKEN=<your-huggingface-token>

   hf download Chhagan005/Chhagan-DocVL-Qwen3 --local-dir /tmp/my-vlm-lora
   ```

2. Load the LoRA adapter:

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

3. Run inference with the LoRA adapter:

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

4. Compare with the base model:

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

5. Unload the LoRA adapter:

   ```bash
   curl -X DELETE http://localhost:8081/v1/loras/my-vlm-lora | jq .
   ```

6. To stop the server, press `Ctrl+C` in the terminal running `lora_agg.sh`. The trap handler cleans
   up child processes.

### Vision-Language Launch Options

`lora_agg.sh` accepts a base model and extra vLLM arguments:

```bash
./lora_agg.sh --model llava-hf/llava-1.5-7b-hf            # Use a different base model
./lora_agg.sh -- --enforce-eager                            # Pass extra vLLM args
./lora_agg.sh -- --mm-processor-kwargs '{"max_pixels": 1003520}'  # Cap image resolution
```

| Variable | Default | Description |
|---|---|---|
| `DYN_MODEL_NAME` | `Qwen/Qwen3-VL-2B-Instruct` | Base VLM model |
| `DYN_HTTP_PORT` | `8000` | Frontend HTTP port |
| `DYN_SYSTEM_PORT` | `8081` | Worker system/admin port |
| `DYN_LORA_PATH` | `/tmp/dynamo_loras_multimodal` | Local LoRA adapter cache |
| `DYN_MAX_LORA_RANK` | `64` | Maximum LoRA rank supported |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

Base models supported by the script:

| Model | Notes |
|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | Default. Good for single-GPU testing. |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Higher quality, needs 24 GB+ VRAM. |
| `llava-hf/llava-1.5-7b-hf` | LLaVA architecture, 4096 max context. |

### Validate the Vision-Language Endpoints

`validate_lora_agg.sh` tests the LoRA endpoints against a running server:

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
- End-to-end LoRA lifecycle: load, verify in `/v1/models`, infer, unload (when `--lora-path` is
  provided)
- Base model multimodal inference

## Image Generation LoRA with vLLM-Omni

The `omni` example launches the Dynamo frontend and a vLLM-Omni worker in aggregated mode, with
dynamic LoRA load and unload enabled for image generation. For the `dynamo.vllm.omni` flags, including
`--lora-path` for adapters fused at startup, see
[vLLM-Omni Configuration](../../reference/backends/vllm-omni-configuration.mdx).

### Launch the Omni Server

```bash
cd examples/backends/vllm/launch/lora/omni
./omni_lora_agg.sh
```

The default model is `stabilityai/stable-diffusion-xl-base-1.0`.

### Load an Image Adapter

```bash
curl -s -X POST http://localhost:8081/v1/loras \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sdxl-lora",
    "source": {
      "uri": "file:///path/to/adapter"
    }
  }' | jq .
```

The adapter folder must contain `adapter_config.json` and `adapter_model.safetensors`.

### Compare LoRA and Base Output

Generate the same prompt with a fixed seed from the LoRA and from the base model, then compare the
images:

```bash
PROMPT="A red fox sitting on a snow-covered mountain at sunset, photorealistic"

curl -sS -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"sdxl-lora\",
    \"prompt\": \"${PROMPT}\",
    \"size\": \"1024x1024\",
    \"nvext\": {\"num_inference_steps\": 20, \"seed\": 42}
  }" \
| jq -e -r '.data[0].b64_json' \
| base64 -d > sdxl_lora.png

curl -sS -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"stabilityai/stable-diffusion-xl-base-1.0\",
    \"prompt\": \"${PROMPT}\",
    \"size\": \"1024x1024\",
    \"nvext\": {\"num_inference_steps\": 20, \"seed\": 42}
  }" \
| jq -e -r '.data[0].b64_json' \
| base64 -d > sdxl_base.png

sha256sum sdxl_lora.png sdxl_base.png
cmp -s sdxl_lora.png sdxl_base.png; echo "cmp_exit=$?"
```

`cmp_exit=0` means identical bytes; `cmp_exit=1` means different outputs.

### List and Unload Image Adapters

```bash
curl -s http://localhost:8081/v1/loras | jq .
curl -s -X DELETE http://localhost:8081/v1/loras/sdxl-lora | jq .
```

### Omni Launch Options

| Variable | Default | Description |
|---|---|---|
| `DYN_MODEL_NAME` | `stabilityai/stable-diffusion-xl-base-1.0` | Base model |
| `DYN_HTTP_PORT` | `8000` | Frontend port |
| `DYN_SYSTEM_PORT` | `8081` | Omni system/admin port |
| `DYN_TENSOR_PARALLEL_SIZE` | `1` | Tensor parallel size |
| `DYN_MEDIA_OUTPUT_FS_URL` | `file:///tmp/dynamo_media` | Media output path |

Pass extra Omni arguments after the script options, with or without a `--` separator:

```bash
./omni_lora_agg.sh --gpu-memory-utilization 0.8
```

```bash
./omni_lora_agg.sh -- --gpu-memory-utilization 0.8
```

### Validate the Omni Endpoints

Run the validator against a running Omni server:

```bash
# Terminal 1
./omni_lora_agg.sh

# Terminal 2 (basic endpoint and base model checks)
./validate_omni_lora_agg.sh
```

Run full end-to-end LoRA lifecycle tests with a real adapter path:

```bash
./validate_omni_lora_agg.sh --lora-path /path/to/omni-lora-adapter
```

Optional arguments:

```bash
./validate_omni_lora_agg.sh --frontend-port 8000 --system-port 8081 --base-model stabilityai/stable-diffusion-xl-base-1.0
```

## Configuration

### Environment Variables

The text launch scripts configure S3-compatible storage and LoRA support with these variables:

```bash
# S3-compatible storage backend configuration
export AWS_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1

# Dynamo LoRA configuration
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH=/tmp/dynamo_loras_minio
```

For each variable's description and default, and the other AWS credential and endpoint variables,
see the worker environment variables in [Serve a LoRA Adapter](lora-adapters.md#serve-a-lora-adapter).

### Load Multiple Adapters

Load multiple LoRA adapters on the same worker:

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

### Use a Different Base Model

To use a different base model, set the `MODEL` environment variable for the scripts that read it,
such as the vLLM `agg_lora_hf.sh` and the SGLang `agg_lora.sh`:

```bash
MODEL=meta-llama/Llama-2-7b-hf ./agg_lora_hf.sh
```

The vLLM `agg_lora.sh` sets `MODEL` in the script, so edit that line instead.

Ensure your LoRAs are compatible with the chosen base model.

## API Reference

The LoRA management endpoints are served on the worker system port (`DYN_SYSTEM_PORT`, default 8081
in these examples). The model and inference endpoints are served on the frontend port (default 8000).
Keep the system port on a trusted network; see the warning in
[Serve a LoRA Adapter](lora-adapters.md#serve-a-lora-adapter).

| Operation | Endpoint | Port | Request body | Response |
|---|---|---|---|---|
| Load LoRA | `POST /v1/loras` | System | `{"lora_name": "string", "source": {"uri": "string"}}` | `{"status": "success", "lora_id": int}` |
| List LoRAs | `GET /v1/loras` | System | None | Array of loaded LoRAs |
| Unload LoRA | `DELETE /v1/loras/{lora_name}` | System | None | `{"status": "success", "message": "string"}` |
| List models | `GET /v1/models` | Frontend | None | OpenAI-compatible models list (base model and loaded LoRAs) |
| Chat completions | `POST /v1/chat/completions` | Frontend | OpenAI-compatible chat completion request | OpenAI-compatible chat completion response |
| Image generations | `POST /v1/images/generations` | Frontend | Image generation request with `model`, `prompt`, `size`, and optional `nvext` | Generated images in `data[].b64_json` |

A load request names the adapter and its source:

```json
{
  "lora_name": "my-adapter",
  "source": {
    "uri": "file:///path/to/adapter"
  }
}
```

Supported URI schemes:

- `file://`: local filesystem path
- `s3://`: S3-compatible storage (requires `AWS_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- `hf://`: Hugging Face Hub repository, as used by `agg_lora_hf.sh`

## Troubleshooting

For general LoRA problems, also see [Troubleshooting](lora-adapters.md#troubleshooting) in the LoRA
Adapters guide.

### MinIO Won't Start

- Check whether ports 9000 and 9001 are already in use.
- Ensure Docker is running.
- Check the Docker logs: `docker logs dynamo-minio`
- Stop any existing MinIO containers: `./setup_minio.sh --stop`
- Restart MinIO: `./setup_minio.sh --start`

### LoRA Fails to Load

- For `s3://` sources, verify that the LoRA is uploaded to MinIO:
  `aws --endpoint-url=http://localhost:9000 s3 ls s3://my-loras/`, and check that the AWS credentials
  are set correctly.
- For `file://` sources, verify that the adapter path exists and contains `adapter_config.json` and
  `adapter_model.safetensors`.
- Ensure the LoRA files are compatible with the base model architecture.
- Check that `max-lora-rank` (default 64) is greater than or equal to the adapter's rank.
- Check the worker logs for detailed error messages.

### Inference Fails After Loading a LoRA

- Verify that the model name in the request matches the `lora_name` exactly (case-sensitive).
- Check that the LoRA is loaded: `curl http://localhost:8081/v1/loras | jq .`
- Ensure the base model supports the LoRA rank, and that `max_lora_rank` in the worker config is
  greater than or equal to the LoRA rank.
- Check that the adapter was trained for the same base model.

### Vision-Language Frontend Fails to Start

- Check whether port 8000 is already in use: `lsof -i :8000`
- Set a different port: `DYN_HTTP_PORT=8001 ./lora_agg.sh`

### Out of Memory During Vision-Language Inference

- Reduce `--max-model-len` through extra args: `./lora_agg.sh -- --max-model-len 4096`
- Cap the image resolution: `./lora_agg.sh -- --mm-processor-kwargs '{"max_pixels": 1003520}'`
- Lower GPU memory utilization: `./lora_agg.sh -- --gpu-memory-utilization 0.80`

### Cache Issues

The text examples cache adapters in `/tmp/dynamo_loras_minio/`; the vision-language example uses
`/tmp/dynamo_loras_multimodal/`.

- Inspect the cache: `ls -la /tmp/dynamo_loras_minio/` or `ls -la /tmp/dynamo_loras_multimodal/`
- Clear the cache if needed: `rm -rf /tmp/dynamo_loras_minio/*` or
  `rm -rf /tmp/dynamo_loras_multimodal/*`
- Ensure the cache directory is writable.

## Cleanup

Press `Ctrl+C` in the terminal running the launch script, such as `agg_lora.sh`, to stop the Dynamo
services.

Stop MinIO with the setup script (recommended):

```bash
./setup_minio.sh --stop
```

Or stop it manually with Docker:

```bash
docker stop dynamo-minio
docker rm dynamo-minio
```

Remove the MinIO data and the LoRA caches:

```bash
# Remove MinIO data
rm -rf ~/dynamo_minio_data

# Remove LoRA cache
rm -rf /tmp/dynamo_loras_minio
rm -rf /tmp/dynamo_loras_multimodal
```

Remove the adapter downloaded for the vision-language example:

```bash
rm -rf /tmp/my-vlm-lora
```

## See Also

- [LoRA Adapters](lora-adapters.md): LoRA concepts, backend support, and KV-aware LoRA routing
- [Kubernetes LoRA Adapter Walkthrough](../../kubernetes/operations/lora-adapter-walkthrough.md):
  deploy the same text and vision-language adapters on Kubernetes
- [LoRA Serving examples](../../recipes/examples/lora.mdx): the launch scripts and manifests
