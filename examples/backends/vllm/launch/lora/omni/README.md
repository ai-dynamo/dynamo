# Omni LoRA (Image) with vLLM Backend

This example launches Dynamo frontend + vLLM-Omni worker in aggregated mode, with dynamic LoRA load/unload enabled for image generation.

## Files

- `omni_lora_agg.sh`: Launch frontend and Omni worker with `--enable-lora`
- `validate_omni_lora_agg.sh`: Validate Omni LoRA endpoints and generation flow

## Quick Start

```bash
cd examples/backends/vllm/launch/lora/omni
./omni_lora_agg.sh
```

Default model is `stabilityai/stable-diffusion-xl-base-1.0`.

## Load a LoRA Adapter

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

Adapter folder must contain `adapter_config.json` and `adapter_model.safetensors`.

## Compare LoRA vs Base Output

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

## List and Unload LoRAs

```bash
curl -s http://localhost:8081/v1/loras | jq .
curl -s -X DELETE http://localhost:8081/v1/loras/sdxl-lora | jq .
```

## Configuration

Environment variables:

- `DYN_MODEL_NAME`: Base model (default `stabilityai/stable-diffusion-xl-base-1.0`)
- `DYN_HTTP_PORT`: Frontend port (default `8000`)
- `DYN_SYSTEM_PORT`: Omni system/admin port (default `8081`)
- `DYN_TENSOR_PARALLEL_SIZE`: Tensor parallel size (default `1`)
- `DYN_MEDIA_OUTPUT_FS_URL`: Media output path (default `file:///tmp/dynamo_media`)

Extra Omni args:

```bash
./omni_lora_agg.sh --gpu-memory-utilization 0.8
```

```bash
./omni_lora_agg.sh -- --gpu-memory-utilization 0.8
```

## Notes

- For non-Omni LoRA examples, see `examples/backends/vllm/launch/lora/README.md`.

## Validation Script

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
