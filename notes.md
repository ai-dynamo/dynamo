# Design Notes: Diffusers on Intel XPU

## Decision 1: Endpoint — Use existing `/v1/images/generations`

**Decision:** Use the existing Dynamo images endpoint, NOT a custom one.

**Why:** The Rust frontend already implements `/v1/images/generations` (in `lib/llm/src/http/service/openai.rs:1959`). It:
- Accepts `NvCreateImageRequest` JSON
- Looks up model via `get_images_engine()`
- Sends request to backend via Dynamo RPC
- Folds backend stream into single `NvImagesResponse`
- Supports `nvext` with `num_inference_steps`, `guidance_scale`, `seed`, `negative_prompt`

The worker registers with `ModelType.Images` and the frontend routes automatically.

**Protocol types** are already defined at `components/src/dynamo/common/protocols/image_protocol.py`:
- `NvCreateImageRequest` — input (prompt, model, n, size, nvext, etc.)
- `NvImagesResponse` — output (created, data: list[ImageData])
- `ImageData` — (b64_json, url, revised_prompt)
- `ImageNvExt` — (negative_prompt, num_inference_steps, guidance_scale, seed)

## Decision 2: Worker Architecture

**Pattern:** Mirror the existing video worker (`worker.py`) but swap FastVideo for diffusers.

```python
# Registration
register_llm(
    ModelInput.Text,
    ModelType.Images,      # <-- Images instead of Videos
    endpoint,
    model_name,
    model_name,
)

# Endpoint
@dynamo_endpoint(NvCreateImageRequest, NvImagesResponse)
async def generate_image(self, request: NvCreateImageRequest):
    # ... generate with diffusers pipeline ...
    yield NvImagesResponse(...).model_dump()
```

**Key differences from video worker:**
- Import protocol types from `dynamo.common.protocols.image_protocol` (no need to define our own)
- Use `StableDiffusionXLPipeline` instead of `VideoGenerator`
- Output is base64-encoded PNG (not MP4)
- `ModelType.Images` instead of `ModelType.Videos`
- No `num_frames`, `fps`, `seconds` — image-specific params only
- `size` field parsed as `WxH` (same pattern as video worker)

## Decision 3: Device Management

```python
import torch

device = torch.device("xpu")
assert torch.xpu.is_available(), "Intel XPU not available"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 preferred on Intel XPU
)
pipe = pipe.to(device)
```

**No IPEX needed** — archived March 2026, features upstreamed to PyTorch 2.5+.

**torch.compile:** Supported on XPU via Inductor backend. Optional optimization flag:
```python
pipe.unet = torch.compile(pipe.unet, backend="inductor")
```

## Decision 4: Request/Response Mapping

### Input: NvCreateImageRequest -> diffusers pipe()

| NvCreateImageRequest field | diffusers pipe() param | Notes |
|---|---|---|
| `prompt` | `prompt` | Direct mapping |
| `size` ("WxH") | `width`, `height` | Parse string |
| `n` | `num_images_per_prompt` | Default 1 |
| `nvext.num_inference_steps` | `num_inference_steps` | Default 30 for SDXL |
| `nvext.guidance_scale` | `guidance_scale` | Default 7.5 for SDXL |
| `nvext.seed` | `generator=torch.Generator("xpu").manual_seed(seed)` | Reproducibility |
| `nvext.negative_prompt` | `negative_prompt` | Direct mapping |

### Output: diffusers result -> NvImagesResponse

```python
images = pipe(...).images  # list of PIL.Image
# For each image:
buffer = io.BytesIO()
image.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()

yield NvImagesResponse(
    created=int(time.time()),
    data=[ImageData(b64_json=b64) for b64 in encoded_images],
).model_dump()
```

## Decision 5: Dependencies

```
torch (XPU wheels from https://download.pytorch.org/whl/xpu)
diffusers
transformers
accelerate
safetensors
Pillow
pydantic
uvloop
```

No IPEX. No flash-attention. No CUDA anything.

## Decision 6: File Layout

```
examples/diffusers/xpu/
  README.md           # Full instructions: prereqs, install, run, test, troubleshoot
  worker_xpu.py       # SDXL worker with @dynamo_endpoint
  run_local.sh        # Launch frontend + worker locally
```

## Decision 7: Launch Script Design

Based on existing `local/run_local.sh` pattern:
- Set `DYN_DISCOVERY_BACKEND=file`
- Start `dynamo.frontend` in background
- Start `worker_xpu.py` in background
- Print curl example for testing
- Trap EXIT to clean up both processes

Environment variables:
- `MODEL` — HF model name (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `HTTP_PORT` — frontend port (default: 8000)
- `XPU_DEVICE` — device index (default: 0, via `ONEAPI_DEVICE_SELECTOR`)

## Decision 8: SDXL Defaults

| Parameter | Default | Why |
|---|---|---|
| dtype | `torch.bfloat16` | Better stability than FP16, native XPU support |
| size | 1024x1024 | SDXL native resolution |
| num_inference_steps | 30 | Good quality/speed balance for SDXL |
| guidance_scale | 7.5 | Standard SDXL CFG |
| attention | default (SDPA) | PyTorch scaled_dot_product_attention works on XPU |

## Test Command

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A photo of a cat sitting on a windowsill at sunset",
    "size": "1024x1024",
    "nvext": {
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 42
    }
  }' | python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
img_data = base64.b64decode(resp['data'][0]['b64_json'])
with open('output.png', 'wb') as f:
    f.write(img_data)
print('Saved to output.png')
"
```

## End-to-End Testing Results (2026-04-13, 8x Intel Arc Pro B60)

### Environment Setup
- PyTorch 2.11.0+xpu, diffusers 0.37.1, ai-dynamo 1.0.1 (from PyPI)
- User must be in `render` group (`sudo usermod -aG render $USER`) — GPU render devices are `660 root:render`
- `xpu-smi discovery` shows "No device discovered" but PyTorch detects all 8 GPUs — use PyTorch as the definitive check

### Performance (bf16, Intel Arc Pro B60)

| Test | Size | Steps | Time | Throughput |
|------|------|-------|------|------------|
| Unit test (no Dynamo) | 512x512 | 5 | 4.8s | ~2.1 it/s |
| Integration (first request) | 1024x1024 | 30 | 10.2s | ~9.3 it/s |
| Integration (warmed up) | 1024x1024 | 30 | 8.4s | ~17 it/s |
| Integration (small) | 512x512 | 10 | 1.9s | ~6.9 it/s |

### Validated Features
- Single image generation (n=1): works
- Multiple image generation (n=2): works, returns distinct images
- Seed reproducibility: works
- Error handling: frontend returns HTTP 400 for invalid size format and missing prompt

### Decision 9: Size is an Enum (discovered during testing)

The Rust frontend validates the `size` field as an enum, not arbitrary WxH:
- Allowed values: `256x256`, `512x512`, `1024x1024`, `1792x1024`, `1024x1792`
- The worker's `_parse_size()` function still works correctly for these values
- Updated README and worker docstring to reflect this

### Decision 10: Dynamo Install via PyPI

`pip install ai-dynamo` installs the full runtime (including Rust bindings) from PyPI without needing a Rust toolchain. Updated README to use this as the default install method, with source build as an alternative.
