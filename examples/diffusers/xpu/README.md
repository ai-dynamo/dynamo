# Stable Diffusion XL on Intel XPU

Run text-to-image generation with [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) on Intel XPU using Dynamo's distributed runtime.

This example uses HuggingFace [diffusers](https://github.com/huggingface/diffusers) with native PyTorch XPU support — no IPEX or CUDA dependencies required.

## Prerequisites

### Hardware

- Intel Data Center GPU (Flex, Max series) or Intel Arc GPU (A-series, B-series)
- Minimum 12 GB device memory (SDXL in BF16 uses ~6.5 GB)

### Software

| Component | Requirement |
|-----------|-------------|
| OS | Linux (Ubuntu 22.04+ recommended) |
| Kernel | 6.11+ (Battlemage GPUs require `force_probe`) |
| GPU drivers | Intel compute runtime — `level-zero`, `intel-level-zero-gpu`, `intel-opencl-icd` |
| Python | 3.10+ |

#### Install Intel GPU Drivers

Follow the [Intel GPU driver installation guide](https://dgpu-docs.intel.com/driver/installation.html) for your distribution. On Ubuntu:

```bash
sudo apt update
sudo apt install -y intel-fw-gpu intel-i915-dkms xpu-smi \
    intel-level-zero-gpu level-zero intel-opencl-icd
```

Ensure your user has access to GPU devices.

Verify the GPU is visible:

```bash
xpu-smi discovery
# If xpu-smi shows "No device discovered" but the hardware is present,
# verify via PyTorch instead (see step 2 below).
```

## Setup

### 1. Clone the Repository and Create a Virtual Environment

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch with XPU Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

Verify XPU is detected:

```bash
python3 -c "import torch; print(f'XPU available: {torch.xpu.is_available()}, devices: {torch.xpu.device_count()}')"
```

Expected output:

```
XPU available: True, devices: int (the actual XPU card on your server)
```

### 3. Install Diffusers and Dependencies

```bash
pip install diffusers transformers accelerate safetensors Pillow uvloop
```

### 4. Install Dynamo

```bash
pip install ai-dynamo
```

Alternatively, to install from source (requires Rust toolchain):

```bash
cd lib/bindings/python && pip install maturin && maturin develop && cd -
pip install -e .
```

### 5. Download the Model (Optional)

The model downloads automatically on first run. To pre-download:

```bash
python3 -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"
```

## Run

### Quick Start

```bash
cd examples/diffusers/xpu
./run_local.sh
```

This starts two processes:
- **Frontend** — Dynamo HTTP server on port 8000 (`/v1/images/generations`)
- **Worker** — SDXL pipeline on Intel XPU

Logs are written to `.runtime/logs/`.

### Configuration

Override defaults with environment variables:

```bash
# Use a different model
MODEL=stabilityai/stable-diffusion-xl-base-1.0 ./run_local.sh

# Use FP16 instead of BF16
DTYPE=fp16 ./run_local.sh

# Change the HTTP port
HTTP_PORT=8080 ./run_local.sh

# Enable torch.compile for faster inference (slower first request)
WORKER_EXTRA_ARGS="--enable-torch-compile" ./run_local.sh

# Select a specific XPU device (when multiple GPUs are present)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./run_local.sh
```

### Run Worker Directly (Without Launch Script)

Terminal 1 — Worker:

```bash
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=/tmp/dynamo-discovery
python3 worker_xpu.py --model stabilityai/stable-diffusion-xl-base-1.0
```

Terminal 2 — Frontend:

```bash
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=/tmp/dynamo-discovery
python3 -m dynamo.frontend --http-port 8000 --discovery-backend file
```

## Test

### Generate an Image

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
print(f'Saved output.png ({len(img_data)} bytes)')
"
```

Example output (`seed=42`, 30 steps, 1024x1024, BF16 on Intel Arc Pro B60):

![A photo of a cat sitting on a windowsill at sunset](output.png)

### Generate Multiple Images

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "A watercolor painting of a mountain landscape",
    "size": "1024x1024",
    "n": 2,
    "nvext": {
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 123
    }
  }' | python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
for i, img in enumerate(resp['data']):
    data = base64.b64decode(img['b64_json'])
    with open(f'output_{i}.png', 'wb') as f:
        f.write(data)
    print(f'Saved output_{i}.png ({len(data)} bytes)')
"
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | (required) | Text description of the desired image |
| `model` | string | (required) | Must match the model the worker registered |
| `size` | string | `"1024x1024"` | One of: `256x256`, `512x512`, `1024x1024`, `1792x1024`, `1024x1792` |
| `n` | int | `1` | Number of images to generate (1-10) |
| `nvext.num_inference_steps` | int | `30` | Denoising steps (more = higher quality, slower) |
| `nvext.guidance_scale` | float | `7.5` | CFG scale (higher = more prompt-adherent) |
| `nvext.seed` | int | random | RNG seed for reproducibility |
| `nvext.negative_prompt` | string | none | Text describing what to avoid |

## Troubleshooting

### "Intel XPU is not available"

1. Check that your user has access to GPU devices.
2. Check that GPU drivers are installed:
   ```bash
   xpu-smi discovery
   # Note: xpu-smi may show "No device discovered" even when drivers work.
   # Use PyTorch as the definitive check (step 3).
   ```
3. Check that PyTorch was installed with XPU support:
   ```bash
   python3 -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.device_count())"
   ```
4. If using Battlemage (Arc B-series), ensure kernel 6.11+ with `i915.force_probe=*` boot parameter.

### Out of Memory

- Reduce image size: `"size": "512x512"`
- Use FP16: `DTYPE=fp16 ./run_local.sh`
- Check device memory: `xpu-smi stats -d 0`

### Slow First Request

The first request is slower because:
- Model weights are loaded to XPU on first call (if not pre-loaded)
- With `--enable-torch-compile`, the first request triggers compilation (~1-2 minutes)

Subsequent requests are significantly faster.

### Frontend Cannot Find Model

If you see `model_not_found` in the response, check that:
1. The worker has finished loading (check `.runtime/logs/worker.log` for "SDXL pipeline ready")
2. The `model` field in your request exactly matches the model the worker registered
3. Both frontend and worker use the same discovery directory (`DYN_FILE_KV`)

### Check Logs

```bash
# Worker log
tail -f examples/diffusers/xpu/.runtime/logs/worker.log

# Frontend log
tail -f examples/diffusers/xpu/.runtime/logs/frontend.log
```

## Architecture

```
Client (curl / Python)
    │
    │  POST /v1/images/generations
    ▼
┌──────────────────┐
│  Dynamo Frontend  │  (Rust HTTP server, port 8000)
│  dynamo.frontend  │  Routes to registered image backend
└────────┬─────────┘
         │ Dynamo RPC (file-based discovery)
         ▼
┌──────────────────┐
│  SDXL Worker     │  (worker_xpu.py)
│  Intel XPU       │  StableDiffusionXLPipeline → base64 PNG
└──────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `worker_xpu.py` | SDXL image generation worker with Dynamo endpoint |
| `run_local.sh` | Launch script (starts frontend + worker) |
| `README.md` | This file |
