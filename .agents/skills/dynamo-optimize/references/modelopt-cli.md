# Model Optimizer CLI Reference

Authoritative source: NVIDIA TensorRT Model Optimizer
(`https://github.com/NVIDIA/TensorRT-Model-Optimizer`). The commands
below are stable as of `nvidia-modelopt >= 0.20.0`; verify against the
release the user has installed.

---

## Install

```bash
pip install nvidia-modelopt
```

Pin to the version the target Dynamo release was tested against. The
1.2.0 line tested against the modelopt version pulled by the recipe
under `recipes/<model>/<framework>/<config>/`. Read the recipe's
`requirements.txt` or `Dockerfile.dsv4.*` to find the pin.

---

## Weight-Only FP8

Fastest quantization. No calibration data.

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode fp8 \
  --quant-method per-tensor \
  --output <output-dir> \
  --device cuda
```

Output dir contains:
- `model-*.safetensors` (quantized weights)
- `config.json` (model config)
- `quant_config.json` (consumed by Dynamo backends)
- `tokenizer.json`, `tokenizer_config.json` (copied from source)

---

## FP8 PTQ (with calibration)

Production-grade FP8. Needs calibration data.

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode fp8 \
  --quant-method per-tensor \
  --calib-dataset <path-to-jsonl-or-hf-dataset> \
  --calib-samples 512 \
  --output <output-dir> \
  --device cuda
```

Calibration dataset format: JSONL with one record per line containing a
`text` field, OR an HuggingFace dataset ID with a `text` column.

---

## NVFP4 (Blackwell)

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode nvfp4 \
  --calib-dataset <path> \
  --calib-samples 256 \
  --output <output-dir> \
  --device cuda
```

NVFP4 requires Blackwell-class hardware for **quantization**: the
calibration pass uses NVFP4 tensor cores. Quantization on H100 will
fail at runtime.

---

## INT8 Weight-Only

Pre-Hopper option. No calibration.

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode int8 \
  --quant-method weight-only \
  --output <output-dir> \
  --device cuda
```

---

## AWQ

Activation-aware. Slow but highest accuracy preservation for 4-bit weights.

```bash
modelopt quantize \
  --model <hf-id-or-path> \
  --quant-mode awq \
  --quant-method int4 \
  --calib-dataset <path> \
  --calib-samples 128 \
  --output <output-dir> \
  --device cuda
```

---

## `quant_config.json` Schema

The output `quant_config.json` is what Dynamo backends read. Minimal schema:

```json
{
  "quant_algo": "FP8",
  "kv_cache_quant_algo": null,
  "group_size": null,
  "smoothquant_val": null,
  "has_zero_point": false,
  "pre_quant_scale": null,
  "exclude_modules": ["lm_head"]
}
```

| Field | Meaning |
|---|---|
| `quant_algo` | The technique applied to weights (`FP8`, `NVFP4`, `INT8`, `W4A16`, etc.) |
| `kv_cache_quant_algo` | Optional KV-cache quantization (typically `null` or `FP8`) |
| `group_size` | For grouped quantization techniques (AWQ, GPTQ) |
| `smoothquant_val` | For SmoothQuant variants |
| `exclude_modules` | Layers left in higher precision; typically `lm_head` to preserve accuracy |

The Dynamo worker's `--load-format` flag (vLLM) and `quant_config.json`
discovery (TensorRT-LLM) read this file to select the correct kernels at
load time.

---

## Hand-Off to dynamo-deploy

After quantization, the optimized checkpoint dir replaces the original
model in the deployment manifest:

```yaml
# DGD pattern
spec:
  services:
    VllmDecodeWorker:
      extraPodSpec:
        mainContainer:
          args:
            - --model
            - /opt/optimized-checkpoints/<model>-fp8   # mounted from a PVC
            - --load-format
            - auto
```

For DGDR planning, pass the optimized path as `spec.model`. The
profiler will use the optimized weights to measure the actual
TTFT/ITL the deployment will see.
