# Optimization Known Issues

Stable issue patterns relevant to quantization with Model Optimizer.
Strict 6-element shape per.

---

### NVFP4 quantization fails on non-Blackwell hardware

**Symptom:** `modelopt quantize --quant-mode nvfp4 ...` runs but the calibration step fails with a CUDA kernel error or "unsupported tensor core" message.

**Root cause:** NVFP4 quantization (the calibration pass itself) uses Blackwell-tier FP4 tensor cores. Pre-Blackwell hardware (H100, A100) can run NVFP4-quantized models at lower precision via emulation, but cannot produce an NVFP4 checkpoint.

**Affected:** All modelopt versions that support NVFP4. Hardware: anything pre-Blackwell.

**Fix:** Either (a) provision a Blackwell host for the quantization step (the resulting checkpoint can deploy anywhere), or (b) switch to FP8 PTQ for Hopper-tier deployment.

**Verify:**

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
# Expect "B200", "GB200", or similar Blackwell-tier name for NVFP4 quantization.
```

Source: NVIDIA TensorRT Model Optimizer documentation; (container image conventions).

---

### Optimized checkpoint loads but produces garbage tokens

**Symptom:** Dynamo worker loads the optimized checkpoint without error; inference returns syntactically valid but semantically broken output.

**Root cause:** Most commonly, the calibration dataset distribution did not match the inference workload distribution. Less commonly, a layer was incorrectly excluded from quantization (or incorrectly included).

**Affected:** PTQ / AWQ techniques where calibration data is required.

**Fix:** Re-quantize with a calibration set drawn from the actual inference workload. Move `lm_head` to `exclude_modules` if it was quantized. Verify the source model's `config.json` was copied to the output dir unchanged (modelopt sometimes rewrites it).

**Verify:** Run a short accuracy probe with known-good prompts. If accuracy is still degraded, fall back to a higher-precision technique (FP8 PTQ instead of NVFP4, AWQ instead of weight-only).

Source: Model Optimizer best-practices documentation.

---

### `quant_config.json` not recognized by Dynamo backend

**Symptom:** Worker pod fails to start; logs show "unsupported quant_algo" or "quant_config.json schema mismatch".

**Root cause:** modelopt version produced a `quant_config.json` field that the target backend does not yet support. Backend pins drift faster than the schema; a newer modelopt + older backend is a common cause.

**Affected:** Any modelopt version pairing where the backend is older. Most often when the user pulls latest modelopt but the Dynamo release pins an older backend (e.g. `vllm==0.21.0` per the 1.2.0 line).

**Fix:** Pin modelopt to the version the recipe for this model uses. If no recipe exists, pin to the version released alongside the target backend pin.

**Verify:**

```bash
cat <output-dir>/quant_config.json
# Compare quant_algo field to the backend's supported list.
```

Source: (pin source files), (image registry conventions).
