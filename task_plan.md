# Task Plan: Diffusers on Intel XPU Example for Dynamo

## Goal
Create a complete, tested example at `examples/diffusers/xpu/` that lets developers run Stable Diffusion XL on Intel XPU using Dynamo's distributed runtime.

## Phases

- [x] Phase 1: Research — Understand Dynamo image endpoint, XPU PyTorch status
- [x] Phase 2: Design — Architecture decisions, file layout, API contract
- [x] Phase 3: Implement worker_xpu.py
- [x] Phase 4: Implement run_local.sh
- [x] Phase 5: Write README.md
- [x] Phase 6: Test on XPU server (B60)
- [x] Phase 7: Fix issues found during testing
- [x] Phase 8: PR submission

## Status
**All phases complete.** Commits `857ed8749` (initial example) and `39e8c9128` (testing fixes) on `diffusers_exmple_on_xpu`, pushed to remote.

## Key Findings from Phase 1

### Dynamo Frontend — Image Support EXISTS
- `/v1/images/generations` endpoint is fully implemented in Rust frontend
- `ModelType.Images` (bitflag 1<<5) is available in Python bindings
- `register_llm(ModelInput.Text, ModelType.Images, endpoint, model_path, model_name)` will work
- Request protocol: `NvCreateImageRequest` in `components/src/dynamo/common/protocols/image_protocol.py`
- Response protocol: `NvImagesResponse` with `ImageData` objects (supports `b64_json` and `url`)
- `nvext` field supports: `num_inference_steps`, `guidance_scale`, `seed`, `negative_prompt`

### PyTorch XPU — Ready for Production
- PyTorch 2.11.0 stable has native XPU support — pip index: `https://download.pytorch.org/whl/xpu`
- IPEX is **archived** (March 2026) — no longer needed, features upstreamed to PyTorch
- `torch.compile` works on XPU via Inductor backend (Triton + SYCL)
- Both FP16 and BF16 supported on B60 hardware — BF16 recommended for stability

### HuggingFace Diffusers — XPU Compatible
- `StableDiffusionXLPipeline.to("xpu")` works
- Active XPU CI in diffusers repo with ongoing fixes
- Some tolerance differences vs CUDA, but functionally correct

### Intel GPU Drivers
- Linux kernel 6.11+ with `force_probe` for Battlemage
- Packages: `intel-i915-dkms`, `level-zero`, `intel-level-zero-gpu`, `intel-opencl-icd`
- LTS 2523.x driver stream recommended

## Files to Create
1. `examples/diffusers/xpu/worker_xpu.py` — SDXL worker with Dynamo endpoint
2. `examples/diffusers/xpu/run_local.sh` — Local launch script (frontend + worker)
3. `examples/diffusers/xpu/README.md` — Full instructions

## Open Questions (all resolved)
- [x] Does `NvCreateImageRequest` map cleanly to diffusers generate() params? **YES** — see Decision 4 in notes.md
- [x] What is the exact request/response JSON shape the frontend sends/expects? **YES** — `NvCreateImageRequest` -> worker -> `NvImagesResponse`, protocol in `image_protocol.py`
- [x] Can we reuse the existing frontend module (`dynamo.frontend`) as-is for images? **YES** — `ModelType.Images` auto-enables `/v1/images/generations` route

## Phase 6 Testing Results (2026-04-13)

### Environment (8x Intel Arc Pro B60)
- PyTorch 2.11.0+xpu, diffusers 0.37.1, ai-dynamo 1.0.1 (PyPI)
- User must be in `render` group for GPU access
- `xpu-smi discovery` shows "No device discovered" but PyTorch detects all 8 GPUs fine

### Unit Test (no Dynamo)
- SDXL pipeline loads in bf16, generates 512x512 in 4.8s (5 steps). No issues.

### Integration Test (full Dynamo stack)
- Worker registers, frontend discovers model via file-based discovery
- 1024x1024 30-step: ~10.2s first request, ~8.4s warmed up (~17 it/s)
- 512x512 10-step: 1.9s
- n=2 multiple images: works
- Error handling: frontend returns 400 for invalid size / missing prompt

### Key Finding
- Rust frontend validates `size` as enum (`256x256`, `512x512`, `1024x1024`, `1792x1024`, `1024x1792`), not arbitrary WxH

## Phase 7 Fixes Applied (commit `39e8c9128`)

- README: added `render` group requirement (setup + troubleshooting)
- README: changed Dynamo install to `pip install ai-dynamo` (no Rust toolchain needed)
- README: fixed `size` parameter docs — enum values instead of arbitrary WxH
- README: added `xpu-smi` caveat, reordered troubleshooting steps
- worker_xpu.py: updated size docstring to list valid enum values

## Errors / Blockers
- Port 8000 was in use on test server — used 8001; `run_local.sh` supports `HTTP_PORT` override
- `xpu-smi` not detecting devices — cosmetic issue, PyTorch works fine
