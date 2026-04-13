# Task Plan: Diffusers on Intel XPU Example for Dynamo

## Goal
Create a complete, tested example at `examples/diffusers/xpu/` that lets developers run Stable Diffusion XL on Intel XPU using Dynamo's distributed runtime.

## Phases

- [x] Phase 1: Research — Understand Dynamo image endpoint, XPU PyTorch status
- [x] Phase 2: Design — Architecture decisions, file layout, API contract
- [x] Phase 3: Implement worker_xpu.py
- [x] Phase 4: Implement run_local.sh
- [x] Phase 5: Write README.md
- [ ] Phase 6: Test on XPU server (B60)
- [ ] Phase 7: Fix issues found during testing
- [ ] Phase 8: PR submission

## Status
**Phase 2 complete** — All 8 design decisions documented in notes.md. Ready for implementation.

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

## Errors / Blockers
(none yet)
