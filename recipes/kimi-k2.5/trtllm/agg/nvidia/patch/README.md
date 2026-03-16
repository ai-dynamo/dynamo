# Kimi K2.5 TensorRT-LLM Patch

Kimi K2.5 support has not yet been released in TensorRT-LLM ([tracking PR](https://github.com/NVIDIA/TensorRT-LLM/pull/11816)).

This directory contains a unified diff that registers `KimiK25ForConditionalGeneration` on top of the existing DeepSeek-V3 model code, letting you run Kimi K2.5 on TensorRT-LLM today.

## Quick start

Build a patched image:

```bash
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.0 \
  -t nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.0-patched \
  recipes/kimi-k2.5/trtllm/agg/nvidia/patch/
```

The patch is applied via `patch -p1 --fuzz=0`:
- If the target file has changed upstream, the build **fails loudly** instead of silently producing broken code.
- If the patch is already applied, it is skipped (idempotent).
- A smoke test verifies the class is registered before the build completes.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Single-stage build that applies the patch to a base Dynamo image |
| `kimi.patch` | Unified diff from [upstream PR #11816](https://github.com/NVIDIA/TensorRT-LLM/pull/11816) — adds `KimiK25ForConditionalGeneration` to `modeling_deepseekv3.py` |
