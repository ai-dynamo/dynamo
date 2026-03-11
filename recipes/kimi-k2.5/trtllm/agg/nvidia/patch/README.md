# Kimi K2.5 TensorRT-LLM Patch

Kimi K2.5 support has not yet been released in TensorRT-LLM ([tracking branch](https://github.com/NVIDIA/TensorRT-LLM/compare/main...feat/k25-demo)).

This directory contains patches that register `KimiK25ForConditionalGeneration` on top of the existing DeepSeek-V3 model code and enable `attention_dp` to work with KVBM, letting you run Kimi K2.5 on TensorRT-LLM today.

## Quick start

Patch a Dynamo docker image by running:

```bash
./patch-container.sh <docker-image>
```

For example:

```bash
./patch-container.sh nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag
# produces image:    nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:my-tag-patched
```

The kimi patch is idempotent -- if `KimiK25ForConditionalGeneration` is already registered, it is skipped.

## Files

| File | Description |
|------|-------------|
| `patch-container.sh` | Builds a patched docker image from a base Dynamo image |
| `kimi.patch` | Appended to `modeling_deepseekv3.py` inside the container -- adds a thin `DeepseekV3ForCausalLM` subclass that extracts the Kimi text backbone config and remaps weight prefixes |
| `attention-dp.patch` | Git patch applied to `tensorrt_llm` -- enables `attention_dp` to work with KVBM in `kv_cache_connector.py` and `py_executor_creator.py` |
