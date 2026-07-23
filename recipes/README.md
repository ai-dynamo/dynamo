<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Recipes

Kubernetes deployment and benchmark recipes for serving models with NVIDIA Dynamo.

> [!NOTE]
> Install the [Dynamo Kubernetes Platform](https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/README.md)
> before using these recipes.

## Available Recipes

`Recipe date` is the initial publication date or the date of the latest substantial configuration
or topology change. Version bumps and maintenance-only edits do not change it.

| Recipe date | Model | Framework | Configuration | GPUs | Features |
|---|---|---|---|---|---|
| 2026-07-21 | [DeepSeek-V4-Flash](deepseek-v4/deepseek-v4-flash/) | vLLM, SGLang | Aggregated, disaggregated | 4–28× B200, H200, or GB200 | NVFP4 or FP8, KV-aware routing, MTP, reasoning, tool calling |
| 2026-07-21 | [DeepSeek-V4-Pro](deepseek-v4/deepseek-v4-pro/) | vLLM, SGLang | Aggregated, disaggregated | 8–32× B200, H200, or GB200 | NVFP4 or FP8, KV-aware routing, MTP, reasoning, tool calling |
| 2026-07-20 | [GLM-5.2](glm-5.2/) | SGLang | Aggregated, disaggregated | 16 or 20× B200; 16 or 24× H200 | NVFP4 or FP8, KV-aware routing, MTP, HiCache CPU offload |
| 2026-07-20 | [GPT-OSS-120B](gpt-oss-120b/) | vLLM, TensorRT-LLM | Aggregated, disaggregated | 4–8× B200, H200, or GB200 | MXFP4, FP8 KV cache, KV-aware routing, EAGLE3, reasoning, tool calling |
| 2026-07-16 | [TML Inkling](inkling/) | SGLang | Aggregated | 8× B200 | NVFP4, EAGLE, text/image/audio input, reasoning, tool calling |
| 2026-06-03 | [Kimi-K2.6](kimi-k2.6/) | vLLM | Aggregated chat and agentic | 4× B200 or 8× H200 | NVFP4 or INT4, EAGLE3, LMCache CPU offload, text/image input |
| 2026-04-28 | [Nemotron-3-Nano-Omni](nemotron-3-nano-omni/) | vLLM | Aggregated | 1× NVIDIA GPU | NVFP4, text/image/video/audio input, custom container |
| 2026-06-03 | [Nemotron-3-Super](nemotron-3-super/) | vLLM | Aggregated chat and agentic | 4× B200 or H200 | NVFP4 or FP8, KV-aware routing, MTP, reasoning, tool calling |
| 2026-03-11 | [Nemotron-3-Super-FP8](nemotron-3-super-fp8/) | vLLM, SGLang, TensorRT-LLM | Aggregated, disaggregated | 4× H100 or H200 | FP8, KV-aware routing, NIXL or UCX KV transfer |
| 2026-06-04 | [Nemotron-3-Ultra](nemotron-3-ultra/) | vLLM | Aggregated, disaggregated | 4–8× B200 or H200 | NVFP4, FP8 KV cache, KV-aware routing, MTP, reasoning, tool calling |
| 2026-04-29 | [Qwen3-235B-A22B-FP8](qwen3-235b-a22b-fp8/) | TensorRT-LLM | Aggregated, disaggregated | 16× H100, H200, B100, or B200 | TP4/EP4, KV-aware routing, DEEPGEMM on Blackwell |
| 2026-03-10 | [Qwen3-VL-30B-A3B-FP8](qwen3-vl-30b/) | vLLM | Aggregated | 1× GB200 | Multimodal embedding cache and benchmark comparison |
| 2026-06-26 | [Qwen3-VL-32B-Instruct-FP8](qwen3-vl-32b-fp8/) | vLLM | Aggregated, heterogeneous disaggregated | 1× H100/H200 or 1× Intel XPU + 1× NVIDIA GPU | Multimodal serving, FP8 KV cache, RDMA embedding transfer |

## Use a Recipe

1. Open the model README and choose a configuration for your hardware.
2. Create the namespace, storage, and model registry secrets described by the recipe.
3. Apply the model-cache and deployment manifests.
4. If the recipe includes a `perf.yaml`, apply it to run the benchmark.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the recipe directory structure and validation requirements.
