# Recipe And Benchmark Prototype Tracker

This tracker records the current split between model-first Recipes and feature-first Benchmarks.

## Landing Pages

| Page | Status | Notes |
| --- | --- | --- |
| Recipes landing | Updated | Model-first catalog, no validation-status badges, links users to benchmarks for feature claims. |
| Benchmarks landing | Added | Feature-first index with simple A/B and consolidated benchmark classes. |
| Cross-linking | Updated | Recipe pages link to related benchmark pages; benchmark pages link to the promoted recipe target. |
| Inventory CSV | Updated | `recipe-benchmark-classification.csv` classifies every source row as recipe target, benchmark arm, or deploy-only/missing-perf asset. |

## Model Pages

| Model page | Status | Recipe treatment | Related benchmark |
| --- | --- | --- | --- |
| Qwen3-32B | Updated | 2 recipe target(s) | Qwen3-32B KV routing A/B |
| DeepSeek V3.2 NVFP4 | Updated | 1 recipe target(s) | DeepSeek V3.2 WideEP routing A/B |
| Qwen3-VL-30B | Updated | 1 recipe target(s) | Qwen3-VL embedding cache A/B |
| Kimi-K2.5 NVFP4 | Updated | 1 recipe target(s) | Kimi-K2.5 feature-stack benchmark |
| GLM-5 NVFP4 | Updated | 1 recipe target(s) | None yet |
| GPT-OSS-120B | Updated | 2 recipe target(s) | None yet |
| DeepSeek-R1 | Updated | 4 recipe target(s) | None yet |
| DeepSeek-V4-Pro | Updated | 7 recipe target(s) | None yet |
| DeepSeek-V4-Flash | Updated | 4 recipe target(s) | None yet |
| Llama-3.3-70B FP8 | Updated | 3 recipe target(s) | Llama-3.3-70B topology benchmark |
| Nemotron 3 Nano Omni | Updated | 1 recipe target(s) | None yet |
| Nemotron-3-Super FP8 | Updated | 4 recipe target(s) | None yet |
| Qwen3-235B-A22B FP8 | Updated | 4 recipe target(s) | None yet |
| Qwen3-32B FP8 | Updated | 3 recipe target(s) | None yet |
| Qwen3.6-35B-A3B FP8 | Updated | 1 recipe target(s) | Qwen3.6 frontend/cache benchmark |

## Benchmark Pages

| Benchmark page | Status | Class |
| --- | --- | --- |
| Qwen3-32B KV Routing A/B | Added | Simple A/B test |
| DeepSeek V3.2 WideEP Routing A/B | Added | Simple A/B test |
| Qwen3-VL Embedding Cache A/B | Added | Simple A/B test |
| Kimi-K2.5 Feature-Stack Benchmark | Added | Consolidated benchmark |
| Qwen3.6 Frontend and Cache Benchmark | Added | Consolidated benchmark |
| Llama-3.3-70B Topology Benchmark | Added | Simple topology benchmark |
