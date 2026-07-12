<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TRT-LLM OpenEngine sidecar launches

These scripts run the TRT-LLM HTTP server and its optional OpenEngine sibling
against the same `LLM`, then register a CPU-only `dynamo-openengine-sidecar`
worker with Dynamo.

The examples use the OpenEngine schema release
`cea19cb06acf03c911b84d5c147e519b60dd92a6`. Install the matching local
OpenEngine Python package in the TRT-LLM environment and build the Rust sidecar
against the same local checkout before launching them.

| Script | Model/default | GPUs | Coverage |
| --- | --- | ---: | --- |
| `openengine_agg.sh` | `Qwen/Qwen3-0.6B` | 1 | Aggregate text |
| `openengine_disagg.sh` | `Qwen/Qwen3-0.6B` | 2 | Context-first 1P1D text and LoRA |
| `openengine_multimodal.sh` | `Qwen/Qwen3-VL-2B-Instruct` | 1 | Aggregate image and video |
| `openengine_multimodal_disagg.sh` | `Qwen/Qwen3-VL-2B-Instruct` | 2 | Image/video context-first 1P1D |
| `openengine_audio.sh` | required raw Phi-4-MM clone | 1 | Aggregate audio with bundled `speech-lora` |
| `openengine_lora.sh` | `Qwen/Qwen3-0.6B` | 1 | Aggregate lazy LoRA lifecycle |

The one-node scripts reserve frontend HTTP `8000`, TRT-LLM HTTP `8001`,
OpenEngine gRPC `50051`, and the Dynamo worker system port `8081`. The 1P1D
scripts add TRT-LLM HTTP `8002`, OpenEngine gRPC `50052`, and system port
`8082`. Every value has an environment-variable override in the script.

The sidecar intentionally advertises no frontend media decoder. As a result,
the frontend keeps ordered `http(s)` URLs and `data:` URIs intact and the local
TRT-LLM process owns media fetch/decode. Decoded/RDMA media is rejected rather
than silently copied.

For LoRA, set `DYN_LORA_PATH` to a directory visible at the same absolute path
to the sidecar and TRT-LLM. The provided local scripts share one filesystem;
the Kubernetes examples use a shared PVC. The Qwen configs bound the GPU and
CPU adapter caches to two entries. Phi-4's checkpoint must retain its bundled
`speech-lora/` directory. The TRT-LLM Phi-4-MM loader does not support the
Hugging Face snapshot layout, so `openengine_audio.sh` requires `MODEL_PATH` to
point to a raw git clone of the model repository.
