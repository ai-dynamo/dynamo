---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Encoder Disaggregation
subtitle: Separate vision encoding into a dedicated worker for independent scaling
---

## Overview

Encoder disaggregation separates the vision encoder from the prefill/decode pipeline into its own worker. Instead of running image encoding inline, a dedicated encode worker handles media processing and transfers the resulting embeddings to downstream workers via NIXL (RDMA).

This enables:

- Independent scaling of encode workers based on vision workload
- Reduced GPU memory pressure on prefill/decode workers
- Better GPU utilization by matching worker counts to actual bottlenecks

## When to Use

Use encoder disaggregation when:

- Vision encoding is a bottleneck and you need to scale encoders independently of LLM workers
- You want to run the vision encoder on different hardware (e.g., smaller GPUs for encoding, larger GPUs for LLM inference)
- Your deployment handles high volumes of multimodal requests and encoding throughput is limiting

For simple deployments or development/testing, the aggregated (EPD) pattern is easier to set up.

## Support Matrix

| Backend | E/PD | E/P/D | Notes |
|---------|------|-------|-------|
| **vLLM** | ✅ | ✅ | Separate encode worker currently handles `image_url` inputs; `video_url` inputs stay on the prefill/PD path |
| **TRT-LLM** | ❌ | ✅ | Supports image URLs (via `MultimodalEncoder`) and pre-computed embeddings (via NIXL) |
| **SGLang** | ✅ | ✅ | NIXL for embeddings; bootstrap mechanism for P/D KV transfer |

## Deployment Patterns

**E/PD** — Separate encoder, combined prefill+decode:

```text
Frontend → Processor → Encode Worker → PD Worker → Response
                           (NIXL)
```

The encode worker runs the vision model and transfers embeddings via NIXL to a combined prefill+decode worker.

**E/P/D** — All stages separate:

```text
Frontend → Processor → Encode Worker → Prefill Worker → Decode Worker → Response
                           (NIXL)          (KV transfer)
```

Full disaggregation with separate workers for each stage. The encode worker transfers embeddings to the prefill worker, which then transfers KV cache to the decode worker.

## Launching

### vLLM

```bash
cd $DYNAMO_HOME/examples/backends/vllm

# E/PD
bash launch/disagg_multimodal_e_pd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

# E/P/D
bash launch/disagg_multimodal_epd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
```

### TRT-LLM

```bash
cd $DYNAMO_HOME/examples/backends/trtllm

# E/PD
bash launch/disagg_e_pd.sh

# E/P/D
./launch/epd_multimodal_image_and_embeddings.sh
```

### SGLang

```bash
cd $DYNAMO_HOME/examples/backends/sglang

# E/PD
./launch/multimodal_epd.sh

# E/P/D
./launch/multimodal_disagg.sh
```

See the backend-specific documentation ([vLLM](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md), [TRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-trtllm.md), [SGLang](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-sglang.md)) for full configuration details and component flags.
