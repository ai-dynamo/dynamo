---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Encoder Disaggregation
subtitle: Scale vision encoders separately from LLM workers
---

![In aggregated serving, one worker does media decode, preprocessing, the vision encoder, prefill, and decode. In EPD serving, a prefill worker checks the embedding cache first and calls encoder workers over NIXL on a miss, then runs LLM prefill and sends the KV cache over NIXL to a decode worker.](../../../../assets/img/encoder-disaggregation.svg)

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

For simple deployments or development and testing, the aggregated pattern (EPD, one worker) is easier to set up.

## Deployment Patterns

<CardGroup cols={2}>
  <Card title="E/PD">
    **Separate encoder, combined prefill and decode**

    <Badge intent="info" minimal>Frontend</Badge> → <Badge intent="info" minimal>Processor</Badge> → <Badge intent="info" minimal>Encode Worker</Badge> → <Badge intent="info" minimal>PD Worker</Badge> → <Badge intent="info" minimal>Response</Badge>

    The encode worker runs the vision model and transfers embeddings over NIXL to the combined prefill and decode worker.
  </Card>
  <Card title="E/P/D">
    **Separate encoder, prefill, and decode**

    <Badge intent="info" minimal>Frontend</Badge> → <Badge intent="info" minimal>Processor</Badge> → <Badge intent="info" minimal>Encode Worker</Badge> → <Badge intent="info" minimal>Prefill Worker</Badge> → <Badge intent="info" minimal>Decode Worker</Badge> → <Badge intent="info" minimal>Response</Badge>

    The encode worker transfers embeddings over NIXL to the prefill worker. The prefill worker then transfers the KV cache to the decode worker.
  </Card>
</CardGroup>

## Placement Options

Where the encoder runs is a separate choice from whether it is disaggregated:

- **Aggregated:** every GPU runs one worker that encodes, prefills, and decodes. This is the simplest setup.
- **Colocated encoder:** every GPU runs encoder workers and a prefill and decode (PD) worker side by side, sharing the GPU through CUDA MPS. Encoding no longer queues behind prefill on the same scheduler, and you need no extra hardware.
- **Disaggregated encoder:** encoders run on their own GPUs, which can be lower-tier hardware, and send embeddings to the PD workers over NIXL.

![Three topologies. Aggregated serving runs an aggregated worker on each of four identical GPUs. Colocated encoder runs encoder workers and a PD worker on each of four identical GPUs using MPS. Disaggregated encoder runs encoder workers on two lower-tier GPUs and sends embeddings over NIXL to PD workers on four other GPUs.](../../../../assets/img/encoder-placement-options.svg)

## Launching

<Tabs>
  <Tab title="vLLM" language="vllm">
    ```bash
    cd $DYNAMO_HOME/examples/backends/vllm

    # E/PD
    bash launch/disagg_multimodal_e_pd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

    # E/P/D
    bash launch/disagg_multimodal_epd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    ```
  </Tab>
  <Tab title="SGLang" language="sglang">
    ```bash
    cd $DYNAMO_HOME/examples/backends/sglang

    # E/PD
    ./launch/multimodal_epd.sh

    # E/P/D
    ./launch/multimodal_disagg.sh
    ```
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    ```bash
    cd $DYNAMO_HOME/examples/backends/trtllm

    # E/PD
    bash launch/disagg_e_pd.sh

    # E/P/D
    ./launch/epd_multimodal_image_and_embeddings.sh
    ```
  </Tab>
</Tabs>

See the backend-specific documentation ([vLLM](../../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md), [TRT-LLM](../../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md), [SGLang](../../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md)) for full configuration details and component flags.

## Support Matrix

| Backend | E/PD | E/P/D | Notes |
|---------|------|-------|-------|
| **vLLM** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | Separate encode worker currently handles `image_url` inputs; `video_url` inputs stay on the prefill/PD path |
| **TensorRT-LLM** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | Supports image URLs (via `MultimodalEncoder`) and pre-computed embeddings (via NIXL) |
| **SGLang** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | NIXL for embeddings; bootstrap mechanism for P/D KV transfer |
