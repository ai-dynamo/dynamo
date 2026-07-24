---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal KV Routing
subtitle: Route multimodal requests to workers with the best KV cache overlap
---

## Overview

Multimodal KV routing extends Dynamo's KV-aware router to account for image content when calculating cache overlap. The frontend assigns each image a stable hash and includes its identity in the routing view of the prompt.

When an image appears again, the KV router sends the request to the worker with the most matching KV cache blocks. This increases prefix-cache reuse and avoids repeating prefill work for cached multimodal content.

<Info>
The KV cache stores attention key/value state so a worker can skip repeated prefill work. The embedding cache stores vision encoder outputs so the encoder can skip repeated image processing. You can use both features together. See [Embedding Cache](embedding-cache.md).
</Info>

## When to Use

Use multimodal KV routing when:

- Multiple backend workers serve multimodal requests.
- Images repeat across requests, such as product photos or shared reference images.
- You want to maximize KV cache reuse for multimodal content.

Single-worker deployments do not need routing, and workloads with entirely unique images receive little image-specific cache benefit.

## How It Works

The routing flow in general has three steps:

1. The frontend computes a stable identity for each image.
2. The frontend represents the image in a routing-only token view that matches the backend's cache identity.
3. The KV router selects the worker with the highest block overlap and forwards the same image identity to that worker.

<Tabs>
  <Tab title="vLLM" language="vllm">
    vLLM provides two routing paths. Use the default Rust frontend for supported model families when you want minimal frontend processing. Use the Python chat processor when you need vLLM's broader model support or want the frontend to preprocess images and transfer the processed inputs to workers.

    **Default Rust frontend**

    The frontend calculates only the image identity and routing token layout. The selected worker still runs the model's multimodal processor. This path has lower frontend overhead, but multimodal routing depends on the model being registered with Dynamo's Rust processor registry.

    **Alternative: Python chat processor**

    With `--dyn-chat-processor vllm`, the frontend runs vLLM's full multimodal processor. It supports models known to vLLM without requiring a Dynamo Rust processor specification and can transfer processed inputs through shared memory or NIXL. This shifts preprocessing and transfer work to the frontend.

  </Tab>
  <Tab title="SGLang" language="sglang">
    The frontend hashes each image and converts that hash into the same `pad_value` token that SGLang RadixAttention uses for its prefix-cache key. The matching token view lets the router measure image overlap before selecting a worker.

    Dynamo's SGLang image includes the required hash-forwarding support.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    The frontend hashes each image, represents that identity in its routing token view, and forwards the hash as `multi_modal_uuids`. TensorRT-LLM workers publish matching KV events so the router can identify cached image blocks.

  </Tab>
</Tabs>

## Launch

<Tabs>
  <Tab title="vLLM" language="vllm">
    Use the Python chat processor when you need vLLM's model-native multimodal processor or want to transfer processed multimodal inputs from the frontend. Otherwise, use the default Rust frontend.

    **Default Rust frontend**

    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/vllm/launch/agg_multimodal_router.sh
    ```

    **Alternative: Python chat processor**

    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/vllm/launch/agg_multimodal_router_chat_processor.sh
    ```

    See [vLLM Multimodal](../../knowledge-base/modular-components/backends/v-llm/vllm-multimodal#multimodal-kv-routing) for model support, hashing behavior, transfer modes, and configuration.
  </Tab>
  <Tab title="SGLang" language="sglang">
    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/sglang/launch/agg_multimodal_router.sh
    ```

    The launcher configures KV events and matching frontend and worker block sizes.

    See [SGLang Multimodal](../../knowledge-base/modular-components/backends/sg-lang/sglang-multimodal#multimodal-kv-routing) for prerequisites, configuration, fallback behavior, and verification.
  </Tab>
  <Tab title="TensorRT-LLM" language="trtllm">
    ```bash
    cd $DYNAMO_HOME
    bash examples/backends/trtllm/launch/agg_multimodal_router.sh
    ```

    The launcher enables multimodal serving, KV event publishing, block reuse, and KV-aware routing.

    See [TensorRT-LLM Multimodal](../../knowledge-base/modular-components/backends/tensor-rt-llm/tensorrt-llm-multimodal#multimodal-kv-routing) for worker requirements, supported models, and limitations.
  </Tab>
</Tabs>

## Support Matrix

| Backend | Routing Path | Status | Notes |
|---------|--------------|--------|-------|
| [vLLM](../../knowledge-base/modular-components/backends/v-llm/vllm-multimodal#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Supported families include Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA 1.5, LLaVA-NeXT, Llama 4, Kimi K2.5/K2.6, Qwen3.5, and Qwen3.6. The rest use text-prefix-only routing. |
| [vLLM](../../knowledge-base/modular-components/backends/v-llm/vllm-multimodal#multimodal-kv-routing) | Python chat processor | <Badge intent="success" minimal>Yes</Badge> | Uses vLLM’s own multimodal processor — supports any VLM that vLLM supports. |
| [SGLang](../../knowledge-base/modular-components/backends/sg-lang/sglang-multimodal#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Dynamo's SGLang image includes hash-forwarding support; custom installations must add it. |
| [TensorRT-LLM](../../knowledge-base/modular-components/backends/tensor-rt-llm/tensorrt-llm-multimodal#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Supported model scope is the Qwen2-VL family (Qwen2-VL / Qwen2.5-VL / Qwen3-VL) and Kimi (Kimi-K2.5 / Kimi-K2.6). Other multimodal models fall back to text-prefix routing. |
