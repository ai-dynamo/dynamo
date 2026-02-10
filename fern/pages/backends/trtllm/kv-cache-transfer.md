---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# KV Cache Transfer in Disaggregated Serving

In disaggregated serving architectures, KV cache must be transferred between prefill and decode workers. TensorRT-LLM supports two methods for this transfer:

## Using NIXL for KV Cache Transfer

Start the disaggregated service: See [Disaggregated Serving](./README.md#disaggregated) to learn how to start the deployment.

## Default Method: NIXL
By default, TensorRT-LLM uses **NIXL** (NVIDIA Inference Xfer Library) with UCX (Unified Communication X) as backend for KV cache transfer between prefill and decode workers. [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's high-performance communication library designed for efficient data transfer in distributed GPU environments.

### Specify Backends for NIXL

TensorRT-LLM supports two NIXL communication backends: UCX and LIBFABRIC. By default, UCX is used if no backend is explicitly specified. Dynamo currently only supports the UCX backend, as LIBFABRIC support is still a work in progress. Please do not change the NIXL backend in the Dynamo runtime image.

## Alternative Method: UCX

TensorRT-LLM can also leverage **UCX** (Unified Communication X) directly for KV cache transfer between prefill and decode workers. To enable UCX as the KV cache transfer backend, set `cache_transceiver_config.backend: UCX` in your engine configuration YAML file.

> [!Note]
> The environment variable `TRTLLM_USE_UCX_KVCACHE=1` with `cache_transceiver_config.backend: DEFAULT` does not enable UCX. You must explicitly set `backend: UCX` in the configuration.
