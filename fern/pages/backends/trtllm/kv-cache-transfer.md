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

NIXL supports multiple communication backends that can be configured via environment variables. By default, UCX is used if no backends are explicitly specified.

**Environment Variable Format:**
```bash
DYN_KVBM_NIXL_BACKEND_<BACKEND>=<value>
```

**Supported Backends:**
- `UCX` - Unified Communication X (default)
- `GDS` - GPU Direct Storage

**Examples:**
```bash
# Enable UCX backend (default behavior)
export DYN_KVBM_NIXL_BACKEND_UCX=true

# Enable GDS backend
export DYN_KVBM_NIXL_BACKEND_GDS=true

# Enable multiple backends
export DYN_KVBM_NIXL_BACKEND_UCX=true
export DYN_KVBM_NIXL_BACKEND_GDS=true

# Explicitly disable a backend
export DYN_KVBM_NIXL_BACKEND_GDS=false
```

**Valid Values:**
- `true`, `1`, `on`, `yes` - Enable the backend
- `false`, `0`, `off`, `no` - Disable the backend

> [!NOTE]
> If no `DYN_KVBM_NIXL_BACKEND_*` environment variables are set, UCX is used as the default backend.

## Alternative Method: UCX

TensorRT-LLM can also leverage **UCX** (Unified Communication X) directly for KV cache transfer between prefill and decode workers. To enable UCX as the KV cache transfer backend, set `cache_transceiver_config.backend: UCX` in your engine configuration YAML file.

> [!NOTE]
> The environment variable `TRTLLM_USE_UCX_KV_CACHE=1` with `cache_transceiver_config.backend: DEFAULT` does not enable UCX. You must explicitly set `backend: UCX` in the configuration.
