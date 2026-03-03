---

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

## title: KV Cache Offloading
subtitle: CPU and disk offloading integrations for vLLM in Dynamo

# KV Cache Offloading

Dynamo supports multiple KV cache offloading backends for vLLM, allowing you to extend effective KV cache capacity beyond GPU memory using CPU RAM and disk storage. Each backend integrates through vLLM's connector interface and works with both aggregated and disaggregated serving.


| Backend                 | Source                                           |
| ----------------------- | ------------------------------------------------ |
| **[KVBM](#kvbm)**       | [Dynamo](../../components/kvbm/README.md)        |
| **[LMCache](#lmcache)** | [GitHub](https://github.com/LMCache/LMCache)     |
| **[FlexKV](#flexkv)**   | [GitHub](https://github.com/taco-project/FlexKV) |


## KVBM

[KVBM](../../components/kvbm/README.md) (KV Block Manager) is Dynamo's built-in KV cache offloading system. It provides a three-layer architecture (LLM runtime, logical block management, NIXL transport) with support for CPU and disk cache tiers, and integrates natively with Dynamo's KV-aware routing and disaggregated serving.


| Deployment                 | Launch Script                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------- |
| Aggregated                 | `[agg_kvbm.sh](../../../examples/backends/vllm/launch/agg_kvbm.sh)`                     |
| Aggregated + KV routing    | `[agg_kvbm_router.sh](../../../examples/backends/vllm/launch/agg_kvbm_router.sh)`       |
| Disaggregated (1P1D)       | `[disagg_kvbm.sh](../../../examples/backends/vllm/launch/disagg_kvbm.sh)`               |
| Disaggregated (2P2D)       | `[disagg_kvbm_2p2d.sh](../../../examples/backends/vllm/launch/disagg_kvbm_2p2d.sh)`     |
| Disaggregated + KV routing | `[disagg_kvbm_router.sh](../../../examples/backends/vllm/launch/disagg_kvbm_router.sh)` |


For configuration details, see the [KVBM Guide](../../components/kvbm/kvbm-guide.md).

## LMCache

[LMCache](https://github.com/LMCache/LMCache) is an open-source KV cache engine that provides prefill-once, reuse-everywhere caching with multi-level storage backends (CPU RAM, local storage, Redis, GDS, InfiniStore/Mooncake).


| Deployment                        | Launch Script                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| Aggregated                        | `[agg_lmcache.sh](../../../examples/backends/vllm/launch/agg_lmcache.sh)`                     |
| Aggregated (multiprocess metrics) | `[agg_lmcache_multiproc.sh](../../../examples/backends/vllm/launch/agg_lmcache_multiproc.sh)` |
| Disaggregated                     | `[disagg_lmcache.sh](../../../examples/backends/vllm/launch/disagg_lmcache.sh)`               |


For configuration details, see the [LMCache Integration Guide](../../integrations/lmcache-integration.md).

## FlexKV

[FlexKV](https://github.com/taco-project/FlexKV) is a scalable, distributed KV cache runtime developed by Tencent Cloud's TACO team. It supports multi-level caching (GPU, CPU, SSD), distributed KV cache reuse across nodes, and high-performance I/O via io_uring and GPUDirect Storage.


| Deployment              | Launch Script                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Aggregated              | `[agg_flexkv.sh](../../../examples/backends/vllm/launch/agg_flexkv.sh)`               |
| Aggregated + KV routing | `[agg_flexkv_router.sh](../../../examples/backends/vllm/launch/agg_flexkv_router.sh)` |
| Disaggregated           | `[disagg_flexkv.sh](../../../examples/backends/vllm/launch/disagg_flexkv.sh)`         |


For configuration details, see the [FlexKV Integration Guide](../../integrations/flexkv-integration.md).

## See Also

- **[KVBM Design](../../design-docs/kvbm-design.md)**: Architecture and design of Dynamo's built-in KV cache offloading
- **[KV-Aware Routing](../../components/router/router-guide.md)**: Routing requests based on KV cache state
- **[Disaggregated Serving](../../design-docs/disagg-serving.md)**: Prefill/decode separation architecture

