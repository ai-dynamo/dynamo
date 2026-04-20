<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM launch scripts

Shell scripts that bring up a Dynamo frontend plus vLLM worker(s) on a single machine, for learning and local development.

> See also: the authoritative guide for this backend at [`docs/backends/vllm/`](../../../../docs/backends/vllm/README.md).

## What's here

| Script | Shape |
|---|---|
| `agg.sh` | Aggregated single-node baseline. |
| `agg_router.sh` / `agg_router_replicas.sh` / `agg_router_approx.sh` | Aggregated with KV-aware router variants. |
| `agg_kvbm.sh` / `agg_kvbm_router.sh` | Aggregated with KVBM offloading. |
| `agg_flexkv.sh` / `agg_flexkv_router.sh` | Aggregated with FlexKV integration. |
| `agg_lmcache.sh` / `agg_lmcache_multiproc.sh` | Aggregated with LMCache integration. |
| `agg_spec_decoding.sh` | Aggregated with speculative decoding. |
| `agg_tracing.sh` | Aggregated with OTEL tracing enabled. |
| `agg_request_planes.sh` | Aggregated with split request planes. |
| `agg_multimodal.sh` / `agg_omni*.sh` | Aggregated multimodal (vision / audio / video) variants. |
| `disagg.sh` | Disaggregated prefill / decode baseline. |
| `disagg_router.sh` / `disagg_kvbm*.sh` / `disagg_flexkv.sh` / `disagg_lmcache.sh` | Disaggregated variants with router / KVBM / FlexKV / LMCache. |
| `disagg_multimodal_*.sh` / `disagg_omni_*.sh` | Disaggregated multimodal variants. |
| `dep.sh` | Dynamo expert-parallelism launcher. |

## Prerequisites

- A CUDA-compatible GPU and the vLLM runtime installed (see [Local Installation](../../../../docs/getting-started/local-installation.md)).
- etcd and NATS running locally, or use `--discovery-backend file` for single-machine development.

## See also

- Production-ready K8s deploys: [`recipes/`](../../../../recipes/README.md).
- Example K8s deploys for vLLM: [`examples/backends/vllm/deploy/`](../deploy/).
- Use-case index: [`examples/README.md`](../../../README.md#by-use-case).
