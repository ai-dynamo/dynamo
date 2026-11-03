<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Super VL

See the [Fern recipe guide](https://docs.nvidia.com/dynamo/dev/recipes/nemotron-3-5-super-vl).

## Decode KV Connector

In the disaggregated vLLM recipes, decode workers use `NixlConnector` only, with no CPU KV cache. Prefill workers keep their `SimpleCPUOffloadConnector` CPU cache.

**Why.** On each new turn of a conversation, a decode worker can reuse the KV cache it already holds from the earlier turns, and pull only the new part of the prompt from prefill. Nemotron 3.5 Super VL is a hybrid Mamba-attention model. For hybrid models, the vLLM in `vllm-runtime:1.6.0-nemotron-3.5-super-vl-dev.1` allows that reuse only if every KV connector on the worker supports it. `SimpleCPUOffloadConnector` does not. With a CPU cache on decode, the worker reuses nothing and pulls the whole prompt from prefill on every turn.

**Effect.** On the GB200 benchmark, removing the decode CPU cache cut the data pulled per request from 1.16 GB to 0.17 GB, and freed about 200 GiB of host memory per decode pod. Throughput and latency did not get worse.

**Caveats.**
- Newer vLLM versions may support this reuse with a CPU cache. Re-check when the image changes.
- The published disaggregated results for B200, GB200, B300 and GB300 were measured with the old decode CPU cache, and have not been re-run.
