<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Super VL

See the [Fern recipe guide](https://docs.nvidia.com/dynamo/dev/recipes/nemotron-3-5-super-vl).

## Decode KV Connector

Decode workers in the vLLM disaggregated recipes use `NixlConnector` alone; prefill workers keep the `SimpleCPUOffloadConnector` host tier. Nemotron 3.5 Super VL mixes Mamba and attention layers. In the vLLM build inside `vllm-runtime:1.6.0-nemotron-3.5-super-vl-dev.1`, a decode worker reuses the attention KV of a session's earlier turns, with the latest Mamba state sent by prefill, only when every KV connector reports `supports_divergent_local_hybrid_hits`. `SimpleCPUOffloadConnector` does not, so a decode host tier makes each turn pull the whole prompt's KV from prefill again: an estimated 1.13 GB instead of 0.13 to 0.15 GB at the 222,076-token median prompt. Later vLLM versions may lift this restriction, so re-check it when the image changes. The disaggregated B200, GB200, B300, and GB300 results on the recipe page were published with the earlier decode host tier and have not been re-measured.
