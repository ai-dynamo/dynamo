<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# vLLM

See [docs/backends/vllm/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/overview.md) for documentation.

## Recurrent checkpoint events

For hybrid models, `DYN_VLLM_RECURRENT_KV_EVENTS=1` enables recurrent checkpoint
publication through the direct KV event publisher. Enable it only after all
routers, indexers, and event consumers have been upgraded to support
`kv_cache_group` events. The default remains attention-only publication.

This mode requires engine cache-group metadata and an enabled local indexer
(`enable_local_indexer=True`, the default for Dynamo vLLM workers).
The publisher announces every recurrent group before attention events and keeps
empty groups in recovery dumps. If its recovery endpoint cannot start, the source
is not advertised. State-agent publication does not yet negotiate this capability
and rejects the option explicitly.

The upstream vLLM checkpoint payload stays small: one checkpoint hash and no
copied token range. Dynamo adds the expected recurrent-group count and block
alignment to its internal event tag. GPU reuse requires GPU checkpoints; offload
reuse requires all recurrent groups at a common checkpoint in the restoring tier.
