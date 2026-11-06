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
The supported resume contract is deliberately limited to full-attention and
Mamba groups that all use the same positive block size as the indexer, with
prefix caching in `align` mode, no context parallelism, and no speculative
decoding. `prefix_match_unit` must be unset or equal that common block size.
Sliding-window, chunked-local, mixed-size, and finer-than-indexer layouts fail
at startup when recurrent publication is enabled. Dense workers and publication
with the option disabled keep the existing attention-only behavior.

The engine group utility reports physical sizes, not the effective hash or hit
granularity. Dynamo derives equal hash, scheduler, and hit units only after
validating the fixed-common-unit restrictions above; it does not infer general
hybrid resume behavior from a physical block-size LCM. The runtime metadata
publishes these units and the physical group layout under the
`vllm_fixed_common_unit_v1` contract. Engine versions without the required
configuration capability fields cannot enable this mode.

An optional CPU offload connector must be `OffloadingConnector` with
`self_describing_kv_events=true`, explicit `blocks_per_chunk=1`, and the default
CPU offloading spec. Other connectors and chunk geometries require separate
resume support.

The publisher announces every recurrent group before attention events and keeps
empty groups in recovery dumps. If its recovery endpoint cannot start, the source
is not advertised. State-agent publication does not yet negotiate this capability
and rejects the option explicitly.

The upstream vLLM checkpoint payload stays small: one checkpoint hash and no
copied token range. Dynamo adds the expected recurrent-group count and block
alignment to its internal event tag. GPU reuse requires GPU checkpoints; offload
reuse requires all recurrent groups at a common checkpoint in the restoring tier.
Only published checkpoint events create presence: reaching an attention boundary
does not imply an SSM checkpoint was retained, including during decode under
sparse checkpoint-retention policies.
