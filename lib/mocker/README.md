<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mocker

`dynamo-mocker` is a GPU-free simulation crate for Dynamo's LLM scheduling and KV-cache behavior.
It is used for testing, replay, and benchmarking workflows where you want realistic scheduler and
cache behavior without running a real inference engine.

## What This Crate Provides

- `MockerConfig` combining the AISimulate launch configuration with Dynamo runtime options
- `engine::create_engine` for building a vLLM-style or SGLang-style mock scheduler
- `KvEventPublishers` hooks for emitting router-visible KV cache events
- `loadgen` and `replay` modules for synthetic and trace-driven experiments

## Basic Rust Usage

```rust
use dynamo_mocker::config::MockerConfig;

let config = MockerConfig::from_value(serde_json::json!({
    "engine": {
        "backend": "vllm",
        "block_size": 16,
        "num_gpu_blocks": 1024,
        "max_num_seqs": 32,
        "max_num_batched_tokens": 4096
    },
    "dynamo": {"enable_local_indexer": true}
}))?;
```

AISimulate supplies the engine configuration, defaults, validation, and scheduling implementation.
Dynamo supplies transport, event publishing, and Router/Planner adapters. Python replay entry points
accept the same canonical mapping; no Dynamo engine-argument constructor classes are required.

## Session Lineage in Offline Replay

Set `enable_session_prefix_index=True` on the replay's `KvRouterConfig` to enable
Dynamo's shared `SessionPrefixIndexer`. It records selected-worker G1 matches at
admission and applies attributed G1 stores, removals, clears, and worker removal
after the physical index update. Queued requests recheck G1 residency at admission;
configured KV-event lag delays both indexes together. This is opt-in and does not
change placement or eviction policy.

Session IDs come from replay request metadata and `RouterEvent.session_id`.
Unattributed stores and lower-tier events do not create session lineage. Replay
forwards AISimulate stored-block attribution into router events. The adapter never
infers a store's session from whichever requests happen to be running.
Disaggregated decode placement remains load-only; prefill observes the cache events.

Session lineage uses the production indexer's five-minute wall-clock cleanup
interval, not simulated replay time. Unreferenced nodes can remain until a later
update triggers cleanup or the replay ends. Cleanup timing can affect lineage
reconstructed by later requests, but does not affect current placement or eviction
decisions.

## Further Reading

- [Mocker CLI reference](../../docs/fern/pages/reference/components/mocker-cli-reference.mdx)
- [Replay configuration reference](../../docs/fern/pages/reference/components/dynosim-replay-cli-reference.mdx)
