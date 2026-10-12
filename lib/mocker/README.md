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

## Further Reading

- [Mocker CLI reference](../../docs/fern/pages/reference/components/mocker-cli-reference.mdx)
- [Replay configuration reference](../../docs/fern/pages/reference/components/dynosim-replay-cli-reference.mdx)
