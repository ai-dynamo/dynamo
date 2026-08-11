<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-mocker

`dynamo-mocker` integrates the shared AI Simulate engine and Replayer with Dynamo. It does not own
the generalized scheduler or deterministic replay core.

## What This Crate Owns

- the Live Mocker driver around `aisimulate_core::engine`
- Dynamo request, response, lifecycle, KV-event, and metrics adaptation
- transport, cancellation, and publication for simulated Dynamo workers
- KV router placement and Planner scaling composition for offline replay
- Dynamo-compatible online replay and legacy Rust entrypoints

The dependency direction is:

```text
Dynamo Router / Planner / Live adapters
                    ↓
        aisimulate-core::replay
                    ↓
        aisimulate-core::engine
```

The shared engine owns scheduler behavior, GPU KV-cache accounting, preemption, timing, and
attention data-parallel barriers. The shared Replayer owns deterministic virtual time, logical
workers, aggregated and disaggregated replay, and reports.

## Further Reading

- [AISimulate Core](../../aisimulate/crates/core/README.md)
- [Dynamo Offline Replay Adapters](src/replay/offline/README.md)
- [Live Mocker Architecture](../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/mocker/mocker-engine-architecture.md)
- [Run a Local Live Mocker Deployment](../../docs/fern/pages/cli/operations/simulation-with-dynosim/mocker-live-simulation.mdx)
- [Dynamo Replay Integration](../../docs/fern/pages/developer-guide/knowledge-base/concepts/simulation/dynosim-architecture.md)
