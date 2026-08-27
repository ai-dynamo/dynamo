---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sweep Configuration Providers
subtitle: Extend Sweeper without adding application dependencies to its core
---

> [!WARNING]
> **Experimental.** The provider ABI is versioned but may change before AI Simulate stabilizes.

A `SweepConfigProvider` lets an optional feature package contribute search dimensions without
making `aisimulate` depend on that package.

## Contract

```python
class ExampleProvider:
    name = "example.policy"
    api_version = 1

    def generate_search_space(self, search_spec, context):
        return AdapterSearchPlan(
            fragment=SearchSpaceFragment(
                choices_by_branch={
                    "agg": {"mode": list(search_spec["mode"])}
                }
            )
        )

    def materialize_replay(self, plan, selection, context):
        return AdapterReplaySpec(config={"mode": selection["mode"]})
```

`generate_search_space` receives the complete adapter-owned search-space mapping plus an isolated
`SweepContext`. It may validate composite settings, derive dimensions, and store JSON-compatible
state in `AdapterSearchPlan`.

`materialize_replay` receives that plan, one concrete local selection, and an isolated
`CandidateContext`. It returns the concrete adapter config and any versioned `RuntimeHookSpec`
objects needed by replay.

## Registration

Expose a zero-argument factory from the `aisimulate.sweep_config_providers` entry-point group:

```toml
[project.entry-points."aisimulate.sweep_config_providers"]
"example.policy" = "example_package.simulation:create_provider"
```

The factory and provider implementation are imported only when configuration selects
`adapters.example.policy`. Direct constructor injection takes precedence over installed entry
points.

## ABI Rules

- `name` must equal the configured adapter key.
- `api_version` must match the core provider API version.
- Plans, concrete config, diagnostics, state, and runtime-hook config must be strict JSON values.
- A provider must not mutate `SweepContext`, `CandidateContext`, or previously returned values.
- Runtime hooks must be declared during search preparation so runner compatibility can be checked
  before runner creation.
- Providers run in the main process. Spawned replay workers receive `ReplaySpec` values only.
