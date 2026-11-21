<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-router-policies

`dynamo-router-policies` holds the router plugins Dynamo ships. Each one is written against the same public plugin API a custom policy uses (`dynamo_kv_router::plugins::prelude`), so the crate also serves as a set of worked examples.

| Kind | Type name | Source |
|---|---|---|
| Worker selection | `default`, `dynamo-default-cost-fn`, `kvcr` | `src/default/` |
| Worker selection | `dynamo-sticky-min-load` | `src/sticky_min_load.rs` |
| Worker selection | `dynamo-two-tier-cost-fn` | `src/two_tier_cost_fn.rs` |
| Worker selection | `random`, `round-robin`, `least-loaded` | `src/router_modes/` |
| Worker selection and request classifier | `thunderagent` | `src/thunderagent/` |
| Request classifier | `output_length_prediction` | `src/output_length/` |
| Hint policy | `kv_fetch` | `src/kv_fetch.rs` |

Hosts link the crate directly. `default_registry` gives the default policy and the routing modes; `register` adds the named policies to a catalog.

To write your own policy, see [Custom worker selection](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx). The router's design is in [`lib/router/README.md`](../../router/README.md).
