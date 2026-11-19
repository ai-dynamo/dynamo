<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode-Progress Hotspot Policy

`hotspot-progress` is an experimental policy for a two-worker aggregated pool
serving two siblings with a shared prefix. It can move the first sibling from a
warm worker with a decode backlog to an idle worker. Enable soft session affinity
so the host updates the group's binding when a worker accepts a dispatch.

## Decisions

Each selection uses the current eligible workers, device prefix overlap, and
projected load. The hotspot rule requires a parent session, two distinct workers
with available load observations, one warm prefix, and zero overlap on the other
worker. Call them A (warm) and B (zero overlap).

| Observed state | Selection |
| --- | --- |
| No soft target or target A, enough active requests on A, no active prefill on A, idle B, and fresh progress covering every active request on A | Select B if the remaining budget exceeds the threshold. Otherwise select A. |
| Soft target B, with one active request on B and positive estimated prefill load | Keep B without consulting A's decode progress or active request minimum. |
| Any other state | Apply the example's two-tier fallback to the current overlap and load. |

The retention rule uses aggregate load on B. It cannot identify the prefilling
request or confirm that its KV is ready for reuse. A busy B, an insufficient
warm prefix on A, or any overlap on B returns the choice to the fallback. A soft
target outside the candidates also uses the fallback. The host enforces exact
pins and worker eligibility. The policy reports an error for roles other than
aggregated.

Each request uses the soft target it acquired before selection. In a group
already bound to A, concurrent siblings can both acquire A. Once the first
request books B, the second can still carry A even after the binding changes
to B. Because B is now busy, the second request uses the fallback and may land
on A. Retention applies when the second request acquires B and B still meets the
zero-overlap and prefill conditions above.

The picker requests `CACHE`, `LOAD`, and experimental `DECODE_PROGRESS`. Only
the first-move branch reads progress, once on A after checking the load gates.
Declaring the inputs enables load projection and progress collection for supported
streams.

The fallback mirrors the built-in two-tier policy's fixed defaults: active-request
spread above 32 and ratio above 1.1 takes the least-loaded worker; otherwise a
maximum device-prefix overlap above 0.5 of the request blocks keeps the least-loaded
worker holding that overlap; otherwise it takes the least-loaded worker. The builtin
picker is private, so this example owns the small formula and checks its fallback
choices against the real builtin policy in tests. Ties follow candidate row order.

Dynamo retains eligibility, exact pins, allowlists, booking ownership, dispatch and
affinity state. LOAD and progress are advisory reads. Remaining output budget is
an upper bound in tokens.

## Configuration

[`worker-selection.yaml`](worker-selection.yaml) selects the profile:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `source_active_requests` | 8 | Minimum number of active decoding owners on the warm source |
| `min_prefix_tokens` | 16384 | Minimum warm device-prefix overlap in raw tokens; positive |
| `min_remaining_tokens` | 256 | Strict move threshold for the sum of remaining tokens across the source owners; positive |
| `max_progress_age_ms` | 100 | Maximum age of the oldest source observation; allowed range 1–60000 ms |

For example, eight streams with an output limit of 40 and eight observed tokens
each have a total remaining budget of 256. Equality keeps the warm source. The
profile is carried from the controlled experiment; it is not a universal migration
boundary.

Build the Python extension with the [example catalog](../README.md#run-with-the-python-frontend)
and start an aggregated worker pool. Select the example from the Dynamo root:

```bash
env -u DYN_ROUTER_WORKER_SELECTION_POLICY \
python -m dynamo.frontend \
  --router-mode kv \
  --router-policy-config examples/router/custom-policy-example/hotspot-progress/worker-selection.yaml \
  --router-session-affinity-ttl-secs 60 \
  --router-session-affinity-mode soft \
  --http-port 8000
```

The command clears the global policy override so the YAML's inline parameters
select the policy for the aggregated pool.

The frontend collects progress from single-choice aggregated streams with a
positive effective output limit, including limits derived from remaining context.
Use explicit limits calibrated for your workload with this example. A
context-derived limit can dominate the budget sum even when the request will
finish quickly.
Collection follows the policy declaration independently of output-block accounting.
When progress is unavailable, the first-move branch uses the fallback.
See the [input contract](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx#experimental-decode-progress)
for availability and freshness details.

## CPU Validation

The policy tests cover the first-move count and budget boundaries, fallback,
and unsupported roles. The affinity regression uses the native session affinity
table, the selector, and real slot bookings. It starts with an unbound group,
models an accepted dispatch by booking the selected worker and committing its
affinity hold, then acquires the next sibling's target from the table.

```bash
cargo test -p dynamo-custom-policy-example-hotspot-progress --test policy
```
