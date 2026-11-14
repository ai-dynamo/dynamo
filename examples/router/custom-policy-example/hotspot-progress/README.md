<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decode-Progress Hotspot Policy

`hotspot-progress` uses the remaining declared output budgets on a warm worker to
choose whether siblings should reuse that worker or pay for one cold prefix on an
idle alternative. It reads the experimental decode-progress input from Dynamo's
native tracker and uses the existing soft affinity target to retain a move while
the first sibling is prefilling.

This example carries a calibrated two-worker, fan-out-two rule. The profile requires a parent
session context, one warm exact prefix, eight background decoders on that worker,
and an idle alternative. It supports aggregated workers. Its thresholds need
calibration for the intended workload, model and hardware.

This example depends on the experimental input in [PR #16758](https://github.com/ai-dynamo/dynamo/pull/16758). Build it on the same checkout as that input while the dependency is under review.

## Decisions

| Observed state | Selection |
| --- | --- |
| Supported warm source, fresh complete progress, remaining budget above the threshold | Select the idle alternative |
| Same state, remaining budget at or below the threshold | Keep the warm source |
| Affinity already points to the cold alternative and the first sibling is prefilling | Keep that target after validating the source progress |
| Missing, stale, incomplete or otherwise unsupported state | Apply the example's two-tier fallback |
| Prefill, decode or encode worker role | Return an unsupported-role error |

The picker requests `CACHE`, `LOAD` and experimental `DECODE_PROGRESS`. It reads only
the source rank, once, after the prefix, topology, affinity and load gates pass.
A short background decode can still require that read to learn its remaining
budget. Declaring the inputs also enables host load projection and supported stream
collection, so an early policy fallback does not make the entire request path free.

The fallback mirrors the built-in two-tier policy's fixed defaults: active-request
spread above 32 and ratio above 1.1 takes the least-loaded worker; otherwise a
maximum device-prefix overlap above 0.5 of the request blocks keeps the least-loaded
worker holding that overlap; otherwise it takes the least-loaded worker. The builtin
picker is private, so this example owns the small formula and checks its fallback
choices against the real builtin policy in tests. Ties follow candidate row order.

Dynamo retains eligibility, exact pins, allowlists, booking ownership, dispatch and
affinity state. LOAD and progress are advisory reads. Remaining declared budget is
a token bound, not a prediction of GPU completion time.

## Configuration

[`worker-selection.yaml`](worker-selection.yaml) selects the profile:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `source_active_requests` | 8 | Required number of active decoding owners on the warm source; allowed range 1–4095 |
| `min_prefix_tokens` | 16384 | Minimum warm device-prefix overlap in raw tokens; positive |
| `min_remaining_tokens` | 256 | Strict move threshold for the sum of remaining declared tokens across the source owners; positive |
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

The frontend collects progress from supported single-choice streams with a
positive explicit output limit, `ignore_eos=true` and no other stop conditions.
Collection follows the policy declaration independently of output-block accounting.
A host without a native output stream supplies no progress and uses the fallback.
See the [input contract](https://github.com/JasperChan929/dynamo/blob/5ca7aa7ae16153b8b6ec3807d6f8fb4305cb24cb/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx#experimental-decode-progress)
for availability and freshness details.

## CPU Validation

The three contract tests exercise long-budget movement and target retention,
short/boundary retention, and conservative fallback. They create real native
bookings, read through the normal selector input and compare fallback decisions
with the registered builtin two-tier policy. A paused clock makes freshness checks
deterministic.

```bash
cargo test -p dynamo-custom-policy-example-hotspot-progress --test policy
```

The selection microbenchmark measures structural rejection and a qualifying
source-rank read. Owners, observations and projected loads are created before
measurement. Its static observations use a 60000 ms freshness limit only in the
benchmark.

```bash
cargo bench -p dynamo-custom-policy-example-hotspot-progress --bench selection
```

The benchmark excludes streamed-output collection and GPU execution. Historical
GPU evidence belongs in the PR description and applies only to its recorded source
revisions; these CPU checks establish the example's behavior.
