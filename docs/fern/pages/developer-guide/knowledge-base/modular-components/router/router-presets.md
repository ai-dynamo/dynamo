---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Presets
subtitle: Choose between default KV-aware, load-aware, and agentic routing
---

Router presets group related KV router settings for common workload shapes. Use a
preset when its workload assumptions match your deployment. Use the individual
router options when you need a different cache-locality and load-balancing tradeoff.

## Choose a Preset

| Configuration | Use when | Cache-locality behavior | Load signal |
| --- | --- | --- | --- |
| Default KV-aware routing (`--router-mode kv`) | You need a general-purpose balance between prefix reuse and worker load | Credits device-local prefix overlap | Tracks prefill tokens and active decode blocks |
| `--load-aware` | Cache reuse is unavailable, unreliable, or intentionally disabled | Ignores prefix-cache overlap | Routes with active prefill and decode load |
| `--agentic` | Long-running aggregate agentic workflows repeatedly reuse shared prefixes and cache colocation is preferred | Gives device-local prefix overlap additional credit | Weighs uncached prefill work and active decode requests more heavily |

Both named presets imply `--router-mode kv` when used with `dynamo.frontend`.
The standalone `dynamo.router` command accepts the same preset flags and is already
KV-aware. `--load-aware` and `--agentic` are mutually exclusive.

## Load-Aware

The load-aware preset uses the KV router's active-load model without relying on
prefix-cache reuse. It is useful when the backend does not publish reliable KV
events, transferred blocks are not deduplicated, or requests do not share meaningful
prefixes.

Enable it on the frontend:

```bash
python -m dynamo.frontend \
    --model-name Qwen/Qwen3-32B \
    --load-aware
```

You can also set `DYN_ROUTER_LOAD_AWARE=true`.

The preset applies these settings:

- Sets device-local overlap credit to `0`.
- Disables KV events and KV-reuse assumptions.
- Enables active-block and prefill-token tracking.
- Disables remote and shared cache indexers.
- Preserves the configured prefill load scale and lower-tier cache weights.

## Agentic

**Experimental.** The agentic preset is optimized for long-running aggregate
agentic workflows where repeated prefixes make cache colocation preferable, while
overlapping prefill and decode work still requires load-aware balancing. Examples
include multi-turn sessions, tool-use loops, and subagent fan-out that repeatedly
reuse a long system prompt or conversation prefix.

The preset does not enable session affinity or pin a conversation to one worker. It
uses observed or predicted prefix overlap to prefer a worker that already holds the
request's prefix, then prices the worker's prefill and decode load before selection.

Enable it on the frontend:

```bash
python -m dynamo.frontend \
    --model-name Qwen/Qwen3-32B \
    --agentic
```

You can also set `DYN_ROUTER_AGENTIC=true`.

The preset applies these cost-function weights:

- `overlap_score_credit=2` gives device-local prefix overlap additional credit.
- `prefill_load_scale=4` increases the cost of uncached prompt work.
- `decode_active_request_weight=64` adds 64 block-equivalents for each active
  request on a candidate worker.

The preset takes precedence over the corresponding individual CLI options and
environment variables. It also clears the deprecated overlap-score weight. Queueing,
cache tracking, cache event handling, and other router controls retain their normal
defaults.

These weights were validated with 16-token router blocks. At that block size, the
active-request term contributes 1,024 token-equivalents per active request. Benchmark
representative traffic before using a different block size, model, runtime, hardware,
or deployment topology. The validation workload used aggregated serving; this preset
is not a prefill/decode-disaggregated recommendation.

## Customize the Cost Function

For settings outside these workload assumptions, configure the three terms directly:

```text
prefill_load_scale * adjusted_prefill_blocks
    + potential_decode_blocks
    + decode_active_request_weight * active_requests
```

See [Configuration and Tuning](configuration-and-tuning.md#tuning-guidelines) for
the individual options and [Routing Concepts](routing-concepts.md) for the complete
worker-selection model.
