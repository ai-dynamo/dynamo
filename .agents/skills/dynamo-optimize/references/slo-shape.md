<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SLO Shape — AIPerf 0.8.0

Reference content for `dynamo-optimize/SKILL.md`. Defines the AIPerf SLO
contract the skill enforces: `--goodput` grammar, the metric tag registry,
the output-file schema, per-dimension PASS/FAIL semantics, and rollback
policy on FAIL.

Source of truth: `dynamo-skills/corpus/aiperf/{version,output-schema,goodput-syntax}.yaml`.
Every claim here cites a Section-G row from `dynamo-skills/docs/citations.md`.

## AIPerf version pin

The skill ALWAYS pins `aiperf==0.8.0` regardless of what a recipe's
`perf.yaml` pins (G27). The 2026-05-22 user-locked decision was to roll
forward to 0.8.0 unconditionally, because:

- 0.8.0 is the current PyPI release (2026-05-16, G27).
- 0.8.0 adds metrics that older versions don't expose: `e2e_output_token_throughput`,
  the t-digest aggregator, per-request `output_length`, multi-run
  confidence reporting, MMLU accuracy benchmark, parameter sweeping
  (per 0.8.0 release notes).
- Recipe pins in the tree are stale: 8 perf.yaml files pin `0.6.0`, 4
  pin `0.7.0` (kimi-k2.5 only), 11 pin a `54cd6dc8` git SHA (pre-0.7.0),
  1 pins `4d3fa294` (gpt-oss-120b disagg only), 1 pins `79de74ec` (qwen3.6-35b).
- Single-version contract simplifies the PASS/FAIL output: comparing two
  measurements across different AIPerf versions is invalid (one of the
  refusal conditions in `SKILL.md`).

The known risk: DYN-2878-style transformers-version conflict. The
recipe's base image may ship a `transformers` version that doesn't
co-install with `aiperf==0.8.0`'s dependency tree. When this happens,
the skill surfaces the conflict in the output contract — it does NOT
silently fall back to the recipe's older pin.

## `--goodput` grammar

Per AIPerf 0.8.0 `docs/cli-options.md` (G29):

```
--goodput "<metric_tag>:<value> <metric_tag>:<value> ..."
```

- Combinator: **space-separated** `KEY:VALUE` pairs inside one quoted
  string.
- Value units: in the metric's **display unit** (ms for latency metrics,
  tokens/s for throughput metrics). AIPerf converts to internal units
  automatically.
- Direction (lower-vs-higher-is-better): **inferred** from the metric
  class's `MetricFlags.LARGER_IS_BETTER`. The user does NOT specify
  direction in `--goodput` (G30).
- A request is "good" only if it satisfies ALL declared thresholds.

Unknown tags raise `ValueError: Unknown metric tag(s) in --goodput: <tag>`
at AIPerf startup (G30). Use the supported-tags table below; do not
guess.

### Verbatim AIPerf source for the pass logic (G30)

```python
# src/aiperf/metrics/types/good_request_count_metric.py
def _passes(self, metric_cls, record_value: float, threshold_value: float) -> bool:
    """Compare a record value against its SLO using the metric's directionality."""
    if metric_cls.flags.has_flags(MetricFlags.LARGER_IS_BETTER):
        return record_value >= threshold_value
    return record_value <= threshold_value
```

## Supported `--goodput` metric tags

Confirmed in docs/cli-options.md examples or recipe usage (G31, G32):

| Tag | Units (display) | Direction | Confirmed source |
|---|---|---|---|
| `time_to_first_token` | ms | lower_is_better | recipes/qwen3-32b/vllm/disagg-kv-router/perf.yaml uses `time_to_first_token:2000` |
| `inter_token_latency` | ms | lower_is_better | recipes/qwen3-32b/vllm/disagg-kv-router/perf.yaml uses `inter_token_latency:25` |
| `request_latency` | ms | lower_is_better | docs/cli-options.md example `request_latency:250` |
| `output_token_throughput_per_user` | tokens/s | larger_is_better | docs/cli-options.md example `output_token_throughput_per_user:600` |

Registered tags AIPerf will accept but with weaker provenance (registry
accepts, no docs example, may be filtered at runtime as "not applicable
to the current endpoint/config"):

| Tag | Units (display) | Direction | Notes |
|---|---|---|---|
| `time_to_second_token` | ms | lower_is_better | Useful when speculative decoding is in play. |
| `time_to_first_output_token` | ms | lower_is_better | Distinguishes prefill+first-output-token from prefill+first-token-of-any-kind. |
| `inter_chunk_latency` | ms | lower_is_better | Streaming chunk timing. |
| `e2e_output_token_throughput` | tokens/s | larger_is_better | Added in 0.8.0. Formula: `output_sequence_length / request_latency_seconds`. |
| `request_throughput` | requests/s | larger_is_better | Derived; may be filtered as not-applicable. |

### Pitfalls

| Bad tag | What happens | What you actually want |
|---|---|---|
| `tokens_per_second` | `ValueError: Unknown metric tag` — appears in `src/aiperf/config/slos.py` docstring but is NOT registered. | `output_token_throughput` or `output_token_throughput_per_user` or `total_token_throughput`. |
| `ttft` | `ValueError: Unknown metric tag` — that's a short header, not a tag. | `time_to_first_token`. |
| `itl` | `ValueError: Unknown metric tag` — short header only. | `inter_token_latency`. |

## AIPerf output schema (`profile_export_aiperf.json`)

Every AIPerf run writes `profile_export_aiperf.json` to `--artifact-dir`
(G28). The Dynamo in-tree wrapper at
`/Users/dagil/dynamo/components/src/dynamo/profiler/utils/aiperf.py`
confirms this and the access pattern.

Top-level shape (Pydantic, with `extra='allow'`):

```python
class JsonExportData(AIPerfBaseModel):
    schema_version: str          # "1.3" at 0.8.0
    aiperf_version: str          # "0.8.0"
    # ... benchmark/run identity, input config, run metadata,
    # ... start/end times, telemetry, error summary
    # ... + dynamically-added metric_tag -> JsonMetricResult
```

Per-metric shape (verbatim from `src/aiperf/common/models/export_models.py`):

```python
class JsonMetricResult(AIPerfBaseModel):
    unit: str
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None
    count: int | None = None
    sum: int | float | None = None
```

**Important shape detail (G28):** percentiles are NESTED under the metric
key, NOT separate top-level metrics. The Dynamo in-tree wrapper accesses
`aiperf_result["time_to_first_token"]["avg"]` and
`aiperf_result["time_to_first_token"]["max"]` — this is the canonical
access pattern.

## Per-dimension PASS/FAIL semantics

`scripts/measure_slo.py` parses `profile_export_aiperf.json` and, for
each declared `--goodput` metric tag, emits one line:

```
PASS|<metric>|measured=<value><unit> SLO=<threshold><unit>
FAIL|<metric>|measured=<value><unit> SLO=<threshold><unit>
```

Following the stdout schema in
`dynamo-skill-author/references/body-shape.md`.

Comparison uses the metric class's `MetricFlags.LARGER_IS_BETTER`:

- LARGER_IS_BETTER tags: PASS if `measured >= threshold`.
- Default (smaller is better): PASS if `measured <= threshold`.

The metric used for comparison is the **avg** percentile, NOT p99 — the
recipe and the AIPerf example invocations both compare avgs.
`scripts/measure_slo.py` can be extended to use p95 or p99 with a flag,
but the default matches the `--goodput` semantics in AIPerf itself.

Exit code: `0` only when ALL dimensions PASS. Any FAIL exits non-zero
with the per-dimension lines preserved on stdout.

## Delta vs baseline (optional)

When `--baseline <baseline.json>` is supplied, the script also emits:

```
DELTA|<metric>|post_value=<value> baseline_value=<value> delta=<+/-X%>
```

If the baseline `aiperf_version` differs from `0.8.0`, the script labels
the entire delta block as `INVALID|<metric>|reason=different_aiperf_versions`
and exits non-zero — comparing across versions is one of the skill's
refusal conditions.

## Rollback policy

When ANY dimension FAILs, the skill surfaces these options to the user
(see `SKILL.md` Phase 4 Decision points):

1. `kubectl delete dgd <name>` — full rollback to pre-Phase-4 state.
2. `kubectl patch dgd ...` — apply a smaller change. Common targeted
   patches:
   - Revert `DYN_ROUTER_MODE` to `round-robin` if `kv` was set but the
     workload's prefix reuse turned out below the KV-router beneficial
     threshold (G21).
   - Revert the GPU-count patch if you shrunk below the tested envelope.
   - Revert the image-tag patch if the new image regressed.
3. Switch recipe mode — loop back to Phase 2 with the new workload-class
   signal that the FAIL revealed.
4. Accept the SLO miss explicitly — the skill records the acceptance in
   the output contract but does not silently treat FAIL as PASS.

The skill never auto-rolls-back without user confirmation. Every
rollback is an explicit DESTRUCTIVE or MUTATING action requiring the
Human-in-the-Loop prompt.

## See Also

- [k8s-recipe-workflow.md](k8s-recipe-workflow.md) — Where the SLO fits into the deploy/validate sequence.
- [inference-literature.md](inference-literature.md) — Regression conditions that explain WHY a dimension might FAIL.
- [known-issues.md](known-issues.md) — DYN-2878 transformers conflict, "Unknown metric tag" errors, etc.
