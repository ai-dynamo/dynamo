---
name: audit-aiperf-results
description: >-
  Validate raw AIPerf outputs, verify workload and request integrity, and normalize all reported metrics into an
  auditable benchmark summary. Use after an AIPerf Job completes and before any SLO, gain/loss, or promotion analysis.
---

# Audit AIPerf Results

Decide whether the benchmark evidence is usable. Preserve raw files unchanged.

Read `docs/rules/benchmarking.md`, `docs/guides/result_storage.md`, the canonical benchmark plan, execution ledger,
AIPerf config, and `submodules/aiperf/docs/tutorials/working-with-profile-exports.md`.

## Validate

- Parse per-request `profile_export.jsonl` with AIPerf's native Pydantic models when available. Record the parser and
  runtime version used.
- Parse `profile_export_aiperf.json` and multi-run aggregate/search artifacts when configured.
- Confirm files are readable, non-empty, and internally consistent.
- Confirm the executed config matches the canonical plan and candidate endpoint.
- Confirm trace hash or static-shape identity, schedule mode, endpoint type, model, tokenizer, warmup, seed, request
  count/duration, repetitions, and load controls.
- Separate warmup from profiling records and exclude only data the canonical plan says to exclude.
- Compare attempted, successful, failed, cancelled, and timed-out request counts.
- Check actual ISL/OSL distributions against the input workload and report any output-length shortfall.
- Check timestamps, benchmark duration, fixed-schedule coverage, duplicate/missing request ids, malformed metrics,
  NaN/inf values, units, and impossible negative latencies.
- Recompute user-requested percentiles from raw profiling records when AIPerf does not export them directly.

If the aggregate export is missing but complete raw records exist, one deterministic reconstruction is allowed using
the pinned AIPerf models/metric definitions. Record `valid_with_recovery`, the missing file, method, and generated
summary. Never modify or replace the raw directory.

## Outputs

Write `benchmark_audit.json` with:

- `status`: `valid`, `valid_with_recovery`, or `invalid`;
- benchmark-series id and workload identity;
- expected versus actual requests and phases;
- error/cancellation breakdown;
- integrity checks and recovery actions;
- parser/AIPerf versions;
- blockers and `next_action`: `analyze`, `rerun_benchmark`, or `stop`.

Write `benchmark_summary.json` with normalized benchmark metadata and every numerical metric reported by AIPerf,
including units and available statistics. Include requested custom percentiles and per-GPU derived throughput with the
GPU-count source. Do not include gain/loss interpretation in the summary.

If `status` is `invalid` and the blockers can be repaired without changing the canonical benchmark plan or workload
semantics, set `next_action` to `rerun_benchmark`. Return `benchmark_audit.json` and `benchmark_execution.json` to
`run-aiperf-benchmark`, preserve the invalid run as failed evidence, and rerun the same frozen workload without
overwriting its raw artifacts. After the rerun, invoke `audit-aiperf-results` again before analysis.

If repair would change workload semantics or a bounded rerun repeats the same invalid result, set `next_action` to
`stop`. An invalid audit stops analysis and candidate promotion until a valid rerun exists. Never discard an invalid
run.
