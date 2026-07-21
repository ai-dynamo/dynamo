# Benchmarking Rules

Keep benchmark evidence comparable, reproducible, and tied to the target workload.

## Freeze The Evaluation Contract

- Create one canonical `benchmark_plan.json` for the optimization run and reuse it across deployment iterations.
- Keep the workload source, trace hash, schedule mode, request shape, endpoint type, streaming mode, tokenizer, random
  seed, warmup, AIPerf version, and repeated-run settings fixed when comparing candidates.
- Treat a Mooncake trace as the workload. The reward is the measured objective, such as goodput under SLOs or a
  throughput/latency Pareto result.
- Keep exact fixed-schedule trace replay separate from capacity experiments that replace trace timestamps with
  concurrency or request-rate controls. Never compare them as one benchmark series.
- Do not drop requests, clip output lengths, alter timestamps, or change prompt/hash reuse semantics unless the user
  requested that transformation. Record any transformed workload as a new series.

## Select Workloads In Order

1. Use a user-provided trace when available.
2. Otherwise use the user's exact ISL/OSL or traffic controls.
3. Otherwise select the closest Dynamo recipe trace as a proxy and record its source, hash, fit rationale, and mismatch.

A recipe proxy supports exploration. It does not prove performance for an unspecified production workload.

## Run Discipline

- Run AIPerf inside the target cluster unless the user explicitly requests another placement.
- Do not run competing benchmarks against the same deployment or GPUs.
- Preserve raw AIPerf outputs unchanged and record the exact command, config, image, source/runtime version, endpoint,
  DGD, model, hardware, GPU count, and artifact paths.
- Retry only after identifying a concrete execution failure. Do not weaken the workload to make a failed run pass.
- Use AIPerf repeated-run confidence data for close decisions. If uncertainty overlaps, report the comparison as
  inconclusive rather than inventing a universal significance threshold.
- Only audited, same-series runs may be used for gain/loss claims or candidate promotion.
