# Benchmark Tooling

Use AIPerf as the only benchmark harness for this optimization loop.

## Sources Of Truth

- Read `submodules/aiperf/README.md` for current capabilities.
- If the pinned checkout is absent, initialize it with
  `git submodule update --init --recursive submodules/aiperf`; do not advance it during an optimization run.
- Read only the relevant AIPerf docs for the selected mode, especially YAML configuration, trace replay, goodput,
  multi-run confidence, timeslices, adaptive search, and profile exports.
- Use existing Dynamo recipe `perf.yaml` files and traces as examples, not as universal defaults.
- Record the AIPerf submodule commit and the separately installed AIPerf runtime version. Do not assume they match.

## Workload Modes

| Available input | Primary benchmark |
|---|---|
| User Mooncake trace | Exact fixed-schedule replay |
| User ISL/OSL | Static synthetic workload at the exact target shape |
| Neither | Best-fit recipe trace, explicitly marked `recipe_proxy` |

For a timestamped Mooncake trace, preserve fixed-schedule replay for the fidelity result. If capacity must also be
measured, create a separate benchmark series that deliberately varies request rate or concurrency while preserving the
request-shape and prefix-reuse data.

## Objective

When the target workload includes latency SLOs, prefer goodput and good-request fraction as the optimization objective,
with errors included in feasibility. Without sufficient SLOs, retain a Pareto view over throughput, per-user
throughput, TTFT, ITL, request latency, and errors instead of inventing a scalar score.

AIPerf may search numeric benchmark controls for one fixed endpoint. The Dynamo optimization loop remains responsible
for changing and redeploying `deploy.yaml` between candidates.

## Files

- `perf.yaml` is the run-scoped Kubernetes Job used to execute AIPerf.
- `aiperf-config.yaml` is the native configuration consumed by `aiperf profile --config`.
- `workload_trace.jsonl` is an input workload. `profile_export.jsonl` is AIPerf per-request output. Do not call them the
  same kind of trace.

Prefer a native AIPerf YAML config over a long generated CLI. The Kubernetes Job may mount the config or create it from
a ConfigMap. Run AIPerf headlessly and write all raw outputs beneath the candidate's `benchmark/raw_aiperf/` directory.

## Required Recording

Record the benchmark plan id, workload source/hash, exact AIPerf config and command, AIPerf source/runtime versions,
endpoint/model/tokenizer, DGD and image identity, hardware/GPU count, schedule/load controls, warmup, repetitions,
random seed, all reported numerical metrics and units, errors, duration, and raw artifact paths.
