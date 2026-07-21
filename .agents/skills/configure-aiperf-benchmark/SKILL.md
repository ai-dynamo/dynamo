---
name: configure-aiperf-benchmark
description: >-
  Select and freeze an AIPerf workload, objective, load policy, and Kubernetes execution manifest for a successfully
  deployed Dynamo candidate. Use when creating iteration 0 benchmark_plan.json or rendering later same-series
  perf.yaml and aiperf-config.yaml files.
---

# Configure AIPerf Benchmark

Create a reproducible benchmark without changing the deployed candidate.

Read `docs/rules/benchmarking.md`, `docs/guides/benchmark_tooling.md`, the target workload, deployment ledger, and the
relevant docs under `submodules/aiperf/docs/` before selecting flags. Inspect matching Dynamo recipe `perf.yaml` files
when available.

## Workflow

1. Require a successful smoke test and resolve the in-cluster frontend endpoint, served model, tokenizer, Kubernetes
   context, namespace, artifact collection path, and GPU count.
2. At iteration 0, select the workload in this order:
   - user-provided Mooncake trace;
   - exact user-provided ISL/OSL and traffic controls;
   - closest Dynamo recipe trace as a `recipe_proxy`.
3. Validate a selected trace as JSONL. Record its path, SHA256, row count, timestamp range, ISL/OSL distribution,
   prefix/hash information, and any rows outside the served context limit. Do not silently filter or clip rows.
4. Select the benchmark series:
   - `trace_fidelity`: preserve timestamps and fixed-schedule behavior;
   - `static_shape`: preserve the exact synthetic ISL/OSL target;
   - `capacity`: vary concurrency or request rate as a separate series.
5. Select the objective:
   - with target SLOs, use goodput/good-request fraction and the stated attainment constraints;
   - without sufficient SLOs, preserve a throughput/latency/error Pareto view.
6. Use AIPerf native sweeps or adaptive search for numeric load controls when helpful. Do not use AIPerf search to
   change deployment settings or mutate the evaluation workload.
7. Pin the AIPerf runtime version or source commit. Record the `submodules/aiperf` commit separately.
8. Write the canonical plan to `EXP_ROOT/inputs/benchmark_plan.json`. For iteration > 0, load this file and refuse
   workload-semantic changes; only endpoint, Job identity, namespace, or artifact wiring may differ.
9. Write run-scoped `<DEPLOY_ROOT>/benchmark/aiperf-config.yaml` and `<DEPLOY_ROOT>/benchmark/perf.yaml`.

## Canonical Plan

`benchmark_plan.json` must identify:

- plan id and benchmark-series id;
- workload source: `user_trace`, `user_static`, or `recipe_proxy`;
- trace path/hash or exact synthetic distribution;
- fixed-schedule, concurrency, request-rate, request-count/duration, warmup, repetitions, confidence level, and seed;
- endpoint type, streaming behavior, model, and tokenizer;
- target metrics, SLOs, goodput thresholds, and optimization direction;
- AIPerf submodule commit and required runtime version;
- proxy rationale and limitations when applicable.

## Manifest Rules

- `perf.yaml` is a Kubernetes Job, not the AIPerf-native config.
- Run AIPerf inside the cluster and use the target context and namespace.
- Prefer an existing compatible recipe Job as a starting point. Modify only the run-scoped copy.
- Ensure raw artifacts remain available after Job completion, either on a PVC or in the completed pod until copied.
- Read referenced secret names from existing manifests; never copy secret values into benchmark files.
- Do not embed a local host path that is unavailable to the benchmark pod.

Stop with a concrete configuration blocker when no trace, static shape, or defensible recipe proxy can be selected.
