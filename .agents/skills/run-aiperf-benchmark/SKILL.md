---
name: run-aiperf-benchmark
description: >-
  Launch, monitor, debug, and collect one run-scoped AIPerf Kubernetes benchmark without changing its workload
  semantics. Use after perf.yaml and aiperf-config.yaml have been configured for a smoke-tested Dynamo deployment.
---

# Run AIPerf Benchmark

Execute the configured Job and preserve operational evidence. Do not interpret performance.

Read `docs/rules/execution.md`, `docs/rules/logging.md`, `docs/rules/benchmarking.md`, and the benchmark plan first.

## Preflight

- Confirm the selected Kubernetes context and namespace.
- Server-dry-run `perf.yaml` and verify its Job name from `metadata.name`.
- Verify the frontend service is reachable in cluster and `/v1/models` exposes the configured served model.
- Verify the AIPerf config, workload trace, tokenizer, PVC/mounts, image, and referenced secrets are available to the
  benchmark pod.
- Confirm no other benchmark is targeting the same candidate.

## Execute And Monitor

1. Apply the run-scoped Job.
2. Wait with a bounded timeout and coarse polling.
3. Monitor Job conditions, pod phase/restarts, recent events, and the relevant AIPerf log tail.
4. On completion, copy the complete AIPerf output directory unchanged into
   `<DEPLOY_ROOT>/benchmark/raw_aiperf/` before deleting the Job or pod.
5. Record exact commands, Job/pod names, start/end timestamps, AIPerf source/runtime versions, exit status, retry
   history, and artifact paths in `benchmark_execution.json`.

## Debug By Ownership

| Failure | Action |
|---|---|
| Endpoint, model, or deployed workload failure | Record a handoff to `recipe-deployer`; do not patch the DGD here |
| Invalid trace or AIPerf settings | Return to `configure-aiperf-benchmark` |
| Repairable invalid audit | Consume the audit blockers, rerun the same frozen workload without overwriting prior raw artifacts, record the retry, then return to `audit-aiperf-results` |
| Job scheduling, mount, image, or artifact-copy issue | Repair only the run-scoped benchmark manifest and record why |
| AIPerf client/tool failure | Use the pinned AIPerf docs/source, repair without changing benchmark semantics |
| Repeated identical failure | Stop and record the blocker |

Do not reduce request count, remove difficult trace rows, relax SLOs, lower load, switch schedule mode, or alter
ISL/OSL to make a failed benchmark finish. Such a change creates a different benchmark plan and comparison series.

## Output Status

Set `benchmark_execution.json.status` to `completed`, `failed`, or `blocked`. A completed Job is not automatically a
valid benchmark; validity belongs to `audit-aiperf-results`.

Keep logs only when they explain a failure beyond the concise execution ledger. Do not retain routine successful pod
logs or broad cluster snapshots.
