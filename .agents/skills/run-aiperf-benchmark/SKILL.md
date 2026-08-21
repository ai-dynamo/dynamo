---
name: run-aiperf-benchmark
description: >-
  Launches, monitors, debugs, and collects one run-scoped AIPerf Kubernetes benchmark without changing its workload
  semantics. Use after perf.yaml and aiperf-config.yaml have been configured for a smoke-tested Dynamo deployment.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - aiperf
    - benchmarking
---

# Run AIPerf Benchmark

Execute the configured Job and preserve operational evidence. Do not interpret performance.

Read `agent-docs/rules/execution/deployment.md`, `agent-docs/rules/execution/logging.md`,
`agent-docs/rules/execution/run-artifacts.md`, `agent-docs/rules/benchmarking/benchmark-isolation.md`,
`agent-docs/rules/benchmarking/comparison-uncertainty.md`,
`agent-docs/rules/benchmarking/evidence-eligibility.md`, `agent-docs/rules/benchmarking/series-boundaries.md`, and the
exact active benchmark plan first.

## Preflight

- Confirm the selected Kubernetes context and namespace.
- Verify the plan path, SHA256, series ID, and performance question against the run-scoped configuration.
- Server-dry-run `perf.yaml` and verify its Job name from `metadata.name`.
- Verify the frontend service is reachable in cluster and `/v1/models` exposes the configured served model.
- Verify the AIPerf config, workload trace, tokenizer, PVC/mounts, image, and referenced secrets are available to the
  benchmark pod.
- Confirm the AIPerf measurement is configured to finish in 30 minutes or less and uses one repetition by default.
- For an additional valid-run repetition, require the prior analysis to record why the repeat is necessary and worth
  its GPU cost. Do not repeat only to obtain confidence intervals. The once-per-series noise-floor pilot (n=3)
  required by `comparison-uncertainty.md` is pre-authorized and arrives through the normal `repeat_decision:
  necessary` path with a rationale naming the series pilot; it needs no further justification.
- Confirm no other benchmark is targeting the same candidate.
- Every kubectl invocation this skill drives pins `--context "${KUBE_CONTEXT}"` and `-n "${NAMESPACE}"`,
  bounds every wait with an explicit timeout, distinguishes job `Failed` from timeout (poll the job's true
  condition rather than waiting on `Complete` alone), and chains commands so a failure never flows into the next
  step — the same discipline `deploy-dynamo-recipe`'s scripted blocks encode.
- Record neighbour occupancy in `benchmark_execution.json` at run start and end (the pods scheduled on the serving
  nodes, e.g. `kubectl --context "${KUBE_CONTEXT}" get pods -A -o wide` filtered to those nodes), so
  `comparison-uncertainty.md`'s like-for-like and transition checks have a recorded condition to read.

## Execute And Monitor

1. Apply the run-scoped Job.
2. Wait with a bounded timeout and coarse polling.
3. Monitor Job conditions, pod phase/restarts, recent events, and the relevant AIPerf log tail.
4. On completion, copy the complete AIPerf output directory unchanged into
   `<DEPLOY_ROOT>/benchmark/raw_aiperf/` before deleting the Job or pod.
5. Record exact commands, Job/pod names, start/end timestamps, configured and actual measurement duration, whether this
   was the default run or an approved repeat, repeat rationale, AIPerf source/runtime versions, exit status, retry
   history, and artifact paths in `benchmark_execution.json`.

## Debug By Ownership

| Failure | Action |
|---|---|
| Endpoint, model, or deployed workload failure | Record a handoff to `recipe-deployer`; do not patch the DGD here |
| Invalid trace or AIPerf settings | Return to `configure-aiperf-benchmark` |
| Repairable invalid evidence | Consume the audit blockers from `analyze-aiperf-results`, rerun the active series unchanged without overwriting prior raw artifacts, record the retry, then return to `analyze-aiperf-results` |
| Job scheduling, mount, image, or artifact-copy issue | Repair only the run-scoped benchmark manifest and record why |
| AIPerf client/tool failure | Use the pinned AIPerf docs/source, repair without changing benchmark semantics |
| Repeated identical failure | Stop and record the blocker |

Do not reduce request count, remove difficult trace rows, relax SLOs, lower load, switch schedule mode, or alter
ISL/OSL to make a failed benchmark finish. Such a change creates a different benchmark plan and comparison series.
Do not launch another valid run because a small delta was classified as noise or because confidence intervals are
absent.

## Output Status

Set `benchmark_execution.json.status` to `completed`, `failed`, or `blocked`. A completed Job is not automatically a
valid benchmark; validity belongs to the audit phase of `analyze-aiperf-results`. Record the exact benchmark-plan path,
SHA256, series ID, and performance question in the execution record.

Keep logs only when they explain a failure beyond the concise execution ledger. Do not retain routine successful pod
logs or broad cluster snapshots.
