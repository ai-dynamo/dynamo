<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experiment Workloads

## Standalone Planner Control Loop

`batch_planner_control_loop.py` is the single-pool Planner POC runner. It wires
the public `BatchGatewayJobSource`, `LlmdAsyncPrometheusSource`,
`BatchSchedulingCollector`, `plan_batch_schedule`, and
`RedisLeasedDrainLimitActuator` APIs without joining the normal Planner
lifecycle.

The default is a read-only dry run. It reads Batch Gateway and Prometheus,
evaluates the policy for a bounded number of iterations, and writes one strict
JSON object per decision. It never changes Redis unless
`--apply-drain-limit` is present. Replica floors are recorded as advisory;
this runner never scales replicas.

Run from the experiment root after the Batch Gateway and Prometheus endpoints
are reachable:

```bash
../../../../../.venv/bin/python workloads/batch_planner_control_loop.py \
  --pool dynamo-batch \
  --work-class gsm8k-max-output-128 \
  --tenant planner-poc-baseline \
  --prometheus-url http://127.0.0.1:19090 \
  --prometheus-observation-window-seconds 90 \
  --safe-rps-per-ready-replica 10 \
  --ready-replicas 1 \
  --online-offered-rps 0 \
  --iterations 6 \
  --interval 10 \
  --drain-lease-seconds 30
```

`--pool` and `--work-class` are applied to jobs collected from this
single-pool POC. `--safe-rps-per-ready-replica`, `--ready-replicas`, and
`--online-offered-rps` are explicit static assumptions, not discovered
measurements. If `--max-replicas` is omitted, it defaults to the supplied ready
replica count.

Batch Gateway listing is scoped by `X-MaaS-Username`, so this POC defaults to
the same `planner-poc-baseline` tenant used by the baseline submission harness.
The Redis drain cap, however, applies to the whole worker pool. A cap derived
from one tenant's visible jobs would be unsafe if other tenants share that pool.
Mixed-tenant pools are therefore out of scope; use one dedicated tenant and
worker pool for this POC.

`--prometheus-url` must target a real Prometheus HTTP API that serves
`/api/v1/query`, such as a local port-forward from `19090` to the cluster's
Prometheus service. Do not point it at llm-d Async's raw container
`:9090/metrics` listener; that endpoint exposes text metrics but cannot execute
the instant PromQL queries used by `LlmdAsyncPrometheusSource`.
The dispatch-rate range must span at least two Prometheus scrapes. The POC
defaults to 90 seconds so it remains meaningful with a 30-second scrape; if a
PodMonitor is enabled, prefer a 5-10 second scrape interval.

By default, decisions go to a unique
`results/raw/<UTC-run-id>/control-loop-decisions.jsonl` path. Each line contains
the collected observation, drain and advisory replica decision, diagnostics,
actuation status, and any sanitized error. Endpoint URLs, Redis keys, and
credentials are not recorded. Existing output paths are never overwritten.

### Apply the Leased Drain Limit

Apply mode needs the optional `redis` Python package, which is not part of the
current Dynamo environment. Layer it onto the project environment for the
invocation:

```bash
uv pip install --python ../../../../../.venv/bin/python 'redis>=5,<7'

../../../../../.venv/bin/python \
  workloads/batch_planner_control_loop.py \
  --pool dynamo-batch \
  --work-class gsm8k-max-output-128 \
  --tenant planner-poc-baseline \
  --prometheus-url http://127.0.0.1:19090 \
  --prometheus-observation-window-seconds 90 \
  --safe-rps-per-ready-replica 10 \
  --ready-replicas 1 \
  --online-offered-rps 0 \
  --iterations 6 \
  --interval 10 \
  --drain-lease-seconds 30 \
  --apply-drain-limit \
  --redis-url redis://127.0.0.1:6379/0 \
  --redis-control-key 'llm-d-async:drain-limit:dynamo-batch'
```

The controlled overlay uses worker pool `dynamo-batch` and control key
`llm-d-async:drain-limit:dynamo-batch`. The key is deliberately required; the
runner does not guess it. Verify that both values still match the deployed
dispatcher configuration before apply mode. Use a credentialless local Redis
port forward where possible. The runner does not print or record the Redis URL,
but command-line credentials can still be visible in shell history and process
listings.

The controlled Helm overlay installs the `redis-leased-rate` gate on
`ap.workerPools[0]`. This is a worker-pool admission boundary: it limits batch
requests entering `dynamo-batch`. It is intentionally not a `queuesConfig`
queue gate, where an `ActionWait` result can fall through to later dispatch
stages. The pool ID in Prometheus, the policy input, and the Redis key must all
name this same worker pool.

Every successful apply iteration publishes a fresh decision ID and absolute
lease expiry. The interval must be shorter than the lease duration. A
post-start collection or policy failure causes one best-effort zero-RPS pause
lease in apply mode, records whether that pause succeeded, and exits nonzero.
Dry-run failures never invoke the actuator. An actuation failure is recorded
and exits nonzero without attempting a second mutation; any prior lease is
allowed to expire into llm-d Async's fail-closed behavior.

Prometheus must expose all metrics required by
`LlmdAsyncPrometheusSource`, including backlog-source availability and both raw
and wall-clock lease-validity signals. Missing or ambiguous series abort the
observation instead of being treated as zero.

The source also evaluates `time() - min(timestamp(...))` for the mandatory
backlog-availability series. This puts scrape age in the PromQL value instead of
trusting an instant query's evaluation timestamp. A missing, future, or older
than `--max-observation-age-seconds` anchor aborts collection; apply mode then
publishes its existing best-effort zero-RPS pause. The backlog gauge itself is
updated on Async's configured broker-poll cadence, so that cadence remains a
separate, bounded source of observation lag.

Before any live dry run, verify that Prometheus is scraping the controlled
llm-d Async pods and that its HTTP API returns the required series. The target
cluster's PodMonitor CRD and cross-namespace Prometheus selectors were verified,
so the controlled overlay enables a 5-second PodMonitor. A rendered resource is
not sufficient evidence: confirm the live target is healthy before applying a
drain decision.

### Canonical Controlled Run

The successful 2026-08-28 treatment used a dedicated test cluster. Set the
context for your deployment and run these forwards in separate terminals:

```bash
export KUBE_CONTEXT=your-kube-context

kubectl --context "${KUBE_CONTEXT}" port-forward -n default \
  service/qwen3-0-6b-batch-frontend 18000:8000
kubectl --context "${KUBE_CONTEXT}" port-forward -n default \
  service/batch-gateway-apiserver 18001:8000
kubectl --context "${KUBE_CONTEXT}" port-forward -n monitoring \
  service/kube-prometheus-stack-prometheus 19090:9090
kubectl --context "${KUBE_CONTEXT}" port-forward -n default \
  service/batch-gateway-valkey 16379:6379
```

From the Dynamo repository root, the bounded apply controller was:

```bash
.venv/bin/python -u \
  examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/workloads/batch_planner_control_loop.py \
  --pool dynamo-batch \
  --work-class gsm8k-qwen3 \
  --tenant planner-poc-baseline \
  --batch-base-url http://127.0.0.1:18001 \
  --prometheus-url http://127.0.0.1:19090 \
  --prometheus-observation-window-seconds 90 \
  --safe-rps-per-ready-replica 15 \
  --ready-replicas 1 \
  --online-offered-rps 0 \
  --iterations 40 \
  --interval 2 \
  --drain-lease-seconds 10 \
  --max-observation-age-seconds 15 \
  --min-replicas 0 \
  --max-replicas 1 \
  --max-batch-admission-rps 5 \
  --apply-drain-limit \
  --redis-url redis://127.0.0.1:16379/0 \
  --redis-control-key llm-d-async:drain-limit:dynamo-batch
```

Controller run `20260828T183813Z-planner-loop-15424f` was paired with this
command from the experiment root:

```bash
./workloads/run_baseline.sh \
  --run-kind planner-controlled \
  --paired-controller-run-id 20260828T183813Z-planner-loop-15424f \
  --context "${KUBE_CONTEXT}" \
  --namespace default \
  --tenant planner-poc-baseline \
  --batch-base-url http://127.0.0.1:18001 \
  --batch-size 100 \
  --max-tokens 128 \
  --poll-interval-seconds 2 \
  --timeout-seconds 600 \
  --expected-gate-type redis-leased-rate
```

For a new treatment, replace the paired controller ID with the new ID printed
by the controller. The controller must already be running before submission.

Inspect decisions with:

```bash
jq -c '{iteration,status,decision,diagnostics,error,fail_closed_pause}' \
  results/raw/<run-id>/control-loop-decisions.jsonl
```

## Native Planner-Controlled Run

Use `--run-kind planner-native` when the normal Planner tick owns observation,
policy, leased drain actuation, and replica effects. This mode does not accept a
paired standalone controller ID. It records `control_plane.mode=native-planner`
and uses `planner-native` in both the run ID and submitted request IDs, so the
treatment cannot be mistaken for either the stock baseline or the standalone
controller experiment.

A live native run also has a stricter evidence contract. Before submission, the
harness requires a Running pod matching the native Planner regex and captures
the explicitly named ConfigMap mounted by that pod. At the end, logs captured
since the run began must contain at least two `Batch scheduling decision:`
records. This proves recurring native ticks made batch decisions during the
workload; merely having a Planner pod in the namespace is insufficient.

With the POC deployment, run from the experiment root while the Batch API port
forward is active:

```bash
./workloads/run_baseline.sh \
  --run-kind planner-native \
  --native-planner-configmap qwen3-0-6b-batch-planner-config \
  --native-planner-pod-name-regex 'qwen3-0-6b-batch-planner.*planner' \
  --context "${KUBE_CONTEXT}" \
  --namespace default \
  --tenant planner-poc-baseline \
  --batch-base-url http://127.0.0.1:18001 \
  --batch-size 100 \
  --max-tokens 128 \
  --poll-interval-seconds 2 \
  --timeout-seconds 600
```

For `planner-native`, the default expected gate type is `redis-leased-rate`,
the default decision log expression is `Batch scheduling decision:`, and the
minimum match count is two. Native runs require that gate type and do not allow
`--skip-gate-verification`. Use `--native-planner-decision-log-regex` or
`--native-planner-min-decision-logs` only when the deployed logging contract or
tick cadence intentionally differs. The generic `--pod-name-regex` must still
select the Planner pod; its default already includes this POC's DGD name.

## Evidence-Preserving Batch Baseline

Run this harness against the existing Batch Gateway, llm-d Async, and Dynamo
deployment. It never applies or changes Kubernetes resources. The live path
requires read access to the selected namespace plus existing local port forwards for
the Batch API and, when online load is enabled, the Dynamo frontend.

## What the Harness Records

Each invocation creates `results/raw/<UTC-run-id>/` before it does any work. A
run contains:

- exact normalized GSM8K Batch JSONL and its source/output checksums;
- file upload, Batch creation, progress, terminal state, and retrieved result
  files;
- optional per-online-request HTTP status, Time To First Token (TTFT), latency,
  and token usage;
- selected pod specifications, image digests, referenced ConfigMaps, events,
  current and previous logs, Kubernetes versions, and pod-proxy metric snapshots;
- optional periodic snapshots from explicit unauthenticated metric URLs;
- sanitized stdout, stderr, every captured command's stdout/stderr, and each exit
  code.

Native Planner runs additionally store the expected and observed Planner pods,
mounted/captured ConfigMap, Planner image digest records, per-log command exit
codes, and in-run decision-log match count under
`kubernetes.{start,end}.native_planner` in `metadata.json`.

The pod selector defaults to names containing `batch-gateway`,
`async-dispatch`, or `qwen3-0-6b-batch`. Override `--pod-name-regex` if the
deployed names differ.

## Canonical Autonomous Zero-Worker Run

Canonical run `20260828T213549Z-planner-native-1e3ff8` used one explicit
pre-run setup mutation to establish the worker DGDSA at zero:

```bash
kubectl patch dgdsa \
  qwen3-0-6b-batch-vllmdecodeworker \
  --namespace default \
  --subresource scale \
  --type merge \
  --patch '{"spec":{"replicas":0}}'
```

That patch completed at 21:28:07Z, more than seven minutes before the evidence
window. Before T0, verify DGDSA spec/status are both zero, no worker pod exists,
the Planner is Ready, the Redis lease is zero, and the Gateway tenant has no
active job. Start read-only DGDSA watch, Planner log-follow, and periodic
DGD/worker/Redis observers before submission. Do not run `kubectl patch`,
`apply`, `delete`, or `scale` between T0 and T1. The canonical raw evidence
directory includes the exact three observer scripts used.

With Batch Gateway on local port 18001 and Async metrics on 19092, the workload
command was:

```bash
./workloads/run_baseline.sh \
  --run-kind planner-native \
  --native-planner-configmap qwen3-0-6b-batch-planner-config \
  --native-planner-pod-name-regex 'qwen3-0-6b-batch-planner.*planner' \
  --native-planner-min-decision-logs 3 \
  --context "${KUBE_CONTEXT}" \
  --namespace default \
  --tenant planner-poc-baseline \
  --batch-base-url http://127.0.0.1:18001 \
  --batch-size 100 \
  --max-tokens 128 \
  --poll-interval-seconds 2 \
  --timeout-seconds 900 \
  --expected-gate-type redis-leased-rate \
  --metrics-url async=http://127.0.0.1:19092/metrics \
  --metrics-interval-seconds 2
```

After terminal completion, retain observers for at least two more Planner
ticks, capture final DGD/DGDSA/Redis/Async state, and stop only the exact
observer PIDs. Verify the resulting evidence directory:

```bash
python3 workloads/verify_native_planner_e2e.py \
  --run-dir results/raw/20260828T213549Z-planner-native-1e3ff8 \
  --evidence-dir \
    results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence
```

The verifier checks 15 invariants: terminal 100/100/0 and output validity,
DGDSA zero-to-one watch evidence, Planner scaling logs, policy ordering,
closed admission through worker readiness, dispatch only after a positive
lease, exact Async counter deltas, empty terminal queues, and a fresh
authoritative terminal zero lease. It deliberately does not claim worker
scale-down; the final worker replica remains one while the floor and cap return
to zero.

## Run a Local Preflight

Validate the scripts and deterministic workload without contacting Kubernetes or
an API:

```bash
./workloads/run_baseline.sh \
  --preflight-only \
  --skip-cluster-preflight \
  --skip-api-preflight
```

This creates a raw preflight run but no external traffic.

## Run a Read-Only Live Preflight

Start the existing Batch API port forward in another terminal:

```bash
kubectl port-forward -n default service/batch-gateway-apiserver 8001:8000
```

Then verify namespace access, selected pods, effective gate configuration, and
the Batch API without creating a job:

```bash
./workloads/run_baseline.sh --preflight-only
```

The default requires evidence of `gate_type=constant`. Use
`--expected-gate-type` to name another already deployed fast gate. Do not bypass
gate verification unless you preserve and review the effective config manually.

## Run the Batch-Only Baseline

Keep the Batch API port forward running, then submit the default deterministic
100-request slice:

```bash
./workloads/run_baseline.sh \
  --batch-size 100 \
  --max-tokens 128 \
  --timeout-seconds 1800
```

The harness uses the existing gate and deployment without Planner actions.

## Add Concurrent Online Load

Forward the existing Dynamo frontend in another terminal:

```bash
kubectl port-forward -n default service/qwen3-0-6b-batch-frontend 8000:8000
```

Run fixed streaming online requests at two requests per second for two minutes:

```bash
./workloads/run_baseline.sh \
  --batch-size 100 \
  --online-rate 2 \
  --online-duration-seconds 120 \
  --online-max-inflight 32
```

Online requests use streaming so TTFT and end-to-end latency remain separate.
The scheduler is open loop. A request that cannot acquire an in-flight slot is
recorded as `max_inflight` instead of silently changing the offered rate.

## Add Periodic Metric Snapshots

Forward an existing metric endpoint and pass an unauthenticated URL:

```bash
./workloads/run_baseline.sh \
  --metrics-url async=http://127.0.0.1:9092/metrics \
  --metrics-interval-seconds 15
```

Metric URLs must not contain credentials or query parameters. Independently, the
harness attempts read-only Kubernetes pod-proxy snapshots for selected container
ports named `metrics` and common metric ports.

## Compile a Run

Use the run ID printed by the harness:

```bash
python3 workloads/compile_run.py \
  --run-id 20260828T120000Z-baseline-a1b2c3
```

The compiler writes a new `results/compiled/<run-id>-summary/` directory. It
checks monotonic progress and duplicate online request indexes, writes CSV
projections, calculates nearest-rank latency percentiles, and records checksums
for every source artifact it used.

## Workload Contract

The immutable source defaults to the user's converted GSM8K main/test JSONL.
Each run selects records in source order and rewrites only:

- `custom_id`, to the existing stable baseline identifier for stock/standalone
  runs or a distinct `planner-native` identifier for native control;
- `model`;
- `max_tokens`;
- `temperature`;
- `stream=false`.

The message content is preserved, so GSM8K prompt lengths still vary. “Fixed
shape” here means an identical deterministic record slice and request
configuration across comparable runs, not identical token counts.

## Credential Handling

The harness does not enumerate the process environment, query Kubernetes Secret
objects, or read Hugging Face credentials. It uses a fixed non-credential
authorization placeholder accepted by this validation deployment. Captured text
is sanitized for credential-shaped values before it is written.
