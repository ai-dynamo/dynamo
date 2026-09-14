<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experiment Workloads

## Reproduce the CPU-Mocker Campaign

`run_local_mocker_impact_campaign.zsh` reproduces the five-repetition campaign
used by the Planner Gym impact report. It applies the committed CPU-Mocker and
namespace-local Prometheus manifests, alternates the committed stock and native
Planner Async overlays, runs every cell, and invokes the strict campaign
compiler. It assumes the parent Batch Gateway and Valkey deployment already
exist in the selected namespace.

### Prepare Planner Gym

The reportable campaign used the Planner Gym source at revision
`cd8e42b04aa8389d5f66d8d765513bd5308f3fdc` from
`https://github.com/NVIDIA-dev/planner`. It used Python 3.12 and AIPerf 0.12.0.
Create a clean checkout and online-backend environment next to the Dynamo
checkout:

```bash
export DYNAMO_ROOT="$(git rev-parse --show-toplevel)"
export WORKAREA="$(dirname "${DYNAMO_ROOT}")"
export PLANNER_GYM_REPOSITORY="${WORKAREA}/planner"
export PLANNER_GYM_ROOT="${PLANNER_GYM_REPOSITORY}/gym"
export PLANNER_GYM_REVISION=cd8e42b04aa8389d5f66d8d765513bd5308f3fdc

git clone https://github.com/NVIDIA-dev/planner.git "${PLANNER_GYM_REPOSITORY}"
git -C "${PLANNER_GYM_REPOSITORY}" checkout --detach \
  "${PLANNER_GYM_REVISION}"
python3.12 -m venv "${PLANNER_GYM_ROOT}/.venv"
uv pip install --python "${PLANNER_GYM_ROOT}/.venv/bin/python" \
  -e "${PLANNER_GYM_ROOT}" 'aiperf==0.12.0'

test "$(git -C "${PLANNER_GYM_REPOSITORY}" rev-parse HEAD)" = \
  "${PLANNER_GYM_REVISION}"
test -z "$(git -C "${PLANNER_GYM_REPOSITORY}" status --short)"
"${PLANNER_GYM_ROOT}/.venv/bin/aiperf" --version
```

The campaign driver rejects a different or dirty Planner Gym checkout. Set
`PLANNER_GYM_REVISION` explicitly only when starting a new campaign that will
record and report a different source revision.

### Generate the Frozen Trace

Generate the 1,080-request, 180-second Mooncake trace with every byte-affecting
parameter explicit:

```bash
export DATA_DIR=/absolute/path/to/campaign-inputs
export TRACE="${DATA_DIR}/planner-gym-impact-low-high-low-5-8-5.jsonl"
mkdir -p "${DATA_DIR}"

"${DYNAMO_ROOT}/.venv/bin/python" \
  "${DYNAMO_ROOT}/examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/workloads/generate_planner_gym_impact_trace.py" \
  --output "${TRACE}" \
  --phase low-1:60:5 \
  --phase high:60:8 \
  --phase low-2:60:5 \
  --seed 20260918 \
  --input-tokens 512 \
  --output-tokens 64 \
  --input-jitter-tokens 64 \
  --output-jitter-tokens 8 \
  --max-model-length 4096 \
  --block-size 512 \
  --prefix-mode none \
  --shared-prefix-blocks 0
```

The expected trace SHA-256 is
`b8dcdd7c630831dbd37f4d49a4d7f347c256941626f347d737ee50162d9ad314`.
The report also used a converted OpenAI Batch JSONL dataset with SHA-256
`488c0fb6b64ce349e8d8da1ea001ccec4f8339194949aec7a03b6fcddaaeab8f`.
Set `DATASET` to that immutable file. The driver checks both hashes before it
changes cluster state. A different trace or dataset defines a new campaign;
set `TRACE_SHA256` or `DATASET_SHA256` to the new recorded values and do not
compare it as another repetition of the frozen campaign.

### Run All 15 Cells

Load or publish one immutable Dynamo image containing the frontend, Mocker, and
Planner source under test. The Mocker manifest defaults to an `emptyDir` model
cache and no image pull secret. To use existing cluster resources, set
`MODEL_CACHE_CLAIM` and `IMAGE_PULL_SECRET`; the renderer replaces the defaults
in the generated manifest without changing the checked-in template. The
`hf-token-secret` prerequisite is intentionally not optional: this campaign
preserves the parent Batch backend contract, and the frontend and Mocker use
the secret for authenticated Hub access when populating a cold model cache.
Create that secret in `NAMESPACE` before starting the driver.

```bash
export KUBE_CONTEXT=your-kube-context
export NAMESPACE=your-namespace
export DYNAMO_IMAGE=your-registry.example/dynamo:immutable-campaign-build
export ASYNC_IMAGE=your-registry.example/llm-d-async:immutable-campaign-build
export DATASET=/absolute/path/to/gsm8k-main-train.jsonl

"${DYNAMO_ROOT}/examples/deployments/llm-d-batch-gateway/planner-scheduling/experiment/workloads/run_local_mocker_impact_campaign.zsh"
```

The optional cluster-resource form is:

```bash
export MODEL_CACHE_CLAIM=your-model-cache-pvc
export IMAGE_PULL_SECRET=your-registry-pull-secret
```

Current captures preserve standard Docker, containerd, and CRI-O SHA-256
`imageID` values, so they normally need no attestation. For a legacy artifact
or a nonstandard runtime whose collector sanitized those values, the driver
inspects the start/end image inventories that the compiler will actually
consume. If, and only if, the exact `DYNAMO_IMAGE` reference has a sanitized
ID, provide the resolved imageID by itself (not `IMAGE_REF=IMAGE_ID`):

```bash
export DYNAMO_IMAGE_ID_ATTESTATION='docker-pullable://your-registry.example/dynamo@sha256:...'
```

The driver maps that value to `DYNAMO_IMAGE` for the compiler. It does not pass
an unused attestation when those inventories contain resolved IDs, and it
fails closed if any other image reference has a sanitized ID.

The driver owns local ports 18000, 18001, 18002, and 19091 by default. Override
them with `FRONTEND_PORT`, `BATCH_PORT`, `ASYNC_METRICS_PORT`, and
`PROMETHEUS_PORT`. It preserves its rendered manifests, per-cell logs, and
`runs.tsv` in the printed campaign-state directory. Set `CAMPAIGN_STATE_DIR`
or `COMPILED_OUTPUT` before launch to choose stable output locations.

The chronological arm order is frozen as follows:

1. Repetition 1: online-only, stock, Planner-native.
2. Repetition 2: stock, Planner-native, online-only.
3. Repetition 3: Planner-native, online-only, stock.
4. Repetition 4: online-only, stock, Planner-native.
5. Repetition 5: stock, Planner-native, online-only.

After all 15 cells pass, the driver calls
`compile_planner_gym_batch_impact.py` with every run directory in that order,
`--minimum-repetitions 5`, `--cv-threshold-percent 5`, and
`--minimum-batch-overlap-fraction 0.95`. Keep every attempted cell, including a
failed one; chronological ordinal pairing is valid only when attempts are not
silently removed.

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
uv pip install --python ../../../../../.venv/bin/python 'redis>=6.2,<9'

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
`ap.workerPools[0]`. This worker-pool admission boundary limits batch requests
entering `dynamo-batch`. A separate `prometheus-query` gate remains on the
`queuesConfig` entry so the broker does not dequeue before the frontend is
ready. The pool ID in Prometheus, the policy input, and the Redis key must all
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

For any new harness run, choose the deployed namespace and prepare an OpenAI
Batch JSONL dataset outside this repository. Each record must contain
`custom_id`, `method`, `url`, and an OpenAI request `body`; the harness validates
and normalizes the selected records without modifying the source file. With the
POC deployment, run from the experiment root while the Batch API port-forward
is active:

```bash
export NAMESPACE=your-namespace
export KUBE_CONTEXT=your-kube-context
export DATASET=/absolute/path/to/gsm8k-main-test.jsonl
: "${NAMESPACE:?set NAMESPACE to the deployed example namespace}"
: "${KUBE_CONTEXT:?set KUBE_CONTEXT to the target cluster context}"
test -r "${DATASET}" || { echo "dataset is not readable: ${DATASET}" >&2; exit 1; }

./workloads/run_baseline.sh \
  --run-kind planner-native \
  --native-planner-configmap qwen3-0-6b-batch-planner-config \
  --native-planner-pod-name-regex 'qwen3-0-6b-batch-planner.*planner' \
  --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" \
  --dataset "${DATASET}" \
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

If you are starting here rather than from the native-run section, set the same
required inputs first:

```bash
export NAMESPACE=your-namespace
export DATASET=/absolute/path/to/gsm8k-main-test.jsonl
: "${NAMESPACE:?set NAMESPACE to the deployed example namespace}"
test -r "${DATASET}" || { echo "dataset is not readable: ${DATASET}" >&2; exit 1; }
```

## What the Harness Records

Each invocation creates `results/raw/<UTC-run-id>/` before it does any work. A
run contains:

- exact normalized GSM8K Batch JSONL and its source/output checksums;
- file upload, Batch creation, progress, terminal state, and retrieved result
  files;
- optional per-online-request HTTP status, Time To First Token (TTFT), latency,
  and token usage;
- selected pod specifications, image references, runtime-reported image IDs,
  referenced ConfigMaps, events, current and previous logs, Kubernetes versions,
  and pod-proxy metric snapshots;
- optional periodic snapshots from explicit unauthenticated metric URLs;
- sanitized stdout, stderr, every captured command's stdout/stderr, and each exit
  code.

Native Planner runs additionally store the expected and observed Planner pods,
mounted/captured ConfigMap, Planner image-reference/image-ID records, per-log
command exit codes, and in-run decision-log match count under
`kubernetes.{start,end}.native_planner` in `metadata.json`.

The pod selector defaults to names containing `batch-gateway`,
`async-dispatch`, or `qwen3-0-6b-batch`. Override `--pod-name-regex` if the
deployed names differ.

## Canonical Autonomous Zero-Worker Run

Canonical run `20260828T213549Z-planner-native-1e3ff8` used one explicit
pre-run setup mutation to establish the worker DGDSA at zero:

```bash
kubectl patch dgdsa \
  qwen3-0-6b-batch-worker \
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
    results/raw/20260828T213549Z-planner-native-1e3ff8/autonomous-scale-evidence \
  --worker-component VllmDecodeWorker \
  --adapter-name qwen3-0-6b-batch-vllmdecodeworker \
  --expected-lease-duration-seconds 60
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
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
  --preflight-only \
  --skip-cluster-preflight \
  --skip-api-preflight
```

This creates a raw preflight run but no external traffic.

## Run a Read-Only Live Preflight

Start the existing Batch API port forward in another terminal:

```bash
kubectl port-forward -n "${NAMESPACE}" service/batch-gateway-apiserver 8001:8000
```

Then verify namespace access, selected pods, effective gate configuration, and
the Batch API without creating a job:

```bash
./workloads/run_baseline.sh \
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
  --preflight-only
```

The default requires evidence of `gate_type=constant`. Use
`--expected-gate-type` to name another already deployed fast gate. Do not bypass
gate verification unless you preserve and review the effective config manually.

## Run the Batch-Only Baseline

Keep the Batch API port forward running, then submit the default deterministic
100-request slice:

```bash
./workloads/run_baseline.sh \
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
  --batch-size 100 \
  --max-tokens 128 \
  --timeout-seconds 1800
```

The harness uses the existing gate and deployment without Planner actions.

## Add Concurrent Online Load

Forward the existing Dynamo frontend in another terminal:

```bash
kubectl port-forward -n "${NAMESPACE}" service/qwen3-0-6b-batch-frontend 8000:8000
```

Run fixed streaming online requests at two requests per second for two minutes:

```bash
./workloads/run_baseline.sh \
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
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
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
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
