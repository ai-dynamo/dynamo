<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Fault Tolerance Test Suite

As a large scale distributed inference serving framework in addition
to providing high throughput and low latency, Dynamo needs to
provide fault detection, resilency, and quick recovery in the face of
unforseen failures. In order to test Dynamo we are developing a test
suite to inject and measure the impact of different types of failure
conditions.

The suite has two layers, both pytest-driven and both running against a
real `DynamoGraphDeployment` in a Kubernetes namespace:

1. **Event-based harness** (current) — `tests/fault_tolerance/deploy/scenario.py`
   plus `events.py`. Each test composes a deployment + a list of timed
   events (`StartLoad`, `DeletePod`, `TerminateProcess`,
   `NetworkPartition`, …) and emits per-test FT reports plus a
   combined comparison table.
2. **Legacy scenario-driven harness** — `test_deployment.py` parametrised
   by the table in `scenarios.py`. Documented further down under
   *Legacy scenario harness*.

New tests should target the event-based harness; everything below up
to *Legacy scenario harness* describes that path.

## Event-based test harness

### Component architecture

```mermaid
graph TD
    subgraph Test["Test (pytest)"]
        spec[DeploymentSpec.from_backend]
        events[events: WaitForModelReady, StartLoad, fault, StopLoad, ...]
        checks[checks: MinRequests, MaxErrors, ZeroErrors, ...]
        reports[reports: FaultToleranceReport, ErrorBreakdownReport]
    end
    Test --> Scenario[run_scenario]
    Scenario --> MD[ManagedDeployment]
    Scenario --> Load[ManagedLoad]
    Scenario --> EventLoop[event.timed_execute]
    EventLoop --> MD
    EventLoop --> Load
    EventLoop --> NPClass[NetworkPartition]
    NPClass --> NetPolicy[NetworkPolicy + conntrack flusher pod]
    MD --> K8s[Kubernetes API]
    MD --> PVC[(RWX log PVC)]
    Load --> AIPerf[aiperf Job in-cluster]
    AIPerf --> Frontend[Frontend pod]
    AIPerf --> ServerMetrics[server_metrics_export.json &lt;-- aiperf scrapes /metrics ]
    PVC --> LogExtract[end-of-run log extract Job]
    LogExtract --> Outputs[test_outputs/test_name/...]
    reports --> Outputs
    Combined[scripts/aggregate-ft-reports.sh] --> Outputs
```

### Components

| Module | Role |
| --- | --- |
| `tests/utils/managed_deployment.py` (`ManagedDeployment`) | Async context manager. Materialises the log-collection PVC, applies the `DynamoGraphDeployment`, waits for `Ready=True`, and tears everything down on exit. Captures per-pod manifests, container logs (via PVC tee), and `/metrics` snapshots. |
| `tests/utils/managed_load.py` (`ManagedLoad`, `LoadConfig`) | Spawns an in-cluster aiperf load `Job` against the frontend service. aiperf's `--server-metrics` flag is **on by default** so the run also writes `server_metrics_export.json` (timestamped Prometheus scrape of the frontend) alongside `profile_export_aiperf.json`. |
| `tests/fault_tolerance/deploy/events.py` | Timed events (load, faults, waits) implementing a common `Event.execute()` template with `started_at` / `ended_at` brackets that reports use to slice metrics around faults. |
| `tests/fault_tolerance/deploy/checks.py` | Pluggable assertions (`MinRequests`, `MaxErrors`, `LoadStopped`, `LoadCompleted`, `ZeroErrors`). Run after reports so a failed assertion never blocks report generation. |
| `tests/fault_tolerance/deploy/reports.py` | Markdown + JSON sidecar reports. `FaultToleranceReport` produces the per-fault Timing / Counts / Latency table; `ErrorBreakdownReport` summarises aiperf error types. |
| `tests/fault_tolerance/deploy/scenario.py` (`run_scenario`) | Glue. Drives `ManagedDeployment` + events + reports + checks. |

### Data flow during a run

```mermaid
graph LR
    subgraph cluster["dynamo-test namespace"]
        Aiperf[aiperf Job]
        FE[Frontend pod]
        W[Worker pod]
        PVC[(RWX log PVC<br/>service_logs/)]
    end
    subgraph dyn-system["dynamo-system namespace"]
        ETCD[(etcd)]
        NATS[(NATS)]
    end
    subgraph hostnet["host network privileged pod"]
        FlushPod[netshoot conntrack -D]
    end

    Aiperf -->|HTTP /v1/chat/completions| FE
    FE <-->|TCP request plane| W
    W -->|register / events| ETCD
    W -->|publish / events| NATS
    Aiperf -->|/metrics scrape every ~400ms| FE
    Aiperf -->|profile_export_aiperf.json| Outputs[test_outputs/]
    Aiperf -->|server_metrics_export.json| Outputs
    FE -->|tee stdout/stderr| PVC
    W -->|tee stdout/stderr| PVC
    PVC -->|end-of-run extract Job| Outputs
    FlushPod -.->|conntrack -D for source/target| Kernel[(node kernel<br/>conntrack table)]
```

The TCP request plane (`RequestPlaneMode::Tcp`, default) is direct
pod-to-pod; NATS / etcd carry events and discovery, not per-request
traffic. That distinction matters for partition design — the
NetworkPolicy must target the request-plane edge, not the NATS edge.

### Lifecycle

```mermaid
sequenceDiagram
    participant Pytest
    participant Scenario as run_scenario
    participant MD as ManagedDeployment
    participant K8s as Kubernetes API
    participant Worker
    participant Frontend
    participant Load as ManagedLoad (aiperf)

    Pytest->>Scenario: events, checks, reports, deployment_spec
    Scenario->>MD: __aenter__
    MD->>K8s: scrub namespace, create RWX log PVC
    MD->>K8s: apply DynamoGraphDeployment
    MD->>K8s: poll until Ready=True
    Scenario->>Scenario: WaitForModelReady (port-forward to Frontend, poll /v1/models)
    Scenario->>Load: StartLoad (aiperf Job, scrapes Frontend /metrics)
    Load->>Frontend: HTTP requests
    Frontend->>Worker: TCP request plane
    Scenario->>K8s: fault event (DeletePod / TerminateProcess / NetworkPartition / ...)
    Scenario->>Scenario: WaitForRecovery / Wait
    Scenario->>Load: StopLoad
    Scenario->>MD: __aexit__
    MD->>K8s: extract logs from PVC, capture pod manifests + /metrics, delete deployment
    Scenario->>Scenario: generate reports (Markdown + JSON sidecars)
    Scenario->>Scenario: run checks (assertions)
```

Reports run **before** checks so an assertion failure never suppresses
the run's data. Per-test artefacts land at
`<DYN_TEST_OUTPUT_PATH>/<test_name>/`:

```
test_outputs/test_network_partition_vllm/
├── frontend/
│   ├── <pod>.yaml
│   └── <pod>_<ts>.log              # tee'd from PVC
├── vllmdecodeworker/
│   ├── <pod>.yaml
│   └── <pod>_<ts>.log
├── load/
│   ├── profile_export_aiperf.json  # request-level latencies, errors
│   ├── server_metrics_export.json  # aiperf's Frontend /metrics scrape
│   ├── server_metrics_export.csv
│   ├── inputs.json
│   └── aiperf.log
├── fault_tolerance_report.md       # per-test Timing/Counts/Latency
├── fault_tolerance_report.json     # machine-readable sidecar
└── error_breakdown_report.md       # aiperf error types
```

### Supported fault events

| Event | What it does | Notes |
| --- | --- | --- |
| `DeletePod(services=[...])` | `pod.delete(force=...)`. Worker / frontend / planner — anything with the right `nvidia.com/dynamo-component` label. | Snapshots the pod manifest + `/metrics` `before_delete` so post-mortem comparisons are possible. |
| `TerminateProcess(services, process_name, signal)` | Exec into the pod and `kill -<signal> <pid>` of a matching process (`dynamo.vllm`, `VLLM::EngineCore`, `sglang::scheduler`, …). | The kubelet restart-in-place behaviour depends on the process being PID 1 vs. a child; subprocess kills are no-ops at the container level (documented in *Backend-Specific Validations* further below). |
| `StallProcess(services, process_name, duration=None)` | `SIGSTOP` the matching process; sleep `duration` seconds (or until `stop()` if `duration` is None); `SIGCONT` to resume. | Models a "hung worker": the process stays alive, TCP sockets and conntrack entries are preserved, kubelet doesn't notice — but it stops servicing requests. Exercises frontend timeout / health-check behaviour rather than crash recovery. |
| `NetworkPartition(source, target, duration=None)` | Applies a `NetworkPolicy` selecting `target` pods and denying ingress from `source` pods, then schedules a privileged hostNetwork `conntrack -D` flusher Pod on the source pod's node so already-established TCP sockets are severed (k8s `NetworkPolicy` is connection-tracked; without the flush a pooled socket trivially survives). Heals on `stop()` or after `duration` seconds. | Requires a CNI that enforces NetworkPolicy (k3s kube-router, Calico, Cilium). The flusher uses `localhost:32000/netshoot:latest` by default; override with `DYN_CONNTRACK_IMAGE`. |
| `RollingUpgrade(services, env_var=...)` | Stamps a unique env var on each named service then publishes the change so the operator rolls the underlying Deployments. | Drives the "redeploy under load" scenario. |
| `RunCommand(services, command)` | Exec arbitrary command in matched pods. | Escape hatch for one-off injections (e.g., `pkill -STOP`, `nvidia-smi -i 0 -r`). |
| `Wait(duration)`, `WaitForRecovery(timeout)`, `WaitForModelReady(timeout)`, `WaitForLogPattern(pattern, timeout)` | Time / state synchronisation between events. | `WaitForModelReady` port-forwards to the frontend and polls `/v1/models` until the deployed model is registered, so vllm/sglang model load (which can run tens of seconds beyond pod-Ready) doesn't bleed into the load window. |
| `StartLoad(load_config)`, `StopLoad`, `WaitForLoadCompletion` | aiperf-driven load lifecycle. | aiperf's `--server-metrics` is on by default; aiperf scrapes the Frontend's `/metrics` for the duration of the run and writes timestamped stats per metric. |

### Per-fault state transitions

These diagrams show what the worker pod (or the worker→frontend
connection) goes through during each fault mode. The "Healthy"
state is "pod Ready, model registered, frontend can route requests."

#### DeletePod

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> PodTerminating: DeletePod (force)
    PodTerminating --> PodGone: grace period elapsed
    PodGone --> PodPending: operator reschedules
    PodPending --> ContainerCreating: pulling image
    ContainerCreating --> Running: container start
    Running --> ModelLoading: vllm engine init + KV warmup
    ModelLoading --> Healthy: model re-registered with frontend
```

The frontend deregisters the worker on disconnect and re-discovers it
when it reappears in etcd. Failed-after errors are dominated by
`Model not found` 404s during the re-registration window.

#### TerminateProcess (engine kill)

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> ProcessKilled: kill -SIGKILL dynamo.vllm
    ProcessKilled --> ContainerExited: PID 1 exits
    ContainerExited --> ContainerRestarting: kubelet restartPolicy
    ContainerRestarting --> Running: same pod, new container
    Running --> ModelLoading: vllm reload
    ModelLoading --> Healthy: model re-registered
```

Pod IP is **preserved** (kubelet restarts the container in place), so
the frontend's TCP socket fails fast on the existing connection rather
than waiting for re-discovery — but the model must still reload, so
recovery time is comparable to `DeletePod`.

#### StallProcess (engine hang)

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> SIGSTOPSent: kill -SIGSTOP dynamo.vllm
    SIGSTOPSent --> Stopped: process suspended; pod alive, sockets open
    Stopped --> RequestsBacklogged: in-flight requests stall on the worker; new requests queue at frontend
    RequestsBacklogged --> ClientTimeouts: aiperf request_timeout fires; 5xx returned
    ClientTimeouts --> SIGCONTSent: duration expires (event.stop)
    SIGCONTSent --> Resumed: process scheduled again; queued work drains
    Resumed --> Healthy
```

The pod, container, TCP sockets, and conntrack entries all persist
unchanged through the stall — kubelet doesn't notice and there is no
container restart. Distinct from `TerminateProcess` (crash) and
`NetworkPartition` (network cut): only `StallProcess` exercises the
"worker accepted my request and went silent" failure mode.

#### NetworkPartition + conntrack flush

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> Loading: StartLoad (aiperf concurrency=4)
    Loading --> SteadyState: in-flight≈4, queue=0
    SteadyState --> PolicyApplied: NetworkPolicy applied (source=Frontend, target=Worker)
    PolicyApplied --> ConntrackFlushed: hostNetwork flusher Pod runs conntrack -D
    ConntrackFlushed --> SocketsSevered: established TCP dies; new connect attempts denied
    SocketsSevered --> QueueBuilds: HTTP queue grows; in-flight pinned at 4 retrying
    QueueBuilds --> WorkerDeregistered: frontend marks model unavailable
    WorkerDeregistered --> NotFoundErrors: 404 'Model not found' to clients
    NotFoundErrors --> PolicyLifted: duration expires (event.stop)
    PolicyLifted --> Reconnecting: new TCP handshake succeeds
    Reconnecting --> SteadyState: queue drains; throughput burst (3-4× baseline)
```

Without the conntrack flush, the system stays in `SteadyState` for
the full partition window — the connection-tracked TCP socket survives
the new policy and requests keep flowing. The flush is what makes the
partition observable as a fault.

### Example test

```python
async def test_worker_kill_vllm(namespace, image, skip_service_restart, storage_class):
    spec = DeploymentSpec.from_backend("vllm", "agg")
    served_model = spec["VllmDecodeWorker"].model

    await run_scenario(
        deployment_spec=spec,
        events=[
            WaitForModelReady(timeout=600),
            StartLoad(load_config=LoadConfig(
                model_name=served_model,
                tokenizer=served_model,
                duration_minutes=2,
                concurrency=4,
                input_tokens_mean=128,
                output_tokens_mean=32,
            )),
            Wait(duration=20),
            DeletePod(services=["VllmDecodeWorker"]),
            WaitForRecovery(timeout=300),
            Wait(duration=20),
            StopLoad(),
        ],
        checks=[MinRequests(min_count=20), MaxErrors(max_errors=1_000_000)],
        reports=[FaultToleranceReport(), ErrorBreakdownReport()],
        namespace=namespace,
        image=image,
        test_name="test_worker_kill_vllm",
        skip_service_restart=skip_service_restart,
        storage_class=storage_class,
    )
```

### Combined comparison report

`scripts/aggregate-ft-reports.sh` (in the dev vault) walks the output
tree and merges every `fault_tolerance_report.json` + `error_breakdown_report.md`
+ `server_metrics_export.json` into one Markdown comparison file. This
is what gets shipped to slides.

```text
# Fault-tolerance comparison — 4 scenarios

## Timing across scenarios

| Scenario · Failure                                                          | Startup (s) | Recovery (s) |
|:---------------------------------------------------------------------------|------------:|:-------------|
| test_sanity_vllm · none                                                    |       81.59 | N/A          |
| test_worker_kill_vllm · DeletePod(VllmDecodeWorker)                        |       81.03 | 81.53        |
| test_engine_kill_vllm · TerminateProcess(VllmDecodeWorker)                 |       81.63 | 94.04        |
| test_network_partition_vllm · NetworkPartition(Frontend->VllmDecodeWorker) |      191.64 | 20.00        |

## Request counts (pre/post fault)

| Scenario · Failure                                                          | Success Before | Failed Before | Success After | Failed After |
|:---------------------------------------------------------------------------|---------------:|--------------:|--------------:|-------------:|
| test_sanity_vllm · none                                                    |             10 |             0 |   N/A         |   N/A        |
| test_worker_kill_vllm · DeletePod(VllmDecodeWorker)                        |            190 |             0 |   408         | 271457       |
| test_engine_kill_vllm · TerminateProcess(VllmDecodeWorker)                 |            207 |             0 |   279         | 312521       |
| test_network_partition_vllm · NetworkPartition(Frontend->VllmDecodeWorker) |            144 |             0 |   609         |   7070       |

## Latency (avg ms)

| Scenario · Failure                                                          | Before (ms) | After (ms) |
|:---------------------------------------------------------------------------|------------:|:-----------|
| test_sanity_vllm · none                                                    |      182.36 | N/A        |
| test_worker_kill_vllm · DeletePod(VllmDecodeWorker)                        |      186.13 | 180.59     |
| test_engine_kill_vllm · TerminateProcess(VllmDecodeWorker)                 |      185.10 | 184.13     |
| test_network_partition_vllm · NetworkPartition(Frontend->VllmDecodeWorker) |      190.93 | 186.60     |

## Frontend metrics during run (from aiperf /metrics scrape)

| Scenario · Failure                                                          | Inflight max | Inflight avg | Queued max | Queued p95 | Disconnected (max) |
|:---------------------------------------------------------------------------|-------------:|-------------:|-----------:|-----------:|-------------------:|
| test_sanity_vllm · none                                                    |            2 |          0.8 |          0 |          0 |                  0 |
| test_worker_kill_vllm · DeletePod(VllmDecodeWorker)                        |            4 |         0.96 |          4 |          0 |                  0 |
| test_engine_kill_vllm · TerminateProcess(VllmDecodeWorker)                 |            4 |         0.73 |          1 |          0 |                  0 |
| test_network_partition_vllm · NetworkPartition(Frontend->VllmDecodeWorker) |            4 |         3.60 |          4 |          3 |                  0 |
```

The Frontend-metrics section comes from aiperf's built-in
`--server-metrics` collector, which scrapes the inference endpoint's
`/metrics` for the full duration of the run; `LoadConfig` doesn't need
to opt in. Each scenario's `error_breakdown_report.md` is also
inlined verbatim in the combined output so the dominant error type per
fault (404 "Model not found" during partition, "Connection refused"
during conntrack flush, "TCP request timed out" during pod kill) is
visible at a glance.

### Current gaps

What this harness covers today, and what it doesn't:

#### Fault types
- ✅ **Pod delete** (`DeletePod`) — worker, frontend, planner.
- ✅ **Process kill** (`TerminateProcess`) — engine cores, schedulers, detokenizers, …
- ✅ **Binary network partition** (`NetworkPartition` + conntrack flush)  — same-namespace, source pod ↛ target pod TCP cut. Mid-load partition is visible (queue grows, 404 storms, recovery on lift).
- ❌ **Partial-network faults** — packet loss, latency, jitter, bandwidth caps. Requires Chaos Mesh; pending the fault-injection-service work below.
- ❌ **Cross-namespace partitions** — partition `Worker → NATS` (in `dynamo-system`) or `Frontend → etcd`. The current `NetworkPartition` only matches by `nvidia.com/dynamo-component` within the test namespace.
- ❌ **GPU faults** — XID errors, ECC errors, CUDA toggle, kernel-launch interception, NVSentinel-driven recovery. Coming via fault-injection-service `oviya/fault-injection-api/gpu`.
- ❌ **Disk / IO faults** — disk pressure, slow IO, ENOSPC on log PVC, etcd disk full.
- ❌ **Memory pressure / OOM** — no controlled OOM injection.
- ❌ **Time-skew / clock drift** — no NTP-misbehaviour scenarios.

#### Topology coverage
- ✅ **Aggregated** (`agg`) — single Frontend + single Worker. All four fault modes proven.
- ❌ **Disaggregated** (`disagg-prefill-... -decode-...`) — mocker tests exist in `scenarios.py` (legacy), but no event-based scratch tests for prefill+decode topologies.
- ❌ **Multi-replica** — every test uses `replicas=1`. Tests like "kill 1 of 3 decode workers and assert no error spike" are not written.
- ❌ **Multi-model / multi-deployment** — single `DynamoGraphDeployment` per test. No cross-deployment fault scenarios.
- ❌ **Router / KV-aware-router faults** — KV router has metric publishing over NATS; no test asserts behaviour when those events drop.
- ❌ **Planner faults** — planner is rarely deployed in fault tests.

#### Load profile coverage
- ✅ **Steady-state low-concurrency** (`concurrency=4`, ~5 min runs).
- ❌ **Bursty / variable-rate load** — aiperf supports `--request-rate` shapes; not yet exercised.
- ❌ **Long-running soak** — no overnight tests; today's runs are minutes, not hours.
- ❌ **Realistic prompt distributions** — synthetic fixed-length prompts only; no traces, no Mooncake replay.
- ❌ **Streaming-only / completion-only mix** — `streaming=True` is hardcoded; no test asserts non-streaming behaviour under faults.
- ❌ **Per-request timeout sensitivity sweeps** — `request_timeout_seconds` is set per-test, not parametrised.

#### Observability / assertions
- ✅ **Frontend `/metrics` time-series** via aiperf `--server-metrics` (in-cluster scrape; ~400 ms cadence).
- ✅ **Worker `/metrics` and stdout** captured to PVC + extracted at exit.
- ✅ **Per-fault Markdown + JSON sidecars** (`fault_tolerance_report.{md,json}`), aiperf error breakdown (`error_breakdown_report.md`), and a multi-test combined comparison via `scripts/aggregate-ft-reports.sh`.
- ❌ **Time-correlated single timeline** — frontend metrics, worker engine logs, partition events, and aiperf request timeline are in separate files; no overlay tool.
- ❌ **SLA assertions** — `MaxErrors=1_000_000` is the practical default for vllm fault tests because aiperf retries 30 k+ on a closed socket. Real per-mode SLA bounds (e.g. "p99 latency ≤ 1.2× baseline within 60 s of recovery") need richer checks.
- ❌ **Recovery quality assertions** — recovery time is captured but not bounded by mode (target: pod-kill ≤ 90 s, partition recovery ≤ 5 s post-lift).
- ⚠️ **Misleading metric**: `dynamo_frontend_model_migration_total` increments on the *initial-attempt* migratable failure even when `migration_limit=0` (no retry actually performed). Reported during the May 5 partition validation; needs renaming or a guard upstream.
- ⚠️ **Framework `_get_pod_metrics` snapshot has a sync/async dual-method bug** — the cleanup-time scrape lands in a capitalized `Frontend/` directory while everything else is lowercased, so the snapshot file is silently lost. aiperf's continuous scrape compensates but the framework's own snapshot is unreliable. Tracked as a follow-up.

#### Infrastructure constraints
- ⚠️ **Privileged + hostNetwork pod** required for `NetworkPartition` (the `conntrack -D` flusher). Won't run on clusters with cluster-wide PSA or OPA blocking those. Mitigations: namespace exception, vcluster, or move to the fault-injection-service (which centralises the privileged daemon).
- ⚠️ **In-cluster image registry** — defaults assume `localhost:32000` (k3s + local docker registry). On other clusters, override `DYN_CONNTRACK_IMAGE` and the test image references.
- ⚠️ **CNI requirement** — k3s kube-router, Calico, or Cilium needed for NetworkPolicy enforcement. Plain Flannel silently no-ops the policy.

### Future direction — Fault Injection Service

The current `NetworkPartition` event hits two structural limits:

1. It only does binary cuts. Packet loss, latency injection, jitter,
   bandwidth caps require Chaos Mesh.
2. It needs cluster privileges to schedule the conntrack flush Pod.
   On clusters with cluster-wide PSA / OPA policies that ban
   `privileged` + `hostNetwork`, the framework can't run as-is — see
   the *vcluster* and *namespace exception* notes in the deploy
   playbook.

The follow-on `Fault Injection Service` work (Oviya Seeniraj —
`oviya/fault-injection/dev` and friends on
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)) replaces both
hops with a cluster-installed service:

- **API service** (`tests/fault_tolerance/hardware/fault-injection-service/api-service`) —
  HTTP API installed once into `fault-injection-system`. Tests POST
  fault specs at it instead of creating raw NetworkPolicies.
- **Network fault-injector DaemonSet** (`agents/network-fault-injector/`) —
  per-node agent with the privileges the API does not. The API
  delegates flush / partition / packet-loss work to the agent over a
  signed channel.
- **Chaos Mesh integration** for partial-loss and delay scenarios
  (`NetworkChaos` CRs are issued by the API).
- **GPU faults** (`oviya/fault-injection-api/gpu`,
  `oviya/fault-injection/cuda-injection-tools`,
  `oviya/fault-injection/xid79-nvsentinel-test`) — CUDA toggle, kernel
  launch interception, NVSentinel-driven recovery from XID 79.
- **Python client** (`oviya/fault-injection/client-library`) — drop-in
  replacement for `NetworkPartition` and friends. Test code imports
  the client; everything else moves into the API.
- Pre-built example tests in `examples/`:
  `test_partition_worker_to_nats.py`,
  `test_partition_frontend_to_nats.py`,
  `test_partition_worker_to_frontend.py`,
  `test_specific_pod_to_pod_blocking.py`,
  `test_nats_packet_loss_50_percent.py`.

Once that work merges, the event-based harness should re-implement
`NetworkPartition`, the conntrack flush, and any future GPU / packet-
loss fault as thin wrappers around the client library — keeping the
scenario / event abstraction here while moving the privileged
machinery and Chaos Mesh dependency out of the test process.

## Legacy scenario harness

What follows is the original parametrised scenario design driving
`test_deployment.py` via `scenarios.py`. New tests should prefer the
event-based harness above; this section remains for callers that
still depend on the legacy parameter sweep.

## Test Architecture

The fault tolerance test suite is designed as a set of pytest
configurations that launch typical dynamo deployments in a Kubernetes
environment and then inject failures by terminating processes or
pods. To test the recovery time and impact of failures, a set number of
clients are launched in parallel using **AI-Perf (aiperf)** for load generation.
Each client sends synthetic requests with configurable token patterns.
Log files are stored for each pod as well as for each client and inspected
using a post-processing script that parses AI-Perf metrics.

> [!NOTE]
> Test pass / failure is not an indication of SLA for recovery or resilience
> It only indicates is the test was executed and data was collected

###  Test Sequence Diagram

```mermaid
sequenceDiagram
    participant Tester as Test Runner
    participant DynamoKubernetes as Dynamo Kubernetes Platform
    participant DynamoDeployment as Dynamo Deployment
    participant Clients as Client Processes
    participant Logs as Log Files
    participant Parser as Results Parser

    Tester->>DynamoKubernetes: Deploy Dynamo graph (Frontend + Workers)
    DynamoKubernetes->>DynamoDeployment: Create pods/services (Frontend, Workers)
    DynamoDeployment->>Tester: Signal ready (all pods running)
    Tester->>Clients: Launch clients (concurrent requests)
    Clients->>DynamoDeployment: Send requests via Port Forwarding to Frontend
    Tester->>DynamoDeployment: Inject failures (delete pods/terminate processes)
    Clients->>Logs: Log request results to files
    DynamoDeployment->>Logs: Save pod logs
    Tester->>DynamoKubernetes: Teardown deployment (delete pods/services)
    DynamoKubernetes->>DynamoDeployment: Delete resources
    Tester->>Parser: Parse logs
    Parser->>Tester: Generate results table
```

### Test Scenarios

The test suite is organized around three core components: **Deployments**, **Client Load**, and **Failures**. Each scenario combines these elements to simulate fault conditions and measure system resilience.

#### Deployments

Deployments represent specific graphs that are deployed using the Dynamo Kubernetes Platform.

Below are some representative examples of the generated scenarios:

| Example Scenario Name                         | Backend | Type   | TP | DP | Description                                             |
|-----------------------------------------------|---------|--------|----|----|---------------------------------------------------------|
| `vllm-agg-tp-1-dp-1`                          | vllm    | agg    | 1  | 1  | Basic aggregated worker.                                |
| `vllm-agg-tp-1-dp-2`                          | vllm    | agg    | 1  | 2  | Aggregated worker with Data Parallelism.                |
| `sglang-agg-tp-4-dp-1`                        | sglang  | agg    | 4  | 1  | Aggregated SGLang worker with Tensor Parallelism.       |
| `sglang-disagg-prefill-tp-2-decode-tp-2-dp-1`   | sglang  | disagg | 2  | 1  | Disaggregated SGLang workers with Tensor Parallelism.   |

The full test matrix is generated from these parameters, creating comprehensive test coverage across all configurations.

#### Client Load (AI-Perf Configuration)

- **Load Generator**: AI-Perf (`aiperf`) with synthetic token generation
- **Concurrent Clients**: 10 clients by default, adjustable per scenario
- **Requests per Client**: 150 requests per client (configurable)
- **Input/Output Token Configuration**:
  - Input tokens: mean=100, stddev=0 (consistent length)
  - Output tokens: mean=100, stddev=0 (consistent length)
- **Concurrency**: Sequential requests (concurrency=1) per client
- **Retry Logic**: 3 retry attempts for fault tolerance
- **Streaming Support**: Optional `--streaming` flag for TTFT/ITL metrics
- **No warmup**: warmup-request-count=0 to avoid initial failures

#### Failures

Failures are injected into deployed pods either by using pod delete or
sending signals to specified processes.

The following failure types are defined in `scenarios.py`:

| Failure Name                  | Description                                        | Injection Method              | Applicable Backends |
|-------------------------------|----------------------------------------------------|-------------------------------|---------------------|
| `none`                        | No failure injection (baseline).                   | N/A                           | All                 |
| `frontend`                    | Terminate frontend process.                        | `SIGINT` to `dynamo.frontend` | All                 |
| `frontend_pod`                | Delete frontend pod.                               | Kubernetes API pod deletion   | All                 |
| `decode_worker`               | Terminate decode worker process.                   | `SIGKILL` to `dynamo.<backend>` | All                 |
| `decode_worker_pod`           | Delete decode worker pod.                          | Kubernetes API pod deletion   | All                 |
| `prefill_worker`              | Terminate prefill worker process.                  | `SIGKILL` to `dynamo.<backend>` | All                 |
| `prefill_worker_pod`          | Delete prefill worker pod.                         | Kubernetes API pod deletion   | All                 |
| `vllm_decode_engine_core`     | Terminate VLLM decode engine core process.         | `SIGKILL` to `VLLM::EngineCore` | vllm only           |
| `vllm_prefill_engine_core`    | Terminate VLLM prefill engine core process.        | `SIGKILL` to `VLLM::EngineCore` | vllm only           |
| `sglang_decode_scheduler`     | Terminate SGLang decode scheduler process.         | `SIGKILL` to `sglang::scheduler`| sglang only         |
| `sglang_decode_detokenizer`   | Terminate SGLang decode detokenizer process.       | `SIGKILL` to `sglang::detokenizer`| sglang only         |
| `sglang_prefill_scheduler`    | Terminate SGLang prefill scheduler process.        | `SIGKILL` to `sglang::scheduler`| sglang only         |
| `sglang_prefill_detokenizer`  | Terminate SGLang prefill detokenizer process.      | `SIGKILL` to `sglang::detokenizer`| sglang only         |

#### Token Overflow Tests

In addition to process and pod failures, the suite includes tests for **token overflow**, where the model receives an input prompt larger than its configured `max_seq_len`. These tests are crucial for verifying that the system can gracefully reject invalid requests without crashing.

- **Failure Injection**: Unlike other tests, this failure is injected from the **client side**. The `aiperf` client is configured to send a batch of requests with oversized token lengths.
- **Two-Phase Execution**: These tests run in two distinct phases, creating separate log directories for each:
  1.  **`overflow` Phase**: Sends oversized requests. The expected outcome is a high rate of failed requests (rejections) as the server correctly identifies and blocks them.
  2.  **`recovery` Phase**: Immediately after the overflow phase, sends valid, normal-sized requests. The expected outcome is a high success rate, confirming that the system has recovered and remains operational.

The combined results of these two phases demonstrate both the system's ability to reject invalid inputs and its stability after handling them.

#### Example Scenario Breakdown

**Scenario**: `sglang-agg-tp-2-dp-1-decode_worker`

- **Backend**: `sglang`
- **Deployment**: Aggregation with 1 decoder worker replica, using 2 GPUs for tensor parallelism (`agg-tp-2-dp-1`).
- **Client Load**: 10 clients, 100 requests each, max request rate 1/sec.
- **Failure**: Terminates 1 decoder worker process 30 seconds into the test.

#### Example Scenario Execution:

Run standard deployments and failure scenarios (excludes custom builds by default):

```bash
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v --namespace ${NAMESPACE}
```

To include all scenarios including custom builds (e.g., MoE models):

```bash
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v --namespace ${NAMESPACE} --include-custom-build
```

### Test Results Directory

For each test scenario a directory of log files is created and post-processed to summarize the test. The directory structure differs based on which client type is used.

#### AI-Perf Client Output Structure (Default)

```
test_fault_scenario[sglang-agg-tp-1-dp-1-frontend]
.
├── client_0/
│   └── attempt_0/
│       ├── profile_export_aiperf.json    # AI-Perf metrics in JSON format
│       ├── profile_export_aiperf.csv     # AI-Perf metrics in CSV format
│       ├── genai_perf.log                # AI-Perf execution log
│       └── logs/
│           └── aiperf.log                # Detailed AI-Perf logs
├── client_1/
│   ├── attempt_0/                        # First attempt (may fail during fault)
│   └── attempt_1/                        # Retry attempt after failure
│       └── [same structure as above]
├── [client_2 through client_9...]
├── Frontend/
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.log
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.metrics.log
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.previous.log  # Pre-restart logs
│   └── fault-tolerance-test-frontend-576bd784dc-jv68q.yaml
├── decode/                                # Or VllmDecodeWorker for vLLM backend
│   └── [same structure as Frontend]
└── test.log.txt
```

| File/Directory Name                | Description                                                                                      |
|------------------------------------|------------------------------------------------------------------------------------------------|
| **client_N/attempt_M/**            | AI-Perf results for client N, attempt M (supports multiple retry attempts)                      |
| **profile_export_aiperf.json**     | Complete AI-Perf metrics including latencies (P50/P90/P99), throughput, token counts           |
| **profile_export_aiperf.csv**      | Tabular format of key metrics for easy analysis                                                |
| **genai_perf.log**                 | AI-Perf execution output (stdout/stderr)                                                       |
| **{Service}/*.log**                | Current container log for pod (Frontend, decode, etc.)                                         |
| **{Service}/*.previous.log**       | Previous container log before restart (contains pre-fault logs)                                |
| **{Service}/*.metrics.log**        | Prometheus metrics from `/metrics` endpoint                                                    |
| **{Service}/*.yaml**               | Pod specification and status transitions                                                       |
| **test.log.txt**                   | Primary test execution log (deployment timing, fault injection, recovery events)               |

#### Legacy Client Output Structure (with `--client-type legacy`)

```
test_fault_scenario[sglang-agg-tp-1-dp-1-frontend]
.
├── client_0.log.txt                       # JSONL format: one request per line
├── client_1.log.txt                       # Direct HTTP request/response logs
├── client_2.log.txt
├── client_3.log.txt
├── client_4.log.txt
├── client_5.log.txt
├── client_6.log.txt
├── client_7.log.txt
├── client_8.log.txt
├── client_9.log.txt
├── Frontend/
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.log
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.metrics.log
│   ├── fault-tolerance-test-frontend-576bd784dc-jv68q.previous.log  # Pre-restart logs
│   └── fault-tolerance-test-frontend-576bd784dc-jv68q.yaml
├── decode/                                # Or VllmDecodeWorker for vLLM backend
│   └── [same structure as Frontend]
└── test.log.txt
```

| File Name                      | Description                                                                                      |
|--------------------------------|------------------------------------------------------------------------------------------------|
| **client_N.log.txt**           | JSONL format logs with one request/response per line (per-request retry support)               |
| **{Service}/*.log**            | Current container log for pod (Frontend, decode, etc.)                                         |
| **{Service}/*.previous.log**   | Previous container log before restart (contains pre-fault logs)                                |
| **{Service}/*.metrics.log**    | Prometheus metrics from `/metrics` endpoint                                                    |
| **{Service}/*.yaml**           | Pod specification and status transitions                                                       |
| **test.log.txt**               | Primary test execution log (deployment timing, fault injection, recovery events)               |

**Example JSONL content in `client_N.log.txt`:**
```json
{"time": "2025-10-03T10:30:45", "results": [{"status": 200, "request_elapsed_time": 1.23, "url": "http://localhost:8000/v1/chat/completions", "pod": "frontend-pod"}], "total_time": 1.25}
{"time": "2025-10-03T10:30:47", "results": [{"status": 200, "request_elapsed_time": 1.18, "url": "http://localhost:8000/v1/chat/completions", "pod": "frontend-pod"}], "total_time": 1.20}
```

## Validation Framework

### Overview

The fault tolerance test suite includes an automated validation framework that verifies both the test execution and the results. Validation runs automatically after each test completes, ensuring that:

1. **The failure was actually injected** (Stage 1: Scenario Verification)
2. **The system recovered appropriately** (Stage 2: Results Verification)

### Two-Stage Validation Approach

#### Stage 1: Scenario Verification

Verifies that the test scenario executed correctly by checking Kubernetes events and pod states:

**For Pod Deletions (`*_pod` failures):**
- Confirms specific pods were deleted via K8s events (`Killing`, `Terminating`)
- Validates pod recreation and lifecycle transitions
- Logs deletion confirmation with timestamps

**For Process Terminations (non-`*_pod` failures):**
- Checks container restart counts (`restartCount` field)
- Looks for container restart events (`Started`, `BackOff`, `CrashLoopBackOff`)
- **Main process terminations** (e.g., `decode_worker`) → container restarts (verifiable via `restartCount`)
- **Subprocess terminations** (e.g., `sglang_*_scheduler`, `sglang_*_detokenizer`) → no container restart (subprocess becomes zombie/defunct). These produce warnings but are documented known limitations (see Backend-Specific Validations below)

**Example Stage 1 Output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STAGE 1: SCENARIO VERIFICATION                            ║
║          (Verify test scenario executed correctly)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────
1.1 Verifying Specific Pod Deletion via K8s Events
────────────────────────────────────────────────────────────────────────────────
Target pod(s) for deletion: ['fault-tolerance-test-0-vllmdecodeworker-abc123']

✓ DELETION CONFIRMED: [Normal] Killing - Stopping container main
✓ Pod fault-tolerance-test-0-vllmdecodeworker-abc123 deletion verified via K8s events
✓ STAGE 1.1 PASSED: Pod deletion confirmed via K8s events
```

#### Stage 2: Results Verification

Validates system behavior based on deployment redundancy:

**High Availability (DP > 1):**
- Success rate: ≥99%
- Recovery time: <60 seconds
- Minimal impact on ongoing requests

**Single Worker (DP = 1):**
- Success rate: ≥10% (allows for failures during recovery)
- Recovery time: <180 seconds
- System eventually recovers

**Baseline (No Failures):**
- Success rate: 100%
- No failed requests

**Example Stage 2 Output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    STAGE 2: RESULTS VERIFICATION                             ║
║                 (Single worker - no redundancy)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────
2.1 Basic Recovery Check
────────────────────────────────────────────────────────────────────────────────
✓ System recovered: 1470 requests succeeded

────────────────────────────────────────────────────────────────────────────────
2.2 Success Rate Validation (Single Worker)
────────────────────────────────────────────────────────────────────────────────
Success rate: 98.00% (1470/1500 requests)
✓ STAGE 2.2 PASSED: Success rate meets threshold (10%)

────────────────────────────────────────────────────────────────────────────────
2.3 Recovery Time Validation
────────────────────────────────────────────────────────────────────────────────
Recovery time: 150.52 seconds
✓ STAGE 2.3 PASSED: Recovery time within acceptable range (180s max)
```

### Validation Architecture

The validation system uses a **factory pattern** for flexible, extensible validation:

```
┌─────────────────────────────────────────────────────────────┐
│              test_deployment.py                              │
│           (validation_context fixture)                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ After test completes
                       │
          ┌────────────▼─────────────┐
          │   checker_factory.py     │
          │   (Checker Factory)      │
          └──────┬───────────┬───────┘
                 │           │
    ┌────────────▼───┐  ┌───▼────────────────┐
    │ Scenario       │  │ Results            │
    │ Checkers       │  │ Checkers           │
    │ (Stage 1)      │  │ (Stage 2)          │
    └────────┬───────┘  └───┬────────────────┘
             │              │
    ┌────────▼───────┐  ┌───▼────────────────┐
    │ k8s_utils.py   │  │ validation_checks  │
    │ - Pod events   │  │ - Success rate     │
    │ - Restart cnt  │  │ - Recovery time    │
    └────────────────┘  └────────────────────┘
```

### Factory Functions

#### `get_checkers_for_scenario(test_name, scenario)`

Determines which checkers to run based on:

1. **Explicit checkers** in `scenario.checkers` (highest priority)
   - Allows scenarios to specify custom checker lists
2. **Pattern matching** on test name:
   - Delegates to `get_scenario_checker()` for Stage 1
   - Delegates to `get_results_checker()` for Stage 2

#### `get_scenario_checker(test_name, scenario)`

Selects scenario checker (Stage 1) based on test name pattern:

- `*-none]` → `NoFailureChecker` (baseline)
- `*_pod]` → `PodDeletionChecker` (pod deletions)
- `*decode_worker]`, `*prefill_worker]`, `*frontend]`, `*scheduler]`, `*detokenizer]`, `*engine_core]` → `ProcessTerminationChecker` (process terminations)

#### `get_results_checker(test_name, scenario)`

Selects results checker (Stage 2) based on deployment redundancy:

- `*-none]` → `BaselineResultsChecker` (100% success required)
- **DP > 1** → `HighAvailabilityResultsChecker` (≥90% success, ≤60s recovery)
- **DP = 1** → `SingleWorkerResultsChecker` (≥10% success, ≤180s recovery)

### Backend-Specific Validations

#### SGLang Subprocess Limitations

SGLang has a **known limitation** where subprocess termination leads to zombie processes without automatic recovery:

**Affected Failures:**
- `sglang_decode_scheduler` - Scheduler subprocess becomes `<defunct>`
- `sglang_decode_detokenizer` - Detokenizer subprocess becomes `<defunct>`

**Expected Behavior:**
```
Process killed → becomes zombie (PID exists with Z state, <defunct>)
Container does NOT restart (main process PID 1 still running)
No new subprocess spawned
System does NOT recover automatically
```

**Validation Approach:**
- Confirms container restart count = 0 (subprocess kill, not container crash)
- Documents limitation in test output
- Does not expect recovery

#### vLLM Disaggregated Prefill Worker Resilience

vLLM decode workers use `--kv-connector-role kv_both` by default, allowing them to handle both prefill and decode operations. When a prefill worker fails, decode workers automatically take over prefill requests, resulting in 100% success rate with minimal impact.

**Expected Behavior:** Prefill worker failures don't cause request failures - this is vLLM's built-in fault tolerance, not a test issue.

### Summary Results

Results are parsed from AI-Perf metrics and presented in table format after each test. The parsing script (`parse_results.py`) extracts comprehensive metrics for each scenario:

#### Per-Test Output Format
```
============================================================
FAULT TOLERANCE TEST SUMMARY - AI-PERF
============================================================
╒═══════════════════════════════════╤════════════════════════════════════════════════════╕
│ Metric                            │ Value                                              │
╞═══════════════════════════════════╪════════════════════════════════════════════════════╡
│ Test Directory                    │ test_fault_scenario[sglang-agg-tp-1-dp-1-frontend] │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Number of Clients                 │ 10                                                 │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ === Deployment Metrics ===        │                                                    │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Startup Time                      │ 69.00 sec                                          │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Recovery Time                     │ 2.00 sec                                           │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ === Request Metrics ===           │                                                    │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Total Requests                    │ 1500                                               │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Successful Requests               │ 1470                                               │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Failed Requests                   │ 30                                                 │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Success Rate                      │ 98.00%                                             │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ === Latency Metrics (seconds) === │                                                    │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Mean Latency                      │ 0.502                                              │
├───────────────────────────────────┼────────────────────────────────────────────────┤
│ P50 Latency                       │ 0.396                                              │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ P90 Latency                       │ 0.422                                              │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ P99 Latency                       │ 0.761                                              │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ === Throughput Metrics ===        │                                                    │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Total Throughput                  │ 19.72 req/s                                        │
├───────────────────────────────────┼────────────────────────────────────────────────────┤
│ Avg Client Throughput             │ 1.97 req/s                                         │
╘═══════════════════════════════════╧════════════════════════════════════════════════════╛
```

| Metric Category       | Metrics Included                                                            |
|-----------------------|-----------------------------------------------------------------------------|
| **Deployment Metrics**| Startup Time, Recovery Time                                                |
| **Request Metrics**   | Total/Successful/Failed Requests, Success Rate                             |
| **Latency Metrics**   | Mean, P50, P90, P99 latencies (in seconds)                                |
| **Token Metrics**     | TTFT (Time to First Token), ITL (Inter-Token Latency) when streaming enabled |
| **Throughput Metrics**| Total and per-client request throughput                                    |

## Example Deployment Architectures

The following architectures are tested with various failure scenarios:

### Aggregated Workers

#### No Redundancy

To demonstrate the failure and recovery time in the case that there is
a single instance of each process we ran a simmple "agg-tp-1-dp-1" configuration.

```mermaid
graph LR
    Client["Client"]
    Frontend["Frontend"]

    Client --> Frontend
    Frontend --> DecodePool

    %% Decode Worker Pool (vertical layout)
    subgraph DecodePool["Decode Worker Pool"]
        direction TB
        subgraph Decode1["Decode 1"]
            direction TB
            D1GPU0["GPU 0"]
        end
    end

    %% Styling
    style DecodePool stroke:#000,stroke-width:2px
```





#### Redundant Workers (Over Provisoned)

To demonstrate the failure and recovery time in the case that there
are multiple instances of each process (except for the frontend) we
ran a simple "agg-tp-1-dp-2" configuration.

```mermaid
graph LR
    Client["Client"]
    Frontend_1["Frontend_1"]
    Frontend_2["Frontend_2"]

    Client --> Frontend_1
    Client --> Frontend_2

    Frontend_1 --> DecodePool
    Frontend_2 --> DecodePool

    subgraph DecodePool["Decode Worker Pool"]
        direction LR
        subgraph Decode1["Decode 1"]
            direction TB
            D1GPU0["GPU 0"]
        end
        subgraph Decode2["Decode 2"]
            direction TB
            D2GPU0["GPU 0"]
        end
    end

    style DecodePool stroke:#000,stroke-width:2px
```
1. By immediately detecting a decode worker failure, Dynamo can limit
   the failures and reroute requests to healthy workers with minimal
   impact.

### Disaggregated Workers

#### No Redunancy

To demonstrate the failure and recovery time in the case of a
disaaggregated deployment with a single instance for each process in
the graph we ran a simple `disagg-tp-1-dp-1` configuration.

```mermaid
graph LR
    Client["Client"]
    Frontend["Frontend"]

    Client --> Frontend
    Frontend <--> DecodePool

    %% Prefill Worker Pool (horizontal layout)
    subgraph PrefillPool["Prefill Worker Pool"]
        direction LR
        subgraph Prefill1["Prefill 1"]
            direction TB
            P1GPU0["GPU 0"]
   		end
    end

    %% Decode Worker Pool (vertical layout)
    subgraph DecodePool["Decode Worker Pool"]
        direction TB
        subgraph Decode1["Decode 1"]
            direction TB
            D1GPU0["GPU 0"]
        end
    end


    DecodePool --> PrefillPool
    PrefillPool -.-> DecodePool

    %% Styling
    style PrefillPool stroke:#0066cc,stroke-width:2px
    style DecodePool stroke:#000,stroke-width:2px
```

#### Summary:


1. Prefill worker engine failure causes decode engine failure.

2. When prefill workers fail gracefully, decode workers will automatically do prefill as well.


#### Redundant Workers

To demonstrate the failure and recovery time in the case that there
are multiple instances of each process (except for the frontend and
decode worker) we ran a simple "disagg-tp-1-dp-2"
configuration.


```mermaid
graph LR
    Client["Client"]
    Frontend_1["Frontend 1"]
	Frontend_2["Frontend 2"]

    Client --> Frontend_1
    Client --> Frontend_2

    Frontend_1 <--> DecodePool
	Frontend_2 <--> DecodePool

    %% Prefill Worker Pool (horizontal layout)
    subgraph PrefillPool["Prefill Worker Pool"]
        direction LR
        subgraph Prefill1["Prefill 1"]
            direction TB
            P1GPU0["GPU 0"]
		end
        subgraph Prefill2["Prefill 2"]
            direction TB
            P2GPU0["GPU 0"]
		end

    end

    %% Decode Worker Pool (vertical layout)
    subgraph DecodePool["Decode Worker Pool"]
        direction TB
        subgraph Decode1["Decode 1"]
            direction TB
            D1GPU0["GPU 0"]
        end
    end


	DecodePool --> PrefillPool
    PrefillPool -.-> DecodePool

    %% Styling
    style PrefillPool stroke:#0066cc,stroke-width:2px
    style DecodePool stroke:#000,stroke-width:2px
```



#### Summary:


1. Redundant prefill workers are able to absorb the load.

2. When prefill workers go down, decode workers can also do prefill locally.

## Quick Start

### Install Dynamo Platform

Follow the [instructions](../../../docs/kubernetes/installation-guide.md) to install `Dynamo` in your Kubernetes cluster.

### Mount Workspace and Kube Config

Ensure you are able to run a `Dynamo` deployment directly from your host.

Then run the development container mounting the workspace and your kube config.

```
./container/run.sh --mount-workspace -it -v ~/.kube:/root/.kube
```

### Run the tests

#### Default: Run Standard Tests Only

By default, tests requiring custom builds (e.g., MoE models) are **automatically excluded**:

```bash
# Standard tests only
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE}
```

#### Include Custom Build Tests

To run ALL tests including those requiring custom builds (e.g., MoE models):

```bash
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  --include-custom-build
```

#### Run Only Custom Build Tests

To run ONLY tests that require custom builds:

```bash
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  -m "custom_build"
```

#### List Available Tests

```bash
# See which tests will run by default (excludes custom_build)
pytest tests/fault_tolerance/deploy/test_deployment.py --collect-only -q

# See which tests are excluded
pytest tests/fault_tolerance/deploy/test_deployment.py --collect-only -m "custom_build" -q
```

> **Note:** Tests requiring custom builds are marked with `@pytest.mark.custom_build` and include:
> - MoE (Mixture-of-Experts) models like DeepSeek-V2-Lite
> - Tests requiring special Docker image configurations
> - Any scenario with `requires_custom_build=True` in scenarios.py


### Note on Running with Additional Credentials

When running on an cluster that requires additional authentication (such as `AKS`) in addition you will need
to authenticate and install cli as appropriate in to the container. As an example, before running the tests you
in an `AKS` cluster you would need to do the following:

```
# In case you have multiple configs
export KUBECONFIG=~/.kube/dynamo-kubeconfig

curl -sL https://aka.ms/InstallAzureCLIDeb
az aks install-cli
az login
```
## Dual Client Implementation for Fault Tolerance Tests

### Overview

This document describes the implementation of dual client support for fault tolerance tests, allowing tests to use either the **AI-Perf** client or the **legacy custom client**.

### Motivation

A requirement to support both clients simultaneously for:
- Comparing performance and results between implementations
- Gradual migration path from legacy to AI-Perf
- Supporting different use cases (AI-Perf for comprehensive metrics, legacy for simple testing)

### Architecture

The implementation uses a **factory pattern** to cleanly separate client implementations and parsers while providing a unified interface.

```
┌─────────────────────────────────────────────────────────────┐
│                    test_deployment.py                        │
│                     (Test Runner)                            │
└──────────────────────┬──────────────────────┬────────────────┘
                       │                      │
                       ├──────────────────────┤
                       │                      │
          ┌────────────▼─────────┐ ┌─────────▼──────────┐
          │  client_factory.py   │ │ parse_factory.py   │
          │  (Client Selection)  │ │ (Parser Selection) │
          └──────┬───────┬───────┘ └──────┬──────┬──────┘
                 │       │                │      │
         ┌───────▼───┐ ┌─▼──────────┐ ┌──▼──────▼───────────┐
         │ client.py │ │legacy_     │ │parse_   │legacy_    │
         │ (AI-Perf) │ │client.py   │ │results  │parse_     │
         └───────────┘ └────────────┘ │.py      │results.py │
                                      └─────────┴───────────┘
```

### Usage

#### Running Tests with Command-Line Option

The client type can be dynamically selected using the `--client-type` pytest argument:

##### **Using AI-Perf Client (Default)**
```bash
# Default - no flag needed
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE}

# Or explicitly specify
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  --client-type aiperf
```

##### **Using Legacy Client**
```bash
pytest tests/fault_tolerance/deploy/test_deployment.py -s -v \
  --namespace ${NAMESPACE} \
  --image ${IMAGE} \
  --client-type legacy
```

##### **Single Test with Legacy Client**
```bash
pytest tests/fault_tolerance/deploy/test_deployment.py::test_fault_scenario[vllm-agg-tp-1-dp-1-none] -s -v \
  --namespace test-ft \
  --image your-image:tag \
  --client-type legacy
```
Legacy Output Format

```bash
PASSED[TEST] 2025-10-03T21:02:21 INFO root: Using legacy parser for results

Test Group: vllm-agg-tp-1-dp-2
╒═══════════════════╤═══════════╤═══════════╤══════════╤═══════════╤══════════╤═══════════╤═══════════╤════════════╕
│      Failure      │   Startup │   Success │   Failed │   Success │   Failed │   Latency │   Latency │   Recovery │
│                   │           │    Before │   Before │     After │    After │    Before │     After │            │
╞═══════════════════╪═══════════╪═══════════╪══════════╪═══════════╪══════════╪═══════════╪═══════════╪════════════╡
│ decode_worker_pod │    178.00 │    149.00 │     0.00 │   1349.00 │     2.00 │      1.19 │      1.19 │     163.90 │
╘═══════════════════╧═══════════╧═══════════╧══════════╧═══════════╧══════════╧═══════════╧═══════════╧════════════╛

================================================================================
[TEST] 2025-10-03T21:02:22 INFO root: Using legacy parser for results

Test Group: vllm-agg-tp-1-dp-2
╒═══════════════════╤═══════════╤═══════════╤══════════╤═══════════╤══════════╤═══════════╤═══════════╤════════════╕
│      Failure      │   Startup │   Success │   Failed │   Success │   Failed │   Latency │   Latency │   Recovery │
│                   │           │    Before │   Before │     After │    After │    Before │     After │            │
╞═══════════════════╪═══════════╪═══════════╪══════════╪═══════════╪══════════╪═══════════╪═══════════╪════════════╡
│ decode_worker_pod │    178.00 │    149.00 │     0.00 │   1349.00 │     2.00 │      1.19 │      1.19 │     163.90 │
╘═══════════════════╧═══════════╧═══════════╧══════════╧═══════════╧══════════╧═══════════╧═══════════╧════════════╛

```


