> Implementation draft (2026-10-07): the opt-in Python frontend helper now exposes
> a private Unix-socket API. `torch_profile` combines frontend Linux perf and
> selected worker torch.profiler hooks. Enable it with `--profiling-config`.
> The implemented schema, usage and integration limitations are in
> [profiling/README.md](components/src/dynamo/frontend/profiling/README.md).
> Nsight adapters and runtime OTel detail switching are deferred. The remaining
> document is the broader design proposal, including future contract extensions.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Draft on demand profiling interface

Status: proposed PoC contract, 2026-10-07. Source baseline: Dynamo main
`16b2ae50032edbdde70a9e03a7e7056fac3b7c7d`.
Proposal: https://docs.google.com/document/d/1YQFg5XcN25P_1U_IA9t2C27wJgf1TXKbb38aWRxx_lE/edit

An external telemetry service or operator submits a bounded capture plan to the
Dynamo frontend. The frontend hosts the coordinator role: resolve exact processes,
validate current capabilities, dispatch capture, track status, and return artifact
references. Anomaly detection and deeper analysis remain outside inference.

This contract applies equally to frontend processes and model workers. A frontend
is an explicit target with its own capabilities, not an implicit worker alias.
The control interface is independent of the lifecycle instrumentation feature gate.

## Existing integration points

At this baseline, worker profiling already has local engine controls:

- `lib/runtime/src/system_status_server.rs` serves `/engine/{*path}` through
  the runtime's engine route registry.
- `lib/runtime/src/engine_routes.rs` stores callbacks registered from Python.
- vLLM's `handlers.py::start_profile` delegates to
  `engine_client.start_profile(profile_prefix=...)`; `stop_profile` delegates
  to the engine client. Failures can be returned as JSON `status: error` even
  with HTTP 200, so an adapter must inspect both.
- SGLang's `request_handlers/handler_base.py` forwards a `ProfileReq` to
  `tokenizer_manager.start_profile` and exposes a stop handler.
- `lib/llm/src/http/service/frontend_extension.rs` and `service_v2.rs`
  provide frontend routing integration points.

These controls provide start/stop hooks. The coordinated session API, process
inventory, leases, capability reporting, and artifact manifests below are proposed
additions. Engine profiler configuration still has to be prepared at startup.

## Control flow

```text
OTel evidence -> external agent -> frontend coordinator -> selected targets
                                    |                         |
                                    | <- acknowledgements ----|
                                    | <- status and artifacts |
                                    v
                              client polls session
```

OTLP remains the evidence export path. The client uses HTTP JSON to request
profiling; telemetry ingestion itself does not trigger profiling inside Dynamo.
A manual client uses the same API as an analysis agent.

For the PoC, use one explicitly addressed coordinator frontend and polling.
Another frontend replica is a separate target. Do not resolve "frontend" to whichever
replica a load balancer happens to choose. Return the session owner's identity;
clients poll that coordinator. A later deployment can share session storage.

Expose these routes on an opt-in administration surface under
`/admin/profiling/v1`, with deployment access controls. Do not expose privileged
capture controls through ordinary chat request headers.

## Frontend API

| Method | Path under /admin/profiling/v1 | Result |
| --- | --- | --- |
| GET | /targets | Process inventory and capture capabilities |
| POST | /plans/validate | Check a plan without allocating or starting capture |
| POST | /sessions | Recheck, reserve targets, and start an asynchronous session |
| GET | /sessions/{capture_id} | Aggregate and per-target state, timing, errors, artifacts |
| POST | /sessions/{capture_id}/stop | Idempotent early stop and finalization |
| GET | /sessions/{capture_id}/artifacts | Manifest of finalized or incomplete artifacts |

Validation is advisory and subject to races. Session creation repeats checks
against current process epochs and conflicts. HTTP 202 means accepted, not that
every target is recording. Poll per-target acknowledgements to establish coverage.
GET session already includes the manifest; the artifacts route provides a compact
view for an analyzer. Artifact bytes are fetched from configured storage separately.

Return structured errors with `code`, `message`, and optional `target_id`.
Use 400 for malformed plans, 404 for unknown sessions/targets, 409 for an epoch,
active-capture, or idempotency conflict, 422 for unsupported configurations, and
429 for coordinator budget/cooldown limits. Runtime failures after acceptance
appear in session status.

## Which processes and what to capture

Inventory entries include:

- `target_id`: opaque stable identifier within the deployment.
- `process_epoch`: unique startup identity; required in every target request.
- `kind`: `frontend` or `worker`.
- `worker_id`, `role`, `model`, `backend`: when applicable.
- `capabilities`: supported collector IDs, activity sets, output formats,
  collection scope, configured/available state, maximum duration, and constraints.
- `active_capture_id` and `cooldown_until`: when applicable.
- `observed_at`: inventory freshness.

The coordinator maintains the routable administration address internally. External
clients supply target IDs rather than arbitrary host URLs. Workers announce their
control address and capabilities through discovery or an explicitly configured
PoC inventory; existing model discovery alone must not be assumed to include them.

The PoC uses `torch.profiler` for both vLLM and SGLang, exposed through the common
collector ID `torch_profiler`. Each backend adapter controls its engine's existing
profiler hooks and translates supported options into that backend's configuration.
Profiler hooks are prepared at startup; recording starts only for an accepted
capture and stops at the collection deadline or an explicit stop.

Nsight Systems (`nsys`) is an additional collector, using either Nsight Cloud
or a target-local session supervisor as its execution provider. Advertise support
only for prepared targets. Frontend CPU profiling needs an independently supported
collector: a Rust frontend may use an available Nsight CPU configuration but cannot
inherit `torch.profiler` support from a Python backend.

Capabilities describe actual scope: an engine invocation may capture multiple
tensor/data-parallel child processes or ranks. The PoC selects that whole engine
group and reports member coverage; it must reject unsupported rank-level selection.
GPU counters, stacks, shape recording, and memory profiling are separate optional
capabilities, not promised by every trace capture.

## When to start

The external agent decides when always-on evidence warrants escalation and submits
a plan with `start.mode: immediate`, a maximum start delay, and a collection
duration. Targets start only after local validation, reservation, and lease setup.
Actual start times are reported individually; this is not a synchronized cluster
barrier. A target that cannot start before its deadline must fail without starting.

For the PoC, capture the next workload window. A completed anomalous request cannot
be profiled retroactively. Trigger trace/request IDs correlate why capture was
requested; they do not filter process-wide GPU activity or prove that the original
request is present in the capture.

Later extensions may add scheduled start or request-bound arming. These require
explicit support and timeout semantics; do not implement them by silently rerouting
production requests. Live stall detection needs progress evidence because finished
OTel spans may arrive only after the stall ends.

## Example plan

```json
{
  "schema_version": "v1",
  "client_request_id": "investigation-42",
  "trigger": {
    "reason": "decode_latency_anomaly",
    "trace_ids": ["4bf92f3577b34da6a3ce929d0e0e4736"],
    "request_ids": ["req-123"],
    "evidence_uri": "evidence://investigations/42"
  },
  "start": {
    "mode": "immediate",
    "max_start_delay_ms": 5000
  },
  "duration_ms": 10000,
  "targets": [
    {
      "target_id": "decode-17",
      "process_epoch": "decode-17-startup-8",
      "collector": "torch_profiler",
      "options": {
        "activities": ["cpu", "cuda"],
        "record_shapes": false,
        "with_stack": false,
        "profile_memory": false
      }
    }
  ],
  "artifact_sink": "investigation-store"
}
```

`artifact_sink` names a server-configured destination; no caller-supplied paths,
upload credentials, or executable profiler commands. Bound target counts, durations,
evidence lists, and option sizes. Backend adapters translate supported typed options
into the existing engine hooks; reject unknown fields and unsupported options.

To include a frontend, add its exact ID and epoch with an advertised collector,
for example `cpu_sampling` once that adapter exists. Reject the entire plan at
preflight if any requested collector is unavailable. Do not silently omit it.

Scope `client_request_id` to the caller. Same key and normalized plan returns the
same capture ID without restarting or extending capture; changed content returns
409. Define and advertise the PoC's bounded idempotency retention window.

## Target protocol

The coordinator needs the same semantics locally and remotely:

```text
capabilities() -> TargetCapabilities
prepare(capture_id, expected_epoch, collector, options, start_deadline,
        lease_expiry, artifact_sink) -> Reservation
start(capture_id, reservation_id) -> StartAcknowledgement
status(capture_id) -> TargetCaptureStatus
stop(capture_id, reason) -> StopAcknowledgement
artifacts(capture_id) -> ArtifactManifest
```

These are proposed logical operations, not new callbacks already present in Dynamo.
A frontend target calls its local adapter; remote targets expose the protocol on
their runtime administration server. Preparation reserves ownership but never
starts the profiler. Start/stop retries are idempotent and fenced by capture ID,
reservation, and process epoch. A delayed command must not start after expiry
or stop a newer capture.

The target owns a local watchdog and a monotonic collection deadline. Automatic
stop must survive coordinator/client disconnect. Setup cannot claim success before
the watchdog is armed. Clock conversion must account for cross-host skew; the
maximum local duration remains enforceable independently of wall-clock timestamps.

One active capture per actual profiler resource, including engine child groups.
Prepare all targets before start. Preflight/preparation failure releases reservations
without recording. A start failure stops already-started targets and retains partial
artifacts. Distributed start is not atomic; report the intervals that actually ran.

## Session and artifact status

A session progresses through `preparing -> starting -> recording -> stopping ->
finalizing -> completed`. It may terminate as `failed` or `cancelled`.
Per-target state is authoritative when targets start at different times.
`completed` requires confirmed stop, baseline restoration, and manifest finalization
for every selected target. A control timeout alone does not prove a profiler stopped.

Status includes requested duration, owner epoch, target states, actual start/end
timestamps, expiry, stop reason, failure code, restoration state, and manifest.
Coverage is `complete`, `partial`, or `unknown`, independently of session state.
An unreachable target remains unconfirmed/unknown; polling exposes the gap.

Each artifact entry includes capture ID, target/process/rank identity, collector,
format, actual capture interval, size, checksum when finalized, storage URI,
availability state, and trigger evidence references. Include missing members and
clock uncertainty. "Profiler stopped" is not evidence that a valid file was uploaded.

Do not label a raw torch.profiler process capture as request-exact evidence.
Optional request/NVTX correlation must describe which associations are observed
and which remain unavailable.

## Expiry and baseline restoration

A hard deadline bounds recording. Finalization has its own bounded deadline.
User stop, expiry, start failure, and coordinator shutdown use the same owned stop
path. Targets restore the pre-capture policy, release reservations, and enforce
cooldown. New requests snapshot any active detail policy; existing request state
is not rewritten. Baseline OTel settings stay unchanged unless a distinct,
explicitly supported detail-policy operation is added.

If an engine or profiler is blocked, restoration may be unconfirmed. Report that
state, prevent another overlapping capture, and surface the failure rather than
claiming expiry guaranteed successful cleanup. Target watchdog placement and
independence from the blocked engine loop need implementation validation.

The PoC can keep session state in coordinator memory, with an advertised retention
limit. After coordinator restart, target leases still expire locally. Old session
IDs must return a clear lost-state result rather than fabricated completion; durable
reconciliation and failover are later work.

## Nsight Systems and Nsight Cloud

Add `nsys` alongside `torch_profiler`. The collector describes the evidence;
a server-configured execution provider describes how capture is controlled:

| Collector | Execution provider | Intended use |
| --- | --- | --- |
| torch_profiler | engine_hooks | vLLM/SGLang engine profiling |
| nsys | nsight_cloud | Prepared cloud deployment with Nsight capture infrastructure |
| nsys | local_nsys | Prepared local process session for the DLC PoC |

Nsight Cloud includes injection, coordination, storage, and remote inspection
components. Dynamo should retain workload selection, capture budgets, and OTel
correlation while delegating supported Nsight execution to that infrastructure.
This is the proposed integration boundary, not an implemented connection.
See [Nsight Cloud](https://developer.nvidia.com/nsight-cloud).

### Nsight Cloud provider

The documented operator workflow supports `session-begin`, `profiler-start`
with a duration, `profiler-stop`, `session-end`, and artifact listing/download.
Use these semantics to map Dynamo capture IDs to provider session and collection IDs;
a provider session may contain multiple collections. The exact service integration
must be checked against the deployed release before choosing an API or CLI bridge.
See [Nsight Operator control workflow](https://docs.nvidia.com/nsight-operator/UserGuide/index.html).

The adapter resolves each target ID and process epoch to provider-managed
pod/container/process identity and capture tags. Check the resolved scope before
dispatch: a broad tag must not capture unselected workers. If the provider cannot
isolate the requested process group, report the actual selectable group as a
capability and reject a narrower plan.

Readiness requires the target to have been prepared for capture by the deployment.
Do not promise late CUDA injection into an arbitrary running process.
Kubernetes preparation and access follow the
[Nsight Operator deployment guide](https://docs.nvidia.com/nsight-operator/GettingStarted/index.html).
The current DLC worktree does not establish that Nsight Cloud is deployed.

Use provider duration enforcement plus a separately owned stop/reconciliation path.
Confirm which component enforces the deadline and whether it survives coordinator
loss; advertising automatic expiry requires that verification. Map provider errors
and artifact availability to the existing per-target states. Keep provider credentials
in configured infrastructure; clients choose target IDs, not provider endpoints.

### Local Nsight provider for DLC

Nsight Systems supports launching a process into an interactive session without
recording, then manually starting and stopping collection. Sequential collections
are documented. Use a target-local supervisor to own that prepared session,
dispatch bounded capture, and finalize reports while inference continues.
See [Nsight Systems interactive collection](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-interactive-cli-command-sequences).

An illustrative sequence is `nsys launch <application>`, `nsys start`,
`nsys stop`. The implementation must address the exact owned session, use
server-selected trace options/output paths, and validate the installed CLI version.
Do not run unscoped start/stop commands in a multi-worker environment or terminate
the inference process to finish a capture.

An alternative for CUDA workers is a prepared CUDA-profiler-API capture range
with engine-side start/stop calls. CPU-only frontend collection needs a suitable
session control path; CUDA calls are not a universal frontend trigger.

### Shared contract additions

Capability responses include `execution_provider`, `prepared`,
`supported_activities`, actual process-group scope, profiler version, and
`deadline_enforcement`. An unavailable provider returns `not_prepared` or
`provider_unavailable`; the client can revise the plan. Do not silently substitute
torch.profiler for Nsight or switch execution providers after acceptance.

For `collector: nsys`, use typed options such as `trace: [cuda, nvtx, osrt]`
when supported. Frontend targets advertise their own CPU options. Sampling, GPU
metrics, and trace activities require separate capability checks.

Session/manifests retain provider session ID, collection ID, per-process coverage,
and report availability. Preserve finalized `.nsys-rep` files and configured
storage references; an intermediate file is not a finalized report. Optional
viewer links are separate from machine-readable artifact locations.

Treat torch.profiler and Nsight CUDA capture on the same process group as conflicting
for the PoC, including externally initiated profiling. Sharing profiling resources
requires separate validation before concurrent capture can be advertised.
Prepared infrastructure may add overhead even with recording disabled; measure
that baseline separately from recording overhead.

## First implementation slice

1. Define shared typed plan, capability, status, and manifest models.
2. Add the opt-in frontend API and exact target inventory.
3. Implement a reusable target session manager with reservation, local expiry,
   idempotent start/stop, and restoration reporting.
4. Implement vLLM and SGLang adapters using `torch.profiler` through the existing
   engine start/stop hooks, including supported option translation and artifacts.
   Add the `nsys` collector contract and a prepared local session provider for DLC;
   provide a Nsight Cloud execution adapter when its deployment/API is available.
5. Validate frontend CPU collection through a supported `nsys` configuration,
   retaining capability rejection until it works.
6. Validate the external-agent flow using scripted evidence, followed by a real
   bounded workload capture on allocated compute nodes.

Validation cases: unsupported frontend collector, stale epoch, conflict,
duplicate create/start/stop, prepare failure, partial start, start deadline expiry,
coordinator loss, worker restart, stop failure, artifact upload failure, cooldown,
and correct multi-rank coverage. Verify automatic stop and final artifacts with
real engine adapters, not only mocked acknowledgements.

This draft changes no serving behavior. A tracked issue/DEP and normal review are
needed before an architectural change is proposed upstream.
