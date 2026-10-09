<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# On-demand profiling administration

The frontend receives client requests on a private Unix socket and passes an
explicit runtime snapshot to `dynamo.profiling.Coordinator`. The coordinator
manages capture leases, worker clock measurements, collector cleanup, and
artifact validation. HTTP routing and process hosting live in this directory;
the reusable coordinator and adapters live in `dynamo/profiling`. See the
[interface design](../../profiling/DESIGN.md) for the module boundaries and
capture lifecycle.

The default host runs inside the frontend process:

```bash
python -m dynamo.frontend --profiling-config /private/profiling/config.json
```

The socket directory and output directory must be owned by the serving user and
have mode 0700. The socket has mode 0600. The interface is separate from the
inference port; the existing Rust frontend extension API only supports static GET
routes. The interface does not provide public POST routes or remote
authentication.

## Configuration

```json
{
  "socket_path": "/private/profiling/admin.sock",
  "output_dir": "/private/profiling/sessions",
  "frontend_perf_enabled": true,
  "max_duration_ms": 60000,
  "start_timeout_s": 10,
  "finalize_timeout_s": 1200,
  "clock_probe_timeout_s": 1,
  "clock_probe_samples": 3,
  "artifact_validation_timeout_s": 120,
  "max_trace_bytes": 8589934592,
  "lifecycle_evidence_location": "https://existing-otel-backend.example/traces",
  "workers": [
    {
      "target_id": "decode-0",
      "process_epoch": "<worker control-process boot-id:pid:starttime>",
      "backend": "vllm",
      "control_url": "http://decode-0-admin:9090",
      "verify_runtime": true,
      "artifact_location": "/shared/native-profiles/decode-0",
      "expected_rank_ids": [
        "dp0_pp0_tp0_dcp0_ep0_rank0"
      ],
      "start_options": {}
    },
    {
      "target_id": "decode-1",
      "process_epoch": "<decode-1 epoch from profiling_info>",
      "backend": "vllm",
      "control_url": "http://decode-1-admin:9090",
      "verify_runtime": true,
      "artifact_location": "/shared/native-profiles/decode-1",
      "expected_rank_ids": ["dp0_pp0_tp0_dcp0_ep0_rank0"],
      "start_options": {}
    }
  ]
}
```

Destinations, worker URLs, rank membership, and profiler options are trusted
operator configuration. Clients cannot choose arbitrary PIDs, commands, imports,
or output paths. `RuntimeSnapshot.workers` is the coordinator's authoritative
inventory; the frontend initially obtains it from configuration. Automatic
Dynamo discovery integration is a follow-up.

`verify_runtime` defaults to false for compatibility with older runtime images.
For a worker with the new callback, POST `{}` to
`/engine/control/profiling_info` to obtain its startup epoch and enable the flag.
The callback reports the worker control process, backend, PID, hostname, and
clock timestamps; it does not attest the individual engine-rank processes.
When enabled, a failed probe or changed epoch prevents profiling start. A second
measurement after stop checks identity and records offset change.
Epochs include boot ID, PID, and process start tick, so replicas starting in the
same tick remain distinct. Obtain the opaque epoch from the callback; refresh
inventory after restarting a worker or upgrading from the earlier epoch format.

The vLLM engine must be launched with torch profiling enabled and its output
path configured, for example:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/shared/native-profiles","torch_profiler_dump_cuda_time_total":false}'
```

The last flag avoids expensive native CUDA summary processing observed in the
live validation. Raw Kineto serialization can still pause inference. SGLang uses
its existing start/stop controls and backend-specific operator `start_options`.

## Client requests

```bash
curl --unix-socket /private/profiling/admin.sock \
  http://localhost/admin/profiling/v1/targets

curl --unix-socket /private/profiling/admin.sock \
  -H 'Content-Type: application/json' \
  -d '{"schema_version":1,"client_request_id":"investigation-42","mode":"torch_profile","targets":[{"target_id":"decode-0","process_epoch":"<epoch from inventory>"}],"duration_ms":10000,"trace_ids":["<otel trace id>"],"request_ids":["<request id>"]}' \
  http://localhost/admin/profiling/v1/sessions
```

| Method | Path under `/admin/profiling/v1` | Result |
| --- | --- | --- |
| GET | `/targets` | Configured inventory and identity requirements |
| POST | `/plans/validate` | Validate a plan without starting capture |
| POST | `/sessions` | HTTP 202 with capture ID and initial manifest |
| GET | `/sessions/{id}` | State, errors, and evidence |
| POST | `/sessions/{id}/stop` | HTTP 202; idempotent early stop |
| GET | `/sessions/{id}/artifacts` | Current manifest and artifact validation |

A request ID identifies one immutable plan. Retrying it returns the same session
and deadline. One coordinator admits one active/finalizing session; concurrent
captures receive 409. Default retained-session limit is 64. State is in memory;
restart preserves on-disk manifests but does not recover leases or idempotency.

`torch_profile` includes frontend Linux perf even for a worker-only request.
Set `frontend_perf_enabled: false` temporarily when perf attachment is denied;
the manifest records the omission. An explicit frontend target is rejected in
that configuration. Stop calls run concurrently so slow worker export does not
extend frontend perf capture. Remote starts that time out still receive cleanup.
Backend HTTP 200 responses with `status:error` are failures. Finalization has a
coordinator timeout; failed or timed-out cleanup blocks new captures. Reconcile
the worker profiler before restarting the coordinator; restart alone is not
cleanup proof.

The example selects only decode-0; decode-1 stays in inventory and receives no
clock probes or profiling start/stop calls for this session. To capture only the
selected worker, also set the operator's `frontend_perf_enabled` to false.
Selection is at the Dynamo worker endpoint boundary. One endpoint backed by a
multi-rank vLLM engine group still profiles its internal ranks together. Separate
DP1 endpoints avoid the shared-group collective coupling observed in validation.

## Duration and completion

`duration_ms` requests a **best-effort lease from acceptance**. Clock probes and
collector startup consume part of that lease. There is no synchronized start or
exact recording-duration guarantee. Early stop shortens the lease; retries retain
the original deadline. Embedded event-loop stalls can delay stop dispatch.

Startup, stop/export, and artifact validation use separate operator timeouts:
`start_timeout_s`, `finalize_timeout_s`, and `artifact_validation_timeout_s`.
Export can continue after native recording ends; the session remains finalizing
and holds the lease until cleanup and validation finish. The manifest's
`timing_contract` describes these semantics and limits.

Torch profile `started_at_ns` and `stopped_at_ns` are **control-response receipt
times**, with `timestamp_scope` recording that distinction. In particular,
`stopped_at_ns` can include a long synchronous trace export. The separate
`start_requested_at_ns` and `stop_requested_at_ns` describe coordinator dispatch.
Neither pair is a substitute for native recorder bounds.

## Evidence and clock alignment

Every session writes an atomic `manifest.json`. Control completion and artifact
verification are separate: `state:completed` may have
`artifact_status:unverified`. Missing or corrupt accessible rank files produce a
failed session and partial artifact status.

The built-in validator reads shared-filesystem vLLM Kineto files using a unique
session-specific `profile_prefix`. It validates the entire JSON/gzip stream,
counts events and CUDA kernels, hashes uncompressed bytes, checks for files
changing during validation, and compares rank identities with
`expected_rank_ids`. Old files and duplicate rank files cannot establish complete
coverage. Validation runs in a thread with byte and cooperative time limits.
When a unique native PyTorch Profiler marker and `baseTimeNanoseconds` are
present, each artifact also reports `recording_bounds`: observed start, end, and
duration in integer nanoseconds. Missing or ambiguous bounds remain unknown
without invalidating an otherwise valid trace. These are raw worker trace clock
values; host offset estimates do not attest GPU-rank clock synchronization.

Without configured rank membership, individually valid files still have unknown
coverage. Unmounted/external stores, SGLang artifacts, and Linux perf data remain
unverified by this validator. AsyncLLM CPU traces lack vLLM's profile prefix and
are excluded from verified engine-rank coverage; they require a separate
association policy. Other stores/formats can implement `ArtifactValidator`.

Clock measurements estimate worker-minus-coordinator offset and uncertainty
from four timestamps, choosing the lowest round-trip sample. They never change
system clocks. Start/end estimates are metadata for analysis, not a simultaneous
start guarantee or rewritten trace timestamps. They describe the control-process
host; multi-host engine-rank alignment remains unknown. Clock steps invalidate a
sample. Disabled legacy agents have explicit unknown alignment.

Lifecycle OTel evidence is included through trigger trace/request IDs and a
reference to the existing telemetry backend. Coverage stays unknown; this does
not assume local OTLP archival or change instrumentation verbosity.

## Hosting and extension points

```python
from dynamo.profiling import Coordinator, RuntimeSnapshot

runtime = RuntimeSnapshot(frontend_pid, frontend_epoch, hostname, workers)
coordinator = Coordinator(config, runtime, registry=registry,
                          clock_sampler=clock_sampler,
                          artifact_validator=artifact_validator)
coordinator.validate(plan)
session = coordinator.submit(plan)
coordinator.stop(session.session_id)
await coordinator.close()
```

The coordinator copies supplied configuration and runtime containers. Each
accepted session owns its runtime snapshot, and manifest/inventory responses
return detached containers. Changing a caller's options, labels, or response
does not change the accepted capture.

`CollectorRegistry` registers trusted mode factories returning collectors with
`start()`, `stop()`, and evidence. Collectors own bounded finalization and must
attempt cleanup after uncertain starts. `ClockSampler` and `ArtifactValidator`
are injectable protocols. The built-in registry supports `torch_profile`.
`nsys` remains rejected until a prepared-session provider is installed; a fake
provider test verifies the registration boundary, not Nsight execution.

The optional standalone host calls the same module:

```bash
python -m dynamo.frontend.profiling --config /private/profiling/config.json \
  --frontend-pid "$FRONTEND_PID" --frontend-epoch "$FRONTEND_EPOCH"
```

Run either the embedded host or the standalone host for a socket, not both.
The standalone host watches the target frontend epoch and drains on shutdown.
Embedded capture deadlines depend on the frontend event loop; a stalled loop can
delay stop. Independent worker leases and restart recovery remain follow-ups.

Nsight Cloud can be a provider behind this registry. Its documented coordinator
controls prepared agents and collections, and its OTLP proxy can mirror traces.
The deployed API, agent scope, and stop behavior must be validated before adding
that adapter. See the [Nsight Operator architecture](https://docs.nvidia.com/nsight-operator/Architecture/index.html)
and [user guide](https://docs.nvidia.com/nsight-operator/UserGuide/index.html).

## Validation status

The embedded implementation was exercised with the real GH200 AgentX workload
on Dynamo/runtime `1.6.0.dev20261007` and vLLM `0.30.0`. Independent DP1/TP1
replicas produced both selected GPU traces in one session, and a second run
selected only decode-0 while decode-1 continued serving. The latter produced a
9.800296-second native GPU trace with 90,480 CUDA kernels, no profiling traces on
unselected workers, and 69 observed lifecycle OTel spans across three probes.
All four workload jobs and their recorded steps completed successfully.

These live runs validate checkpoint `fa32249977`. The metadata/snapshot
refinements at `0e6ec5e1f3` were then validated with the same selected-worker
workload: one 9.957628-second GPU trace, 104,400 CUDA kernels, 69 observed lifecycle
OTel spans, and no profiler starts, stops, or traces on unselected workers. All
four jobs completed successfully; the measurement/grace phase completed 83
requests, cancelled one at the grace timeout, and recorded zero request errors.
The synchronous stop/export response arrived 60.753 seconds after the lease
deadline. Full-tree pre-commit and 405 applicable pre-merge tests passed.
Frontend perf remained disabled in live runs. Actual Nsight collection and SGLang
export validation are pending. These are functional validations, not overhead comparisons.
