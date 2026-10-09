<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# On-demand profiling administration (PoC)

The frontend receives client requests on a private Unix socket and passes an
explicit runtime snapshot to `dynamo.profiling.Coordinator`. The coordinator
manages capture leases, worker clock measurements, collector cleanup, and
artifact validation. HTTP routing and process hosting live in this directory;
the reusable coordinator and adapters live in `dynamo/profiling`.

The default host runs inside the frontend process:

```bash
python -m dynamo.frontend --profiling-config /private/profiling/config.json
```

The socket directory and output directory must be owned by the serving user and
have mode 0700. The socket has mode 0600. The interface is separate from the
inference port; the existing Rust frontend extension API only supports static GET
routes. No public POST route or remote authentication layer is added by this PoC.

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
      "target_id": "decode",
      "process_epoch": "<worker control-process boot-id:starttime>",
      "backend": "vllm",
      "control_url": "http://worker-admin:9091",
      "verify_runtime": true,
      "artifact_location": "/shared/native-profiles",
      "expected_rank_ids": [
        "dp0_pp0_tp0_dcp0_ep0_rank0",
        "dp1_pp0_tp0_dcp0_ep1_rank0"
      ],
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

The vLLM engine must be launched with torch profiling enabled and its output
path configured, for example:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/shared/native-profiles","torch_profiler_dump_cuda_time_total":false}'
```

The last flag avoids expensive native CUDA summary processing observed in the
live PoC. Raw Kineto serialization can still pause inference. SGLang uses its
existing start/stop controls and backend-specific operator `start_options`.

## Client requests

```bash
curl --unix-socket /private/profiling/admin.sock \
  http://localhost/admin/profiling/v1/targets

curl --unix-socket /private/profiling/admin.sock \
  -H 'Content-Type: application/json' \
  -d '{"schema_version":1,"client_request_id":"investigation-42","mode":"torch_profile","targets":[{"target_id":"decode","process_epoch":"<epoch from inventory>"}],"duration_ms":10000,"trace_ids":["<otel trace id>"],"request_ids":["<request id>"]}' \
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
