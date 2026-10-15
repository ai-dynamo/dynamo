<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# On-demand profiling administration

The frontend's private Unix socket calls the reusable `dynamo.profiling`
coordinator. See the [interface design](../../profiling/DESIGN.md) for module
boundaries. Start the embedded host with:

```bash
python -m dynamo.frontend --profiling-config /private/profiling/config.json
```

Socket/output directories must belong to the serving user with mode 0700; the
socket has mode 0600. The listener is separate from the inference port and does
not provide remote authentication.

## Configuration

```json
{
  "socket_path": "/private/profiling/admin.sock",
  "output_dir": "/private/profiling/sessions",
  "frontend_perf_enabled": true,
  "lifecycle_evidence_location": "https://existing-otel-backend.example/traces",
  "workers": [{
    "target_id": "decode-0",
    "process_epoch": "<epoch from profiling_info>",
    "backend": "vllm",
    "control_url": "http://decode-0-admin:9090",
    "verify_runtime": true,
    "artifact_location": "/shared/native-profiles/decode-0",
    "expected_rank_ids": ["dp0_pp0_tp0_dcp0_ep0_rank0"],
    "start_options": {}
  }]
}
```

Worker URLs, options, paths, and rank membership are operator-owned. Clients
cannot supply commands, imports, PIDs, or output paths. Add independent endpoints
to `workers` as needed; `RuntimeSnapshot.workers` is authoritative for the callable
API. Automatic discovery is deferred. All fields/defaults are defined in
[models.py](../../profiling/models.py).

POST `{}` to each worker's `/engine/control/profiling_info` for its opaque startup
epoch before enabling `verify_runtime` (default false for older images). Epochs
include boot ID, PID, and process start tick. Refresh inventory after a restart.
Failed probes or changed epochs prevent start; post-stop probes verify identity
again. Clock estimates describe the control-process host, not individual ranks.

Launch vLLM with torch profiling and its output directory configured:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/shared/native-profiles","torch_profiler_dump_cuda_time_total":false}'
```

Disabling CUDA summary processing avoids its observed export cost; raw Kineto
serialization can still pause inference. SGLang uses its existing start/stop
controls and backend-specific `start_options`.

## Client API

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
| GET | `/targets` | Inventory and identity requirements |
| POST | `/plans/validate` | Validate without collection |
| POST | `/sessions` | HTTP 202 with ID and manifest |
| GET | `/sessions/{id}` | State, errors, and evidence |
| POST | `/sessions/{id}/stop` | HTTP 202; idempotent early stop |
| GET | `/sessions/{id}/artifacts` | Current manifest and validation |

A request ID identifies an immutable plan; retries return the same session and
deadline. Conflicting plans or concurrent captures receive 409. One lease remains
active through finalization. Retention defaults to 64 sessions; manifests survive
restart, but leases and idempotency state are not recovered.

Only selected worker endpoints receive probes/start/stop. Shared multi-rank
engine groups still profile together; use independent DP1 endpoints for separate
selection. `torch_profile` implicitly includes frontend Linux perf even for a
worker-only plan. Set `frontend_perf_enabled: false` to omit it; the manifest
records the omission and explicit frontend targets are then rejected.

## Timing and evidence

`duration_ms` is a best-effort lease from acceptance: probes and startup count
toward it. There is no exact-duration or synchronized-start guarantee, and an
embedded event-loop stall can delay stop. Early stop shortens the lease. Startup,
export/stop, and validation use `start_timeout_s`, `finalize_timeout_s`, and
`artifact_validation_timeout_s`, respectively. Selected targets are probed,
started, stopped and validated concurrently, so slow worker export does not
extend frontend perf. A stop before recording begins fails the session with
explicit missing coverage. Uncertain starts still receive
cleanup; failed/timed-out cleanup blocks new captures. Reconcile worker state
before restarting the coordinator.

Each session atomically writes `manifest.json`. HTTP 200 with `status:error`
is a control failure. Control completion and artifact verification are separate:
a completed session can remain unverified. Worker `started_at_ns`/`stopped_at_ns`
are response-receipt times, including synchronous export; separate request times
record dispatch. `timing_contract` and native `recording_bounds` describe their
scope. Unique PyTorch Profiler markers and `baseTimeNanoseconds` yield integer
nanosecond bounds; missing or ambiguous bounds remain unknown.

Shared-filesystem vLLM validation uses a unique session prefix, parses the entire
JSON/gzip stream in a thread with byte/time limits, and checks hashes, file
stability, and expected rank membership. Missing/corrupt/duplicate files fail;
valid files with unknown membership remain unverified. Linux perf, SGLang,
unmounted stores, and AsyncLLM CPU-file association remain unverified. Clock
probes estimate worker-minus-coordinator offsets and uncertainty without changing
clocks or rewriting traces; rank synchronization remains unattested.

OTel evidence consists of trigger trace/request IDs and an existing backend
reference. Coverage is unknown; local archival and runtime verbosity changes are
not required by the interface.

## Investigation bundle

After a session finishes, build its bundle outside the serving frontend:

```bash
python -m dynamo.profiling.bundle /private/profiling/sessions/<session-id> \
  --otel-jsonl /otel/collector-export.jsonl
```

The command writes `<session>/bundle/` with `manifest.json`, `bundle.json`
(file hashes and separate profiler/lifecycle coverage),
`lifecycle/traces.otlp.jsonl` (selected traces in collector export-batch form),
`lifecycle/trace-index.json`, `control/events.jsonl`, and `timeline.json.gz`
for Perfetto, and packages it with the raw native traces (Kineto rank traces,
Nsight `.nsys-rep` reports and their SQLite exports) under `profiles/<target>/`
as `<session>/bundle.tar.gz` with a `.sha256` sidecar.

A request is selected when its `worker.operation.*` span ran on a captured
target during that target's recording interval, or when the plan names its
trace ID. Worker spans are assigned through the router decision
(`kv_router.route_request` worker ID, equal to `handle_payload` instance ID)
and joined to the captured process by the PID prefix of `dynamo.process.epoch`;
a PID claimed by more than one target stays unmapped. Spans and native events
share the coordinator clock using each target's measured offset. Engine work
inside an operation span is shared batch work and is not attributed per kernel.
For Kineto sessions, `timeline.json.gz` is one overall Perfetto timeline for
the window: every selected request as an OTel span tree, the recording windows
and control events, and each captured worker's Kineto events over its recording,
written once. `--per-request` adds `timelines/<request>.json.gz` with only the
native events overlapping that request's operation spans.

For Nsight sessions the default output is `reports/`, for the Nsight Systems
multi-report view: every target's native report, and in the frontend's report
(or, without one, each worker's) the selected requests added as NVTX ranges in a
"Dynamo requests" group above all processes. Each request is a row named by
session, request and route, with its span tree nested parent-first and colored
by kind; recording windows and control events have rows of their own. Times are
converted with the measured clock offsets and clipped to the report's session,
so its time window is unchanged (spans entirely outside it are counted, not
drawn). The ranges are written with Nsight Systems' `nsys_writer` (needs the
`nvtx` package, `ai-dynamo[profiling]`) or, as a fallback, `ImportNvtxt`, which
places them below the processes; `reports/<target>.nvtxt` keeps them as text.
Point `--nsys-root` at the Nsight Systems install (default: `$NSYS_HOME` or the
install of `nsys` on `PATH`). `--perfetto` also writes the Perfetto timeline,
which is the fallback when neither tool is available. The frontend's detailed
Nsight host trace (OS runtime calls only where CPU sampling is not permitted) is
validated and packaged but not drawn.

## Callable API and hosting

```python
from dynamo.profiling import Coordinator, RuntimeSnapshot

runtime = RuntimeSnapshot(frontend_pid, frontend_epoch, hostname, workers)
coordinator = Coordinator(config, runtime, registry=registry,
                          clock_sampler=clock_sampler,
                          artifact_validator=artifact_validator)
session = coordinator.submit(plan)
coordinator.stop(session.session_id)
await coordinator.close()
```

The coordinator copies input snapshots and detaches manifest/inventory responses.
Trusted registry factories return collectors with `start()`, `stop()`, and
`evidence`; `ClockSampler` and `ArtifactValidator` are injectable protocols.
Built-in `nsys` requests require operator-configured prepared launches. Runtime
control uses the same client API; it invokes NVTX or backend profiling functions
without spawning Nsight CLI commands.
See the [Nsight Operator architecture](https://docs.nvidia.com/nsight-operator/Architecture/index.html)
and [user guide](https://docs.nvidia.com/nsight-operator/UserGuide/index.html).

The optional standalone host uses the same module, watches the frontend epoch,
and drains on shutdown. Run one host per socket:

```bash
python -m dynamo.frontend.profiling --config /private/profiling/config.json \
  --frontend-pid "$FRONTEND_PID" --frontend-epoch "$FRONTEND_EPOCH"
```

## Nsight collection

Install `ai-dynamo[profiling]` for the optional frontend NVTX dependency. At
process launch, prepare the frontend with CPU sampling, scheduling, OS runtime
tracing, and NVTX capture. Prepare each independent worker replica separately
with CUDA/NVTX tracing and CUDA-profiler capture. For vLLM, select its `cuda`
profiler; for SGLang the adapter sends `activities: ["CUDA_PROFILER"]`.
The backend must dispatch start/stop into its GPU execution processes. A vLLM
launch configured for Nsight advertises only Nsight worker capture; Torch capture
requires a separate vLLM launch with its Torch profiler configured.

These are deployment launch options; client requests never execute them:

```text
frontend:
  --trace=nvtx,osrt --sample=process-tree --cpuctxsw=process-tree
  --capture-range=nvtx --nvtx-capture=ROI@app
  --capture-range-end=repeat:100:async --export=sqlite
worker:
  --trace=cuda,nvtx,osrt
  --capture-range=cudaProfilerApi
  --capture-range-end=repeat:100:async --export=sqlite
```

Use an Nsight version supporting asynchronous repeat export; the frontend and
workers continue running between windows. The deployment controls collectors,
output paths, and the maximum number of windows. Configure a distinct, fresh
absolute output prefix for every process-tree launch:

```json
{
  "nsys_frontend_output_prefix": "/shared/run/frontend/frontend",
  "workers": [{
    "target_id": "decode-0",
    "process_epoch": "<worker-control-epoch>",
    "backend": "vllm",
    "control_url": "http://worker:9090",
    "nsys_output_prefix": "/shared/run/decode-0/worker",
    "verify_runtime": true,
    "expected_rank_ids": ["dp0_pp0_tp0_dcp0_ep0_rank0"]
  }]
}
```

These fields supplement the existing socket/output configuration. The prefix
must match that launch's Nsight output setting, and its directory must be mounted
in the coordinator. A single coordinator must own all capture triggers for those
launches. Use `mode: "nsys"` in the existing client plan and select `frontend`
explicitly when wanted; frontend perf is not implicit in this mode. An explicit
client stop or the existing lease deadline closes the frontend range and sends
worker stop requests concurrently.

The coordinator waits for new `.nsys-rep`/`.sqlite` pairs, checks final export
metadata, SQLite integrity, relevant activity, file stability, and SHA-256.
Existing reports cannot satisfy another window. `verified` establishes the
reported validation scope; it does not prove every configured event was collected.
Frontend evidence requires CPU activity beyond NVTX and records missing samples,
context switches, and Nsight diagnostics. Host policy may permit OS runtime
tracing while denying CPU sampling. Worker evidence requires CUDA kernel or memory activity and,
when rank membership is configured, the matching GPU process count. Rank identities
remain unattested; Nsight may combine a replica's ranks into one report.
Native report bounds and host clock estimates are recorded separately.

Frontend NVTX control requires the embedded host. A standalone coordinator can
control workers, but needs a future IPC adapter to trigger frontend NVTX.
Export failure blocks further admission until the operator reconciles the launch,
preventing a late export from being assigned to a later capture. Report transfer
and Nsight Cloud upload remain separate extensions.
