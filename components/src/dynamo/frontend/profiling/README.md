# On-demand profiling interface PoC

The frontend owns a helper process with a private Unix-socket administration API.
A client or telemetry service can request an immediate, bounded capture while
inference is serving. The helper's event loop handles capture deadlines even when
the frontend's inference loop stalls. Enable it with `--profiling-config config.json`.
The helper starts before the frontend input service and finalizes on frontend shutdown
or detected parent exit. The option is off by default.

`torch_profile` combines frontend `linux_perf` with `torch_profiler` on
explicitly selected vLLM/SGLang engine groups. Selecting only workers still includes
the frontend. Selecting only the frontend records perf without a worker capture.
Perf attaches to the existing frontend PID and threads with 99 Hz user-space
CPU-clock sampling and DWARF call stacks. It starts disabled and confirms counter
enablement through the perf control ACK before recording the start boundary.
It does not require launching the frontend under perf. The helper runs as the same
user; deployment must allow perf attachment. Permission/start failures fail the
session and retain diagnostics; there is no silent worker-only fallback.

For temporary worker-only testing where perf is denied, set the operator config
`"frontend_perf_enabled": false`. The default is `true`. This explicitly skips
frontend perf; target inventory and plan validation report no frontend collector,
and manifests record `frontend.profiling_enabled: false` with
`omission_reason: "disabled_by_operator_config"`. Requests that explicitly select
the frontend return 422 while disabled. Worker leases, cleanup and OTel correlation
remain active. Re-enable frontend perf for complete frontend/worker captures.

## Operator configuration

Use private directories owned by the serving user with permissions `0700`.
The socket is owner-only. This PoC API is intended for a local client or a trusted
telemetry sidecar sharing the socket directory. Network authentication and a public
inference-port route are deferred.

```json
{
  "socket_path": "/run/dynamo-profiling/admin.sock",
  "output_dir": "/var/lib/dynamo-profiling",
  "frontend_perf_enabled": true,
  "max_duration_ms": 60000,
  "start_timeout_s": 10.0,
  "finalize_timeout_s": 1200.0,
  "workers": [
    {
      "target_id": "decode-0",
      "process_epoch": "operator-verified-worker-generation",
      "backend": "vllm",
      "control_url": "http://decode-0:9090",
      "start_options": {"profile_prefix": "incident"},
      "artifact_location": "shared-storage://decode-0/profiles"
    }
  ],
  "lifecycle_evidence_location": "telemetry-backend://deployment/traces"
}
```

For SGLang, `start_options` must match its existing `ProfileReq` fields. For vLLM,
configure its torch profiler and output directory at engine launch. Paths and URLs
are trusted configuration, never request parameters. Worker artifact destinations
must be readable by the investigator; the helper does not download engine files.
Operators must reserve these hooks exclusively for this helper during the PoC.
Existing worker hooks have no lease/ownership check or restart-safe epoch validation.
An ambiguous start is followed by a best-effort stop; timeouts leave coverage unknown.
A worker must be reconfigured after restart. These limitations prevent production
use until backend lease and self-expiring capture contracts are added.

## Client requests

Discover current targets and their process epochs:

```bash
curl --unix-socket /run/dynamo-profiling/admin.sock \
  http://localhost/admin/profiling/v1/targets
```

Use the returned epochs in a plan. POST to `/plans/validate` first if desired:

```json
{
  "schema_version": 1,
  "client_request_id": "incident-unique-id",
  "mode": "torch_profile",
  "duration_ms": 10000,
  "targets": [
    {"target_id": "decode-0", "process_epoch": "operator-verified-worker-generation"}
  ],
  "trace_ids": ["trigger-trace-id"],
  "request_ids": ["trigger-request-id"],
  "reason": "elevated decode latency"
}
```

```bash
curl --unix-socket /run/dynamo-profiling/admin.sock \
  -H 'Content-Type: application/json' --data-binary @plan.json \
  http://localhost/admin/profiling/v1/sessions
```

HTTP 202 returns a session ID; it acknowledges acceptance rather than collector
startup or artifact readiness. Poll `GET /sessions/{session_id}` and retrieve the
manifest using `GET /sessions/{session_id}/artifacts`. Stop early with
`POST /sessions/{session_id}/stop` (202). All paths share `/admin/profiling/v1`.
Reusing a request ID with the same plan returns the same session and never extends
its deadline. Changing the plan or requesting a concurrent session returns 409.
Invalid plans return 400; stale epochs return 409. Validation checks the plan and
inventory, not perf permissions or remote backend readiness.

The capture budget starts at session acceptance and includes
collector startup. States are `accepted`, `starting`, `recording`, `finalizing`,
`completed` or `failed`. The lease triggers stop independently of the client.
All collectors receive stop concurrently; backend serialization can keep the
session finalizing without extending frontend perf recording. Only the helper's
own perf recorder receives SIGINT. After a 30 second flush timeout it kills that
recorder and marks the artifact partial. It never signals the inference process.
One session, including finalization, reserves the inventory. The bounded in-memory
idempotency index is retained until helper restart (64 sessions by default); hitting
that limit rejects new sessions rather than forgetting request IDs.

## Evidence and deferred work

Each session writes `manifest.json`. When frontend perf is enabled, it also writes
`frontend.perf.data` and `frontend.perf.log`.
The manifest pairs collector intervals and identities with trigger trace/request
IDs and an optional reference to the existing lifecycle OTel telemetry backend.
OTel coverage remains `unknown` until an evidence-provider adapter queries/exports
retained spans and confirms coverage. No local OTLP archive is assumed. A completed
capture does not imply verified worker files or complete lifecycle trace evidence;
these have separate explicit coverage fields. Cross-host clock alignment is unknown.
A later artifact provider can add exported lifecycle spans, file checksums and
retrieval locations to this same bundle contract.

This draft rejects `nsys` with 422. A future adapter must control a prepared Nsight
session on processes launched under nsys; spawning an nsys child cannot instrument
an already-running frontend parent. Nsight Cloud integration is a separate provider.
Runtime lifecycle OTel verbosity changes are also deferred.

Other follow-ups: dynamic worker discovery, remote epoch verification, backend
self-expiring leases, helper restart recovery, per-resource reservations, cooldowns,
artifact storage quotas/uploads, and a network-authenticated service API. Killing
the helper or whole container cannot be handled by this process-local watchdog.

Interface tests exercise validation, idempotency, deadline cleanup, stop during
finalization, both engine adapters, HTTP-200 backend errors, and perf control wiring
with fake collectors. Actual perf permissions and live engine exports need a Linux
serving deployment smoke test.

Perf command semantics: [upstream perf record manual](https://man7.org/linux/man-pages/man1/perf-record.1.html).
