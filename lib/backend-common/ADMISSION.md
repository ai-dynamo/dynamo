<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Controller-managed admission

`WorkerConfig.controller_managed` defaults to false. When enabled, an observed
engine needs controller permission and a matching ready observation to join
discovery. Adapters must implement `verify_serving_admission`; its default
fails closed. Only the shared worker writes discovery.

## API

Enable the worker's system listener with `DYN_SYSTEM_PORT` and restrict access
to trusted controllers. These POST routes exist only in managed mode:

| Route | JSON body |
| --- | --- |
| `/engine/admission/status` | `{}` |
| `/engine/admission/admit` | Example below |
| `/engine/admission/withdraw` | `{"session":"<from status>","revision":2}` |

```json
{
  "session": "<from status>",
  "revision": 1,
  "expected": {
    "engine_instance_id": 42,
    "observation_epoch": 1,
    "weight_version": "17"
  }
}
```

Validate the worker association, take instance/epoch from status's `observed`,
and supply the controller's expected version, not blindly the reported version.
Use one command writer and increasing positive revisions per session. Exact
retries return status without reviving failed/revoked permission; stale or
conflicting commands are rejected. A valid newer command clears prior permission
even if verification fails; retry failures with a new revision.

Admission checks observations before/after live verification (ten-second limit).
Status separates `admitted` (permission), `engine_ready` (observation), and
`published` (local discovery result; null means unknown/shutting down).
After a command, wait for its `applied_revision` and intended `published` value
(and `admitted=true` for admission); check `revision`/`last_error` for
superseding commands or failures. HTTP success is not a publication barrier.

Instance/version changes, observation loss, producer closure, or a new stream
epoch revoke permission. Producers must advance epochs on reconnect, even when
disconnect updates coalesce. Sidecar restart creates a new session. Pause/health
changes preserve a matching grant, allowing automatic recovery; resume alone
cannot create a grant.

Before updating weights, stop/resolve submissions, withdraw and await publication,
then pause/update/commit/resume the engine and admit the validated instance/version.
Withdrawal does not pause/drain execution. Session tokens are not authentication;
discovery acknowledgments are not router-cache convergence, a per-request fence,
or proof of tensor/TP-rank consistency. Observed mode does not support the separate
`--enable-rl` endpoint.
