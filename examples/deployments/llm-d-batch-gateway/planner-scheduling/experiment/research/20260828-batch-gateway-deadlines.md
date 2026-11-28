<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Batch Gateway v0.3 deadline compatibility

- Date: 2026-08-28
- Scope: Planner POC reads through the public Batch Gateway API only.

## Public deadline fields

Batch Gateway v0.3 normally returns `expires_at: null` until a later status
transition. Planner therefore prefers a non-null `expires_at`, but otherwise
derives a deadline as `created_at + completion_window`. The fallback accepts
the positive subset of Go `time.ParseDuration` used by the Gateway, including
decimal and compound `h`, `m`, `s`, `ms`, `us`, `µs`, `μs`, and `ns` values.
Missing, malformed, non-positive, negative, or overflowing inputs fail closed.

The Gateway records `created_at` at whole-second precision, while its internal
SLO is based on a nearby `time.Now() + duration` value with finer precision.
The public-field fallback is consequently conservative: it is normally up to
about one second earlier because of timestamp truncation (plus any time spent
by the Gateway between its two clock reads). This is acceptable for the POC
because it can only cause slightly earlier admission or scaling. It does not
add support for an arbitrary user-selected due date; the submitted
`completion_window` remains the only public deadline input.

## Request-count compatibility and trust boundary

In Gateway v0.3, a nonterminal job can report
`total=completed=failed=0` until its first result completes, even after requests
are queued. Zero at that point means the public count is not known yet, not that
the job has no work. Waiting for a live count would deadlock when Planner's
initial drain limit is zero.

For this POC, the trusted job submitter must record the immutable input record
count in OpenAI Batch metadata at creation time:

```json
{"metadata":{"planner_request_count":"1000"}}
```

The value is a string in canonical base-10 form matching `[1-9][0-9]*` and may
not exceed signed int64. For a nonterminal job whose live `total` is zero,
Planner requires and substitutes this declaration. Once the Gateway exposes a
positive live total, Planner requires equality if the declaration is present,
including for terminal history; a mismatch fails the whole observation closed.
The Gateway's live or final counter remains authoritative once it is positive.

The trust boundary is the authenticated tenant's submitter: Gateway v0.3 has no
API for changing Batch metadata after creation, but it does not independently
verify that this declared count matches the uploaded JSONL. The POC is scoped
to one active batch job in the selected Gateway tenant and worker pool. General
multi-job or untrusted-tenant planning needs an independently verified input
record count rather than submitter metadata.

## Local source evidence

- `internal/apiserver/batch/batch_handler.go` records `created_at` with
  `time.Now().UTC().Unix()` and stores the internal SLO from a later
  `time.Now().UTC().Add(completionDuration)` call.
- `internal/shared/openai/batch.go` validates `completion_window` with Go
  `time.ParseDuration` and exposes nullable `expires_at`.

These files were inspected in the local llm-d Batch Gateway checkout at
revision `229674c` on 2026-08-28.
