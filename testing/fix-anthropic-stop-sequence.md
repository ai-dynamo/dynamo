<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Anthropic stop-sequence reporting test contract

Tracked issue: https://github.com/ai-dynamo/dynamo/issues/15795

## Functional behavior

- A matched string stop produces `stop_reason: stop_sequence` and the exact
  matched string in `stop_sequence`, for unary and streaming Messages responses.
- A natural stop or numeric token stop remains `end_turn` with no named sequence.
- Length, tool-call, and content-filter finish reasons retain their existing
  meanings even if unrelated string metadata is present.
- Native stop reporting works without client `nvext` opt-in and when client
  extensions are disabled. Existing permitted request fields remain intact.
- The OpenAI route and worker wire format do not change.

## Unit tests

Exercise the unary converter and production SSE converter with matched string,
natural stop, numeric stop, length, tool-call, and content-filter outputs.
Check serialized output as well as typed state.

## Integration tests

Use the existing HTTP service test infrastructure to send unary and streaming
Messages requests with `stop_sequences`. Confirm the request selects the existing
internal matched-stop metadata and the HTTP response reports it. Include a
client-extensions-disabled case and controls for other finish reasons.

## Smoke and E2E tests

The same HTTP tests exercise the public route and terminal SSE events.
No GPU model or live provider is needed for this conversion change. GPU inference
and an external consumer rerun are outside this validation scope.

## Manual reproduction

The issue links the existing pinned reproduction and raw wire evidence.
Do not replace that reproduction. Regression tests must fail on the unmodified
base and pass with the fix. Run formatting, relevant Rust tests, Clippy, and the
configured pre-commit checks before publishing the PR.
