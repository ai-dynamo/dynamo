<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bounded vLLM extension admission

This change admits a bounded sampling subset on chat/completion token pipelines,
preserves its meaning through preprocessing and selected-worker dispatch, and
rejects unsupported combinations before they silently degrade. It is not a claim
of complete vLLM server parity or complete N-2 deployment conformance.

## Scope and dependencies

The review base is the explicit B1/B2/B4 integration commit
`df905cc155ef929e9787b3d84fe676b0531d402e`. It includes pinned field inventory,
public error-parameter transport, chat request ownership and finite logprob
projection. The runtime catalog endpoint and schema exposure are a separate
dependent change; this branch does not expose that endpoint.

Known native fields come from the generated inventory; inventory membership is
not evidence of support. Unsupported known semantic fields are rejected even
under the migration policy that ignores other unsupported fields. Exact vLLM
0.29.0 and 0.30.0 identities select the inspected streaming-prompt rules;
unidentified targets are not guessed from model names, and other versions do
not inherit a verified profile.

## Public fields and internal transport

Clients use root-level `allowed_token_ids`, `bad_words_token_ids`, and
`logprob_token_ids`. The first and third accept token-ID lists; bad-word IDs
accept lists of token-ID sequences. Null and empty values retain the explicit
normalization rules and native adapter checks; booleans are not integer IDs.

The internal extension envelope is versioned separately from the public API.
Readers validate schema, names and shapes, reject conflicting canonical values,
and bound extension data to 64 KiB, depth 8 and 16,384 values. Writers retain
compatible legacy sampling copies where required. This is not unrestricted
forwarding of arbitrary JSON into engine keyword arguments.

Admission uses the selected pipeline's runtime capability and processor identity.
Routing also validates the selected target, including a prefill hop; accepting a
field at the HTTP boundary alone does not authorize sending it to any worker.
An explicit processor identity describes the implementation, not its capability.
Existing WorkerSet admission/checksum and generation fences remain authoritative.

## Full-vocabulary prompt logprobs

Public `prompt_logprobs=-1` is mapped to an internal unsigned sentinel only
through guarded handling. It requires a supported token pipeline and runtime
capability with schema version 1, `wire_count=u32_max`, a valid vocabulary size,
and an engine limit permitting the complete vocabulary. Missing capability is
not permission. Streaming full-vocabulary requests are rejected; ordinary
finite-count behavior remains covered by the dependency branch.

For identified vLLM targets with inspected rules, positive prompt counts with
streaming are also rejected. Zero is a distinct value, not a positive count.
Response placement and finite projection are described in
[the projection guide](logprob-response-projection.md).

## Explicit legacy deployments

`--legacy-vllm-targets` accepts frontend startup JSON declarations for exact
namespace, component, endpoint, model, worker role and Dynamo release. These are
operator-owned configuration, not request parameters or fabricated worker
advertisements. No wildcard or backend-name inference is allowed. Configuration
is bounded to 128 declarations and 64 KiB and is passed explicitly through the
Python/Rust boundary rather than written back into environment variables.

The temporary declarations cover Dynamo 1.4.0 and 1.5.0 token-worker lowering
during the current N-2 window. They do not enable full-vocabulary output or
establish whole-server compatibility. Version-specific readers/writers remain
at transport boundaries, with removal marked for Dynamo 1.8.

## Developer validation and limits

The relevant existing tests are:

- Rust admission/profile, backend-extension, legacy-config, selected-hop and
  discovery unit tests;
- `frontend_native_field_admission_http` and
  `protocol_logprob_token_selection` integration tests;
- Python extension/handler, explicit legacy and processor identity contracts;
- the opt-in native HTTP and separately pinned mixed-release HTTP regressions.

Pure schema/adapter tests do not prove GPU inference or a mixed-release service
boundary. The release tests require actual pinned wheels and engine environments;
a skipped opt-in test is not a pass. The branch review package records the exact
executed cases, source state and limitations. Other servers, deployment modes,
fields and upstream drift candidates remain continuation work.

Compatibility rejection details and metrics are bounded and payload-free.
Do not add request values, arbitrary field names, model names or worker identity
as metric labels. The custom Python backend call site now reuses the shared
exception mapper; compile coverage and shared-helper tests alone do not prove
the complete custom-engine or N-2 public-parameter delivery path.
