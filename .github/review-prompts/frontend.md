<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend codeowner review

Review the proposed change for frontend defects. Read the diff and surrounding code to establish what the change affects. Focus on the affected HTTP and gRPC handlers, protocol conversions, preprocessing, frontend Python processors, and their directly connected callers and backend adapters. Read enough surrounding code to establish the production path; do not scan unrelated code.

Apply the following concerns only where the change makes them relevant. They describe recurring frontend codeowner feedback, not a requirement to find an issue in every category.

## Request semantics

1. Preserve the request's meaning across the Rust frontend, Python chat processors, protocol
   converters, and supported backend adapters. Trace explicit values, omitted defaults,
   configuration overrides, model configuration, and parser-adjusted options to their real
   consumer, including dynamically discovered models.

2. Check the pinned dependency's actual types and behavior before alleging a compatibility
   defect. Do not silently discard a requested constraint, accept a parser name that downstream
   code treats differently, or reject a value that the declared interface supports.

3. Validate incompatible guided-decoding and forced-tool options before backend generation
   starts. Treat protocol-specific behavior and documented staged scope as authoritative; another
   backend's different policy alone is not a defect.

## Streaming

4. Check streamed output as an ordered sequence of client-visible events, not only as a final
   aggregate. Follow reasoning, visible text, and tool calls through parser state changes,
   buffering, cancellation, and end-of-stream.

5. Preserve supported literal text and complete Unicode characters while removing only actual
   control tokens.

6. Tool-call assembly must retain choice and call identity, combine argument fragments correctly,
   distinguish genuine replay from conflicting identity, and dispatch only when the relevant
   protocol permits it.

7. Do not turn truncated or filtered output into a successful finish, invent complete arguments
   from ambiguous fragments, or emit successful finalization after a terminal error. Compare
   streaming and non-streaming behavior against each interface's own contract rather than
   requiring identical representations across protocols.

## Errors

8. Preserve typed failures through conversion, transport, aggregation, and HTTP or gRPC mapping.
   Distinguish absent optional data from present but malformed data.

9. Verify which failures must be returned before streaming headers and how failures after headers
   are represented. Do not convert an invalid request into HTTP 500, turn unsupported
   functionality into the wrong status by blanket conversion, or replace a failure with empty
   successful output.

10. Verify the existing source error chain and classification before proposing changes; do not
    infer an error type from arbitrary message text.

11. Keep error bodies and logs bounded and avoid including entire schemas or inline media.

## Usage and timing

12. Keep token usage and timing observations tied to the original generated tokens and their
    arrival. Check multiple choices, hidden reasoning, parser buffering, local stop detection,
    cancellation, and empty terminal chunks when those paths change. Establish whether a backend
    reports per-choice counts or a request total before changing aggregation.

13. Observe TTFT and inter-token timing before an operation that buffers or folds the stream, and
    do not retokenize projected text as a substitute for original token IDs when that changes the
    count.

## Media and transport

14. For changed media-fetch and transport paths, follow policy enforcement through redirects, DNS
    resolution, the actual connection, proxy routing, and connection reuse. Verify that
    per-request policy reaches the connection and that supported paths cannot bypass a newly
    introduced media-size limit.

15. Preserve hostname identity for TLS checks and pooling, bounded connection timeouts, supported
    address fallback, and HTTP protocol negotiation. Report a concrete reachable bypass or
    compatibility regression, not a hypothetical security checklist item.

## Work per request

16. Inspect work introduced on every request or output chunk. A schema clone, repeated parser
    cleanup, serialization immediately followed by deserialization, or routine per-request logging
    deserves a finding only when the changed code establishes redundant work with a concrete cost
    or material payload-dependent growth. Name the affected path and avoid speculative performance
    claims.

## Test evidence

17. Use tests as evidence for the production contract. A helper test does not establish that an
    option survives the Python-to-Rust boundary, that a real handler preserves SSE order, or that
    an error reaches the client with the expected status. When a changed test claims such
    coverage, verify its actual exercised path and assertions.

18. Recommend the smallest test that distinguishes the demonstrated failure, using existing test
    infrastructure where possible. Do not report missing tests alone, stylistic preferences,
    redundant comments, or opportunities for general cleanup.

## Reporting findings

Report only high-confidence, actionable defects introduced by the change. Treat these concerns as directions for investigation, not mandatory findings. Exclude summaries, praise, style preferences, speculative failures, unrelated existing problems, and requests for tests without an identified behavioral defect. When review history is available, read the discussion and author explanations, verify them against the code, and account for withdrawn concerns and explicitly staged follow-ups. Do not repeat an underlying defect already raised there, even if it remains unfixed or unresolved. Keep one finding per underlying defect.

For each finding, identify the file and relevant lines, explain the triggering condition and concrete consequence, and describe the narrow correction. Verify any identifiers or replacement code you propose, including relevant ownership and lifetime constraints. If no defect qualifies, state that no findings were identified.
