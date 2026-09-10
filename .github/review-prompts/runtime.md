<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime codeowner review

Review the proposed change for runtime defects. Read the diff and surrounding code to establish what the change affects. Concentrate on changed runtime, transport, discovery, Rust/Python binding, and shared backend code assigned to the runtime group. Follow directly affected callers and callees to establish behavior; do not scan unrelated code or assume every concern below applies.

## Error propagation and admission

- [ ] Trace errors from their origin through serialization, acknowledgments, response-stream
      prologues, deserialization, and the caller's retry decision. Verify that overload,
      cancellation, and unavailable-worker errors retain the distinctions the actual caller
      requires. A correctly typed local error is insufficient if a later boundary turns it into an
      unrelated type or a string that its consumer never interprets.

- [ ] Check the actual transport paths before claiming that TCP and NATS have equivalent behavior:
      admission can occur after an upstream semaphore or response-stream allocation, so its
      effective capacity and retained resources depend on that ordering. Report only a concrete
      consequence supported by the changed path and its callers.

## Discovery and lifecycle

- [ ] When discovery or worker availability changes, distinguish an uninitialized discovery source
      from an authoritative empty snapshot. Trace removal of the final worker, stale scheduler
      candidates, and subsequent restoration. Check whether removed or expired state continues to
      affect selection or imposes repeated work on every request.

- [ ] Inspect the lifetime and ownership of the changed request state, permits, buffers, and tasks
      across errors and early termination; identify a reachable leak, stale-state use, or incorrect
      release before reporting it.

- [ ] Distinguish explicit runtime shutdown from object destruction: a detached cancellation token
      reached only by Drop can leave a listener accepting after shutdown when Python still holds the
      runtime object.

- [ ] Preserve the intended drain ordering when checking how listeners eventually stop. A
      momentarily empty channel does not prove a drain is complete while producers can still send;
      verify that shutdown first stops admission and then accounts for already accepted work.

## Performance

- [ ] Check performance changes against all supported callers of the changed helper. Buffer reuse
      can increase allocations for one-frame responses or for callers that still require an owned
      vector. A conversion that returns a copy is not a no-op merely because the input already has
      the requested format.

- [ ] Look for new per-request allocations, locks, clock reads, or scans whose concrete cost follows
      from the implementation, especially when state could instead be updated when discovery
      changes. Do not demand a benchmark or a preferred design as a finding without establishing a
      regression in a supported path.

## Python and Rust interfaces

- [ ] Follow public options and types across Python signatures, bindings, Rust configuration, and
      their actual consumers. Preserve the established positional argument prefix unless the change
      explicitly and consistently revises that contract.

- [ ] An accepted option must reach the component that implements it or be rejected where
      unsupported. Check that runtime construction and bridge initialization actually use the
      configured runtime or thread count.

- [ ] Before alleging a compatibility break, inspect the declaration and supported callers; do not
      turn naming, typing style, or an intentional staged migration into a defect.

## HTTP media fetching

- [ ] For changes to shared HTTP media fetching, trace the actual outbound connection rather than
      stopping at initial URL validation. Check redirects, proxy routing, DNS resolution, connect
      deadlines, TLS hostname verification, and per-call policy wherever the diff changes those
      behaviors. A validator cannot protect a fetch performed by another client that bypasses it.

- [ ] Preserve logical hostname and policy identity when pooling connections or pinning dial
      addresses, and confirm the documented behavior of the pinned HTTP library before reporting a
      bypass or regression.

## Telemetry

- [ ] For runtime telemetry changes, verify that metric-family metadata is committed only when the
      corresponding family is accepted, and that rejected duplicates cannot change a surviving
      family's type or unit.

- [ ] Check that asynchronous spans receive their intended parent when constructed; entering a span
      only when a future is polled does not reparent an already-created child. Report demonstrably
      incorrect classification, exported data, or parentage, rather than requesting additional
      instrumentation that the pull request explicitly defers.

## Reporting findings

Report only high-confidence, actionable defects introduced by the change. Treat these concerns as directions for investigation, not mandatory findings. Exclude summaries, praise, style preferences, speculative failures, unrelated existing problems, and requests for tests without an identified behavioral defect. When review history is available, read the discussion and author explanations, verify them against the code, and account for withdrawn concerns and explicitly staged follow-ups. Do not repeat an underlying defect already raised there, even if it remains unfixed or unresolved. Keep one finding per underlying defect.

For each finding, identify the file and relevant lines, explain the triggering condition and concrete consequence, and describe the narrow correction. Verify any identifiers or replacement code you propose, including relevant ownership and lifetime constraints. If no defect qualifies, state that no findings were identified.
