<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tracked POC Evidence

Raw and compiled experiment directories are intentionally ignored by Git. This
directory retains the small, sanitized, reviewable evidence needed for a fresh
checkout to understand and verify the reported result.

- `20260828-canonical-controlled-pair.json` is the bidirectional pairing
  manifest for the canonical controller and workload runs. It records outcome
  rollups and SHA-256 digests of the local raw artifacts.
- `20260828-canonical-native-planner.json` is the canonical native control-plane
  manifest. It records the pre-run zero-worker setup, mutation-free experiment
  driver window, worker DGDSA `0 -> 1`, readiness-gated `0 -> 5 -> 0` admission,
  100/100 result, terminal worker/floor/lease state, validation rollups, and
  SHA-256 digests of the ignored raw and compiled artifacts.
- The exact successful port forwards, controller invocation, and workload
  invocation are preserved in the
  [workload guide](../../workloads/README.md#canonical-controlled-run).
- Async image source/build reconstruction is recorded in
  [the provenance note](../../research/20260828-async-image-provenance.md).

Each manifest is a sanitized synthesis, not a replacement for its raw run. A
reviewer who has the ignored artifacts can recompute every listed digest. The
standalone-controlled manifest records post-expiry Redis state, but no request
was submitted after expiry. The native manifest records a fresh terminal zero
lease; Async's idle drain gauge is explicitly last-evaluated rather than
authoritative.
