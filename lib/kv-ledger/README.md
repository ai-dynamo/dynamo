<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-ledger

`dynamo-kv-ledger` keeps one ordered KV cache event stream as what a consumer that joins late needs: the most recent batches for replay, and a block ledger that a snapshot is cut from. Dynamo's worker-side `LocalKvIndexer` is built on it, and an engine's own Rust frontend can embed it to serve replay and snapshots itself.

It is synchronous, single-owner and transport-neutral: no threads, locks, channels or async, and no Dynamo types. It depends on `rustc-hash` only and builds on Rust 1.92 and later.

| The host serves | It calls |
|---|---|
| the live stream | `Replay::apply`, then publishes the batch it passed in |
| replay from a sequence | `Replay::since` |
| a snapshot, then the batches after it | `Replay::snapshot` |

`Replay` folds each batch into a state the host chooses. `Ledger` is that state for one owner's blocks, keyed by the engine's own block hash: `Ledger::chains` gives a tier's blocks parents first, which is the order a prefix index loads them in.

A block is on a tier or it is not. An engine that keeps duplicate blocks under one hash and removes them one at a time (vLLM) has to count its removes before they reach the ledger, as Dynamo's publisher does.
