<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-kv-ledger

`dynamo-kv-ledger` keeps one ordered KV cache event stream as what a consumer that joins late needs: the most recent batches for replay, and a block ledger that a copy of the state is cut from. Dynamo's worker-side `LocalKvIndexer` is built on it, and an engine's own Rust frontend can embed it to serve replay and snapshots itself.

The ledger keeps one value of the host's per block (Dynamo keeps a token hash), in the order blocks were stored. What belongs to an event and not to a block, such as a session id, stays in the replayed batches.

It is synchronous, single-owner and transport-neutral, with no Dynamo types. It depends on `rustc-hash` only and builds on Rust 1.92 and later. The API and its contract are in the crate docs (`src/lib.rs`).
