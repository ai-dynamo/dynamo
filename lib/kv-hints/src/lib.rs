// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV transfer hints: what the Dynamo router hands an engine so the selected worker can fetch
//! a KV prefix from another worker or cache owner, and the parts of that decision that are
//! plain data. The counterpart of `dynamo-kv-events` (what engines publish to the router):
//! engines and their sidecars consume this crate, the router's stages and kernel produce with
//! it.
//!
//! - [`wire`]: the hint message and its versioned actions ([`KvHint`], [`KvHintAction`],
//!   [`KvSourceLocationsPayload`]), the runtime-config keys a worker advertises the capability
//!   under, and, re-exported from the indexer, [`KvTransferCandidateSource`], who holds a
//!   prefix a target could fetch.
//! - [`transfer`]: the source choice. A worker's hashed hint identity
//!   ([`WorkerKvTransferKeys`]), the [`KvTransferTarget`] view a selection reads, the
//!   [`KvTransferScope`] a summary answers, and [`KvTransferSourceSummary`], which groups the
//!   [`ResolvedTransferSource`]s so the source for any target is found in O(1).
//! - [`keys`]: the key builder over a worker's runtime-config fields
//!   ([`WorkerKvTransferKeys::from_worker`]), the once-per-config [`WorkerKvTransferKeysCache`],
//!   the two plain data types a worker's runtime config speaks
//!   ([`KvHintTransferWorkerMetadata`], [`KvTransferEnforcement`]), and the indexer's label
//!   hashing ([`label_key`], [`topology_keys`]), re-exported so both sides hash alike.
//!
//! Deliberately elsewhere: the transfer candidate vocabulary (`KvTransferCandidateSource`,
//! `label_key`, `topology_keys`), `KvTransferCandidates`, `CacheOwnerKvTransferKeys`, and the
//! residency owner key are `dynamo-kv-indexer`'s, because the indexer produces them with a
//! match (the producer owns its output); this crate depends on the indexer and re-exports the
//! three names it has always exposed, and the indexer never names this crate.
//! `resolve_transfer_sources`, which joins the indexer's candidates with worker keys under a
//! selection scope, the `WorkerConfigLike` adapter (`WorkerKvTransferKeysExt`), and the attach
//! point that puts the hint on the selected request stay in `dynamo-scheduler`.
//!
//! Among router crates this one depends on `dynamo-router-protocol` and `dynamo-kv-indexer`;
//! no transport, no stage logic of its own. The scheduler kernel, `dynamo-kv-router`, and the
//! hosts consume it; no stage depends on it.

pub mod keys;
pub mod transfer;
pub mod wire;

pub use keys::{
    KvHintTransferWorkerMetadata, KvTransferEnforcement, WorkerKvTransferKeysCache, label_key,
    topology_keys,
};
pub use transfer::{
    KvTransferScope, KvTransferSourceChoice, KvTransferSourceSummary, KvTransferTarget,
    ResolvedTransferSource, WorkerKvTransferKeys,
};
pub use wire::{
    KV_HINT_TRANSFER_CAPABILITY_KEY, KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, KvHint, KvHintAction, KvSourceLocationsPayload,
    KvTransferCandidateSource,
};
