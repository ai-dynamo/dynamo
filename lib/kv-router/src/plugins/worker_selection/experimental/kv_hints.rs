// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Policy hook for the router's KV fetch hint.
//!
//! When workers consume KV hints, a booked selection can carry a `kv.fetch` hint that tells the
//! selected worker to fetch a longer cached prefix from another worker before prefill. By default
//! the router fetches from the source with the longest prefix. A [`KvTransferPolicy`] lets a
//! plugin keep, redirect, or drop that fetch:
//!
//! 1. The plugin factory builds its [`WorkerSelectionPolicy`] as usual and attaches the fetch
//!    policy with [`with_kv_transfer_policy`].
//! 2. Filters, scorers, and the picker select a worker. The fetch policy cannot change it.
//! 3. When at least one eligible source holds a longer prefix than the selected worker, the router
//!    calls [`KvTransferPolicy::decide`] with every eligible source, longest prefix first.
//! 4. The router builds the hint from the returned [`KvTransferAction`].
//!
//! ```
//! use dynamo_kv_router::KvRouterConfig;
//! use dynamo_kv_router::plugins::worker_selection::experimental::kv_hints::{
//!     KvTransferAction, KvTransferInput, KvTransferPolicy, with_kv_transfer_policy,
//! };
//! use dynamo_kv_router::plugins::worker_selection::{WorkerPicker, WorkerSelectionPolicy};
//!
//! /// Fetch only when the default source adds at least `min_blocks` blocks.
//! struct MinFetch {
//!     min_blocks: u32,
//! }
//!
//! impl KvTransferPolicy for MinFetch {
//!     fn decide(&mut self, input: KvTransferInput<'_>) -> KvTransferAction {
//!         if input.additional_blocks() >= self.min_blocks {
//!             KvTransferAction::Default
//!         } else {
//!             KvTransferAction::Skip
//!         }
//!     }
//! }
//!
//! fn build(config: &KvRouterConfig, picker: Box<dyn WorkerPicker>) -> WorkerSelectionPolicy {
//!     let policy = WorkerSelectionPolicy::new(config.clone(), "decode", Vec::new(), picker);
//!     with_kv_transfer_policy(policy, Box::new(MinFetch { min_blocks: 64 }))
//! }
//! ```

use crate::kv_hints::KvTransferCandidateSource;
use crate::plugins::worker_selection::inputs::WorkerTable;
use crate::plugins::worker_selection::{WorkerMetadata, WorkerSelectionPolicy};
use crate::protocols::WorkerWithDpRank;

/// What the router does with its KV fetch hint for the selected worker.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvTransferAction {
    /// Keep the router's choice: fetch from the source with the longest prefix.
    #[default]
    Default,
    /// Do not attach a `kv.fetch` hint to this request.
    Skip,
    /// Fetch from `input.sources()[index]`. An out-of-range index attaches no hint.
    FetchFrom(usize),
}

/// One source the selected worker can fetch from: another worker, or a cache owner such as a
/// KV pool attached to a worker.
#[derive(Clone, Copy, Debug)]
pub struct KvTransferSource {
    pub(crate) source: KvTransferCandidateSource,
    pub(crate) worker: Option<WorkerWithDpRank>,
    pub(crate) prefix_blocks: usize,
}

impl KvTransferSource {
    /// The worker that holds this KV: the source worker itself, or the worker a cache owner is
    /// attached to. None for a cache owner attached to no worker.
    pub fn worker(self) -> Option<WorkerWithDpRank> {
        self.worker
    }

    /// Whether a cache owner holds this KV, rather than a worker's own cache.
    pub fn is_cache_owner(self) -> bool {
        matches!(self.source, KvTransferCandidateSource::CacheOwner(_))
    }

    /// Root-aligned prefix blocks this source holds. Always greater than the selected worker's
    /// [`KvTransferInput::local_prefix_blocks`].
    pub fn prefix_blocks(self) -> u32 {
        u32::try_from(self.prefix_blocks).unwrap_or(u32::MAX)
    }
}

/// The fetch options for one selection: the selected worker and every source it can fetch from.
#[derive(Clone, Copy)]
pub struct KvTransferInput<'a> {
    pub(crate) target: WorkerWithDpRank,
    pub(crate) local_prefix_blocks: u32,
    pub(crate) sources: &'a [KvTransferSource],
    pub(crate) workers: &'a (dyn WorkerTable + Sync),
}

impl<'a> KvTransferInput<'a> {
    /// The selected worker, which would fetch.
    pub fn target(self) -> WorkerWithDpRank {
        self.target
    }

    /// Root-aligned prefix blocks the selected worker holds in its device and host-pinned tiers.
    pub fn local_prefix_blocks(self) -> u32 {
        self.local_prefix_blocks
    }

    /// Every source the selected worker can fetch from, never empty. Sources are ordered longest
    /// prefix first; index 0 is the router's default choice.
    pub fn sources(self) -> &'a [KvTransferSource] {
        self.sources
    }

    /// Prefix blocks held by the router's default source, `sources()[0]`.
    pub fn source_prefix_blocks(self) -> u32 {
        self.sources[0].prefix_blocks()
    }

    /// Prefix blocks the default source can add beyond the selected worker's own prefix.
    pub fn additional_blocks(self) -> u32 {
        self.source_prefix_blocks() - self.local_prefix_blocks
    }

    /// Facts a worker advertised at registration, such as its transfer domain or topology.
    /// None for a worker the host does not know.
    pub fn worker_metadata(self, worker: WorkerWithDpRank) -> Option<WorkerMetadata<'a>> {
        self.workers
            .get(worker.worker_id)
            .map(|config| WorkerMetadata { config })
    }
}

/// Decides whether and where the selected worker fetches KV through the router's hint.
///
/// Each routing partition owns one instance. Concurrent selections call it one at a time.
pub trait KvTransferPolicy: Send {
    /// Called once per booked selection after the picker, only when at least one source holds a
    /// longer prefix than the selected worker. The decision does not change the selected worker.
    fn decide(&mut self, input: KvTransferInput<'_>) -> KvTransferAction;
}

/// Attach a fetch-hint policy to a worker-selection policy.
///
/// The selector runs in the scheduler queue, but hints are built after it returns. When the host
/// creates the routing partition, it moves `transfer` out of `policy` and calls it where hints are
/// built. This is a free function so the unstable hook stays off `WorkerSelectionPolicy`'s API.
pub fn with_kv_transfer_policy(
    policy: WorkerSelectionPolicy,
    transfer: Box<dyn KvTransferPolicy>,
) -> WorkerSelectionPolicy {
    policy.with_kv_transfer_policy(transfer)
}
