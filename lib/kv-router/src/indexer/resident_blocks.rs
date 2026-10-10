// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-rank resident block counts, published off the request path.
//!
//! Event threads own the per-rank block maps, so reading them is a queued round
//! trip. A poller makes that trip and publishes the result here, where worker
//! selection can read it without waiting on the indexer.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use rustc_hash::FxHashMap;

use super::WorkerLookupStats;
use crate::protocols::WorkerWithDpRank;

/// How often routing hosts republish [`ResidentBlockCounts`].
#[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
pub(crate) const RESIDENT_BLOCK_COUNTS_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Blocks the indexer tracked for each worker rank when polled.
#[derive(Debug)]
pub(crate) struct ResidentBlockCounts {
    counts: FxHashMap<WorkerWithDpRank, u64>,
}

impl ResidentBlockCounts {
    /// Blocks tracked for `worker`; zero when the indexer tracks none for it.
    pub(crate) fn get(&self, worker: WorkerWithDpRank) -> u64 {
        self.counts.get(&worker).copied().unwrap_or(0)
    }
}

impl From<WorkerLookupStats> for ResidentBlockCounts {
    fn from(stats: WorkerLookupStats) -> Self {
        let mut counts = FxHashMap::default();
        for (worker, blocks) in stats.worker_blocks {
            *counts.entry(worker).or_default() += blocks as u64;
        }
        Self { counts }
    }
}

/// The latest published [`ResidentBlockCounts`], shared by the poller and its readers.
#[derive(Clone, Default)]
pub(crate) struct ResidentBlockCountsHandle {
    latest: Arc<ArcSwapOption<ResidentBlockCounts>>,
}

impl ResidentBlockCountsHandle {
    /// The latest counts, or `None` before the first poll or after the poller stops.
    pub(crate) fn load(&self) -> Option<Arc<ResidentBlockCounts>> {
        self.latest.load_full()
    }

    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn publish(&self, counts: ResidentBlockCounts) {
        self.latest.store(Some(Arc::new(counts)));
    }

    /// Withdraw the counts so readers see `None` rather than a frozen snapshot.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn clear(&self) {
        self.latest.store(None);
    }
}
