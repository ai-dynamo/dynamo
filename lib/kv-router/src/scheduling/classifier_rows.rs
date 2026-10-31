// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kernel's implementation of the classifier's [`WorkerView`] over the rows worker
//! selection builds, so admission reads the same columns a filter or scorer would.

use crate::plugins::request_classifier::{WorkerCacheView, WorkerLoadView, WorkerView};
use crate::plugins::worker_selection::{WorkerCandidate, WorkerInfo, WorkerInputRows};

/// Rows built for one classification, read through the classifier's view.
pub(crate) struct ClassifierRows(pub(crate) WorkerInputRows);

impl ClassifierRows {
    fn with_row<R>(&self, row: usize, read: impl FnOnce(WorkerCandidate<'_>) -> R) -> Option<R> {
        self.0.read(|candidates| candidates.get(row).map(read))
    }
}

impl WorkerView for ClassifierRows {
    fn len(&self) -> usize {
        self.0.read(|candidates| candidates.len())
    }

    fn info(&self, row: usize) -> Option<WorkerInfo> {
        self.with_row(row, |candidate| candidate.info())
    }

    fn load(&self, row: usize) -> Option<WorkerLoadView> {
        self.with_row(row, |candidate| {
            candidate.load().map(|load| WorkerLoadView {
                available: load.is_available(),
                active_prefill_tokens: load.active_prefill_tokens(),
                active_decode_blocks: load.active_decode_blocks(),
                decode_cost_blocks: load.decode_cost_blocks(),
                active_requests: load.active_requests(),
            })
        })
        .flatten()
    }

    fn cache(&self, row: usize) -> Option<WorkerCacheView> {
        self.with_row(row, |candidate| {
            candidate.cache().map(|cache| {
                let (effective_overlap_blocks, estimated_cached_tokens) =
                    cache.accounting_cache_estimate();
                WorkerCacheView {
                    effective_overlap_blocks,
                    estimated_cached_tokens,
                    device_overlap_blocks: cache.device_overlap_blocks(),
                    host_overlap_blocks: cache.host_overlap_blocks(),
                    disk_overlap_blocks: cache.disk_overlap_blocks(),
                }
            })
        })
        .flatten()
    }
}
