// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kernel's implementation of the classifier's [`WorkerView`] over the rows worker
//! selection builds, so admission reads the same columns a filter or scorer would.

use dynamo_admission::{WorkerCacheView, WorkerLoadView, WorkerView};
use dynamo_worker_selection::{WorkerCandidate, WorkerInfo, WorkerInputRows};

/// Rows built for one classification, read through the classifier's view.
pub struct ClassifierRows(pub WorkerInputRows);

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
            candidate.load().map(|load| {
                let mut view = WorkerLoadView::default();
                view.available = load.is_available();
                view.active_prefill_tokens = load.active_prefill_tokens();
                view.active_decode_blocks = load.active_decode_blocks();
                view.decode_cost_blocks = load.decode_cost_blocks();
                view.active_requests = load.active_requests();
                view
            })
        })
        .flatten()
    }

    fn cache(&self, row: usize) -> Option<WorkerCacheView> {
        self.with_row(row, |candidate| {
            candidate.cache().map(|cache| {
                let (effective_overlap_blocks, estimated_cached_tokens) =
                    cache.accounting_cache_estimate();
                let mut view = WorkerCacheView::default();
                view.effective_overlap_blocks = effective_overlap_blocks;
                view.estimated_cached_tokens = estimated_cached_tokens;
                view.device_overlap_blocks = cache.device_overlap_blocks();
                view.host_overlap_blocks = cache.host_overlap_blocks();
                view.disk_overlap_blocks = cache.disk_overlap_blocks();
                view
            })
        })
        .flatten()
    }
}
