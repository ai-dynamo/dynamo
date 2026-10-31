// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worker table a classifier reads: one row per eligible rank, holding the column
//! groups the classifier declared. Admission owns the view so that it never names worker
//! selection's rows; the kernel implements it over them.

use dynamo_router_protocol::columns::WorkerInfo;
use dynamo_router_protocol::load::RankLoad;

/// Every worker rank eligible for one request, lent to [`super::RequestClassifier::classify`]
/// for the duration of the call. Views are read-only snapshots, so they are `Sync` and a
/// classification future that borrows one stays `Send`. Rows are in unspecified order. A column group the
/// classifier did not declare through [`super::RequestClassifier::required_worker_inputs`]
/// reads as `None` on every row.
pub trait WorkerView: Sync {
    /// The number of rows.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The rank's identity and advertised capacity; `None` past the last row.
    fn info(&self, row: usize) -> Option<WorkerInfo>;

    /// The rank's booked load, when `LOAD` was declared.
    fn load(&self, row: usize) -> Option<WorkerLoadView>;

    /// The rank's cache estimate for this request, when `CACHE` was declared.
    fn cache(&self, row: usize) -> Option<WorkerCacheView>;
}

/// A rank's booked load as the classifier sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkerLoadView {
    /// Whether the host observed a projection for this rank; false flags a default row.
    pub available: bool,
    pub active_prefill_tokens: usize,
    pub active_decode_blocks: usize,
    pub decode_cost_blocks: f64,
    pub active_requests: usize,
}

impl RankLoad for WorkerLoadView {
    fn active_prefill_tokens(&self) -> usize {
        self.active_prefill_tokens
    }

    fn active_decode_blocks(&self) -> usize {
        self.active_decode_blocks
    }
}

/// A rank's cache estimate for this request as the classifier sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorkerCacheView {
    pub effective_overlap_blocks: f64,
    pub estimated_cached_tokens: usize,
    pub device_overlap_blocks: f64,
    pub host_overlap_blocks: f64,
    pub disk_overlap_blocks: f64,
}

/// A view with no rows, for classifiers invoked without a worker table.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoWorkers;

impl WorkerView for NoWorkers {
    fn len(&self) -> usize {
        0
    }

    fn info(&self, _row: usize) -> Option<WorkerInfo> {
        None
    }

    fn load(&self, _row: usize) -> Option<WorkerLoadView> {
        None
    }

    fn cache(&self, _row: usize) -> Option<WorkerCacheView> {
        None
    }
}
