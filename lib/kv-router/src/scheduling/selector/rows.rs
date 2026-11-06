// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate rows the selection engine builds for each selection, and the borrowed views of them
//! that worker-selection components and request classifiers read.

use crate::plugins::worker_selection::{
    WorkerCandidates, WorkerCapacity, WorkerInfo, WorkerInputs, WorkerLoadInput,
    WorkerTransferInput,
};
use crate::protocols::{SharedCacheHits, WorkerWithDpRank};

/// Host-owned row materialized once for the union of component input requirements.
pub(crate) struct CandidateData {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) inputs: WorkerInputs,
    pub(crate) cache: WorkerCacheData,
    pub(crate) load: WorkerLoadInput,
    pub(crate) transfer: Option<WorkerTransferInput>,
    pub(crate) preferred_taint_multiplier: Option<f64>,
}

/// Request-wide facts borrowed once for all cache views in a selection.
pub(crate) struct CacheSnapshot<'a> {
    pub(crate) shared_hits: Option<&'a SharedCacheHits>,
    pub(crate) has_tier_matches: bool,
}

/// Rows and request-wide cache facts owned by a reader outside worker selection.
pub(crate) struct WorkerInputRows {
    inputs: WorkerInputs,
    rows: Vec<CandidateData>,
    // Index-aligned with `rows`. The rows outlive the worker map, so they carry each rank's
    // capacity; the rank itself is already in the row.
    capacities: Vec<WorkerCapacity>,
    shared_hits: Option<SharedCacheHits>,
    has_tier_matches: bool,
}

impl WorkerInputRows {
    pub(crate) fn new(
        inputs: WorkerInputs,
        rows: Vec<CandidateData>,
        capacities: Vec<WorkerCapacity>,
        cache_snapshot: CacheSnapshot<'_>,
    ) -> Self {
        debug_assert_eq!(rows.len(), capacities.len());
        Self {
            inputs,
            rows,
            capacities,
            shared_hits: if inputs.contains(WorkerInputs::CACHE) {
                cache_snapshot.shared_hits.cloned()
            } else {
                None
            },
            has_tier_matches: cache_snapshot.has_tier_matches,
        }
    }

    /// Rows for a reader that declared no inputs.
    pub(crate) fn empty() -> Self {
        Self {
            inputs: WorkerInputs::NONE,
            rows: Vec::new(),
            capacities: Vec::new(),
            shared_hits: None,
            has_tier_matches: false,
        }
    }

    /// Lend the rows as a view restricted to the declared inputs.
    pub(crate) fn read<R>(&self, read: impl FnOnce(WorkerCandidates<'_>) -> R) -> R {
        let cache_snapshot = CacheSnapshot {
            shared_hits: self.shared_hits.as_ref(),
            has_tier_matches: self.has_tier_matches,
        };
        let info = |row: usize| WorkerInfo::new(self.rows[row].worker, self.capacities[row]);
        read(WorkerCandidates::new(
            &self.rows,
            self.inputs,
            &cache_snapshot,
            &info,
        ))
    }
}

impl std::fmt::Debug for WorkerInputRows {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerInputRows")
            .field("inputs", &self.inputs)
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}

/// Numeric cache row retained in host buffers between selections.
#[derive(Clone, Copy, Default)]
pub(crate) struct WorkerCacheData {
    pub(crate) effective_overlap_blocks: f64,
    pub(crate) estimated_cached_tokens: usize,
    pub(crate) device_overlap_blocks: f64,
    pub(crate) host_overlap_blocks: f64,
    pub(crate) disk_overlap_blocks: f64,
}

// See the `WorkerLoadInput` size assertion.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CandidateData>() == 128);

impl CandidateData {
    pub(crate) fn with_inputs_from(&self, additional: &Self, inputs: WorkerInputs) -> Self {
        debug_assert_eq!(self.worker, additional.worker);
        Self {
            worker: self.worker,
            inputs,
            cache: if inputs.contains(WorkerInputs::CACHE) {
                if self.inputs.contains(WorkerInputs::CACHE) {
                    self.cache
                } else {
                    additional.cache
                }
            } else {
                WorkerCacheData::default()
            },
            load: if inputs.contains(WorkerInputs::LOAD) {
                if self.inputs.contains(WorkerInputs::LOAD) {
                    self.load
                } else {
                    additional.load
                }
            } else {
                WorkerLoadInput::default()
            },
            // Filled for scorers once every row is built.
            transfer: None,
            preferred_taint_multiplier: self
                .preferred_taint_multiplier
                .or(additional.preferred_taint_multiplier),
        }
    }
}
