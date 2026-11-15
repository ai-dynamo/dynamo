// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-rank resident block counts for worker-selection policies.
//!
//! A component that declares [`RESIDENT_BLOCKS`] next to [`WorkerInputs::CACHE`] can read how
//! many blocks the router's indexer holds for each worker rank, including cached blocks that no
//! running request uses, and charge a placement for the cache it would evict. Use
//! [`WorkerInputs::LOAD`] for work in flight.
//!
//! The indexer's event threads keep the counts live, so a count is current as of the last event
//! its thread applied, the same freshness as the `CACHE` overlap scores. It is unweighted and not
//! clamped to the rank's capacity. In KV-event mode it is the device-tier blocks the indexer
//! tracks, and every router replica sees the same counts. In approximate modes it is the blocks
//! this replica routed to the rank within the TTL, or the distinct full blocks its LRU policy
//! retains.
//!
//! A read is `None` when the component did not declare [`RESIDENT_BLOCKS`], when the host has no
//! local indexer, when the indexer has held no blocks for the rank while counting or has removed
//! it, and after an indexer thread stops. A rank cleared of blocks reads `Some(0)`. Only filters,
//! scorers, and pickers receive counts: a request classifier that declares [`RESIDENT_BLOCKS`]
//! reads `None`.
//!
//! [`RESIDENT_BLOCKS`] requires [`WorkerInputs::CACHE`], because the counts come from the indexer
//! that `CACHE` starts. The host checks this for every policy it builds for a routing partition.
//!
//! Plugins reach these items as `dynamo_kv_router::plugins::worker_selection::experimental`;
//! the example names the stage crate directly.
//!
//! ```
//! use dynamo_worker_selection::experimental::resident_blocks::{RESIDENT_BLOCKS, ResidentBlocks};
//! use dynamo_worker_selection::{
//!     WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
//!     WorkerSelectionPolicyError,
//! };
//!
//! /// Charges one unit per block a placement could evict.
//! struct EvictionPressure;
//!
//! impl WorkerScorer for EvictionPressure {
//!     fn required_worker_inputs(&self) -> WorkerInputs {
//!         WorkerInputs::CACHE | RESIDENT_BLOCKS
//!     }
//!
//!     fn score(
//!         &self,
//!         _context: &WorkerSelectionContext<'_>,
//!         candidates: WorkerCandidates<'_>,
//!         costs: &mut [f64],
//!     ) -> Result<(), WorkerSelectionPolicyError> {
//!         for (candidate, cost) in candidates.iter().zip(costs) {
//!             // None means the host has no count for this rank; treat it as neutral.
//!             *cost = candidate.resident_blocks().unwrap_or(0) as f64;
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use crate::{WorkerCandidate, WorkerInputView, WorkerInputs};

/// Request per-rank counts of the blocks the router's indexer tracks. Requires
/// [`WorkerInputs::CACHE`].
pub const RESIDENT_BLOCKS: WorkerInputs = WorkerInputs::RESIDENT_BLOCKS;

mod sealed {
    pub trait Sealed {}
}

/// Reads a candidate's resident block count in a filter or scorer.
pub trait ResidentBlocks: sealed::Sealed {
    /// Blocks the router's indexer holds for this worker rank, or `None` when this component did
    /// not declare [`RESIDENT_BLOCKS`] or the host has no count for the rank (see the module docs).
    fn resident_blocks(self) -> Option<u64>;
}

impl sealed::Sealed for WorkerCandidate<'_> {}

impl ResidentBlocks for WorkerCandidate<'_> {
    fn resident_blocks(self) -> Option<u64> {
        self.resident_block_count()
    }
}

/// Reads resident block counts by candidate row in a picker.
pub trait ResidentBlocksByRow: sealed::Sealed {
    /// [`ResidentBlocks::resident_blocks`] for the candidate at `row`, or `None` for an
    /// out-of-range row.
    fn resident_blocks(self, row: usize) -> Option<u64>;
}

impl sealed::Sealed for WorkerInputView<'_> {}

impl ResidentBlocksByRow for WorkerInputView<'_> {
    fn resident_blocks(self, row: usize) -> Option<u64> {
        self.resident_block_count(row)
    }
}
