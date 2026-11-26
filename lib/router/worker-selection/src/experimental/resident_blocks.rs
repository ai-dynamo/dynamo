// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-rank resident block counts, and KVCR occupancy, for worker-selection policies.
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
//! Plugins reach these items as `dynamo_kv_router::plugins::experimental::resident_blocks`;
//! the examples name the stage crate directly.
//!
//! ```
//! use dynamo_worker_selection::experimental::resident_blocks::{RESIDENT_BLOCKS, ResidentBlocks};
//! use dynamo_worker_selection::{
//!     WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionPolicyError,
//!     WorkerSelectionRequest,
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
//!         _request: &WorkerSelectionRequest<'_>,
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
//!
//! # KVCR occupancy
//!
//! A KVCR instance (cache owner) is the host-memory KV cache attached to one worker rank. The
//! same declaration lets a component read how many host-pinned blocks the router has indexed for
//! a worker's instance, and any component can read the capacity the worker advertises for it.
//! Occupancy is the count divided by the capacity.
//!
//! The count is the number of blocks the router's lower-tier index holds for the instance. The
//! index follows the instance's KV events and the count is read live on each call, so two reads
//! in one selection can differ. Events are best effort, so the count can differ from the
//! instance's real contents: blocks it is still filling and non-main KV groups are missing, and a
//! lost Removed event counts until the instance's next recovery. The count is unweighted and not
//! clamped to the capacity.
//!
//! A count is `None` when the component did not declare [`RESIDENT_BLOCKS`], when no instance is
//! attached to the worker, before the router has received a host-pinned KV event, and when the
//! host has no local lower-tier index. A request classifier reads `None`. The host looks a count
//! up only when a component calls the accessor, so a policy that reads only resident blocks pays
//! nothing for it.
//!
//! The capacity needs no input group. It is what the worker's runtime config advertises, in
//! router KV blocks, and it changes only when the worker registers again. A vLLM worker
//! advertises its KVCR tier's `secondary_g2_slots` times the offload chunk size; a pool served by
//! `kvcr-service` advertises none.
//!
//! ```
//! use dynamo_worker_selection::experimental::resident_blocks::{
//!     CacheOwnerCapacity, RESIDENT_BLOCKS, ResidentBlocks,
//! };
//! use dynamo_worker_selection::{
//!     WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionPolicyError,
//!     WorkerSelectionRequest,
//! };
//!
//! /// Charges up to one unit for a full KVCR instance.
//! struct KvcrOccupancy;
//!
//! impl WorkerScorer for KvcrOccupancy {
//!     fn required_worker_inputs(&self) -> WorkerInputs {
//!         WorkerInputs::CACHE | RESIDENT_BLOCKS
//!     }
//!
//!     fn score(
//!         &self,
//!         _request: &WorkerSelectionRequest<'_>,
//!         candidates: WorkerCandidates<'_>,
//!         costs: &mut [f64],
//!     ) -> Result<(), WorkerSelectionPolicyError> {
//!         for (candidate, cost) in candidates.iter().zip(costs) {
//!             let capacity = candidate.info().cache_owner_capacity_blocks();
//!             // None means the host has no count or no capacity for this rank; treat it as neutral.
//!             *cost = candidate
//!                 .cache_owner_blocks()
//!                 .zip(capacity)
//!                 .map_or(0.0, |(blocks, capacity)| (blocks as f64 / capacity as f64).min(1.0));
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use crate::{WorkerCandidate, WorkerInfo, WorkerInputView, WorkerInputs};

/// Request per-rank counts of the blocks the router's indexer tracks. Requires
/// [`WorkerInputs::CACHE`].
pub const RESIDENT_BLOCKS: WorkerInputs = WorkerInputs::RESIDENT_BLOCKS;

mod sealed {
    pub trait Sealed {}
}

/// Reads a candidate's block counts in a filter or scorer.
pub trait ResidentBlocks: sealed::Sealed {
    /// Blocks the router's indexer holds for this worker rank, or `None` when this component did
    /// not declare [`RESIDENT_BLOCKS`] or the host has no count for the rank (see the module docs).
    fn resident_blocks(self) -> Option<u64>;

    /// Host-pinned blocks the router's index holds for the KVCR instance attached to this worker
    /// rank, or `None` when this component did not declare [`RESIDENT_BLOCKS`] or the host has no
    /// count for the rank (see the module docs).
    fn cache_owner_blocks(self) -> Option<u64>;
}

impl sealed::Sealed for WorkerCandidate<'_> {}

impl ResidentBlocks for WorkerCandidate<'_> {
    fn resident_blocks(self) -> Option<u64> {
        self.resident_block_count()
    }

    fn cache_owner_blocks(self) -> Option<u64> {
        self.cache_owner_block_count()
    }
}

/// Reads block counts by candidate row in a picker.
pub trait ResidentBlocksByRow: sealed::Sealed {
    /// [`ResidentBlocks::resident_blocks`] for the candidate at `row`, or `None` for an
    /// out-of-range row.
    fn resident_blocks(self, row: usize) -> Option<u64>;

    /// [`ResidentBlocks::cache_owner_blocks`] for the candidate at `row`, or `None` for an
    /// out-of-range row.
    fn cache_owner_blocks(self, row: usize) -> Option<u64>;
}

impl sealed::Sealed for WorkerInputView<'_> {}

impl ResidentBlocksByRow for WorkerInputView<'_> {
    fn resident_blocks(self, row: usize) -> Option<u64> {
        self.resident_block_count(row)
    }

    fn cache_owner_blocks(self, row: usize) -> Option<u64> {
        self.cache_owner_block_count(row)
    }
}

/// Reads the capacity a worker rank advertises for its KVCR instance.
pub trait CacheOwnerCapacity: sealed::Sealed {
    /// Host-pinned capacity of the KVCR instance attached to this worker rank, in router KV
    /// blocks, or `None` when the worker does not advertise one (see the module docs).
    fn cache_owner_capacity_blocks(&self) -> Option<u64>;
}

impl sealed::Sealed for WorkerInfo {}

impl CacheOwnerCapacity for WorkerInfo {
    fn cache_owner_capacity_blocks(&self) -> Option<u64> {
        self.capacity().cache_owner_capacity_blocks()
    }
}
