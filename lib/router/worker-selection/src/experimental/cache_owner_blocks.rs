// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVCR occupancy for worker-selection policies.
//!
//! A KVCR instance (cache owner) is the host-memory KV cache attached to one worker rank. A
//! component that declares [`WorkerInputs::CACHE`](crate::WorkerInputs::CACHE) can read how many
//! host-pinned blocks the router has indexed for a worker's instance, and any component can read
//! the capacity the worker advertises for it. Occupancy is the count divided by the capacity.
//!
//! The count is the number of blocks the router's lower-tier index holds for the instance. The
//! index follows the instance's KV events and the count is read live on each call, so two reads
//! in one selection can differ. Events are best effort, so the count can differ from the
//! instance's real contents: blocks it is still filling and non-main KV groups are missing, and a
//! lost Removed event counts until the instance's next recovery. The count is unweighted and not
//! clamped to the capacity.
//!
//! A count is `None` when the component did not declare `CACHE`, when no instance is attached to
//! the worker, before the router has received a host-pinned KV event, and when the host has no
//! local lower-tier index. Only filters, scorers, and pickers receive counts: a request
//! classifier reads `None`. The host looks a count up only when a component calls the accessor,
//! so a policy that never reads one pays nothing.
//!
//! The capacity needs no input group. It is what the worker's runtime config advertises, in
//! router KV blocks, and it changes only when the worker registers again. A vLLM worker
//! advertises its KVCR tier's `secondary_g2_slots` times the offload chunk size; a pool served by
//! `kvcr-service` advertises none.
//!
//! Plugins reach these items as `dynamo_kv_router::plugins::worker_selection::experimental`;
//! the example names the stage crate directly.
//!
//! ```
//! use dynamo_worker_selection::experimental::cache_owner_blocks::{
//!     CacheOwnerBlocks, CacheOwnerCapacity,
//! };
//! use dynamo_worker_selection::{
//!     WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
//!     WorkerSelectionPolicyError,
//! };
//!
//! /// Charges up to one unit for a full KVCR instance.
//! struct KvcrOccupancy;
//!
//! impl WorkerScorer for KvcrOccupancy {
//!     fn required_worker_inputs(&self) -> WorkerInputs {
//!         WorkerInputs::CACHE
//!     }
//!
//!     fn score(
//!         &self,
//!         _context: &WorkerSelectionContext<'_>,
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

use crate::{WorkerCandidate, WorkerInfo, WorkerInputView};

mod sealed {
    pub trait Sealed {}
}

/// Reads the block count of a candidate's KVCR instance in a filter or scorer.
pub trait CacheOwnerBlocks: sealed::Sealed {
    /// Host-pinned blocks the router has indexed for the KVCR instance attached to this worker
    /// rank, or `None` when this component did not declare `CACHE` or the host has no count for
    /// the rank (see the module docs).
    fn cache_owner_blocks(self) -> Option<u64>;
}

impl sealed::Sealed for WorkerCandidate<'_> {}

impl CacheOwnerBlocks for WorkerCandidate<'_> {
    fn cache_owner_blocks(self) -> Option<u64> {
        self.cache_owner_block_count()
    }
}

/// Reads KVCR block counts by candidate row in a picker.
pub trait CacheOwnerBlocksByRow: sealed::Sealed {
    /// [`CacheOwnerBlocks::cache_owner_blocks`] for the candidate at `row`, or `None` for an
    /// out-of-range row.
    fn cache_owner_blocks(self, row: usize) -> Option<u64>;
}

impl sealed::Sealed for WorkerInputView<'_> {}

impl CacheOwnerBlocksByRow for WorkerInputView<'_> {
    fn cache_owner_blocks(self, row: usize) -> Option<u64> {
        self.cache_owner_block_count(row)
    }
}

/// Reads the capacity a worker rank advertises for its KVCR instance.
pub trait CacheOwnerCapacity: sealed::Sealed {
    /// Host-pinned capacity of the KVCR instance attached to this rank, in router KV blocks, or
    /// `None` when the worker advertises none.
    fn cache_owner_capacity_blocks(&self) -> Option<u64>;
}

impl sealed::Sealed for WorkerInfo {}

impl CacheOwnerCapacity for WorkerInfo {
    fn cache_owner_capacity_blocks(&self) -> Option<u64> {
        self.capacity().cache_owner_capacity_blocks()
    }
}
