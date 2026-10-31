// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Row columns derived from what the kernel holds and the stage does not name: worker
//! configuration and the slot tracker's load projection.

use rustc_hash::FxHashMap;

use crate::protocols::{WorkerConfigLike, WorkerWithDpRank};
use dynamo_slot_tracker::WorkerLoadProjection;
use dynamo_worker_selection::{WorkerCapacity, WorkerInfo, WorkerLoadInput};

/// Reads a worker's advertised capacity from its runtime config.
pub trait WorkerCapacityExt {
    /// Read the capacity a worker advertises in its runtime config.
    fn of<C: WorkerConfigLike>(config: &C) -> Self;
}

impl WorkerCapacityExt for WorkerCapacity {
    fn of<C: WorkerConfigLike>(config: &C) -> Self {
        Self::new(config.max_num_batched_tokens(), config.total_kv_blocks())
    }
}

/// Reads a rank's static info from its worker's runtime config.
pub trait WorkerInfoExt {
    fn of<C: WorkerConfigLike>(worker: WorkerWithDpRank, config: &C) -> Self;
}

impl WorkerInfoExt for WorkerInfo {
    fn of<C: WorkerConfigLike>(worker: WorkerWithDpRank, config: &C) -> Self {
        Self::new(worker, WorkerCapacity::of(config))
    }
}

/// Builds the `LOAD` column from the slot tracker's projection.
pub trait WorkerLoadInputExt: Sized {
    /// Convert an observed load projection into a `LOAD` row.
    fn from_projection(load: WorkerLoadProjection) -> Self;

    /// The `LOAD` row for `worker`: its projection in `loads` when the host observed one,
    /// otherwise `unobserved()` flagged unavailable. Worker selection, the policy-class
    /// queue, and offline replay all build rows here or in [`Self::from_observed`], so they
    /// read the same row for the same worker by construction.
    fn for_worker(
        loads: &FxHashMap<WorkerWithDpRank, WorkerLoadProjection>,
        worker: WorkerWithDpRank,
        unobserved: impl FnOnce() -> WorkerLoadProjection,
    ) -> Self {
        Self::from_observed(loads.get(&worker).copied(), unobserved)
    }

    /// The `LOAD` row for a rank's observed projection, or `unobserved()` flagged
    /// unavailable when there is none.
    fn from_observed(
        load: Option<WorkerLoadProjection>,
        unobserved: impl FnOnce() -> WorkerLoadProjection,
    ) -> Self;
}

impl WorkerLoadInputExt for WorkerLoadInput {
    fn from_projection(load: WorkerLoadProjection) -> Self {
        Self {
            available: true,
            active_prefill_tokens: load.active_prefill_tokens,
            active_decode_blocks: load.active_decode_blocks,
            decode_cost_blocks: load.potential_decode_blocks() as f64,
            active_requests: load.active_requests,
        }
    }

    fn from_observed(
        load: Option<WorkerLoadProjection>,
        unobserved: impl FnOnce() -> WorkerLoadProjection,
    ) -> Self {
        match load {
            Some(load) => Self::from_projection(load),
            None => Self {
                available: false,
                ..Self::from_projection(unobserved())
            },
        }
    }
}
