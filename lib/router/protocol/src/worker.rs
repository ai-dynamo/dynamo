// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker and rank identities.

use serde::{Deserialize, Serialize};

/// A worker identifier.
pub type WorkerId = u64;

/// A data parallel rank identifier.
pub type DpRank = u32;

/// A worker identifier combined with its data parallel rank.
/// Used for routing decisions in data parallel setups.
/// dp_rank = 0 indicates either DP not enabled or the first rank.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithDpRank {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
}

/// A worker affinity target that may apply to every data-parallel rank of a worker.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerAffinityTarget {
    pub worker_id: WorkerId,
    pub dp_rank: Option<DpRank>,
}

impl WorkerAffinityTarget {
    pub fn new(worker_id: WorkerId, dp_rank: Option<DpRank>) -> Self {
        Self { worker_id, dp_rank }
    }
}

impl From<WorkerWithDpRank> for WorkerAffinityTarget {
    fn from(worker: WorkerWithDpRank) -> Self {
        Self::new(worker.worker_id, Some(worker.dp_rank))
    }
}

impl WorkerWithDpRank {
    pub fn new(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self { worker_id, dp_rank }
    }

    pub fn from_worker_id(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}
