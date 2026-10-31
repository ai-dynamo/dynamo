// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The column vocabulary stages share: which optional worker-signal groups a component
//! declares, and the static facts about a rank every column carries. Worker selection
//! builds its candidate rows from these; admission's classifier reads workers through them.

use std::ops::BitOr;

use crate::load::WorkerCapacity;
use crate::worker::WorkerWithDpRank;

/// Optional worker-signal groups a stage component declares: a classifier, filter, scorer,
/// or picker asks for the columns it reads, and the kernel builds rows with exactly those.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerInputs(u8);

impl WorkerInputs {
    /// Request no optional worker inputs.
    pub const NONE: Self = Self(0);
    /// Request worker KV-cache overlaps and shared-cache lookup results.
    pub const CACHE: Self = Self(1 << 0);
    /// Request active-load inputs.
    pub const LOAD: Self = Self(1 << 1);
    /// Request preferred-taint routing metadata.
    pub const PREFERRED_TAINT: Self = Self(1 << 2);
    /// Request host-owned active-request counts.
    pub const OCCUPANCY: Self = Self(1 << 5);
    /// Request the KV prefix each worker could fetch from another worker or cache.
    pub const KV_TRANSFER: Self = Self(1 << 6);
    /// Every optional column but the fetch column, which only scorers may declare.
    pub const ALL: Self = Self(Self::CACHE.0 | Self::LOAD.0 | Self::PREFERRED_TAINT.0);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    /// The groups in `self` that are not in `other`.
    pub fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for WorkerInputs {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Static facts about one worker rank: its identity and the capacity its runtime config
/// advertises. Every component can read it without declaring an input group. It changes
/// only when the worker registers again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerInfo {
    worker: WorkerWithDpRank,
    capacity: WorkerCapacity,
}

impl WorkerInfo {
    #[inline]
    pub fn new(worker: WorkerWithDpRank, capacity: WorkerCapacity) -> Self {
        Self { worker, capacity }
    }

    /// Worker ID and data-parallel rank.
    #[inline]
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    #[inline]
    pub fn max_num_batched_tokens(&self) -> Option<u64> {
        self.capacity.max_num_batched_tokens()
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    #[inline]
    pub fn total_kv_blocks(&self) -> Option<u64> {
        self.capacity.total_kv_blocks()
    }

    /// Both advertised values, as a fullness check takes them.
    #[inline]
    pub fn capacity(&self) -> WorkerCapacity {
        self.capacity
    }
}
