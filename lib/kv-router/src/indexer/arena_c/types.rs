// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroU32;

use crate::protocols::{ExternalSequenceBlockHash, WorkerId, WorkerWithDpRank};

#[derive(Clone, Copy, Debug)]
pub(super) enum WorkerRemovalTarget {
    WorkerId(WorkerId),
    DpRank(WorkerWithDpRank),
}

impl WorkerRemovalTarget {
    pub(super) fn matches(self, worker: WorkerWithDpRank) -> bool {
        match self {
            Self::WorkerId(worker_id) => worker.worker_id == worker_id,
            Self::DpRank(target) => worker == target,
        }
    }
}

/// Where a lane-map entry says a block is: a position `(run, offset)`.
///
/// Two integers and no generation: a stale entry is caught by the run's `DEAD` flag, by
/// forwarding, or by the external-hash check (H1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct BlockPos {
    pub(super) run: NonZeroU32,
    pub(super) offset: u32,
}

impl BlockPos {
    #[inline]
    pub(super) fn new(run: u32, offset: u32) -> Self {
        Self {
            run: NonZeroU32::new(run).expect("positions name allocated runs"),
            offset,
        }
    }

    #[inline]
    pub(super) fn run(self) -> u32 {
        self.run.get()
    }
}

// A lane-map slot stays at 16 bytes.
const _: () = assert!(std::mem::size_of::<Option<(ExternalSequenceBlockHash, BlockPos)>>() == 16);
