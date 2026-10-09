// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::MatchDetails;
use crate::indexer::harness::HarnessBackend;

impl HarnessBackend for ArenaIndexC {
    fn harness_new() -> Self {
        Self::new()
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        Some(self.find_match_details_impl(sequence, false, false))
    }

    fn harness_rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        self.slots
            .table(&crossbeam_epoch::pin())
            .slot_of(rank)
            .map(Slot::index)
    }

    fn harness_structure_size(&self) -> Option<usize> {
        Some(self.reachable_runs().len())
    }
}
