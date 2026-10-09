// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ArenaIndex;
use crate::indexer::MatchDetails;
use crate::indexer::harness::HarnessBackend;
use crate::protocols::{LocalBlockHash, WorkerWithDpRank};

impl HarnessBackend for ArenaIndex {
    fn harness_new() -> Self {
        Self::new()
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        Some(self.find_match_details(sequence, false, false))
    }

    fn harness_rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        self.rank_slot(rank)
    }

    fn harness_structure_size(&self) -> Option<usize> {
        Some(self.live_run_count())
    }
}
