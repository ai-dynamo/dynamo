// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::harness::HarnessBackend;

impl HarnessBackend for ConcurrentRadixTreeCompressed {
    fn harness_new() -> Self {
        Self::new()
    }

    fn harness_match_details(&self, sequence: &[LocalBlockHash]) -> Option<MatchDetails> {
        Some(self.find_match_details_impl(sequence, false))
    }

    fn harness_rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        self.slot_for_test(rank).map(|slot| slot.index())
    }

    fn harness_structure_size(&self) -> Option<usize> {
        Some(self.raw_child_edge_count())
    }
}
