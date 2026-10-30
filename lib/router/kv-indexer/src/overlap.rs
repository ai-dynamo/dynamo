// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device-tier overlap scores: the indexer's primary output.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use dynamo_router_protocol::worker::WorkerWithDpRank;

/// Scores representing the overlap of workers (with their dp_rank).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapScores {
    /// Map of worker (with dp_rank) to score.
    pub scores: FxHashMap<WorkerWithDpRank, u32>,
    /// List of frequencies that the blocks have been accessed. Entries with value 0 are omitted.
    pub frequencies: Vec<usize>,
}

impl Default for OverlapScores {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapScores {
    /// Create a new `OverlapScores`.
    ///
    /// ### Returns
    ///
    /// A new `OverlapScores`.
    pub fn new() -> Self {
        Self {
            scores: FxHashMap::default(),
            frequencies: Vec::new(),
        }
    }

    /// Update the scores with a set of workers.
    ///
    /// ### Arguments
    ///
    /// * `workers` - An iterator over `WorkerWithDpRank` references.
    pub fn update_scores<'a, I>(&mut self, workers: I)
    where
        I: IntoIterator<Item = &'a WorkerWithDpRank>,
    {
        for worker in workers {
            let score = self.scores.entry(*worker).or_insert(0);
            *score += 1;
        }
    }
}
