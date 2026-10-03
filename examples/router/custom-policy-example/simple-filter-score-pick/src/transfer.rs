// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV fetch-hint policy for the `simple-filter-score-pick` policy.

use dynamo_kv_router::plugins::worker_selection::WorkerSelectionContext;
use dynamo_kv_router::plugins::worker_selection::experimental::{
    KvTransferAction, KvTransferInput, KvTransferPolicy,
};

/// Skips the router's KV fetch hint when the fetch adds too few blocks.
pub(crate) struct MinimumFetchPolicy {
    pub(crate) min_fetch_blocks: u32,
}

impl KvTransferPolicy for MinimumFetchPolicy {
    fn decide(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: KvTransferInput,
    ) -> KvTransferAction {
        if input.additional_blocks() >= self.min_fetch_blocks {
            KvTransferAction::Default
        } else {
            KvTransferAction::Skip
        }
    }
}
