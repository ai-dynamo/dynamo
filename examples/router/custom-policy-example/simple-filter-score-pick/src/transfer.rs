// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV fetch-hint policy for the `simple-filter-score-pick` policy.

use dynamo_kv_router::plugins::worker_selection::experimental::{
    KvTransferAction, KvTransferInput, KvTransferPolicy,
};

/// Fetches only when a source adds at least `min_fetch_blocks`, preferring a source in the
/// selected worker's KV transfer domain.
pub(crate) struct MinimumFetchPolicy {
    pub(crate) min_fetch_blocks: u32,
}

impl KvTransferPolicy for MinimumFetchPolicy {
    fn decide(&mut self, input: KvTransferInput<'_>) -> KvTransferAction {
        let domain = |worker| {
            input
                .worker_metadata(worker)
                .and_then(|metadata| metadata.kv_transfer_domain())
        };
        let target_domain = domain(input.target());
        let useful = |prefix_blocks: u32| {
            prefix_blocks - input.local_prefix_blocks() >= self.min_fetch_blocks
        };
        // Sources arrive longest prefix first.
        let sources = input.sources();
        sources
            .iter()
            .position(|source| {
                useful(source.prefix_blocks())
                    && target_domain.is_some()
                    && source.worker().and_then(domain) == target_domain
            })
            .or_else(|| {
                sources
                    .iter()
                    .position(|source| useful(source.prefix_blocks()))
            })
            .map_or(KvTransferAction::Skip, KvTransferAction::FetchFrom)
    }
}
