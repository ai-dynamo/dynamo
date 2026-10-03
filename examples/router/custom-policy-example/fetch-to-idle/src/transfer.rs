// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV fetch policy for the `fetch-to-idle` policy.

use dynamo_kv_router::plugins::worker_selection::experimental::{
    KvTransferAction, KvTransferInput, KvTransferPolicy,
};

/// Fetches the prefix when it saves at least `min_fetch_blocks`. Among sources holding the
/// longest prefix, prefers a KV pool over a worker's own cache, so a busy worker's GPU does not
/// serve the transfer.
pub(crate) struct FetchFromHolder {
    pub(crate) min_fetch_blocks: u32,
}

impl KvTransferPolicy for FetchFromHolder {
    fn decide(&mut self, input: KvTransferInput<'_>) -> KvTransferAction {
        let sources = input
            .sources()
            .iter()
            .map(|source| (source.prefix_blocks(), source.is_cache_owner()));
        choose_source(input.local_prefix_blocks(), self.min_fetch_blocks, sources)
            .map_or(KvTransferAction::Skip, KvTransferAction::FetchFrom)
    }
}

/// The index of the source to fetch from, or None to skip the fetch. `sources` yields
/// `(prefix_blocks, is_cache_owner)`, longest prefix first.
pub(crate) fn choose_source(
    local_prefix_blocks: u32,
    min_fetch_blocks: u32,
    sources: impl IntoIterator<Item = (u32, bool)>,
) -> Option<usize> {
    let mut sources = sources.into_iter().enumerate();
    let (_, (longest, is_cache_owner)) = sources.next()?;
    if longest.saturating_sub(local_prefix_blocks) < min_fetch_blocks {
        return None;
    }
    if is_cache_owner {
        return Some(0);
    }
    Some(
        sources
            .take_while(|(_, (prefix_blocks, _))| *prefix_blocks == longest)
            .find(|(_, (_, is_cache_owner))| *is_cache_owner)
            .map_or(0, |(index, _)| index),
    )
}
