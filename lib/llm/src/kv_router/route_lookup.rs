// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    time::{Duration, Instant},
};

use dynamo_kv_router::{
    SharedKvCache,
    indexer::KvRouterError,
    protocols::{LocalBlockHash, SharedCacheHits},
};
use tracing::Instrument;

use super::{Indexer, indexer::TieredMatchDetails, metrics};

pub(super) struct TieredLookupResult {
    pub(super) tiered_matches: TieredMatchDetails,
    pub(super) shared_cache_hits: Option<SharedCacheHits>,
    pub(super) indexer_duration: Duration,
    pub(super) shared_cache_duration: Option<Duration>,
    pub(super) retained_block_hashes: Option<Vec<LocalBlockHash>>,
}

pub(super) fn split_retained_block_hashes(
    block_hashes: Vec<LocalBlockHash>,
    supports_overlap_refresh: bool,
    return_routing_hashes: bool,
) -> (Option<Vec<LocalBlockHash>>, Option<Vec<LocalBlockHash>>) {
    debug_assert!(supports_overlap_refresh || return_routing_hashes);

    if supports_overlap_refresh && return_routing_hashes {
        return (Some(block_hashes.clone()), Some(block_hashes));
    }
    if supports_overlap_refresh {
        return (Some(block_hashes), None);
    }

    debug_assert!(return_routing_hashes);
    (None, Some(block_hashes))
}

pub(super) async fn query_tiered_matches(
    indexer: &Indexer,
    shared_cache: Option<&dyn SharedKvCache>,
    tokens: &[u32],
    block_size: u32,
    block_hashes: Vec<LocalBlockHash>,
    retain_block_hashes: bool,
) -> Result<TieredLookupResult, KvRouterError> {
    if retain_block_hashes {
        let (tiered_matches, shared_cache_hits, indexer_duration, shared_cache_duration) =
            query_retained(indexer, shared_cache, tokens, block_size, &block_hashes).await?;

        return Ok(TieredLookupResult {
            tiered_matches,
            shared_cache_hits,
            indexer_duration,
            shared_cache_duration,
            retained_block_hashes: Some(block_hashes),
        });
    }

    let (tiered_matches, shared_cache_hits, indexer_duration, shared_cache_duration) =
        query_owned(indexer, shared_cache, tokens, block_size, block_hashes).await?;

    Ok(TieredLookupResult {
        tiered_matches,
        shared_cache_hits,
        indexer_duration,
        shared_cache_duration,
        retained_block_hashes: None,
    })
}

async fn query_retained(
    indexer: &Indexer,
    shared_cache: Option<&dyn SharedKvCache>,
    tokens: &[u32],
    block_size: u32,
    block_hashes: &[LocalBlockHash],
) -> Result<
    (
        TieredMatchDetails,
        Option<SharedCacheHits>,
        Duration,
        Option<Duration>,
    ),
    KvRouterError,
> {
    let block_count = block_hashes.len();
    let Some(shared_cache) = shared_cache else {
        let t = Instant::now();
        let tiered = indexer
            .find_matches_by_tier_ref(block_hashes)
            .instrument(tracing::info_span!(
                "kv_router.find_matches",
                block_count,
                block_size,
                retained = true,
                shared_cache = false,
            ))
            .await?;
        return Ok((tiered, None, t.elapsed(), None));
    };

    let indexer_fut =
        indexer
            .find_matches_by_tier_ref(block_hashes)
            .instrument(tracing::info_span!(
                "kv_router.find_matches",
                block_count,
                block_size,
                retained = true,
                shared_cache = true,
            ));
    join_indexer_and_shared_cache(indexer_fut, shared_cache, tokens, block_size, block_count).await
}

async fn query_owned(
    indexer: &Indexer,
    shared_cache: Option<&dyn SharedKvCache>,
    tokens: &[u32],
    block_size: u32,
    block_hashes: Vec<LocalBlockHash>,
) -> Result<
    (
        TieredMatchDetails,
        Option<SharedCacheHits>,
        Duration,
        Option<Duration>,
    ),
    KvRouterError,
> {
    let block_count = block_hashes.len();
    let Some(shared_cache) = shared_cache else {
        let t = Instant::now();
        let tiered = indexer
            .find_matches_by_tier(block_hashes)
            .instrument(tracing::info_span!(
                "kv_router.find_matches",
                block_count,
                block_size,
                retained = false,
                shared_cache = false,
            ))
            .await?;
        return Ok((tiered, None, t.elapsed(), None));
    };

    let indexer_fut = indexer
        .find_matches_by_tier(block_hashes)
        .instrument(tracing::info_span!(
            "kv_router.find_matches",
            block_count,
            block_size,
            retained = false,
            shared_cache = true,
        ));
    join_indexer_and_shared_cache(indexer_fut, shared_cache, tokens, block_size, block_count).await
}

async fn join_indexer_and_shared_cache<I>(
    indexer_fut: I,
    shared_cache: &dyn SharedKvCache,
    tokens: &[u32],
    block_size: u32,
    block_count: usize,
) -> Result<
    (
        TieredMatchDetails,
        Option<SharedCacheHits>,
        Duration,
        Option<Duration>,
    ),
    KvRouterError,
>
where
    I: Future<Output = Result<TieredMatchDetails, KvRouterError>>,
{
    let shared_fut = shared_cache
        .check_blocks(tokens, block_size)
        .instrument(tracing::info_span!(
            "kv_router.shared_cache_check",
            token_count = tokens.len(),
            block_count,
            block_size,
        ));

    let indexer_timed = async {
        let t = Instant::now();
        let r = indexer_fut.await;
        (r, t.elapsed())
    };
    let shared_timed = async {
        let t = Instant::now();
        let r = shared_fut.await;
        (r, t.elapsed())
    };

    let ((indexer_result, indexer_duration), (shared_result, shared_cache_duration)) =
        tokio::join!(indexer_timed, shared_timed);
    let tiered_matches = indexer_result?;
    let shared_cache_hits = match shared_result {
        Ok(hits) => Some(hits),
        Err(e) => {
            tracing::warn!(error = %e, "Shared cache query failed, ignoring");
            if let Some(m) = metrics::RoutingOverheadMetrics::get() {
                m.inc_shared_cache_errors();
            }
            None
        }
    };

    Ok((
        tiered_matches,
        shared_cache_hits,
        indexer_duration,
        Some(shared_cache_duration),
    ))
}
