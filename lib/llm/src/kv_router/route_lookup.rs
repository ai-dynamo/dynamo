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

#[allow(clippy::too_many_arguments)]
pub(super) async fn query_tiered_matches(
    indexer: &Indexer,
    shared_cache: Option<&dyn SharedKvCache>,
    tokens: &[u32],
    block_size: u32,
    block_hashes: Vec<LocalBlockHash>,
    cache_namespace: Option<&str>,
    retain_block_hashes: bool,
    gms_content_hashes_hex: Option<&[String]>,
) -> Result<TieredLookupResult, KvRouterError> {
    if retain_block_hashes {
        let (tiered_matches, shared_cache_hits, indexer_duration, shared_cache_duration) =
            query_retained(
                indexer,
                shared_cache,
                tokens,
                block_size,
                &block_hashes,
                cache_namespace,
                gms_content_hashes_hex,
            )
            .await?;

        return Ok(TieredLookupResult {
            tiered_matches,
            shared_cache_hits,
            indexer_duration,
            shared_cache_duration,
            retained_block_hashes: Some(block_hashes),
        });
    }

    let (tiered_matches, shared_cache_hits, indexer_duration, shared_cache_duration) = query_owned(
        indexer,
        shared_cache,
        tokens,
        block_size,
        block_hashes,
        cache_namespace,
        gms_content_hashes_hex,
    )
    .await?;

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
    cache_namespace: Option<&str>,
    gms_content_hashes_hex: Option<&[String]>,
) -> Result<
    (
        TieredMatchDetails,
        Option<SharedCacheHits>,
        Duration,
        Option<Duration>,
    ),
    KvRouterError,
> {
    let indexer_fut = async move {
        if let Some(gms_content_hashes_hex) = gms_content_hashes_hex {
            indexer
                .find_matches_by_tier_with_gms_placements(
                    block_hashes.to_vec(),
                    Some(gms_content_hashes_hex),
                )
                .await
        } else {
            indexer.find_matches_by_tier_ref(block_hashes).await
        }
    }
    .instrument(tracing::info_span!("kv_router.find_matches"));

    let Some(shared_cache) = shared_cache else {
        let t = Instant::now();
        let tiered = indexer_fut.await?;
        return Ok((tiered, None, t.elapsed(), None));
    };
    join_indexer_and_shared_cache(
        indexer_fut,
        shared_cache,
        tokens,
        block_size,
        cache_namespace,
    )
    .await
}

async fn query_owned(
    indexer: &Indexer,
    shared_cache: Option<&dyn SharedKvCache>,
    tokens: &[u32],
    block_size: u32,
    block_hashes: Vec<LocalBlockHash>,
    cache_namespace: Option<&str>,
    gms_content_hashes_hex: Option<&[String]>,
) -> Result<
    (
        TieredMatchDetails,
        Option<SharedCacheHits>,
        Duration,
        Option<Duration>,
    ),
    KvRouterError,
> {
    let indexer_fut = async move {
        if let Some(gms_content_hashes_hex) = gms_content_hashes_hex {
            indexer
                .find_matches_by_tier_with_gms_placements(
                    block_hashes,
                    Some(gms_content_hashes_hex),
                )
                .await
        } else {
            indexer.find_matches_by_tier(block_hashes).await
        }
    }
    .instrument(tracing::info_span!("kv_router.find_matches"));

    let Some(shared_cache) = shared_cache else {
        let t = Instant::now();
        let tiered = indexer_fut.await?;
        return Ok((tiered, None, t.elapsed(), None));
    };
    join_indexer_and_shared_cache(
        indexer_fut,
        shared_cache,
        tokens,
        block_size,
        cache_namespace,
    )
    .await
}

async fn join_indexer_and_shared_cache<I>(
    indexer_fut: I,
    shared_cache: &dyn SharedKvCache,
    tokens: &[u32],
    block_size: u32,
    cache_namespace: Option<&str>,
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
        .check_blocks(tokens, block_size, cache_namespace)
        .instrument(tracing::info_span!("kv_router.shared_cache_check"));

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
