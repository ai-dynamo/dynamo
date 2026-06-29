// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-query overlap scores at dequeue time.
//!
//! When a request has been parked in the scheduler queue, the KV cache state on each worker
//! may have changed significantly while it waited. The `tier_overlap_blocks` and
//! `effective_overlap_blocks` computed at enqueue time can be stale enough to pick the
//! wrong worker on dispatch.
//!
//! [`SchedulerQueue`](super::queue::SchedulerQueue) holds an optional refresher. When set,
//! the queue calls [`OverlapScoresRefresh::refresh`] for any request that waited longer than
//! the configured threshold, replacing the per-tier and effective-overlap fields on the
//! request with a fresh read from the indexer.
//!
//! The scope of refresh is intentionally narrow: it updates dispatch-time worker
//! selection/load inputs after an entry reaches the front of the queue, but it does not
//! re-key or reinsert that entry in the priority queue. Queue ordering therefore remains
//! based on the enqueue-time key, even when refreshed overlap data changes the eventual
//! worker chosen for dispatch.
//!
//! Refresh failures are non-fatal: an implementation can return `None` and the queue will
//! dispatch with the (stale) original scores rather than dropping the request.

use std::time::Duration;

use async_trait::async_trait;

use crate::config::KvRouterConfig;
use crate::indexer::TieredMatchProvider;
use crate::protocols::LocalBlockHash;

use super::overlap::{OverlapAnalysis, OverlapSignals};

/// Result of a successful overlap refresh.
///
/// Carries everything required to overwrite the overlap-related fields on a
/// [`SchedulingRequest`](super::types::SchedulingRequest) at dequeue time.
pub type RefreshedOverlap = OverlapSignals;

/// Re-query overlap scores for a request that has been waiting in the scheduler queue.
///
/// Implementations are expected to be cheap to clone (typically `Arc`-wrapped) and to never
/// panic. Returning `None` indicates the refresh failed; the queue will then dispatch with
/// the original scores.
#[async_trait]
pub trait OverlapScoresRefresh: Send + Sync {
    async fn refresh(&self, block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap>;
}

pub struct TieredOverlapRefresher<P> {
    provider: P,
    config: KvRouterConfig,
    block_size: u32,
}

impl<P> TieredOverlapRefresher<P> {
    pub fn new(provider: P, config: KvRouterConfig, block_size: u32) -> Self {
        Self {
            provider,
            config,
            block_size,
        }
    }
}

#[async_trait]
impl<P: TieredMatchProvider> OverlapScoresRefresh for TieredOverlapRefresher<P> {
    async fn refresh(&self, block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
        if block_hashes.is_empty() {
            return None;
        }
        let tiered = match self.provider.find_tiered_matches(block_hashes).await {
            Ok(tiered) => tiered,
            Err(error) => {
                tracing::warn!(%error, "overlap refresh: tiered match query failed");
                return None;
            }
        };
        Some(OverlapAnalysis::new(&self.config, self.block_size, &tiered).signals())
    }
}

/// Default wait threshold after which a dequeued request gets a fresh overlap-score lookup.
/// Override with `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS`. A value of `0` disables refresh.
pub const DEFAULT_OVERLAP_REFRESH_AFTER_SECS: u64 = 10;

pub fn read_overlap_refresh_after() -> Option<Duration> {
    let raw = match std::env::var("DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS") {
        Ok(v) => v,
        Err(_) => return Some(Duration::from_secs(DEFAULT_OVERLAP_REFRESH_AFTER_SECS)),
    };
    let parsed: f64 = match raw.parse() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                value = %raw,
                "invalid DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS, falling back to default {DEFAULT_OVERLAP_REFRESH_AFTER_SECS}s"
            );
            return Some(Duration::from_secs(DEFAULT_OVERLAP_REFRESH_AFTER_SECS));
        }
    };
    if !parsed.is_finite() {
        tracing::warn!(
            value = %raw,
            "non-finite DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS, disabling overlap refresh"
        );
        return None;
    }
    if parsed <= 0.0 {
        return None;
    }
    match Duration::try_from_secs_f64(parsed) {
        Ok(d) => Some(d),
        Err(err) => {
            tracing::warn!(
                value = %raw,
                error = %err,
                "DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS is out of range; disabling overlap refresh"
            );
            None
        }
    }
}

pub fn should_refresh_overlap(
    has_refresher: bool,
    refresh_after: Option<Duration>,
    block_hashes: Option<&[LocalBlockHash]>,
    enqueue_at: tokio::time::Instant,
    now: tokio::time::Instant,
) -> bool {
    let Some(refresh_after) = refresh_after else {
        return false;
    };
    if !has_refresher {
        return false;
    }
    let Some(block_hashes) = block_hashes else {
        return false;
    };
    if block_hashes.is_empty() {
        return false;
    }
    now.saturating_duration_since(enqueue_at) >= refresh_after
}

pub async fn refresh_overlap<RF: OverlapScoresRefresh + ?Sized>(
    refresher: Option<&RF>,
    refresh_after: Option<Duration>,
    block_hashes: Option<&[LocalBlockHash]>,
    enqueue_at: tokio::time::Instant,
    now: tokio::time::Instant,
) -> Option<RefreshedOverlap> {
    if !should_refresh_overlap(
        refresher.is_some(),
        refresh_after,
        block_hashes,
        enqueue_at,
        now,
    ) {
        return None;
    }
    refresher?.refresh(block_hashes?).await
}

/// Default no-op refresher used when dequeue-time overlap refresh is not configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopOverlapScoresRefresh;

#[async_trait]
impl OverlapScoresRefresh for NoopOverlapScoresRefresh {
    async fn refresh(&self, _block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{KvRouterError, MatchDetails, TieredMatchDetails};
    use crate::protocols::{OverlapScores, WorkerWithDpRank};
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    struct CountingRefresher {
        calls: AtomicUsize,
    }

    struct FakeProvider {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait]
    impl TieredMatchProvider for FakeProvider {
        async fn find_tiered_matches(
            &self,
            _sequence: &[LocalBlockHash],
        ) -> Result<TieredMatchDetails, KvRouterError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                return Err(KvRouterError::IndexerOffline);
            }
            let worker = WorkerWithDpRank::new(4, 0);
            let mut scores = OverlapScores::new();
            scores.scores.insert(worker, 2);
            Ok(TieredMatchDetails {
                device: MatchDetails {
                    overlap_scores: scores,
                    ..Default::default()
                },
                lower_tier: HashMap::new(),
                gms_placements: HashMap::new(),
            })
        }
    }

    #[async_trait]
    impl OverlapScoresRefresh for CountingRefresher {
        async fn refresh(&self, _block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Some(RefreshedOverlap {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::new(),
                effective_cached_tokens: HashMap::new(),
            })
        }
    }

    #[test]
    fn should_refresh_only_after_threshold_with_hashes() {
        let enqueue_at = tokio::time::Instant::now();
        let now = enqueue_at + Duration::from_secs(5);
        let hashes = [LocalBlockHash(1)];
        assert!(!should_refresh_overlap(
            true,
            Some(Duration::from_secs(10)),
            Some(&hashes),
            enqueue_at,
            now,
        ));
        assert!(should_refresh_overlap(
            true,
            Some(Duration::from_secs(1)),
            Some(&hashes),
            enqueue_at,
            now,
        ));
    }

    #[tokio::test]
    async fn refresh_overlap_is_side_effect_free_when_not_eligible() {
        let refresher = CountingRefresher {
            calls: AtomicUsize::new(0),
        };
        let enqueue_at = tokio::time::Instant::now();
        let now = enqueue_at + Duration::from_secs(1);
        let hashes = [LocalBlockHash(1)];
        assert!(
            refresh_overlap(
                Some(&refresher),
                Some(Duration::from_secs(10)),
                Some(&hashes),
                enqueue_at,
                now,
            )
            .await
            .is_none()
        );
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn tiered_refresher_converts_matches_and_handles_empty_or_failed_queries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let refresher = TieredOverlapRefresher::new(
            FakeProvider {
                calls: Arc::clone(&calls),
                fail: false,
            },
            KvRouterConfig::default(),
            16,
        );
        let worker = WorkerWithDpRank::new(4, 0);

        assert!(refresher.refresh(&[]).await.is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let refreshed = refresher.refresh(&[LocalBlockHash(1)]).await.unwrap();
        assert_eq!(refreshed.tier_overlap_blocks.device[&worker], 2);
        assert_eq!(refreshed.effective_cached_tokens[&worker], 32);

        let failing = TieredOverlapRefresher::new(
            FakeProvider { calls, fail: true },
            KvRouterConfig::default(),
            16,
        );
        assert!(failing.refresh(&[LocalBlockHash(1)]).await.is_none());
    }
}
