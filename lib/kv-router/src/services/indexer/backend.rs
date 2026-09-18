// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::ConcurrentRadixTreeCompressed;
use crate::ThreadPoolIndexer;
use crate::approx::PruneConfig;
use crate::config::{ApproximateCachePolicyKind, KvRouterConfig};
use crate::indexer::{
    ApproximateLruIncarnation, ApproximateLruStats, ApproximateRetentionConfig, KvIndexer,
    KvIndexerInterface, KvIndexerMetrics, KvRouterError, LowerTierIndexers, LowerTierQueryOptions,
    MatchDetails, RoutingDecisionHashes, SyncIndexer, TieredMatchDetails, TieredMatchProvider,
    record_unsupported_residency_event,
};
use crate::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, OverlapScores,
    ResidencyProjection, ResidencyRoutingSnapshot, RouterEvent, WorkerId, WorkerWithDpRank,
};

use super::lookup::{HashInput, merge_side_or_warn};
use super::session_updates::{SessionMutation, SessionUpdateSender};

/// How the device-tier primary is populated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimaryRetention {
    /// Engine KV events are the only writer.
    EventDriven,
    /// No engine events: routing decisions are written into the primary and
    /// expire after the TTL. This is the frontend's `use_kv_events=false`
    /// approximate mode.
    ApproximateTtl(Duration),
}

/// Indexer shape derived from router configuration, mirroring the frontend
/// `KvRouter` indexer resolution so a selection service built from the same
/// `KvRouterConfig` indexes the same way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerPolicy {
    pub primary: PrimaryRetention,
    /// TTL of the predict-on-route side indexer layered over an event-driven
    /// primary (`router_predicted_ttl_secs`). `None` disables it.
    pub side_ttl: Option<Duration>,
}

impl Default for IndexerPolicy {
    fn default() -> Self {
        Self::event_driven()
    }
}

impl IndexerPolicy {
    pub fn event_driven() -> Self {
        Self {
            primary: PrimaryRetention::EventDriven,
            side_ttl: None,
        }
    }

    /// Resolve the indexer shape for `config`.
    ///
    /// Rejects the same combinations the frontend rejects. The local side indexer
    /// is TTL-only, so `router_approximate_cache_policy=lru` is rejected with
    /// `use_kv_events=true` and falls back to TTL with a warning otherwise.
    pub fn from_router_config(config: &KvRouterConfig) -> Result<Self> {
        if config.use_kv_events
            && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
        {
            anyhow::bail!(
                "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
            );
        }
        if config.router_predicted_ttl_secs.is_some() && !config.use_kv_events {
            anyhow::bail!(
                "router_predicted_ttl_secs requires use_kv_events=true; \
                 do not combine a primary approximate indexer with a side approximate indexer"
            );
        }
        if config.use_kv_events {
            return Ok(Self {
                primary: PrimaryRetention::EventDriven,
                side_ttl: config
                    .router_predicted_ttl_secs
                    .map(Duration::from_secs_f64),
            });
        }
        if config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru {
            tracing::warn!(
                "router_approximate_cache_policy=lru is not supported by the selection service yet; falling back to TTL retention"
            );
        }
        Ok(Self {
            primary: PrimaryRetention::ApproximateTtl(Duration::from_secs_f64(
                config.router_ttl_secs,
            )),
            side_ttl: None,
        })
    }
}

/// Predict-on-route side indexer: a short-TTL approximate index that routing
/// decisions populate so a worker gets cache credit for a prompt before the
/// engine's first KV event for it arrives. Always local, never dumped or
/// replayed, and never used to seed lower-tier lookups.
#[derive(Clone)]
pub enum SideIndexer {
    KvIndexer(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

impl SideIndexer {
    /// `cancel` stops the single-threaded indexer's pruning task.
    pub fn new(
        ttl: Duration,
        block_size: u32,
        num_threads: usize,
        metrics: Arc<KvIndexerMetrics>,
        cancel: CancellationToken,
    ) -> Self {
        let prune_config = Some(PruneConfig { ttl });
        if num_threads > 1 {
            return Self::Concurrent(Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                ConcurrentRadixTreeCompressed::new(),
                num_threads,
                block_size,
                Some(metrics),
                prune_config,
            )));
        }
        Self::KvIndexer(KvIndexer::new_with_pruning(
            cancel,
            block_size,
            metrics,
            prune_config,
        ))
    }

    pub(super) async fn find_matches_input(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .find_matches(sequence.into_owned_at_boundary())
                    .await
            }
            Self::Concurrent(indexer) => {
                Ok(indexer.backend().find_matches(sequence.as_slice(), false))
            }
        }
    }

    pub(super) async fn record_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(
                        worker,
                        hashes.local_hashes,
                        hashes.sequence_hashes,
                    )
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .process_routing_decision_hash_slices(
                        worker,
                        &hashes.local_hashes,
                        &hashes.sequence_hashes,
                    )
                    .await
            }
        }
    }

    async fn reset_worker_dp_rank_and_wait(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer(indexer) => indexer.remove_worker(worker_id).await,
            Self::Concurrent(indexer) => indexer.remove_worker(worker_id).await,
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: u32) {
        match self {
            Self::KvIndexer(indexer) => indexer.remove_worker_dp_rank(worker_id, dp_rank).await,
            Self::Concurrent(indexer) => indexer.remove_worker_dp_rank(worker_id, dp_rank).await,
        }
    }
}

/// A primary index served by another process, with transport supplied by the host.
#[async_trait]
pub trait RemotePrimary: Send + Sync {
    async fn find_matches_by_tier(
        &self,
        block_hashes: Vec<LocalBlockHash>,
        device_only: bool,
    ) -> Result<TieredMatchDetails>;

    /// Write a booked routing decision into an approximate remote primary.
    async fn record_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<()>;

    /// Whether the remote primary is fed by engine KV events (a side indexer
    /// may overlay it) rather than by routing decisions.
    fn use_kv_events(&self) -> bool;
}

/// Block-content indexer for one routing partition: a primary device-tier
/// backend plus a per-tier registry of lower-tier indexers (host-pinned,
/// disk, …), or a remote primary, or nothing when routing is load-only.
///
/// `approx` is the optional predict-on-route side indexer, queried alongside
/// the primary and merged by per-worker max. `primary_records_routing_decisions`
/// marks an approximate primary (no engine events) that routing decisions
/// populate directly. At most one of the two is active.
#[derive(Clone)]
pub enum Indexer {
    Single {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
        session_updates: Option<SessionUpdateSender>,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
        session_updates: Option<SessionUpdateSender>,
    },
    /// Primary served by another process. Lower tiers are that process's
    /// concern and arrive in its tiered response; only the side indexer is
    /// local.
    Remote {
        primary: Arc<dyn RemotePrimary>,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    /// No KV awareness.
    None,
}

impl Indexer {
    fn approx(&self) -> Option<&SideIndexer> {
        match self {
            Self::Single { approx, .. }
            | Self::Concurrent { approx, .. }
            | Self::Remote { approx, .. } => approx.as_ref(),
            Self::None => None,
        }
    }

    pub fn supports_overlap_refresh(&self) -> bool {
        matches!(self, Self::Single { .. } | Self::Concurrent { .. })
    }

    pub fn supports_kv_transfer_chain_retention(&self) -> bool {
        matches!(
            self,
            Self::Single {
                approx: None,
                primary_records_routing_decisions: false,
                ..
            } | Self::Concurrent {
                approx: None,
                primary_records_routing_decisions: false,
                ..
            }
        )
    }

    /// Publish a control-plane residency projection for subsequent lookups.
    pub fn set_residency_projection(&self, projection: ResidencyProjection) {
        match self {
            Self::Single { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                lower_tier.set_residency_projection(projection)
            }
            Self::Remote { .. } | Self::None => {}
        }
    }

    pub fn set_residency_routing_snapshot(&self, snapshot: ResidencyRoutingSnapshot) {
        match self {
            Self::Single { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                lower_tier.set_residency_routing_snapshot(snapshot)
            }
            Self::Remote { .. } | Self::None => {}
        }
    }

    /// Apply an event without tier dispatch — kept for callers that have
    /// already determined this is a device-tier event. Most callers should
    /// use [`Self::apply_event_routed`].
    pub async fn apply_event(&self, event: RouterEvent) {
        match self {
            Indexer::Single { primary, .. } => primary.apply_event(event).await,
            Indexer::Concurrent { primary, .. } => primary.apply_event(event).await,
            Indexer::Remote { .. } | Indexer::None => {
                tracing::trace!("Dropping KV event: no local primary indexer");
            }
        }
    }

    /// Apply an event, routing to the device-tier primary when
    /// `event.storage_tier.is_gpu()` and to the appropriate lower-tier
    /// indexer otherwise. `Cleared` events fan out to their applicable physical
    /// indexes according to the event's reset scope. Failures are logged and
    /// the first one is returned after every tier was attempted.
    pub async fn apply_event_routed(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let targets_primary = match event.targets_primary() {
            Ok(targets_primary) => targets_primary,
            Err(_) => {
                match self {
                    Self::Single { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                        lower_tier.record_unsupported_residency_event(&event);
                    }
                    Self::Remote { .. } | Self::None => {}
                }
                return Ok(());
            }
        };
        let is_clear = matches!(&event.event.data, KvCacheEventData::Cleared);
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    let mut reset_error = None;
                    if targets_primary
                        && let Err(error) = primary.apply_event_and_wait(event.clone()).await
                    {
                        tracing::warn!(%error, "Failed to reset primary residency");
                        reset_error = Some(error);
                    }
                    for indexer in lower_tier.all() {
                        if let Err(error) = indexer.apply_event_and_wait(event.clone()).await {
                            tracing::warn!(%error, "Failed to reset lower-tier residency");
                            reset_error.get_or_insert(error);
                        }
                    }
                    if let Some(error) = reset_error {
                        return Err(error);
                    }
                } else if targets_primary {
                    primary.apply_event(event).await;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    let mut reset_error = None;
                    if targets_primary
                        && let Err(error) = primary.apply_event_and_wait(event.clone()).await
                    {
                        tracing::warn!(%error, "Failed to reset primary residency");
                        reset_error = Some(error);
                    }
                    for indexer in lower_tier.all() {
                        if let Err(error) = indexer.apply_event_and_wait(event.clone()).await {
                            tracing::warn!(%error, "Failed to reset lower-tier residency");
                            reset_error.get_or_insert(error);
                        }
                    }
                    if let Some(error) = reset_error {
                        return Err(error);
                    }
                } else if targets_primary {
                    primary.apply_event(event).await;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            }
            Indexer::Remote { .. } | Indexer::None => {
                tracing::trace!("Dropping KV event: no local primary indexer");
            }
        }
        Ok(())
    }

    /// Admit one event with backpressure: a `Cleared` event completes only once
    /// every local tier has applied it (a completion barrier for recovery), a
    /// device event waits for the primary's queue to accept it, and a lower-tier
    /// event is enqueued. Unlike [`Self::apply_event_routed`], the first failure
    /// stops the fan-out.
    pub async fn try_apply_event(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let targets_primary = match event.targets_primary() {
            Ok(targets_primary) => targets_primary,
            Err(_) => {
                match self {
                    Self::Single { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                        lower_tier.record_unsupported_residency_event(&event);
                    }
                    Self::Remote { .. } | Self::None => {
                        record_unsupported_residency_event(None, &event);
                    }
                }
                return Ok(());
            }
        };
        let session_update = if targets_primary {
            match self {
                Self::Single {
                    session_updates, ..
                }
                | Self::Concurrent {
                    session_updates, ..
                } => session_updates.as_ref().and_then(|sender| {
                    SessionMutation::from_event(&event).map(|mutation| (sender.clone(), mutation))
                }),
                Self::Remote { .. } | Self::None => None,
            }
        } else {
            None
        };
        let is_clear = matches!(&event.event.data, KvCacheEventData::Cleared);
        match self {
            Self::Single {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    if targets_primary {
                        primary
                            .reset_worker_dp_rank_and_wait(event.worker_id, event.event.dp_rank)
                            .await?;
                    }
                    for indexer in lower_tier.all() {
                        indexer.apply_event_and_wait(event.clone()).await?;
                    }
                } else if targets_primary {
                    primary
                        .event_sender()
                        .send(event)
                        .await
                        .map_err(|_| KvRouterError::IndexerOffline)?;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .enqueue_event(event)?;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                if is_clear {
                    if targets_primary {
                        primary.apply_event_and_wait(event.clone()).await?;
                    }
                    for indexer in lower_tier.all() {
                        indexer.apply_event_and_wait(event.clone()).await?;
                    }
                } else if targets_primary {
                    primary.enqueue_event(event)?;
                } else {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .enqueue_event(event)?;
                }
            }
            Self::Remote { .. } | Self::None => {
                tracing::trace!("Dropping KV event: no local primary indexer");
            }
        }
        if let Some((sender, update)) = session_update {
            sender.enqueue(update)?;
        }
        Ok(())
    }

    pub fn session_residency_version(&self, worker: WorkerWithDpRank) -> Option<u64> {
        match self {
            Self::Single {
                session_updates, ..
            }
            | Self::Concurrent {
                session_updates, ..
            } => session_updates
                .as_ref()
                .map(|sender| sender.residency_version(worker)),
            Self::Remote { .. } | Self::None => None,
        }
    }

    pub fn enqueue_session_match(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        matched_hash: ExternalSequenceBlockHash,
        residency_version: u64,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::Single {
                session_updates, ..
            }
            | Self::Concurrent {
                session_updates, ..
            } => match session_updates {
                Some(sender) => sender.enqueue(SessionMutation::Matched {
                    worker,
                    session_id: session_id.to_owned(),
                    matched_hash,
                    residency_version,
                }),
                None => Ok(()),
            },
            Self::Remote { .. } | Self::None => Ok(()),
        }
    }

    #[cfg(test)]
    async fn flush_session_updates(&self) -> Result<(), KvRouterError> {
        match self {
            Self::Single {
                session_updates, ..
            }
            | Self::Concurrent {
                session_updates, ..
            } => match session_updates {
                Some(updates) => updates.flush().await,
                None => Ok(()),
            },
            Self::Remote { .. } | Self::None => Ok(()),
        }
    }

    /// Cold-reset one logical rank and wait until every local tier (and the
    /// side indexer) has completed the removal.
    pub async fn reset_worker_dp_rank_and_wait(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::Single {
                primary,
                lower_tier,
                approx,
                ..
            } => {
                primary
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await?;
                for indexer in lower_tier.all() {
                    indexer
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                approx,
                ..
            } => {
                primary
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await?;
                for indexer in lower_tier.all() {
                    indexer
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::Remote { approx, .. } => {
                if let Some(approx) = approx {
                    approx
                        .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                        .await?;
                }
            }
            Self::None => {}
        }
        let session_updates = match self {
            Self::Single {
                session_updates, ..
            }
            | Self::Concurrent {
                session_updates, ..
            } => session_updates.as_ref(),
            Self::Remote { .. } | Self::None => None,
        };
        if let Some(session_updates) = session_updates {
            session_updates.enqueue(SessionMutation::Cleared {
                worker: WorkerWithDpRank::new(worker_id, dp_rank),
            })?;
            session_updates.flush().await?;
        }
        Ok(())
    }

    pub async fn remove_worker(&self, worker_id: WorkerId) {
        if let Some(side) = self.approx() {
            side.remove_worker(worker_id).await;
        }
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                primary.remove_worker(worker_id).await;
            }
            Indexer::Remote { .. } | Indexer::None => {}
        }
    }

    pub async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: u32) {
        if let Some(side) = self.approx() {
            side.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
        match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
                primary.remove_worker_dp_rank(worker_id, dp_rank).await;
            }
            Indexer::Remote { .. } | Indexer::None => {}
        }
    }

    /// Device-tier overlap scores, including side-indexer credit. Tier-aware
    /// callers should use [`Self::find_tiered_matches`].
    pub async fn find_matches(&self, hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        let Some(side) = self.approx() else {
            return self.find_primary_matches(hashes).await;
        };
        let primary = self.find_primary_matches(hashes.clone()).await?;
        let mut merged = MatchDetails::new();
        merged.overlap_scores = primary;
        Ok(
            merge_side_or_warn(Some(side), merged, HashInput::Owned(hashes))
                .await
                .overlap_scores,
        )
    }

    /// Device-tier overlap scores from the primary only (no side-indexer credit).
    async fn find_primary_matches(&self, hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        Ok(match self {
            Indexer::Single { primary, .. } => primary.find_matches(hashes).await?,
            Indexer::Concurrent { primary, .. } => primary.find_matches(hashes).await?,
            Indexer::Remote { primary, .. } => {
                primary
                    .find_matches_by_tier(hashes, true)
                    .await?
                    .device
                    .overlap_scores
            }
            Indexer::None => OverlapScores::default(),
        })
    }

    /// Device match details + per-tier hits, suitable for building the
    /// Mooncake-RFC-shape per-instance breakdown.
    pub async fn find_tiered_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_tiered_matches_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    /// [`Self::find_tiered_matches`] with lower-tier query options. Router-hint
    /// chain retention is honored only by a local event-driven primary (see
    /// [`Self::supports_kv_transfer_chain_retention`]); other shapes ignore it.
    pub async fn find_tiered_matches_with_options(
        &self,
        sequence: Vec<LocalBlockHash>,
        options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        let options = LowerTierQueryOptions {
            retain_kv_transfer_chain: options.retain_kv_transfer_chain
                && self.supports_kv_transfer_chain_retention(),
        };
        self.find_matches_by_tier_with_options(sequence, options)
            .await
    }

    /// Borrowed variant of [`Self::find_tiered_matches_with_options`].
    pub async fn find_tiered_matches_ref_with_options(
        &self,
        sequence: &[LocalBlockHash],
        options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        let options = LowerTierQueryOptions {
            retain_kv_transfer_chain: options.retain_kv_transfer_chain
                && self.supports_kv_transfer_chain_retention(),
        };
        self.find_matches_by_tier_ref_with_options(sequence, options)
            .await
    }

    /// Dump every event tracked by this indexer in a form replayable by a
    /// peer's [`Self::apply_event_routed`]: the primary device-tier dump first,
    /// then every allocated lower-tier indexer's dump with `storage_tier`
    /// retagged to the tier it lives in.
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let (primary_events, lower_tier_entries) = match self {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => (primary.dump_events().await?, lower_tier.entries()),
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => (primary.dump_events().await?, lower_tier.entries()),
            // The remote service owns the events; a peer recovers from it, not
            // from this process.
            Indexer::Remote { .. } => return Ok(Vec::new()),
            Indexer::None => {
                return Err(KvRouterError::Unsupported(
                    "event dumping requires a KV indexer".to_string(),
                ));
            }
        };

        let mut out = primary_events;
        for (tier, indexer) in lower_tier_entries {
            let events = indexer.dump_events().await?;
            for mut event in events {
                event.storage_tier = tier;
                out.push(event);
            }
        }
        Ok(out)
    }

    pub fn uses_approximate_lru(&self) -> bool {
        match self {
            Self::Single { primary, .. } => primary.approximate_lru_enabled(),
            Self::Concurrent { primary, .. } => primary.approximate_lru_enabled(),
            Self::Remote { .. } | Self::None => false,
        }
    }

    pub fn set_approximate_lru_capacity_now(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        capacity: Option<usize>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::Single { primary, .. } => {
                primary.set_approximate_lru_capacity_now(worker, incarnation, capacity)
            }
            Self::Concurrent { primary, .. } => {
                primary.set_approximate_lru_capacity_now(worker, incarnation, capacity)
            }
            Self::Remote { .. } | Self::None => Ok(()),
        }
    }

    pub async fn approximate_lru_stats(&self) -> Result<ApproximateLruStats, KvRouterError> {
        match self {
            Self::Single { primary, .. } => primary.approximate_lru_stats().await,
            Self::Concurrent { primary, .. } => primary.approximate_lru_stats().await,
            Self::Remote { .. } | Self::None => Ok(ApproximateLruStats::default()),
        }
    }
}

#[async_trait]
impl TieredMatchProvider for Indexer {
    async fn find_tiered_matches(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_tiered_matches_ref_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    async fn find_tiered_matches_with_options(
        &self,
        sequence: &[LocalBlockHash],
        options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_tiered_matches_ref_with_options(sequence, options)
            .await
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    /// Construct a STORE [`RouterEvent`] for `local_hashes` that chain off
    /// `prefix_hashes` (the parent path). The event is tagged `storage_tier`.
    /// Mirrors the helper used in the request-plane tests so behavior across
    /// the two indexer surfaces stays comparable.
    pub fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::store_event;
    use super::*;
    use crate::indexer::KvIndexerInterface;
    #[cfg(feature = "metrics")]
    use crate::indexer::{METRIC_EVENT_CLEARED, METRIC_STATUS_OK};
    use crate::protocols::{
        KvCacheEvent, LocalBlockHash, ResidencyDomain, StorageTier, WorkerWithDpRank,
    };

    /// Apply a Device store and a HostPinned store anchored on it. The tiered
    /// query must surface both tier hits, and the device-tier `find_matches`
    /// must still see the device store (i.e. dispatch routed it to the
    /// primary, not to the lower-tier slot).
    #[tokio::test]
    async fn apply_event_routed_dispatches_by_tier() {
        let indexer = create_indexer(4, 1);
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await
            .unwrap();

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        // Flush primary so the in-flight events are observable by the query.
        if let Indexer::Single { primary, .. } = &indexer {
            let _ = primary.flush().await;
        }
        if let Indexer::Single { lower_tier, .. } = &indexer {
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let tiered = indexer.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks for the worker"
        );
        let host_hits = tiered
            .lower_tier
            .get(&StorageTier::HostPinned)
            .expect("host-pinned tier should have been allocated");
        assert_eq!(
            host_hits.hits.get(&worker).copied(),
            Some(1),
            "host-pinned should report 1 additional matched block beyond device"
        );
    }

    /// Dump every tier's events from a populated indexer, replay through a
    /// fresh indexer with `apply_event_routed`, and assert the tiered query
    /// result is identical. This is the peer-recovery round trip: it would
    /// fail if `dump_events` skipped lower-tier events or omitted their tier
    /// tags (peer would replay HostPinned events into the device primary).
    #[tokio::test]
    async fn dump_events_round_trips_through_apply_event_routed() {
        let block_size = 4;
        let worker = WorkerWithDpRank::new(7, 0);
        let source = create_indexer(block_size, 1);

        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        source
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await
            .unwrap();

        if let Indexer::Single {
            primary,
            lower_tier,
            ..
        } = &source
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let dump = source.dump_events().await.unwrap();

        // Sanity: dump must surface events from every tier we fed in.
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Device)),
            "dump must contain Device events"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::HostPinned)),
            "dump must contain HostPinned events with tier retagged"
        );
        assert!(
            dump.iter()
                .any(|e| matches!(e.storage_tier, StorageTier::Disk)),
            "dump must contain Disk events with tier retagged"
        );

        let replayed = create_indexer(block_size, 1);
        for event in dump {
            replayed.apply_event_routed(event).await.unwrap();
        }
        if let Indexer::Single {
            primary,
            lower_tier,
            ..
        } = &replayed
        {
            let _ = primary.flush().await;
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }

        let sequence = vec![
            LocalBlockHash(11),
            LocalBlockHash(12),
            LocalBlockHash(13),
            LocalBlockHash(14),
        ];
        let tiered = replayed.find_tiered_matches(sequence).await.unwrap();

        assert_eq!(
            tiered.device.overlap_scores.scores.get(&worker).copied(),
            Some(2),
            "device should match 2 blocks after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "host-pinned should report 1 additional block after replay"
        );
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|d| d.hits.get(&worker).copied()),
            Some(1),
            "disk should report 1 additional block after replay"
        );
    }

    #[tokio::test]
    async fn clear_reports_partial_failure_after_attempting_every_tier() {
        let indexer = create_indexer(4, 1);
        let worker = WorkerWithDpRank::new(7, 0);
        let (failed_tier, healthy_tier) = match &indexer {
            Indexer::Single { lower_tier, .. } => (
                lower_tier.get_or_create(StorageTier::HostPinned),
                lower_tier.get_or_create(StorageTier::Disk),
            ),
            Indexer::Concurrent { .. } | Indexer::Remote { .. } | Indexer::None => {
                unreachable!("test creates the single indexer")
            }
        };

        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                1,
                &[],
                &[11],
                StorageTier::Disk,
            ))
            .await
            .unwrap();
        let _ = healthy_tier.dump_events().await.unwrap();
        failed_tier.shutdown();

        let error = indexer
            .apply_event_routed(RouterEvent::with_residency_domain(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 2,
                    data: KvCacheEventData::Cleared,
                    dp_rank: worker.dp_rank,
                },
                StorageTier::Device,
                ResidencyDomain::Worker,
            ))
            .await
            .unwrap_err();

        assert!(matches!(error, KvRouterError::IndexerOffline));
        assert!(healthy_tier.dump_events().await.unwrap().is_empty());
    }

    fn policy_indexer(num_threads: usize, policy: IndexerPolicy) -> Indexer {
        create_indexer_with_policy(
            4,
            num_threads,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            &policy,
        )
    }

    async fn flush_side(approx: &Option<SideIndexer>) {
        match approx {
            Some(SideIndexer::KvIndexer(side)) => {
                let _ = side.flush().await;
            }
            Some(SideIndexer::Concurrent(side)) => {
                let _ = side.flush().await;
            }
            None => {}
        }
    }

    /// Wait until every locally enqueued event and routing decision is applied.
    async fn flush(indexer: &Indexer) {
        match indexer {
            Indexer::Single {
                primary, approx, ..
            } => {
                let _ = primary.flush().await;
                flush_side(approx).await;
            }
            Indexer::Concurrent {
                primary, approx, ..
            } => {
                let _ = primary.flush().await;
                flush_side(approx).await;
            }
            Indexer::Remote { approx, .. } => flush_side(approx).await,
            Indexer::None => {}
        }
    }

    impl Indexer {
        pub(crate) async fn flush(&self) {
            flush(self).await
        }
    }

    #[tokio::test]
    async fn event_driven_policy_ignores_routing_decisions() {
        let indexer = policy_indexer(1, IndexerPolicy::event_driven());
        assert!(!indexer.records_routing_decisions());
        assert!(indexer.supports_kv_transfer_chain_retention());
        let worker = WorkerWithDpRank::new(7, 0);
        indexer
            .record_routing_decision_hashes(
                worker,
                RoutingDecisionHashes::from_local_hashes(vec![LocalBlockHash(11)]),
            )
            .await
            .unwrap();
        flush(&indexer).await;
        let tiered = indexer
            .find_tiered_matches(vec![LocalBlockHash(11)])
            .await
            .unwrap();
        assert!(tiered.device.overlap_scores.scores.is_empty());
    }

    #[tokio::test]
    async fn approximate_primary_records_routing_decisions() {
        for num_threads in [1, 2] {
            let indexer = policy_indexer(
                num_threads,
                IndexerPolicy {
                    primary: PrimaryRetention::ApproximateTtl(Duration::from_secs(60)),
                    side_ttl: None,
                },
            );
            assert!(indexer.records_routing_decisions());
            assert!(!indexer.supports_kv_transfer_chain_retention());
            let worker = WorkerWithDpRank::new(7, 0);
            indexer
                .record_routing_decision_hashes(
                    worker,
                    RoutingDecisionHashes::from_local_hashes(vec![
                        LocalBlockHash(11),
                        LocalBlockHash(12),
                    ]),
                )
                .await
                .unwrap();
            flush(&indexer).await;

            let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
            let tiered = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
            assert_eq!(
                tiered.device.overlap_scores.scores.get(&worker).copied(),
                Some(2),
                "{num_threads} thread(s): approximate primary should credit the routed prefix"
            );
            let flat = indexer.find_matches(sequence).await.unwrap();
            assert_eq!(flat.scores.get(&worker).copied(), Some(2));

            // The recorded prefix is dumped like any other primary state, so a
            // recovering peer inherits it.
            assert!(!indexer.dump_events().await.unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn side_indexer_merges_predicted_scores_by_worker_max() {
        for num_threads in [1, 2] {
            let indexer = policy_indexer(
                num_threads,
                IndexerPolicy {
                    primary: PrimaryRetention::EventDriven,
                    side_ttl: Some(Duration::from_secs(60)),
                },
            );
            assert!(indexer.records_routing_decisions());
            let confirmed = WorkerWithDpRank::new(7, 0);
            let predicted = WorkerWithDpRank::new(8, 0);

            // Engine event: worker 7 confirmed holds [11, 12].
            indexer
                .apply_event_routed(store_event(
                    confirmed.worker_id,
                    confirmed.dp_rank,
                    1,
                    &[],
                    &[11, 12],
                    StorageTier::Device,
                ))
                .await
                .unwrap();
            // Routing decisions: worker 8 was just routed [11, 12, 13]; worker 7
            // was routed [11] (shorter than what the engine confirmed).
            indexer
                .record_routing_decision_hashes(
                    predicted,
                    RoutingDecisionHashes::from_local_hashes(vec![
                        LocalBlockHash(11),
                        LocalBlockHash(12),
                        LocalBlockHash(13),
                    ]),
                )
                .await
                .unwrap();
            indexer
                .record_routing_decision_hashes(
                    confirmed,
                    RoutingDecisionHashes::from_local_hashes(vec![LocalBlockHash(11)]),
                )
                .await
                .unwrap();
            flush(&indexer).await;

            let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
            let tiered = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
            let scores = &tiered.device.overlap_scores.scores;
            assert_eq!(
                scores.get(&predicted).copied(),
                Some(3),
                "{num_threads} thread(s)"
            );
            assert_eq!(
                scores.get(&confirmed).copied(),
                Some(2),
                "{num_threads} thread(s): primary wins when it saw the longer prefix"
            );
            // Side scores never seed lower tiers and are never dumped.
            assert!(tiered.lower_tier.values().all(|tier| tier.hits.is_empty()));
            let dumped = indexer.dump_events().await.unwrap();
            assert!(
                dumped
                    .iter()
                    .all(|event| event.worker_id == confirmed.worker_id)
            );

            indexer.remove_worker(predicted.worker_id).await;
            flush(&indexer).await;
            let tiered = indexer.find_tiered_matches(sequence).await.unwrap();
            assert!(!tiered.device.overlap_scores.scores.contains_key(&predicted));
        }
    }

    #[tokio::test]
    async fn remote_primary_merges_local_side_scores() {
        struct TestRemote(Indexer);

        #[async_trait]
        impl RemotePrimary for TestRemote {
            async fn find_matches_by_tier(
                &self,
                hashes: Vec<LocalBlockHash>,
                _device_only: bool,
            ) -> Result<TieredMatchDetails> {
                Ok(self.0.find_tiered_matches(hashes).await?)
            }

            async fn record_routing_decision(
                &self,
                _worker: WorkerWithDpRank,
                _hashes: RoutingDecisionHashes,
            ) -> Result<()> {
                anyhow::bail!("event-driven remote primary must not receive routing decisions")
            }

            fn use_kv_events(&self) -> bool {
                true
            }
        }

        let served_indexer = create_indexer(4, 1);
        let confirmed = WorkerWithDpRank::new(7, 0);
        served_indexer
            .apply_event_routed(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        served_indexer
            .apply_event_routed(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        flush(&served_indexer).await;
        if let Indexer::Single { lower_tier, .. } = &served_indexer {
            for inner in lower_tier.all() {
                let _ = inner.dump_events().await.unwrap();
            }
        }
        let indexer = Indexer::Remote {
            primary: Arc::new(TestRemote(served_indexer)),
            approx: Some(SideIndexer::new(
                Duration::from_secs(60),
                4,
                1,
                Arc::new(KvIndexerMetrics::new_unregistered()),
                CancellationToken::new(),
            )),
            primary_records_routing_decisions: false,
        };
        assert!(indexer.records_routing_decisions());
        assert!(!indexer.supports_kv_transfer_chain_retention());

        // Local events are dropped: the remote service owns the primary.
        indexer
            .apply_event_routed(store_event(
                9,
                0,
                1,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        assert!(indexer.dump_events().await.unwrap().is_empty());

        // A routing decision lands in the local side indexer only.
        let predicted = WorkerWithDpRank::new(8, 0);
        indexer
            .record_routing_decision_hashes(
                predicted,
                RoutingDecisionHashes::from_local_hashes(vec![LocalBlockHash(11)]),
            )
            .await
            .unwrap();
        flush(&indexer).await;

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let tiered = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
        let scores = &tiered.device.overlap_scores.scores;
        assert_eq!(
            scores.get(&confirmed).copied(),
            Some(2),
            "remote device match"
        );
        assert_eq!(scores.get(&predicted).copied(), Some(1), "local side match");
        assert!(!scores.contains_key(&WorkerWithDpRank::new(9, 0)));
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|d| d.hits.get(&confirmed).copied()),
            Some(1),
            "remote lower tiers are carried through"
        );
        let flat = indexer.find_matches(sequence).await.unwrap();
        assert_eq!(flat.scores.get(&confirmed).copied(), Some(2));
        assert_eq!(flat.scores.get(&predicted).copied(), Some(1));
    }

    #[test]
    fn indexer_policy_mirrors_frontend_resolution() {
        let event_driven = KvRouterConfig::default();
        assert_eq!(
            IndexerPolicy::from_router_config(&event_driven).unwrap(),
            IndexerPolicy::event_driven()
        );

        let side = KvRouterConfig {
            router_predicted_ttl_secs: Some(2.5),
            ..Default::default()
        };
        assert_eq!(
            IndexerPolicy::from_router_config(&side).unwrap().side_ttl,
            Some(Duration::from_secs_f64(2.5))
        );

        let approximate = KvRouterConfig {
            use_kv_events: false,
            router_ttl_secs: 30.0,
            ..Default::default()
        };
        assert_eq!(
            IndexerPolicy::from_router_config(&approximate).unwrap(),
            IndexerPolicy {
                primary: PrimaryRetention::ApproximateTtl(Duration::from_secs(30)),
                side_ttl: None,
            }
        );

        let lru_without_events = KvRouterConfig {
            use_kv_events: false,
            router_approximate_cache_policy: ApproximateCachePolicyKind::Lru,
            ..Default::default()
        };
        assert!(matches!(
            IndexerPolicy::from_router_config(&lru_without_events)
                .unwrap()
                .primary,
            PrimaryRetention::ApproximateTtl(_)
        ));

        let lru_with_events = KvRouterConfig {
            router_approximate_cache_policy: ApproximateCachePolicyKind::Lru,
            ..Default::default()
        };
        assert!(IndexerPolicy::from_router_config(&lru_with_events).is_err());

        let side_without_events = KvRouterConfig {
            use_kv_events: false,
            router_predicted_ttl_secs: Some(1.0),
            ..Default::default()
        };
        assert!(IndexerPolicy::from_router_config(&side_without_events).is_err());
    }

    #[cfg(feature = "metrics")]
    #[tokio::test]
    async fn acknowledged_clear_records_primary_event_for_both_indexers() {
        for num_threads in [1, 2] {
            let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
            let indexer = create_indexer_with_metrics(4, num_threads, metrics.clone());
            indexer
                .apply_event_routed(RouterEvent::with_residency_domain(
                    7,
                    KvCacheEvent {
                        event_id: 1,
                        data: KvCacheEventData::Cleared,
                        dp_rank: 0,
                    },
                    StorageTier::Device,
                    ResidencyDomain::Worker,
                ))
                .await
                .unwrap();

            assert_eq!(
                metrics
                    .kv_cache_events_applied
                    .get_metric_with_label_values(&[METRIC_EVENT_CLEARED, METRIC_STATUS_OK])
                    .unwrap()
                    .get(),
                1,
                "clear metric mismatch for {num_threads} indexer thread(s)"
            );
        }
    }
}

pub fn create_indexer(block_size: u32, num_threads: usize) -> Indexer {
    create_indexer_with_metrics(
        block_size,
        num_threads,
        Arc::new(KvIndexerMetrics::new_unregistered()),
    )
}

pub fn create_indexer_with_metrics(
    block_size: u32,
    num_threads: usize,
    metrics: Arc<KvIndexerMetrics>,
) -> Indexer {
    create_indexer_with_policy(
        block_size,
        num_threads,
        metrics,
        &IndexerPolicy::event_driven(),
    )
}

pub fn create_indexer_with_policy(
    block_size: u32,
    num_threads: usize,
    metrics: Arc<KvIndexerMetrics>,
    policy: &IndexerPolicy,
) -> Indexer {
    let approx = policy.side_ttl.map(|ttl| {
        SideIndexer::new(
            ttl,
            block_size,
            num_threads,
            Arc::clone(&metrics),
            CancellationToken::new(),
        )
    });
    let retention = match &policy.primary {
        PrimaryRetention::EventDriven => None,
        PrimaryRetention::ApproximateTtl(ttl) => {
            Some(ApproximateRetentionConfig::Ttl(PruneConfig { ttl: *ttl }))
        }
    };
    let primary_records_routing_decisions = retention.is_some();
    if num_threads > 1 {
        Indexer::Concurrent {
            primary: Arc::new(
                ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                    ConcurrentRadixTreeCompressed::new(),
                    num_threads,
                    block_size,
                    Some(metrics),
                    retention,
                ),
            ),
            lower_tier: LowerTierIndexers::new(num_threads, block_size),
            approx,
            primary_records_routing_decisions,
            session_updates: None,
        }
    } else {
        Indexer::Single {
            primary: KvIndexer::new_with_approximate_retention(
                CancellationToken::new(),
                block_size,
                metrics,
                retention,
            ),
            lower_tier: LowerTierIndexers::new(1, block_size),
            approx,
            primary_records_routing_decisions,
            session_updates: None,
        }
    }
}

#[cfg(test)]
mod session_tests {
    use super::test_util::store_event;
    use super::*;
    use crate::SessionPrefixIndexer;
    use crate::protocols::{KvCacheEvent, KvCacheRemoveData, StorageTier};

    fn make_session_test_indexer() -> (Indexer, Arc<SessionPrefixIndexer>) {
        let session_prefix_index = Arc::new(SessionPrefixIndexer::new());
        let primary = KvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        );
        let indexer = Indexer::Single {
            primary: primary.clone(),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: Some(SessionUpdateSender::for_legacy(
                Arc::clone(&session_prefix_index),
                primary,
            )),
        };
        (indexer, session_prefix_index)
    }

    fn make_concurrent_session_test_indexer() -> (Indexer, Arc<SessionPrefixIndexer>) {
        let session_prefix_index = Arc::new(SessionPrefixIndexer::new());
        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
        ));
        let indexer = Indexer::Concurrent {
            primary: Arc::clone(&primary),
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: Some(SessionUpdateSender::for_concurrent(
                Arc::clone(&session_prefix_index),
                primary,
            )),
        };
        (indexer, session_prefix_index)
    }

    pub(crate) fn remove_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        block_hashes: Vec<ExternalSequenceBlockHash>,
    ) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                dp_rank,
            },
        )
    }

    pub(crate) fn clear_event(worker_id: u64, dp_rank: u32, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank,
            },
        )
    }

    #[tokio::test]
    async fn only_attributed_stored_events_update_session_lineage() {
        let (indexer, session_prefix_index) = make_session_test_indexer();

        indexer
            .try_apply_event(
                store_event(7, 0, 1, &[], &[41], StorageTier::Device).with_session_id("session-1"),
            )
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(7, 0, 2, &[], &[51], StorageTier::HostPinned))
            .await
            .unwrap();
        indexer.flush_session_updates().await.unwrap();

        assert_eq!(
            session_prefix_index
                .get_session_block_lineage("session-1", WorkerWithDpRank::new(7, 0), None)
                .unwrap(),
            vec![vec![ExternalSequenceBlockHash(41)]]
        );
        assert_eq!(session_prefix_index.session_count(), 1);
        assert_eq!(session_prefix_index.node_count(), 1);
        assert!(
            session_prefix_index
                .get_node_from_hash(ExternalSequenceBlockHash(51))
                .is_none()
        );
    }

    #[tokio::test]
    async fn physical_events_update_worker_local_session_frontiers_in_order() {
        let (indexer, session_prefix_index) = make_session_test_indexer();
        let first_worker = WorkerWithDpRank::new(7, 0);
        let second_worker = WorkerWithDpRank::new(8, 0);
        let first_store = store_event(7, 0, 1, &[], &[41, 42, 43], StorageTier::Device)
            .with_session_id("session-1");
        let block_hashes = match &first_store.event.data {
            KvCacheEventData::Stored(stored) => stored
                .blocks
                .iter()
                .map(|block| block.block_hash)
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };
        let second_store = store_event(8, 0, 1, &[], &[41, 42, 43], StorageTier::Device)
            .with_session_id("session-1");

        indexer.try_apply_event(first_store).await.unwrap();
        indexer.try_apply_event(second_store).await.unwrap();
        indexer
            .try_apply_event(remove_event(7, 0, 2, block_hashes[1..].to_vec()))
            .await
            .unwrap();
        indexer.flush_session_updates().await.unwrap();

        assert_eq!(
            session_prefix_index
                .get_session_block_lineage("session-1", first_worker, None)
                .unwrap(),
            vec![vec![block_hashes[0]]]
        );
        assert_eq!(
            session_prefix_index
                .get_session_block_lineage("session-1", second_worker, None)
                .unwrap(),
            vec![block_hashes.clone()]
        );

        indexer.try_apply_event(clear_event(8, 0, 2)).await.unwrap();
        indexer.flush_session_updates().await.unwrap();
        assert!(
            session_prefix_index
                .get_session_block_lineage("session-1", second_worker, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            session_prefix_index.node_count(),
            3,
            "physical removal preserves logical topology"
        );
    }

    #[tokio::test]
    async fn concurrent_physical_events_precede_session_updates() {
        let (indexer, session_prefix_index) = make_concurrent_session_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);
        let store =
            store_event(7, 0, 1, &[], &[41, 42], StorageTier::Device).with_session_id("session-1");
        let block_hashes = match &store.event.data {
            KvCacheEventData::Stored(stored) => stored
                .blocks
                .iter()
                .map(|block| block.block_hash)
                .collect::<Vec<_>>(),
            _ => unreachable!(),
        };

        indexer.try_apply_event(store).await.unwrap();
        indexer
            .try_apply_event(remove_event(7, 0, 2, vec![block_hashes[1]]))
            .await
            .unwrap();
        indexer.flush_session_updates().await.unwrap();

        assert_eq!(
            session_prefix_index
                .get_session_block_lineage("session-1", worker, None)
                .unwrap(),
            vec![vec![block_hashes[0]]]
        );
    }

    #[tokio::test]
    async fn route_match_is_ordered_after_pending_remove_and_store() {
        let (indexer, session_prefix_index) = make_session_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);
        let initial_store =
            store_event(7, 0, 1, &[], &[41], StorageTier::Device).with_session_id("session-old");
        let block_hash = match &initial_store.event.data {
            KvCacheEventData::Stored(stored) => stored.blocks[0].block_hash,
            _ => unreachable!(),
        };

        indexer.try_apply_event(initial_store).await.unwrap();
        indexer.flush_session_updates().await.unwrap();

        indexer
            .try_apply_event(remove_event(7, 0, 2, vec![block_hash]))
            .await
            .unwrap();
        indexer
            .try_apply_event(
                store_event(7, 0, 3, &[], &[41], StorageTier::Device).with_session_id("session-a"),
            )
            .await
            .unwrap();
        let residency_version = indexer.session_residency_version(worker).unwrap();
        indexer
            .enqueue_session_match("session-b", worker, block_hash, residency_version)
            .unwrap();
        indexer.flush_session_updates().await.unwrap();

        for session in ["session-a", "session-b"] {
            assert_eq!(
                session_prefix_index
                    .get_session_block_lineage(session, worker, None)
                    .unwrap(),
                vec![vec![block_hash]]
            );
        }
    }

    #[tokio::test]
    async fn stale_route_match_does_not_undo_removal() {
        let (indexer, session_prefix_index) = make_session_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);
        let store =
            store_event(7, 0, 1, &[], &[41], StorageTier::Device).with_session_id("session-b");
        let block_hash = match &store.event.data {
            KvCacheEventData::Stored(stored) => stored.blocks[0].block_hash,
            _ => unreachable!(),
        };

        indexer.try_apply_event(store).await.unwrap();
        indexer.flush_session_updates().await.unwrap();
        indexer
            .try_apply_event(remove_event(7, 0, 2, vec![block_hash]))
            .await
            .unwrap();
        // The physical removal is queued, but its ordered session mutation has
        // not run yet. A match observed in this window must still be invalidated.
        let stale_version = indexer.session_residency_version(worker).unwrap();
        indexer
            .enqueue_session_match("session-b", worker, block_hash, stale_version)
            .unwrap();
        indexer.flush_session_updates().await.unwrap();

        assert!(
            session_prefix_index
                .get_session_block_lineage("session-b", worker, None)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn rank_reset_waits_for_session_frontier_clear() {
        let (indexer, session_prefix_index) = make_session_test_indexer();
        let reset_rank = WorkerWithDpRank::new(7, 0);
        let retained_rank = WorkerWithDpRank::new(7, 1);
        let block_hash = ExternalSequenceBlockHash(41);

        for rank in [reset_rank, retained_rank] {
            indexer
                .try_apply_event(
                    store_event(
                        rank.worker_id,
                        rank.dp_rank,
                        1,
                        &[],
                        &[block_hash.0],
                        StorageTier::Device,
                    )
                    .with_session_id("session-1"),
                )
                .await
                .unwrap();
        }
        indexer.flush_session_updates().await.unwrap();

        indexer
            .reset_worker_dp_rank_and_wait(reset_rank.worker_id, reset_rank.dp_rank)
            .await
            .unwrap();

        assert!(
            session_prefix_index
                .get_session_block_lineage("session-1", reset_rank, None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            session_prefix_index
                .get_session_block_lineage("session-1", retained_rank, None)
                .unwrap(),
            vec![vec![block_hash]]
        );
    }
}
