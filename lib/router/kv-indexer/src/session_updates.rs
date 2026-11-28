// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::prelude::{
    ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, ResidencyOwnerKeyExt, RouterEvent, StorageTier, WorkerId,
    WorkerWithDpRank,
};
use crate::{
    ConcurrentRadixTreeCompressed, KvIndexer, KvRouterError, LowerTierIndexers,
    SessionPrefixIndexer, SessionResidencyOwner, ThreadPoolIndexer,
};
use dashmap::{DashMap, mapref::entry::Entry};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone)]
pub struct SessionUpdateSender {
    tx: mpsc::UnboundedSender<SessionUpdateMessage>,
    residency_versions: Arc<ResidencyVersions>,
}

#[derive(Default)]
struct ResidencyVersions {
    next: AtomicU64,
    by_rank: DashMap<WorkerWithDpRank, AtomicU64>,
    lifecycle_by_rank: DashMap<WorkerWithDpRank, AtomicU64>,
    lifecycle_by_worker_id: DashMap<WorkerId, AtomicU64>,
    lifecycle_by_cache_owner: DashMap<ResidencyOwnerKey, AtomicU64>,
}

impl ResidencyVersions {
    fn current(&self, worker: WorkerWithDpRank) -> u64 {
        let rank_version = self
            .by_rank
            .get(&worker)
            .map_or(0, |version| version.load(Ordering::Acquire));
        let worker_version = self
            .lifecycle_by_worker_id
            .get(&worker.worker_id)
            .map_or(0, |version| version.load(Ordering::Acquire));
        rank_version.max(worker_version)
    }

    fn advance(&self, worker: WorkerWithDpRank) {
        let next_version = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        match self.by_rank.entry(worker) {
            Entry::Occupied(entry) => {
                entry.get().store(next_version, Ordering::Release);
            }
            Entry::Vacant(entry) => {
                entry.insert(AtomicU64::new(next_version));
            }
        }
    }

    fn current_lifecycle(&self, worker: WorkerWithDpRank) -> u64 {
        let rank_version = self
            .lifecycle_by_rank
            .get(&worker)
            .map_or(0, |version| version.load(Ordering::Acquire));
        let worker_version = self
            .lifecycle_by_worker_id
            .get(&worker.worker_id)
            .map_or(0, |version| version.load(Ordering::Acquire));
        rank_version.max(worker_version)
    }

    fn advance_rank_lifecycle(&self, worker: WorkerWithDpRank) {
        let next_version = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        match self.lifecycle_by_rank.entry(worker) {
            Entry::Occupied(entry) => {
                entry.get().store(next_version, Ordering::Release);
            }
            Entry::Vacant(entry) => {
                entry.insert(AtomicU64::new(next_version));
            }
        }
    }

    fn advance_worker_lifecycle(&self, worker_id: WorkerId) {
        let next_version = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        match self.lifecycle_by_worker_id.entry(worker_id) {
            Entry::Occupied(entry) => {
                entry.get().store(next_version, Ordering::Release);
            }
            Entry::Vacant(entry) => {
                entry.insert(AtomicU64::new(next_version));
            }
        }
    }

    /// Lifecycle of the owner a store names. A cache owner outlives its worker, so only
    /// its own clear advances it.
    fn lifecycle_of(&self, owner: SessionResidencyOwner) -> u64 {
        match owner {
            SessionResidencyOwner::Worker(worker) => self.current_lifecycle(worker),
            SessionResidencyOwner::CacheOwner(owner) => self
                .lifecycle_by_cache_owner
                .get(&owner)
                .map_or(0, |version| version.load(Ordering::Acquire)),
            SessionResidencyOwner::SharedCache(_) => 0,
        }
    }

    fn advance_cache_owner_lifecycle(&self, owner: ResidencyOwnerKey) {
        let next_version = self.next.fetch_add(1, Ordering::AcqRel) + 1;
        self.lifecycle_by_cache_owner
            .entry(owner)
            .or_default()
            .store(next_version, Ordering::Release);
    }
}

enum SessionUpdateMessage {
    Mutation(SessionMutation),
    Flush(oneshot::Sender<()>),
}

pub(super) enum SessionMutation {
    Matched {
        worker: WorkerWithDpRank,
        session_id: String,
        matched_hash: ExternalSequenceBlockHash,
        residency_version: u64,
    },
    /// A routed request's chain, matched against the cache owner projected to the selected
    /// worker. The owner's depth is read when the update applies, behind the lower-tier
    /// barrier, so it already reflects every removal queued before it.
    CacheOwnerMatched {
        owner: ResidencyOwnerKey,
        session_id: String,
        block_hashes: Vec<LocalBlockHash>,
    },
    Stored {
        owner: SessionResidencyOwner,
        tier: StorageTier,
        session_id: String,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: Vec<ExternalSequenceBlockHash>,
        lifecycle_version: u64,
    },
    /// A tier stored blocks without naming a session, such as a device offload to host
    /// memory, a host load-back to the device, or a cache owner's store.
    Inherited {
        owner: SessionResidencyOwner,
        tier: StorageTier,
        block_hashes: Vec<ExternalSequenceBlockHash>,
        lifecycle_version: u64,
    },
    Removed {
        owner: SessionResidencyOwner,
        tier: StorageTier,
        block_hashes: Vec<ExternalSequenceBlockHash>,
    },
    /// A reset of a worker rank, of a cache owner, or of both.
    Cleared {
        worker: Option<WorkerWithDpRank>,
        cache_owner: Option<ResidencyOwnerKey>,
    },
    WorkerRemoved {
        worker_id: WorkerId,
    },
}

impl SessionMutation {
    /// Session frontiers follow device events, worker-owned lower tiers, and cache-owner
    /// tiers. A cache-owner event names its owner in `state_source`; its worker id is not
    /// validated, so it is never read as the owner.
    fn from_event(event: &RouterEvent, residency_versions: &ResidencyVersions) -> Option<Self> {
        let tier = event.storage_tier;
        let owner = || {
            Some(match event.residency_owner().ok()? {
                ResidencyOwner::Worker(worker) => SessionResidencyOwner::Worker(worker),
                owner @ ResidencyOwner::CacheOwner(_) => {
                    SessionResidencyOwner::CacheOwner(owner.compact_key())
                }
            })
        };
        match &event.event.data {
            KvCacheEventData::Stored(stored) => {
                let owner = owner()?;
                let block_hashes = stored.blocks.iter().map(|block| block.block_hash).collect();
                let lifecycle_version = residency_versions.lifecycle_of(owner);
                match event.session_id.clone() {
                    Some(session_id) => Some(Self::Stored {
                        owner,
                        tier,
                        session_id,
                        parent_hash: stored.parent_hash,
                        block_hashes,
                        lifecycle_version,
                    }),
                    None => Some(Self::Inherited {
                        owner,
                        tier,
                        block_hashes,
                        lifecycle_version,
                    }),
                }
            }
            KvCacheEventData::Removed(removed) => Some(Self::Removed {
                owner: owner()?,
                tier,
                block_hashes: removed.block_hashes.clone(),
            }),
            KvCacheEventData::Cleared => {
                let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                let cache_owner = event
                    .state_source
                    .map(|source| ResidencyOwner::cache_owner(source).compact_key());
                // The same scopes the lower-tier index clears.
                match event.reset_scope().ok()?? {
                    ResetScope::All => Some(Self::Cleared {
                        worker: Some(worker),
                        cache_owner,
                    }),
                    ResetScope::Domain(ResidencyDomain::Worker) => Some(Self::Cleared {
                        worker: Some(worker),
                        cache_owner: None,
                    }),
                    ResetScope::Domain(ResidencyDomain::CacheOwner) => Some(Self::Cleared {
                        worker: None,
                        cache_owner: Some(cache_owner?),
                    }),
                }
            }
        }
    }

    /// The worker rank whose index lanes this mutation follows.
    pub(super) fn worker(&self) -> Option<WorkerWithDpRank> {
        match self {
            Self::Matched { worker, .. } => Some(*worker),
            Self::Stored { owner, .. }
            | Self::Inherited { owner, .. }
            | Self::Removed { owner, .. } => owner.as_worker(),
            Self::Cleared { worker, .. } => *worker,
            Self::CacheOwnerMatched { .. } | Self::WorkerRemoved { .. } => None,
        }
    }

    pub(super) fn tier(&self) -> Option<StorageTier> {
        match self {
            Self::Stored { tier, .. }
            | Self::Inherited { tier, .. }
            | Self::Removed { tier, .. } => Some(*tier),
            Self::Matched { .. }
            | Self::CacheOwnerMatched { .. }
            | Self::Cleared { .. }
            | Self::WorkerRemoved { .. } => None,
        }
    }

    /// True when the lower-tier work this mutation follows runs on a cache owner's lane.
    /// An owner clear is not one: its lower-tier work is awaited before it is queued.
    fn follows_cache_owner_lane(&self) -> bool {
        match self {
            Self::CacheOwnerMatched { .. } => true,
            Self::Stored { owner, .. }
            | Self::Inherited { owner, .. }
            | Self::Removed { owner, .. } => {
                matches!(owner, SessionResidencyOwner::CacheOwner(_))
            }
            _ => false,
        }
    }

    async fn apply(
        self,
        index: &SessionPrefixIndexer,
        residency_versions: &ResidencyVersions,
        barrier: &ResidencyBarrier,
    ) {
        match self {
            Self::Matched {
                worker,
                session_id,
                matched_hash,
                residency_version,
            } => {
                let is_resident = if residency_versions.current(worker) == residency_version {
                    true
                } else {
                    match barrier
                        .primary
                        .contains_worker_block(worker, matched_hash)
                        .await
                    {
                        Ok(is_resident) => is_resident,
                        Err(error) => {
                            tracing::warn!(%error, %session_id, ?worker, "failed to revalidate session prefix match");
                            false
                        }
                    }
                };
                if !is_resident {
                    return;
                }
                if let Err(error) =
                    index.update_session_from_match(&session_id, worker, matched_hash)
                {
                    tracing::warn!(%error, %session_id, ?worker, "failed to record session prefix match");
                }
            }
            Self::CacheOwnerMatched {
                owner,
                session_id,
                block_hashes,
            } => {
                for (tier, lower_tier) in barrier.lower_tier.entries() {
                    let Some(matched_hash) = lower_tier
                        .backend()
                        .exact_owner_prefix_end(&block_hashes, owner)
                    else {
                        continue;
                    };
                    if let Err(error) = index.update_session_from_match_in_tier(
                        &session_id,
                        SessionResidencyOwner::CacheOwner(owner),
                        tier,
                        matched_hash,
                    ) {
                        tracing::warn!(%error, %session_id, ?owner, ?tier, "failed to record session cache-owner match");
                    }
                }
            }
            Self::Stored {
                owner,
                tier,
                session_id,
                parent_hash,
                block_hashes,
                lifecycle_version,
            } => {
                if residency_versions.lifecycle_of(owner) != lifecycle_version {
                    return;
                }
                if let Err(error) = index.update_session_from_stored_blocks_in_tier(
                    &session_id,
                    owner,
                    tier,
                    parent_hash,
                    &block_hashes,
                ) {
                    tracing::warn!(%error, %session_id, ?owner, ?tier, "failed to record stored session blocks");
                }
            }
            Self::Inherited {
                owner,
                tier,
                block_hashes,
                lifecycle_version,
            } => {
                if residency_versions.lifecycle_of(owner) != lifecycle_version {
                    return;
                }
                // A cache owner's stores name no session: the frontiers of the worker it
                // is projected to say whose blocks they are.
                let projected = match owner {
                    SessionResidencyOwner::CacheOwner(owner) => barrier
                        .lower_tier
                        .projected_worker(owner)
                        .map(SessionResidencyOwner::Worker),
                    _ => None,
                };
                index.inherit_stored_blocks(owner, tier, projected, &block_hashes);
            }
            Self::Removed {
                owner,
                tier,
                block_hashes,
            } => {
                // Versions guard route-time device matches only.
                if tier.is_gpu()
                    && let Some(worker) = owner.as_worker()
                {
                    residency_versions.advance(worker);
                }
                index.update_session_from_removed_blocks_in_tier(owner, tier, &block_hashes);
            }
            Self::Cleared {
                worker,
                cache_owner,
            } => {
                if let Some(worker) = worker {
                    residency_versions.advance(worker);
                    index.clear_owner_frontiers(SessionResidencyOwner::Worker(worker));
                }
                if let Some(owner) = cache_owner {
                    index.clear_owner_frontiers(SessionResidencyOwner::CacheOwner(owner));
                }
            }
            Self::WorkerRemoved { worker_id } => {
                index.clear_worker_id_frontiers(worker_id);
            }
        }
    }
}

enum PrimaryBarrier {
    Legacy(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

impl PrimaryBarrier {
    async fn wait_for(&self, mutations: &[SessionMutation]) -> Result<(), KvRouterError> {
        match self {
            Self::Legacy(primary) => {
                primary.flush_and_wait().await?;
            }
            Self::Concurrent(primary) => {
                let workers: HashSet<_> = mutations
                    .iter()
                    .filter_map(SessionMutation::worker)
                    .collect();
                for worker in workers {
                    primary.flush_worker_lane_and_wait(worker).await?;
                }
            }
        }
        Ok(())
    }

    async fn contains_worker_block(
        &self,
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
    ) -> Result<bool, KvRouterError> {
        match self {
            Self::Legacy(primary) => primary.contains_worker_block(worker, block_hash).await,
            Self::Concurrent(primary) => primary.contains_worker_block(worker, block_hash).await,
        }
    }
}

/// Orders session mutations after the physical indexes they describe: the primary
/// for every worker in a batch, the lower-tier lane for each `(worker, tier)` a
/// lower-tier mutation names, and every lower-tier lane when one names a cache owner,
/// whose lane is keyed by the owner and not by a worker.
struct ResidencyBarrier {
    primary: PrimaryBarrier,
    lower_tier: LowerTierIndexers,
}

impl ResidencyBarrier {
    async fn wait_for(&self, mutations: &[SessionMutation]) -> Result<(), KvRouterError> {
        self.primary.wait_for(mutations).await?;
        let lanes: HashSet<(WorkerWithDpRank, StorageTier)> = mutations
            .iter()
            .filter_map(|mutation| Some((mutation.worker()?, mutation.tier()?)))
            .filter(|(_, tier)| !tier.is_gpu())
            .collect();
        for (worker, tier) in lanes {
            if let Some(indexer) = self.lower_tier.get(tier) {
                indexer.flush_worker_lane_and_wait(worker).await?;
            }
        }
        if mutations
            .iter()
            .any(SessionMutation::follows_cache_owner_lane)
        {
            for indexer in self.lower_tier.all() {
                indexer.flush_and_wait().await?;
            }
        }
        Ok(())
    }
}

impl SessionUpdateSender {
    pub fn for_legacy(
        index: Arc<SessionPrefixIndexer>,
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
    ) -> Self {
        Self::spawn(index, PrimaryBarrier::Legacy(primary), lower_tier)
    }

    pub fn for_concurrent(
        index: Arc<SessionPrefixIndexer>,
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
    ) -> Self {
        Self::spawn(index, PrimaryBarrier::Concurrent(primary), lower_tier)
    }

    fn spawn(
        index: Arc<SessionPrefixIndexer>,
        primary: PrimaryBarrier,
        lower_tier: LowerTierIndexers,
    ) -> Self {
        let barrier = ResidencyBarrier {
            primary,
            lower_tier,
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let residency_versions = Arc::new(ResidencyVersions::default());
        let task_residency_versions = Arc::clone(&residency_versions);
        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                // Give event ingestion one turn to enqueue the rest of its current batch.
                tokio::task::yield_now().await;
                let mut mutations = Vec::new();
                let mut flushes = Vec::new();
                match first {
                    SessionUpdateMessage::Mutation(mutation) => mutations.push(mutation),
                    SessionUpdateMessage::Flush(flush) => flushes.push(flush),
                }
                while let Ok(message) = rx.try_recv() {
                    match message {
                        SessionUpdateMessage::Mutation(mutation) => mutations.push(mutation),
                        SessionUpdateMessage::Flush(flush) => flushes.push(flush),
                    }
                }

                if !mutations.is_empty() {
                    match barrier.wait_for(&mutations).await {
                        Ok(()) => {
                            for mutation in mutations {
                                mutation
                                    .apply(&index, &task_residency_versions, &barrier)
                                    .await;
                            }
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to order session updates after KV residency updates");
                        }
                    }
                }
                for flush in flushes {
                    let _ = flush.send(());
                }
            }
        });
        Self {
            tx,
            residency_versions,
        }
    }

    pub(super) fn residency_version(&self, worker: WorkerWithDpRank) -> u64 {
        self.residency_versions.current(worker)
    }

    pub(super) fn mutation_from_event(&self, event: &RouterEvent) -> Option<SessionMutation> {
        SessionMutation::from_event(event, &self.residency_versions)
    }

    pub(super) fn enqueue(&self, mutation: SessionMutation) -> Result<(), KvRouterError> {
        // Lifecycle generations advance in ingestion order, so a store ingested after a
        // clear carries the new generation even before the clear itself is applied.
        match &mutation {
            SessionMutation::Cleared {
                worker,
                cache_owner,
            } => {
                if let Some(worker) = worker {
                    self.residency_versions.advance_rank_lifecycle(*worker);
                }
                if let Some(owner) = cache_owner {
                    self.residency_versions
                        .advance_cache_owner_lifecycle(*owner);
                }
            }
            SessionMutation::WorkerRemoved { worker_id } => {
                self.residency_versions.advance_worker_lifecycle(*worker_id);
            }
            _ => {}
        }
        self.tx
            .send(SessionUpdateMessage::Mutation(mutation))
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    pub(super) async fn flush(&self) -> Result<(), KvRouterError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(SessionUpdateMessage::Flush(tx))
            .map_err(|_| KvRouterError::IndexerOffline)?;
        rx.await.map_err(|_| KvRouterError::IndexerDroppedRequest)
    }
}
