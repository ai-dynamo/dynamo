// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, SessionPrefixIndexer,
    indexer::{KvIndexer, KvRouterError, ThreadPoolIndexer},
    protocols::{ExternalSequenceBlockHash, KvCacheEventData, RouterEvent, WorkerWithDpRank},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone)]
pub struct SessionUpdateSender {
    tx: mpsc::UnboundedSender<SessionUpdateMessage>,
    residency_versions: Arc<ResidencyVersions>,
}

#[derive(Default)]
struct ResidencyVersions {
    by_worker: DashMap<WorkerWithDpRank, AtomicU64>,
}

impl ResidencyVersions {
    fn current(&self, worker: WorkerWithDpRank) -> u64 {
        self.by_worker
            .get(&worker)
            .map_or(0, |version| version.load(Ordering::Acquire))
    }

    fn advance(&self, worker: WorkerWithDpRank) {
        match self.by_worker.entry(worker) {
            Entry::Occupied(version) => {
                version.get().fetch_add(1, Ordering::AcqRel);
            }
            Entry::Vacant(entry) => {
                entry.insert(AtomicU64::new(1));
            }
        }
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
    Stored {
        worker: WorkerWithDpRank,
        session_id: String,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: Vec<ExternalSequenceBlockHash>,
    },
    Removed {
        worker: WorkerWithDpRank,
        block_hashes: Vec<ExternalSequenceBlockHash>,
    },
    Cleared {
        worker: WorkerWithDpRank,
    },
}

impl SessionMutation {
    pub(super) fn from_event(event: &RouterEvent) -> Option<Self> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        match &event.event.data {
            KvCacheEventData::Stored(stored) => Some(Self::Stored {
                worker,
                session_id: event.session_id.clone()?,
                parent_hash: stored.parent_hash,
                block_hashes: stored.blocks.iter().map(|block| block.block_hash).collect(),
            }),
            KvCacheEventData::Removed(removed) => Some(Self::Removed {
                worker,
                block_hashes: removed.block_hashes.clone(),
            }),
            KvCacheEventData::Cleared => Some(Self::Cleared { worker }),
        }
    }

    pub(super) fn worker(&self) -> WorkerWithDpRank {
        match self {
            Self::Matched { worker, .. }
            | Self::Stored { worker, .. }
            | Self::Removed { worker, .. }
            | Self::Cleared { worker } => *worker,
        }
    }

    async fn apply(
        self,
        index: &SessionPrefixIndexer,
        residency_versions: &ResidencyVersions,
        barrier: &PrimaryBarrier,
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
                    match barrier.contains_worker_block(worker, matched_hash).await {
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
            Self::Stored {
                worker,
                session_id,
                parent_hash,
                block_hashes,
            } => {
                if let Err(error) = index.update_session_from_stored_blocks(
                    &session_id,
                    worker,
                    parent_hash,
                    &block_hashes,
                ) {
                    tracing::warn!(%error, %session_id, ?worker, "failed to record stored session blocks");
                }
            }
            Self::Removed {
                worker,
                block_hashes,
            } => {
                residency_versions.advance(worker);
                index.update_session_from_removed_blocks(worker, &block_hashes);
            }
            Self::Cleared { worker } => {
                residency_versions.advance(worker);
                index.clear_worker_frontiers(worker);
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
                let workers: HashSet<_> = mutations.iter().map(SessionMutation::worker).collect();
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

impl SessionUpdateSender {
    pub(super) fn for_legacy(index: Arc<SessionPrefixIndexer>, primary: KvIndexer) -> Self {
        Self::spawn(index, PrimaryBarrier::Legacy(primary))
    }

    pub(super) fn for_concurrent(
        index: Arc<SessionPrefixIndexer>,
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
    ) -> Self {
        Self::spawn(index, PrimaryBarrier::Concurrent(primary))
    }

    fn spawn(index: Arc<SessionPrefixIndexer>, barrier: PrimaryBarrier) -> Self {
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

    pub(super) fn enqueue(&self, mutation: SessionMutation) -> Result<(), KvRouterError> {
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
