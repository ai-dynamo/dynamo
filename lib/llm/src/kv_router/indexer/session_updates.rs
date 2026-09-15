// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, SessionPrefixIndexer,
    indexer::{KvIndexer, KvRouterError, ThreadPoolIndexer},
    protocols::{ExternalSequenceBlockHash, KvCacheEventData, RouterEvent, WorkerWithDpRank},
};
use tokio::sync::mpsc;
#[cfg(test)]
use tokio::sync::oneshot;

#[derive(Clone)]
pub struct SessionUpdateSender {
    tx: mpsc::UnboundedSender<SessionUpdateMessage>,
}

enum SessionUpdateMessage {
    Mutation(SessionMutation),
    #[cfg(test)]
    Flush(oneshot::Sender<()>),
}

pub(super) enum SessionMutation {
    Matched {
        worker: WorkerWithDpRank,
        session_id: String,
        matched_hash: ExternalSequenceBlockHash,
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

    fn worker(&self) -> WorkerWithDpRank {
        match self {
            Self::Matched { worker, .. }
            | Self::Stored { worker, .. }
            | Self::Removed { worker, .. }
            | Self::Cleared { worker } => *worker,
        }
    }

    fn apply(self, index: &SessionPrefixIndexer) {
        match self {
            Self::Matched {
                worker,
                session_id,
                matched_hash,
            } => {
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
                index.update_session_from_removed_blocks(worker, &block_hashes);
            }
            Self::Cleared { worker } => {
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
        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                // Give event ingestion one turn to enqueue the rest of its current batch.
                tokio::task::yield_now().await;
                let mut mutations = Vec::new();
                #[cfg(test)]
                let mut flushes = Vec::new();
                match first {
                    SessionUpdateMessage::Mutation(mutation) => mutations.push(mutation),
                    #[cfg(test)]
                    SessionUpdateMessage::Flush(flush) => flushes.push(flush),
                }
                while let Ok(message) = rx.try_recv() {
                    match message {
                        SessionUpdateMessage::Mutation(mutation) => mutations.push(mutation),
                        #[cfg(test)]
                        SessionUpdateMessage::Flush(flush) => flushes.push(flush),
                    }
                }

                if !mutations.is_empty() {
                    match barrier.wait_for(&mutations).await {
                        Ok(()) => {
                            for mutation in mutations {
                                mutation.apply(&index);
                            }
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to order session updates after KV residency updates");
                        }
                    }
                }
                #[cfg(test)]
                for flush in flushes {
                    let _ = flush.send(());
                }
            }
        });
        Self { tx }
    }

    pub(super) fn enqueue(&self, mutation: SessionMutation) -> Result<(), KvRouterError> {
        self.tx
            .send(SessionUpdateMessage::Mutation(mutation))
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    #[cfg(test)]
    pub(super) async fn flush(&self) -> Result<(), KvRouterError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(SessionUpdateMessage::Flush(tx))
            .map_err(|_| KvRouterError::IndexerOffline)?;
        rx.await.map_err(|_| KvRouterError::IndexerDroppedRequest)
    }
}
