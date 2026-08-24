// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    indexer::{
        ApproximateAcquireMode, ApproximateLruBlock, ApproximateLruIncarnation,
        ApproximateLruLease, ApproximateLruReleaseAck, KvIndexer, KvIndexerInterface,
        KvRouterError, RoutingDecisionHashes, ThreadPoolIndexer,
    },
    protocols::{LocalBlockHash, TokensWithHashes, WorkerWithDpRank},
    scheduling::AttemptId,
};
use dynamo_tokens::SequenceHash;
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use super::{Indexer, SideIndexer, remote::RemoteIndexer};

#[derive(Clone)]
pub(crate) struct ApproximateRequestLease {
    lease: ApproximateLruLease,
    mode: Arc<AtomicU8>,
}

impl ApproximateRequestLease {
    pub(crate) async fn acquire(
        &mut self,
        hashes: RoutingDecisionHashes,
        private_blocks: usize,
    ) -> Result<ApproximateAcquireMode, KvRouterError> {
        let blocks = hashes
            .local_hashes
            .iter()
            .zip(&hashes.sequence_hashes)
            .map(|(&local_hash, &sequence_hash)| ApproximateLruBlock {
                local_hash,
                sequence_hash,
            })
            .collect();
        let mode = self.lease.acquire(blocks, private_blocks).await?;
        self.mode.store(
            match mode {
                ApproximateAcquireMode::Lru => 1,
                ApproximateAcquireMode::TtlFallback => 2,
                ApproximateAcquireMode::Ignored => 0,
            },
            Ordering::Release,
        );
        if mode == ApproximateAcquireMode::Lru {
            return Ok(mode);
        }
        if mode == ApproximateAcquireMode::Ignored {
            return Ok(mode);
        }
        Ok(mode)
    }

    pub(crate) fn materialize(
        &self,
        parent_hash: Option<SequenceHash>,
        blocks: Vec<ApproximateLruBlock>,
        start_position: usize,
        private_blocks: usize,
    ) -> Result<(), KvRouterError> {
        if !self.is_active_lru() {
            return Ok(());
        }
        self.lease
            .materialize(parent_hash, blocks, start_position, private_blocks)
    }

    pub(crate) fn begin_finish(&self) -> Result<Option<ApproximateLruReleaseAck>, KvRouterError> {
        self.lease.begin_finish()
    }

    pub(crate) fn release_now(&self) {
        self.lease.release_now();
    }

    pub(crate) fn is_active_lru(&self) -> bool {
        self.mode.load(Ordering::Acquire) == 1
    }
}

#[derive(Clone, Copy)]
pub(super) enum RouteRecordingTarget<'a> {
    Disabled,
    PrimaryLocal(&'a KvIndexer),
    PrimaryConcurrent(&'a ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
    PrimaryRemote(&'a RemoteIndexer),
    SideOverlay(&'a SideIndexer),
}

impl Indexer {
    pub(crate) fn begin_approximate_lru_request(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
    ) -> Option<ApproximateRequestLease> {
        let lease = match self {
            Self::KvIndexer { primary, .. } => {
                primary.begin_approximate_lru_request(worker, incarnation, attempt_id)?
            }
            Self::Concurrent { primary, .. } => {
                primary.begin_approximate_lru_request(worker, incarnation, attempt_id)?
            }
            Self::Remote { .. } | Self::None => return None,
        };
        Some(ApproximateRequestLease {
            lease,
            mode: Arc::new(AtomicU8::new(0)),
        })
    }

    pub(crate) fn records_routing_decisions(&self) -> bool {
        !matches!(self.recording_target(), RouteRecordingTarget::Disabled)
    }

    pub(super) fn recording_target(&self) -> RouteRecordingTarget<'_> {
        match self {
            Self::KvIndexer {
                approx: Some(side), ..
            }
            | Self::Concurrent {
                approx: Some(side), ..
            } => RouteRecordingTarget::SideOverlay(side),
            Self::Remote {
                primary,
                approx: Some(side),
                ..
            } => {
                debug_assert!(
                    primary.use_kv_events(),
                    "remote side indexer requires an event-driven primary"
                );
                RouteRecordingTarget::SideOverlay(side)
            }
            Self::KvIndexer {
                primary,
                primary_records_routing_decisions: true,
                ..
            } => RouteRecordingTarget::PrimaryLocal(primary),
            Self::Concurrent {
                primary,
                primary_records_routing_decisions: true,
                ..
            } => RouteRecordingTarget::PrimaryConcurrent(primary.as_ref()),
            Self::Remote {
                primary,
                primary_records_routing_decisions: true,
                ..
            } => RouteRecordingTarget::PrimaryRemote(primary.as_ref()),
            Self::KvIndexer { .. } | Self::Concurrent { .. } | Self::Remote { .. } | Self::None => {
                RouteRecordingTarget::Disabled
            }
        }
    }

    pub(crate) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.recording_target()
            .record_routing_hashes(
                worker,
                RoutingDecisionHashes {
                    local_hashes,
                    sequence_hashes,
                },
            )
            .await
    }

    pub(crate) async fn record_routing_decision_hashes(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        self.recording_target()
            .record_routing_hashes(worker, hashes)
            .await
    }

    pub(crate) async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let target = self.recording_target();
        if let RouteRecordingTarget::PrimaryConcurrent(primary) = target {
            return primary
                .process_routing_decision_for_request(tokens_with_hashes, worker)
                .await;
        }
        if matches!(target, RouteRecordingTarget::Disabled) {
            return Ok(());
        }

        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
        target
            .record_routing_hashes(
                worker,
                RoutingDecisionHashes {
                    local_hashes,
                    sequence_hashes,
                },
            )
            .await
    }
}

impl<'a> RouteRecordingTarget<'a> {
    async fn record_routing_hashes(
        self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::Disabled => Ok(()),
            Self::PrimaryLocal(primary) => {
                primary
                    .process_routing_decision_with_hashes(
                        worker,
                        hashes.local_hashes,
                        hashes.sequence_hashes,
                    )
                    .await
            }
            Self::PrimaryConcurrent(primary) => {
                primary
                    .process_routing_decision_hash_slices(
                        worker,
                        &hashes.local_hashes,
                        &hashes.sequence_hashes,
                    )
                    .await
            }
            Self::PrimaryRemote(primary) => primary
                .record_hashed_routing_decision(worker, hashes.local_hashes, hashes.sequence_hashes)
                .await
                .map_err(|error| {
                    tracing::warn!(error = %error, "Remote indexer write failed");
                    KvRouterError::IndexerDroppedRequest
                }),
            Self::SideOverlay(side) => side.process_routing_decision_hashes(worker, hashes).await,
        }
    }
}
