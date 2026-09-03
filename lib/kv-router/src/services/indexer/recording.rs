// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Routing-decision recording: which physical index a booked routing decision
//! is written into, and the per-request approximate-LRU lease.

use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use dynamo_tokens::SequenceHash;

use crate::indexer::{
    ApproximateAcquireMode, ApproximateLruBlock, ApproximateLruIncarnation, ApproximateLruLease,
    ApproximateLruReleaseAck, KvIndexer, KvIndexerInterface, KvRouterError, RoutingDecisionHashes,
};
use crate::protocols::{LocalBlockHash, TokensWithHashes, WorkerWithDpRank};
use crate::scheduling::AttemptId;
use crate::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};

use super::backend::{Indexer, RemotePrimary, SideIndexer};

const MODE_INACTIVE: u8 = 0;
const MODE_LRU: u8 = 1;
const MODE_TTL_FALLBACK: u8 = 2;

/// One request's approximate-LRU lease on its primary indexer.
#[derive(Clone)]
pub struct ApproximateRequestLease {
    lease: ApproximateLruLease,
    mode: Arc<AtomicU8>,
}

impl ApproximateRequestLease {
    pub async fn acquire(
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
                ApproximateAcquireMode::Lru => MODE_LRU,
                ApproximateAcquireMode::TtlFallback => MODE_TTL_FALLBACK,
                ApproximateAcquireMode::Ignored => MODE_INACTIVE,
            },
            Ordering::Release,
        );
        Ok(mode)
    }

    pub fn materialize(
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

    pub fn begin_finish(&self) -> Result<Option<ApproximateLruReleaseAck>, KvRouterError> {
        self.lease.begin_finish()
    }

    pub fn release_now(&self) {
        self.lease.release_now();
    }

    pub fn is_active_lru(&self) -> bool {
        self.mode.load(Ordering::Acquire) == MODE_LRU
    }
}

#[derive(Clone, Copy)]
enum RouteRecordingTarget<'a> {
    Disabled,
    PrimaryLocal(&'a KvIndexer),
    PrimaryConcurrent(&'a ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
    PrimaryRemote(&'a dyn RemotePrimary),
    SideOverlay(&'a SideIndexer),
}

impl Indexer {
    pub fn begin_approximate_lru_request(
        &self,
        worker: WorkerWithDpRank,
        incarnation: ApproximateLruIncarnation,
        attempt_id: AttemptId,
    ) -> Option<ApproximateRequestLease> {
        let lease = match self {
            Self::Single { primary, .. } => {
                primary.begin_approximate_lru_request(worker, incarnation, attempt_id)?
            }
            Self::Concurrent { primary, .. } => {
                primary.begin_approximate_lru_request(worker, incarnation, attempt_id)?
            }
            Self::Remote { .. } | Self::None => return None,
        };
        Some(ApproximateRequestLease {
            lease,
            mode: Arc::new(AtomicU8::new(MODE_INACTIVE)),
        })
    }

    /// Whether booked routing decisions should be written into this indexer.
    pub fn records_routing_decisions(&self) -> bool {
        !matches!(self.recording_target(), RouteRecordingTarget::Disabled)
    }

    fn recording_target(&self) -> RouteRecordingTarget<'_> {
        match self {
            Self::Single {
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
            Self::Single {
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
            Self::Single { .. } | Self::Concurrent { .. } | Self::Remote { .. } | Self::None => {
                RouteRecordingTarget::Disabled
            }
        }
    }

    /// Record a booked routing decision. Writes to the side indexer when one
    /// is attached, else to an approximate primary, else is a no-op.
    pub async fn record_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        self.recording_target()
            .record_routing_hashes(worker, hashes)
            .await
    }

    /// Same as [`Self::record_routing_decision`]; kept for callers that name
    /// the hashes explicitly.
    pub async fn record_routing_decision_hashes(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        self.record_routing_decision(worker, hashes).await
    }

    pub async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.record_routing_decision(
            worker,
            RoutingDecisionHashes {
                local_hashes,
                sequence_hashes,
            },
        )
        .await
    }

    pub async fn process_routing_decision_for_request(
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
                .record_routing_decision(worker, hashes)
                .await
                .map_err(|error| {
                    tracing::warn!(error = %error, "Remote indexer write failed");
                    KvRouterError::IndexerDroppedRequest
                }),
            Self::SideOverlay(side) => side.record_routing_decision(worker, hashes).await,
        }
    }
}
