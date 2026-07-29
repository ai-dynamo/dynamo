// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    indexer::{
        KvIndexer, KvIndexerInterface, KvRouterError, RoutingDecisionHashes, ThreadPoolIndexer,
    },
    protocols::{LocalBlockHash, TokensWithHashes, WorkerWithDpRank},
};
use dynamo_tokens::SequenceHash;

use super::{Indexer, SideIndexer, remote::RemoteIndexer};

#[derive(Clone, Copy)]
pub(super) enum RouteRecordingTarget<'a> {
    Disabled,
    PrimaryLocal(&'a KvIndexer),
    PrimaryConcurrent(&'a ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
    PrimaryRemote(&'a RemoteIndexer),
    SideOverlay(&'a SideIndexer),
}

impl Indexer {
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
