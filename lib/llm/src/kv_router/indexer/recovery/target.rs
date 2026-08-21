// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{DpRank, RouterEvent, WorkerId};
use std::future::Future;

use crate::discovery::PublisherId;
use crate::kv_router::Indexer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryResetReason {
    Lifecycle,
    TreeDumpFailed,
    TargetFault,
}

/// Destination semantics required by worker-local KV recovery.
///
/// Ordinary events complete when the destination queue accepts them. Exact-rank
/// reset and replacement are completion barriers. Targets never provide recovery
/// state themselves; the source remains the worker's exact local indexer.
pub(crate) trait RecoveryTarget: Send + Sync + 'static {
    fn admit_event(
        &self,
        publisher_id: PublisherId,
        event: RouterEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn replace_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn reset_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Mark a source's initial recovery attempt complete when it did not install a tree dump.
    /// Targets that batch cold-start replacements use this to close the initial recovery wave.
    fn complete_initial_recovery(
        &self,
        _worker_id: WorkerId,
        _dp_rank: DpRank,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

#[derive(Clone)]
pub(crate) struct IndexerRecoveryTarget {
    indexer: Indexer,
}

impl IndexerRecoveryTarget {
    pub(crate) fn new(indexer: Indexer) -> Self {
        Self { indexer }
    }
}

impl RecoveryTarget for IndexerRecoveryTarget {
    async fn admit_event(
        &self,
        _publisher_id: PublisherId,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        self.indexer
            .try_apply_event(event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        _publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        self.indexer
            .replace_worker_dp_rank_and_wait(worker_id, dp_rank, events)
            .await
            .map_err(Into::into)
    }

    async fn reset_rank(
        &self,
        _publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        _reason: RecoveryResetReason,
    ) -> anyhow::Result<()> {
        self.indexer
            .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
            .await
            .map_err(Into::into)
    }
}
