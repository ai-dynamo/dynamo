// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{DpRank, RouterEvent, WorkerId};
use std::future::Future;

use crate::kv_router::Indexer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryResetReason {
    Lifecycle,
    TreeDumpFailed,
    TargetFault,
}

/// Generation-scoped fence for commands targeting one `(worker, dp_rank)` source.
///
/// The membership coordinator owns this value. Recovery targets must carry it through
/// admission so an accepted command or delayed fault from an old source cannot affect a
/// replacement source for the same logical rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SourceEpoch(u64);

impl SourceEpoch {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Destination semantics required by worker-local KV recovery.
///
/// Ordinary events complete when the destination queue accepts them. Exact-rank
/// reset and replacement are completion barriers. Targets never provide recovery
/// state themselves; the source remains the worker's exact local indexer.
pub(crate) trait RecoveryTarget: Send + Sync + 'static {
    fn admit_event(
        &self,
        source_epoch: SourceEpoch,
        event: RouterEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn reset_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;
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
        _source_epoch: SourceEpoch,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        self.indexer
            .try_apply_event(event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        self.reset_rank(
            source_epoch,
            worker_id,
            dp_rank,
            RecoveryResetReason::Lifecycle,
        )
        .await?;
        for event in events {
            self.admit_event(source_epoch, event).await?;
        }
        Ok(())
    }

    async fn reset_rank(
        &self,
        _source_epoch: SourceEpoch,
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
