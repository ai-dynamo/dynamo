// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_kv_router::protocols::{DpRank, RouterEvent, WorkerId};

use crate::kv_router::Indexer;

/// Destination semantics required by worker-local KV recovery.
///
/// The source remains a worker's exact `LocalKvIndexer`; a target only applies
/// live/ring events and rank lifecycle or replacement operations. Targets do not
/// expose event dumps of their own.
#[async_trait]
pub(crate) trait RecoveryTarget: Send + Sync {
    async fn apply_event(&self, event: RouterEvent) -> anyhow::Result<()>;

    async fn replace_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()>;

    async fn remove_rank(&self, worker_id: WorkerId, dp_rank: DpRank) -> anyhow::Result<()>;

    async fn degraded_reset_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> anyhow::Result<()> {
        self.remove_rank(worker_id, dp_rank).await
    }

    async fn remove_worker(&self, worker_id: WorkerId) -> anyhow::Result<()>;
}

pub(crate) struct IndexerRecoveryTarget {
    indexer: Indexer,
}

impl IndexerRecoveryTarget {
    pub(crate) fn new(indexer: Indexer) -> Self {
        Self { indexer }
    }
}

#[async_trait]
impl RecoveryTarget for IndexerRecoveryTarget {
    async fn apply_event(&self, event: RouterEvent) -> anyhow::Result<()> {
        self.indexer.apply_event(event).await;
        Ok(())
    }

    async fn replace_rank(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        // Preserve the existing router indexer's replacement behavior. Targets
        // with an exact aggregate (such as the DC Relay) override this as one
        // transactional actor command.
        self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
        for event in events {
            self.indexer.apply_event(event).await;
        }
        Ok(())
    }

    async fn remove_rank(&self, worker_id: WorkerId, dp_rank: DpRank) -> anyhow::Result<()> {
        self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
        Ok(())
    }

    async fn remove_worker(&self, worker_id: WorkerId) -> anyhow::Result<()> {
        self.indexer.remove_worker(worker_id).await;
        Ok(())
    }
}
