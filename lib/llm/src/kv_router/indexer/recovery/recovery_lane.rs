// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{hash::Hash, sync::Arc};

use dashmap::DashMap;
use tokio::{sync::Semaphore, task::JoinHandle};
use tokio_util::sync::CancellationToken;

pub(super) const RECOVERY_CONCURRENCY_LIMIT: usize = 16;

struct RecoveryTask {
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

/// Per-source recovery task ownership plus the shared RPC concurrency budget.
///
/// Source-specific identity checks, result application, and live-tail policy stay
/// in their respective coordinators.
pub(super) struct RecoveryLane<K> {
    tasks: DashMap<K, RecoveryTask>,
    semaphore: Arc<Semaphore>,
}

impl<K> RecoveryLane<K>
where
    K: Copy + Eq + Hash,
{
    pub(super) fn new() -> Self {
        Self::with_semaphore(Arc::new(Semaphore::new(RECOVERY_CONCURRENCY_LIMIT)))
    }

    pub(super) fn with_semaphore(semaphore: Arc<Semaphore>) -> Self {
        Self {
            tasks: DashMap::new(),
            semaphore,
        }
    }

    pub(super) fn semaphore(&self) -> Arc<Semaphore> {
        self.semaphore.clone()
    }

    pub(super) fn insert(&self, key: K, cancel: CancellationToken, handle: JoinHandle<()>) {
        self.tasks.insert(key, RecoveryTask { cancel, handle });
    }

    pub(super) fn finish(&self, key: K) {
        let _ = self.tasks.remove(&key);
    }

    pub(super) async fn cancel(&self, key: K) -> Option<tokio::task::JoinError> {
        let (_, task) = self.tasks.remove(&key)?;
        task.cancel.cancel();
        task.handle.abort();
        task.handle.await.err()
    }

    pub(super) async fn cancel_all(&self) {
        let keys: Vec<_> = self.tasks.iter().map(|entry| *entry.key()).collect();
        for key in keys {
            let _ = self.cancel(key).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finishing_stale_generation_keeps_replacement_task() {
        let lane = RecoveryLane::new();
        lane.insert(
            (7u64, 1u64),
            CancellationToken::new(),
            tokio::spawn(async {}),
        );
        let replacement_cancel = CancellationToken::new();
        lane.insert(
            (7, 2),
            replacement_cancel.clone(),
            tokio::spawn(std::future::pending::<()>()),
        );

        lane.finish((7, 1));

        assert!(lane.tasks.contains_key(&(7, 2)));
        lane.cancel_all().await;
        assert!(replacement_cancel.is_cancelled());
    }
}
