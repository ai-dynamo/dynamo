// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker membership for a selection core.
//!
//! A host owns discovery (runtime discovery, a Kubernetes pod reflector, a
//! workers file) and exposes it as a [`WorkerCatalogSource`]: a stream of
//! complete desired snapshots. [`CatalogReconciler`] turns each snapshot into
//! catalog upserts and deletes, retrying workers the core has not yet accepted
//! as schedulable and removing workers that left.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::core::SelectionCore;
use super::error::SelectionError;
use super::types::{WorkerCatalogRecord, WorkerLifecycle, WorkerRequest};
use crate::protocols::WorkerId;

/// Desired worker membership, delivered as complete snapshots rather than
/// deltas so a missed change can never leave a stale worker behind.
#[async_trait]
pub trait WorkerCatalogSource: Send {
    /// The next desired snapshot, or `None` once the source is closed. The
    /// first call returns the current membership without waiting.
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>>;
}

/// Notified after the reconciler applies a catalog change (metrics hooks).
pub trait CatalogObserver: Send + Sync {
    fn upserted(&self, record: &WorkerCatalogRecord);
    fn removed(&self, record: &WorkerCatalogRecord);
}

/// Applies desired snapshots to a core's worker catalog.
pub struct CatalogReconciler {
    core: Arc<SelectionCore>,
    observer: Option<Arc<dyn CatalogObserver>>,
    /// Requests the core reported `Schedulable`. An identical desired record
    /// skips its upsert; anything else is retried on every snapshot.
    converged: HashMap<WorkerId, WorkerRequest>,
    /// Every worker id this reconciler introduced, including ones that never
    /// became schedulable, so stale deletion covers partial upserts.
    tracked: HashSet<WorkerId>,
}

impl CatalogReconciler {
    pub fn new(core: Arc<SelectionCore>) -> Self {
        Self {
            core,
            observer: None,
            converged: HashMap::new(),
            tracked: HashSet::new(),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn CatalogObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Apply one desired snapshot. A snapshot with duplicate worker ids is
    /// rejected before any catalog mutation. A failed upsert or delete stops
    /// the pass; the next snapshot retries it.
    pub async fn apply(&mut self, desired: Vec<WorkerRequest>) -> Result<(), SelectionError> {
        let mut by_id: HashMap<WorkerId, WorkerRequest> = HashMap::with_capacity(desired.len());
        for request in desired {
            let worker_id = request.worker_id;
            if by_id.insert(worker_id, request).is_some() {
                return Err(SelectionError::BadRequest(format!(
                    "duplicate worker_id {worker_id} in membership snapshot"
                )));
            }
        }

        for (worker_id, request) in &by_id {
            // Track before the upsert so a partially applied record is still
            // deleted when the worker leaves the desired set.
            self.tracked.insert(*worker_id);
            if self.converged.get(worker_id) == Some(request) {
                continue;
            }
            let record = self.core.upsert_worker(request.clone()).await?;
            if let Some(observer) = &self.observer {
                observer.upserted(&record);
            }
            if record.lifecycle == WorkerLifecycle::Schedulable {
                self.converged.insert(*worker_id, request.clone());
            } else {
                self.converged.remove(worker_id);
                tracing::warn!(
                    worker_id,
                    lifecycle = ?record.lifecycle,
                    reasons = ?record.not_schedulable_reasons,
                    "worker upserted but not schedulable; retrying on the next membership snapshot"
                );
            }
        }

        let stale: Vec<WorkerId> = self
            .tracked
            .iter()
            .copied()
            .filter(|worker_id| !by_id.contains_key(worker_id))
            .collect();
        for worker_id in stale {
            match self.core.delete_worker(worker_id).await {
                Ok(record) => {
                    if let Some(observer) = &self.observer {
                        observer.removed(&record);
                    }
                }
                Err(SelectionError::NotFound(_)) => {}
                Err(error) => return Err(error),
            }
            self.tracked.remove(&worker_id);
            self.converged.remove(&worker_id);
        }
        Ok(())
    }

    /// Apply every snapshot `source` yields until it closes or `cancel` fires.
    /// The catalog keeps its last state when the source closes; a source that
    /// wants selection to fail closed yields an empty snapshot before closing.
    pub async fn run(mut self, mut source: impl WorkerCatalogSource, cancel: CancellationToken) {
        loop {
            let snapshot = tokio::select! {
                _ = cancel.cancelled() => return,
                snapshot = source.next_snapshot() => snapshot,
            };
            let Some(snapshot) = snapshot else {
                tracing::debug!("worker membership source closed");
                return;
            };
            if let Err(error) = self.apply(snapshot).await {
                tracing::warn!(%error, "membership reconcile failed; retrying on the next snapshot");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KvRouterConfig;
    use crate::services::selection::SelectionCacheConfig;

    fn core() -> Arc<SelectionCore> {
        Arc::new(SelectionCore::new_local(
            KvRouterConfig {
                use_kv_events: false,
                router_queue_threshold: None,
                ..Default::default()
            },
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        ))
    }

    fn schedulable(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            worker_id,
            endpoint: Some(format!("http://10.0.0.{worker_id}:8000")),
            block_size: Some(16),
            ..WorkerRequest::default()
        }
    }

    /// `block_size = 0` fails the schedulable-metadata check regardless of
    /// router or KV-event configuration.
    fn incomplete(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            block_size: Some(0),
            ..schedulable(worker_id)
        }
    }

    fn lifecycle(core: &SelectionCore, worker_id: WorkerId) -> Option<WorkerLifecycle> {
        core.list_workers(None, None)
            .into_iter()
            .find(|record| record.worker_id == worker_id)
            .map(|record| record.lifecycle)
    }

    #[tokio::test]
    async fn incomplete_worker_is_retried_and_deleted_when_it_leaves() {
        let core = core();
        let mut reconciler = CatalogReconciler::new(Arc::clone(&core));

        reconciler.apply(vec![incomplete(1)]).await.expect("apply");
        assert!(reconciler.converged.is_empty());
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Incomplete));

        // The same snapshot re-upserts rather than skipping the unconverged worker.
        reconciler.apply(vec![incomplete(1)]).await.expect("apply");
        assert!(reconciler.converged.is_empty());
        assert!(reconciler.tracked.contains(&1));

        reconciler.apply(Vec::new()).await.expect("apply");
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Unschedulable));
        assert!(reconciler.tracked.is_empty());
    }

    #[tokio::test]
    async fn schedulable_worker_converges_and_changed_record_reupserts() {
        let core = core();
        let mut reconciler = CatalogReconciler::new(Arc::clone(&core));

        reconciler.apply(vec![schedulable(1)]).await.expect("apply");
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Schedulable));
        assert!(reconciler.converged.contains_key(&1));

        let mut moved = schedulable(1);
        moved.endpoint = Some("http://10.0.0.9:8000".to_string());
        reconciler.apply(vec![moved]).await.expect("apply");
        let record = core
            .list_workers(None, None)
            .into_iter()
            .find(|record| record.worker_id == 1)
            .expect("record");
        assert_eq!(record.endpoint.as_deref(), Some("http://10.0.0.9:8000"));
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
    }

    #[tokio::test]
    async fn duplicate_ids_are_rejected_before_any_mutation() {
        let core = core();
        let mut reconciler = CatalogReconciler::new(Arc::clone(&core));

        let error = reconciler
            .apply(vec![schedulable(1), schedulable(1)])
            .await
            .expect_err("duplicates are rejected");
        assert!(
            matches!(error, SelectionError::BadRequest(message) if message.contains("duplicate worker_id 1"))
        );
        assert!(core.list_workers(None, None).is_empty());
    }
}
