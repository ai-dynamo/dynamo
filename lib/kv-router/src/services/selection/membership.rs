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
use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::core::SelectionCore;
use super::error::SelectionError;
use super::types::{WorkerCatalogRecord, WorkerLifecycle, WorkerRequest};
use crate::protocols::WorkerId;

/// Delay before [`CatalogReconciler::run`] re-applies a snapshot whose pass failed.
const RECONCILE_RETRY_DELAY: Duration = Duration::from_secs(5);

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
    /// Passes started by `run`, including retries.
    #[cfg(test)]
    applies: Arc<std::sync::atomic::AtomicUsize>,
}

impl CatalogReconciler {
    pub fn new(core: Arc<SelectionCore>) -> Self {
        Self {
            core,
            observer: None,
            converged: HashMap::new(),
            tracked: HashSet::new(),
            #[cfg(test)]
            applies: Arc::default(),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn CatalogObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Apply one desired snapshot. A snapshot with duplicate worker ids is
    /// rejected before any catalog mutation. A failed upsert or delete does
    /// not stop the pass: every other worker is still upserted and every
    /// worker absent from the snapshot is still deleted. The first error is
    /// returned so the caller retries the snapshot.
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

        let mut to_upsert: Vec<WorkerRequest> = Vec::new();
        for (worker_id, request) in &by_id {
            // Track before the upsert so a partially applied record is still
            // deleted when the worker leaves the desired set.
            self.tracked.insert(*worker_id);
            if self.converged.get(worker_id) != Some(request) {
                to_upsert.push(request.clone());
            }
        }
        // One catalog lock and one scheduler-config publish per partition for
        // the whole snapshot, instead of one of each per worker.
        let ids: Vec<WorkerId> = to_upsert.iter().map(|request| request.worker_id).collect();
        let results = self.core.upsert_workers(to_upsert).await;
        let mut first_error: Option<SelectionError> = None;
        for (worker_id, result) in ids.into_iter().zip(results) {
            let record = match result {
                Ok(record) => record,
                Err(error) => {
                    tracing::warn!(worker_id, %error, "worker upsert failed; continuing the pass");
                    first_error.get_or_insert(error);
                    continue;
                }
            };
            if let Some(observer) = &self.observer {
                observer.upserted(&record);
            }
            if record.lifecycle == WorkerLifecycle::Schedulable {
                let request = by_id.get(&worker_id).expect("upserted id came from by_id");
                self.converged.insert(worker_id, request.clone());
            } else {
                self.converged.remove(&worker_id);
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
                // A worker that was never registered is not an error (idempotent).
                Err(SelectionError::NotFound(_)) => {}
                // Stays tracked so the next pass deletes it.
                Err(error) => {
                    tracing::warn!(worker_id, %error, "stale worker delete failed; continuing the pass");
                    first_error.get_or_insert(error);
                    continue;
                }
            }
            self.tracked.remove(&worker_id);
            self.converged.remove(&worker_id);
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Apply every snapshot `source` yields until it closes or `cancel` fires.
    /// The catalog keeps its last state when the source closes; a source that
    /// wants selection to fail closed yields an empty snapshot before closing.
    ///
    /// A snapshot whose pass failed is kept in a single pending slot and
    /// re-applied after [`RECONCILE_RETRY_DELAY`]; a newer snapshot from the
    /// source supersedes it. Sources block until membership changes, so
    /// without this a failed pass would wait for an unrelated change.
    pub async fn run(mut self, mut source: impl WorkerCatalogSource, cancel: CancellationToken) {
        let mut pending: Option<Vec<WorkerRequest>> = None;
        loop {
            let snapshot = tokio::select! {
                _ = cancel.cancelled() => return,
                snapshot = source.next_snapshot() => {
                    let Some(snapshot) = snapshot else {
                        tracing::debug!("worker membership source closed");
                        return;
                    };
                    snapshot
                }
                _ = tokio::time::sleep(RECONCILE_RETRY_DELAY), if pending.is_some() => {
                    pending.take().expect("guarded by pending.is_some()")
                }
            };
            #[cfg(test)]
            self.applies
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match self.apply(snapshot.clone()).await {
                Ok(()) => pending = None,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        retry_in = ?RECONCILE_RETRY_DELAY,
                        "membership reconcile failed; retrying"
                    );
                    pending = Some(snapshot);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::config::KvRouterConfig;
    use crate::services::selection::SelectionCacheConfig;

    fn core() -> Arc<SelectionCore> {
        Arc::new(
            SelectionCore::try_new_local(
                KvRouterConfig {
                    use_kv_events: false,
                    router_queue_threshold: None,
                    ..Default::default()
                },
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        )
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

    /// Counts observer callbacks so tests can assert convergence (a skipped
    /// upsert) and stale deletion through the public surface.
    #[derive(Default)]
    struct Counter {
        upserted: AtomicUsize,
        removed: AtomicUsize,
    }

    impl Counter {
        fn upserts(&self) -> usize {
            self.upserted.load(Ordering::SeqCst)
        }

        fn removals(&self) -> usize {
            self.removed.load(Ordering::SeqCst)
        }
    }

    impl CatalogObserver for Counter {
        fn upserted(&self, _record: &WorkerCatalogRecord) {
            self.upserted.fetch_add(1, Ordering::SeqCst);
        }

        fn removed(&self, _record: &WorkerCatalogRecord) {
            self.removed.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn reconciler(core: &Arc<SelectionCore>) -> (CatalogReconciler, Arc<Counter>) {
        let counter = Arc::new(Counter::default());
        let reconciler = CatalogReconciler::new(Arc::clone(core))
            .with_observer(Arc::clone(&counter) as Arc<dyn CatalogObserver>);
        (reconciler, counter)
    }

    #[tokio::test]
    async fn incomplete_worker_is_retried_and_deleted_when_it_leaves() {
        let core = core();
        let (mut reconciler, counter) = reconciler(&core);

        reconciler.apply(vec![incomplete(1)]).await.expect("apply");
        assert_eq!(counter.upserts(), 1);
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Incomplete));

        // The same snapshot re-upserts rather than skipping the unconverged worker.
        reconciler.apply(vec![incomplete(1)]).await.expect("apply");
        assert_eq!(counter.upserts(), 2);

        // The never-schedulable worker was still tracked, so leaving deletes it
        // and the catalog no longer lists it.
        reconciler.apply(Vec::new()).await.expect("apply");
        assert_eq!(lifecycle(&core, 1), None);
        assert_eq!(counter.removals(), 1);

        // Nothing is tracked any more: an empty snapshot deletes nothing.
        reconciler.apply(Vec::new()).await.expect("apply");
        assert_eq!(counter.removals(), 1);
    }

    #[tokio::test]
    async fn schedulable_worker_converges_and_changed_record_reupserts() {
        let core = core();
        let (mut reconciler, counter) = reconciler(&core);

        reconciler.apply(vec![schedulable(1)]).await.expect("apply");
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Schedulable));
        assert_eq!(counter.upserts(), 1);

        // An identical desired record is converged and skips its upsert.
        reconciler.apply(vec![schedulable(1)]).await.expect("apply");
        assert_eq!(counter.upserts(), 1);

        let mut moved = schedulable(1);
        moved.endpoint = Some("http://10.0.0.9:8000".to_string());
        reconciler.apply(vec![moved]).await.expect("apply");
        assert_eq!(counter.upserts(), 2);
        let record = core
            .list_workers(None, None)
            .into_iter()
            .find(|record| record.worker_id == 1)
            .expect("record");
        assert_eq!(record.endpoint.as_deref(), Some("http://10.0.0.9:8000"));
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
    }

    #[tokio::test]
    async fn stale_worker_is_deleted_when_another_upsert_fails() {
        let core = core();
        let (mut reconciler, counter) = reconciler(&core);

        reconciler
            .apply(vec![schedulable(1), schedulable(2)])
            .await
            .expect("apply");
        assert_eq!(counter.upserts(), 2);

        // Worker 2 changes and its upsert fails; worker 1 left the snapshot.
        core.fail_upsert_for.lock().insert(2);
        let mut changed = schedulable(2);
        changed.endpoint = Some("http://10.0.0.9:8000".to_string());
        let error = reconciler
            .apply(vec![changed.clone()])
            .await
            .expect_err("failed upsert is reported");
        assert!(matches!(error, SelectionError::Internal(_)), "{error}");

        // The stale worker is still deleted on the pass that observed its absence.
        assert_eq!(lifecycle(&core, 1), None);
        assert_eq!(counter.removals(), 1);
        assert_eq!(reconciler.tracked, HashSet::from([2]));
        // The failed worker is still tracked and is retried once the fault clears.
        core.fail_upsert_for.lock().clear();
        reconciler.apply(vec![changed]).await.expect("apply");
        assert_eq!(counter.upserts(), 3);
    }

    /// Snapshots pushed by the test; closes when the sender drops.
    struct ChannelSource(tokio::sync::mpsc::UnboundedReceiver<Vec<WorkerRequest>>);

    #[async_trait]
    impl WorkerCatalogSource for ChannelSource {
        async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
            self.0.recv().await
        }
    }

    /// Yield until `applies` reaches `expected` without letting the paused
    /// clock auto-advance (the test task never parks).
    async fn wait_for_applies(applies: &AtomicUsize, expected: usize) {
        for _ in 0..1000 {
            if applies.load(Ordering::SeqCst) >= expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "expected {expected} applies, saw {}",
            applies.load(Ordering::SeqCst)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn failed_snapshot_is_retried_after_the_delay() {
        let core = core();
        let (reconciler, counter) = reconciler(&core);
        let applies = Arc::clone(&reconciler.applies);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(reconciler.run(ChannelSource(rx), cancel.clone()));

        core.fail_upsert_for.lock().insert(1);
        tx.send(vec![schedulable(1)]).expect("send");
        wait_for_applies(&applies, 1).await;
        assert_eq!(lifecycle(&core, 1), None);

        // The fault clears, but nothing re-applies before the delay elapses.
        core.fail_upsert_for.lock().clear();
        tokio::time::advance(RECONCILE_RETRY_DELAY - Duration::from_millis(1)).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(applies.load(Ordering::SeqCst), 1);
        assert_eq!(lifecycle(&core, 1), None);

        tokio::time::advance(Duration::from_millis(1)).await;
        wait_for_applies(&applies, 2).await;
        assert_eq!(lifecycle(&core, 1), Some(WorkerLifecycle::Schedulable));
        assert_eq!(counter.upserts(), 1);

        // A successful pass clears the slot: no further retry fires.
        tokio::time::advance(RECONCILE_RETRY_DELAY * 2).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(applies.load(Ordering::SeqCst), 2);

        cancel.cancel();
        task.await.expect("run exits on cancel");
    }

    #[tokio::test(start_paused = true)]
    async fn newer_snapshot_supersedes_the_pending_retry() {
        let core = core();
        let (reconciler, counter) = reconciler(&core);
        let applies = Arc::clone(&reconciler.applies);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(reconciler.run(ChannelSource(rx), cancel.clone()));

        core.fail_upsert_for.lock().insert(1);
        tx.send(vec![schedulable(1)]).expect("send");
        wait_for_applies(&applies, 1).await;
        assert_eq!(lifecycle(&core, 1), None);

        // A newer snapshot inside the retry window replaces the pending one.
        tokio::time::advance(Duration::from_secs(1)).await;
        tx.send(vec![schedulable(2)]).expect("send");
        wait_for_applies(&applies, 2).await;
        assert_eq!(lifecycle(&core, 2), Some(WorkerLifecycle::Schedulable));

        // Even after the fault clears and the window elapses, the superseded
        // snapshot is never re-applied: worker 1 stays absent.
        core.fail_upsert_for.lock().clear();
        tokio::time::advance(RECONCILE_RETRY_DELAY * 2).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(applies.load(Ordering::SeqCst), 2);
        assert_eq!(lifecycle(&core, 1), None);
        assert_eq!(counter.upserts(), 1);

        cancel.cancel();
        task.await.expect("run exits on cancel");
    }

    #[tokio::test]
    async fn initial_snapshot_publishes_once_per_partition() {
        let core = core();
        let (mut reconciler, counter) = reconciler(&core);
        let publishes = || core.publish_count.load(Ordering::SeqCst);

        reconciler
            .apply((1..=4).map(schedulable).collect())
            .await
            .expect("apply");
        assert_eq!(
            publishes(),
            1,
            "one publish for four workers in one partition"
        );
        assert_eq!(counter.upserts(), 4, "observer still fires per worker");
        let schedulable_count = core
            .list_workers(None, None)
            .into_iter()
            .filter(|record| record.lifecycle == WorkerLifecycle::Schedulable)
            .count();
        assert_eq!(schedulable_count, 4);

        // A second partition in the same snapshot publishes once for itself.
        let mut other = schedulable(5);
        other.routing_group = "other".to_string();
        reconciler
            .apply((1..=4).map(schedulable).chain([other]).collect())
            .await
            .expect("apply");
        assert_eq!(publishes(), 2);

        // The single-worker path publishes once per call.
        core.upsert_worker(schedulable(6)).await.expect("upsert");
        assert_eq!(publishes(), 3);
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
