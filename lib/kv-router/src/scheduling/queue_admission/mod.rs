// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::types::SessionContext;

pub mod policy_queue;

pub use policy_queue::{
    PolicyQueue, PolicyQueueEntry, QueueLimitKind, QueueRejection, QueueSnapshot,
};

/// Scheduler-assigned identity for one request managed by a [`QueueAdmissionPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueAdmissionId(u64);

impl QueueAdmissionId {
    /// Create an identity for testing an admission policy outside the scheduler host.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

/// One worker/rank visible to queue admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueAdmissionWorker {
    worker: WorkerWithDpRank,
    capacity_tokens: Option<usize>,
    available: bool,
}

impl QueueAdmissionWorker {
    /// Create a worker snapshot for testing an admission policy outside the scheduler host.
    pub fn new(worker: WorkerWithDpRank, capacity_tokens: Option<usize>, available: bool) -> Self {
        Self {
            worker,
            capacity_tokens,
            available,
        }
    }

    /// Return the worker and data-parallel rank.
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Return the worker's reported KV capacity in tokens.
    ///
    /// `None` means discovery did not report usable KV capacity.
    pub fn capacity_tokens(&self) -> Option<usize> {
        self.capacity_tokens
    }

    /// Return whether the worker is currently available, including transient overload state.
    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// Immutable worker state shared across queue-admission calls.
///
/// The host increments `generation` only when worker topology, capacity, or
/// availability changes. Policies can retain a clone and skip rebuilding their
/// worker state when the generation is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueAdmissionWorkerSnapshot {
    generation: u64,
    workers: Arc<[QueueAdmissionWorker]>,
}

impl QueueAdmissionWorkerSnapshot {
    /// Create a sorted snapshot for testing an admission policy outside the scheduler host.
    pub fn new(generation: u64, mut workers: Vec<QueueAdmissionWorker>) -> Self {
        workers.sort_unstable_by_key(QueueAdmissionWorker::worker);
        Self {
            generation,
            workers: workers.into(),
        }
    }

    /// Return the host generation for this snapshot.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Return all worker/rank entries in stable worker order.
    pub fn workers(&self) -> &[QueueAdmissionWorker] {
        &self.workers
    }

    /// Return one exact worker/rank entry.
    pub fn get(&self, worker: WorkerWithDpRank) -> Option<&QueueAdmissionWorker> {
        self.workers
            .binary_search_by_key(&worker, QueueAdmissionWorker::worker)
            .ok()
            .map(|index| &self.workers[index])
    }

    fn workers_for_id(&self, worker_id: WorkerId) -> &[QueueAdmissionWorker] {
        let start = self
            .workers
            .partition_point(|worker| worker.worker().worker_id < worker_id);
        let end = self
            .workers
            .partition_point(|worker| worker.worker().worker_id <= worker_id);
        &self.workers[start..end]
    }
}

type WorkerEligibilityPredicate<'a> = dyn Fn(WorkerWithDpRank) -> bool + 'a;

#[derive(Clone, Copy, Default)]
struct QueueAdmissionEligibility<'a> {
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
    eligible_workers: Option<&'a HashSet<WorkerWithDpRank>>,
    has_hard_constraints: bool,
    predicate: Option<&'a WorkerEligibilityPredicate<'a>>,
}

impl QueueAdmissionEligibility<'_> {
    fn allows(&self, worker: WorkerWithDpRank) -> bool {
        self.pinned_worker.is_none_or(|pinned| pinned == worker)
            && self
                .allowed_worker_ids
                .is_none_or(|allowed| allowed.contains(&worker.worker_id))
            && self
                .eligible_workers
                .is_none_or(|eligible| eligible.contains(&worker))
            && self.predicate.is_none_or(|predicate| predicate(worker))
    }
}

/// Read-only request facts supplied to a queue admission policy.
pub struct QueueAdmissionRequest<'a> {
    id: QueueAdmissionId,
    request_id: &'a str,
    context_tokens: usize,
    progress: RequestProgress,
    session_context: Option<&'a SessionContext>,
    worker_snapshot: &'a QueueAdmissionWorkerSnapshot,
    eligibility: QueueAdmissionEligibility<'a>,
}

impl<'a> QueueAdmissionRequest<'a> {
    /// Create a request view for testing an admission policy outside the scheduler host.
    pub fn new(
        id: QueueAdmissionId,
        request_id: &'a str,
        context_tokens: usize,
        session_context: Option<&'a SessionContext>,
        worker_snapshot: &'a QueueAdmissionWorkerSnapshot,
    ) -> Self {
        Self {
            id,
            request_id,
            context_tokens,
            progress: RequestProgress::new(context_tokens).0,
            session_context,
            worker_snapshot,
            eligibility: QueueAdmissionEligibility::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_eligibility(
        id: QueueAdmissionId,
        request_id: &'a str,
        context_tokens: usize,
        progress: RequestProgress,
        session_context: Option<&'a SessionContext>,
        worker_snapshot: &'a QueueAdmissionWorkerSnapshot,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<&'a HashSet<WorkerId>>,
        has_hard_constraints: bool,
        predicate: &'a WorkerEligibilityPredicate<'a>,
    ) -> Self {
        Self {
            id,
            request_id,
            context_tokens,
            progress,
            session_context,
            worker_snapshot,
            eligibility: QueueAdmissionEligibility {
                pinned_worker,
                allowed_worker_ids,
                eligible_workers: None,
                has_hard_constraints,
                predicate: Some(predicate),
            },
        }
    }

    pub fn id(&self) -> QueueAdmissionId {
        self.id
    }

    /// Limit a synthetic request to an exact worker/rank set.
    ///
    /// This builder supports policy tests outside the scheduler host. Production
    /// requests receive eligibility from Dynamo's routing constraints.
    pub fn with_eligible_workers(
        mut self,
        eligible_workers: &'a HashSet<WorkerWithDpRank>,
    ) -> Self {
        self.eligibility.eligible_workers = Some(eligible_workers);
        self
    }

    /// Return the caller-supplied request identity.
    ///
    /// The host rejects another admission-managed request with the same identity until this
    /// request reaches a terminal event.
    pub fn request_id(&self) -> &str {
        self.request_id
    }

    pub fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    /// Return the latest logical input-plus-output context for this request.
    ///
    /// The value starts at [`Self::context_tokens`] and advances monotonically
    /// while the response stream crosses output-block boundaries. It can lag by
    /// at most one such boundary; the terminal event carries the authoritative
    /// final context. Policies may retain a clone for the request lifetime.
    pub fn progress(&self) -> &RequestProgress {
        &self.progress
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context
    }

    /// Return the routing partition's current worker snapshot.
    pub fn worker_snapshot(&self) -> &QueueAdmissionWorkerSnapshot {
        self.worker_snapshot
    }

    /// Return the routing partition's current workers.
    ///
    /// Transient overload and hard availability are reported by
    /// [`QueueAdmissionWorker::is_available`]. Use
    /// [`Self::for_each_eligible_worker`] for request eligibility. The worker
    /// picker validates eligibility again before final selection.
    pub fn workers(&self) -> &[QueueAdmissionWorker] {
        self.worker_snapshot.workers()
    }

    /// Return whether this request can use an exact worker/rank.
    pub fn is_worker_eligible(&self, worker: WorkerWithDpRank) -> bool {
        self.worker_snapshot.get(worker).is_some() && self.eligibility.allows(worker)
    }

    /// Visit request-eligible workers without materializing eligibility for every worker.
    ///
    /// Pinned requests use one exact lookup. Allow-list requests visit only the
    /// listed worker IDs. Unrestricted requests visit the shared snapshot directly.
    pub fn for_each_eligible_worker(&self, mut visit: impl FnMut(&QueueAdmissionWorker)) {
        if let Some(pinned) = self.eligibility.pinned_worker {
            if self.eligibility.allows(pinned)
                && let Some(worker) = self.worker_snapshot.get(pinned)
            {
                visit(worker);
            }
            return;
        }

        if let Some(allowed_worker_ids) = self.eligibility.allowed_worker_ids {
            for &worker_id in allowed_worker_ids {
                for worker in self.worker_snapshot.workers_for_id(worker_id) {
                    if self.eligibility.allows(worker.worker()) {
                        visit(worker);
                    }
                }
            }
            return;
        }

        if !self.eligibility.has_hard_constraints && self.eligibility.eligible_workers.is_none() {
            for worker in self.worker_snapshot.workers() {
                visit(worker);
            }
            return;
        }

        for worker in self.worker_snapshot.workers() {
            if self.eligibility.allows(worker.worker()) {
                visit(worker);
            }
        }
    }
}

/// Initial queue admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueAdmissionDecision {
    /// Do not track this request in the custom admission policy.
    Bypass,
    /// Make the request runnable under Dynamo's built-in queue ordering.
    Ready,
    /// Keep the request in host-owned deferred storage until the policy wakes it.
    Defer,
}

/// Lifecycle input delivered serially by the scheduler queue actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueAdmissionEvent<'a> {
    Dispatched {
        id: QueueAdmissionId,
        worker: WorkerWithDpRank,
    },
    Completed {
        request_id: &'a str,
        /// Final input-plus-output context when the response path observed it.
        context_tokens: Option<usize>,
    },
    Aborted {
        request_id: &'a str,
    },
    /// Periodic or topology-driven opportunity to reconsider deferred work.
    Reconcile {
        snapshot: &'a QueueAdmissionWorkerSnapshot,
    },
}

/// Optional programmatic admission policy hosted by [`PolicyQueue`].
///
/// Dynamo keeps ownership of requests and deferred storage. Implementations retain only
/// policy state and append IDs of previously deferred requests that should become ready.
/// A policy can implement session or program fairness by controlling which requests become
/// runnable. It does not reorder or pop runnable requests: policy-class ordering and cross-class
/// deficit round robin remain owned by Dynamo.
pub trait QueueAdmissionPolicy: Send {
    fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision;

    fn on_event(&mut self, _event: QueueAdmissionEvent<'_>, _ready: &mut Vec<QueueAdmissionId>) {}

    /// Return the maximum interval between reconciliation events.
    ///
    /// `None` uses the host interval. A zero duration is ignored.
    fn reconcile_interval(&self) -> Option<Duration> {
        None
    }
}

/// Lock-free access to the latest logical context observed for one request.
#[derive(Debug, Clone)]
pub struct RequestProgress {
    context_tokens: Arc<AtomicUsize>,
}

/// Write capability paired with [`RequestProgress`].
///
/// Updates are monotonic so concurrent or delayed observations cannot move a
/// request's logical context backwards.
#[derive(Debug, Clone)]
pub struct RequestProgressUpdater {
    context_tokens: Arc<AtomicUsize>,
}

impl RequestProgress {
    pub fn new(initial_context_tokens: usize) -> (Self, RequestProgressUpdater) {
        let context_tokens = Arc::new(AtomicUsize::new(initial_context_tokens));
        (
            Self {
                context_tokens: Arc::clone(&context_tokens),
            },
            RequestProgressUpdater { context_tokens },
        )
    }

    #[inline]
    pub fn context_tokens(&self) -> usize {
        self.context_tokens.load(Ordering::Relaxed)
    }
}

impl RequestProgressUpdater {
    #[inline]
    pub fn update_context_tokens(&self, context_tokens: usize) {
        self.context_tokens
            .fetch_max(context_tokens, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkerPlacement {
    /// Preserve the request's existing routing constraints.
    Any,
    /// Add an exact-worker constraint. The router validates it against the
    /// request's existing constraints before dispatch.
    Exact(WorkerWithDpRank),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_progress_is_monotonic() {
        let (progress, updater) = RequestProgress::new(42);

        updater.update_context_tokens(55);
        updater.update_context_tokens(50);

        assert_eq!(progress.context_tokens(), 55);
    }

    #[test]
    fn worker_snapshot_sorts_and_finds_exact_ranks() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let snapshot = QueueAdmissionWorkerSnapshot::new(
            7,
            vec![
                QueueAdmissionWorker::new(second, Some(2_000), true),
                QueueAdmissionWorker::new(first, Some(1_000), false),
            ],
        );

        assert_eq!(snapshot.generation(), 7);
        assert_eq!(snapshot.workers()[0].worker(), first);
        assert_eq!(snapshot.get(second).unwrap().capacity_tokens(), Some(2_000));
        assert!(snapshot.get(WorkerWithDpRank::new(3, 0)).is_none());
    }

    #[test]
    fn synthetic_request_visits_only_eligible_workers() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let snapshot = QueueAdmissionWorkerSnapshot::new(
            1,
            vec![
                QueueAdmissionWorker::new(first, None, true),
                QueueAdmissionWorker::new(second, None, true),
            ],
        );
        let eligible = HashSet::from([second]);
        let request =
            QueueAdmissionRequest::new(QueueAdmissionId::new(1), "request", 16, None, &snapshot)
                .with_eligible_workers(&eligible);
        let mut visited = Vec::new();

        request.for_each_eligible_worker(|worker| visited.push(worker.worker()));

        assert_eq!(visited, [second]);
        assert!(!request.is_worker_eligible(first));
        assert!(request.is_worker_eligible(second));
    }
}
