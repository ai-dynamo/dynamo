// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use rustc_hash::FxHashSet;
use serde::Deserialize;

use crate::protocols::WorkerWithDpRank;

mod controller;

pub use controller::PolicyClassAdmissionStrategies;
pub(crate) use controller::{
    AdmissionTicket, ClassAdmissionAction, PolicyClassAdmissionController,
};

/// Router-assigned identity for one request's admission lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdmissionId(u64);

impl AdmissionId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Lock-free access to the latest logical context observed for one request.
///
/// Strategies may retain this reader after [`PolicyClassAdmissionStrategy::admit`]
/// returns. The request path owns the corresponding updater and publishes
/// monotonic progress without sending commands through the scheduler actor.
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

/// Live worker eligibility for one admitted request.
///
/// The host owns routing constraints and worker state. Strategies may retain
/// this handle when deferred work must be reconsidered against current state.
#[derive(Clone)]
pub struct WorkerEligibility {
    snapshot: Arc<dyn Fn() -> WorkerEligibilitySnapshot + Send + Sync>,
}

impl WorkerEligibility {
    pub fn new(snapshot: impl Fn() -> WorkerEligibilitySnapshot + Send + Sync + 'static) -> Self {
        Self {
            snapshot: Arc::new(snapshot),
        }
    }

    pub fn snapshot(&self) -> WorkerEligibilitySnapshot {
        (self.snapshot)()
    }
}

/// One consistent view of the workers eligible for a request.
#[derive(Clone)]
pub struct WorkerEligibilitySnapshot {
    structural: Arc<FxHashSet<WorkerWithDpRank>>,
    available: Arc<FxHashSet<WorkerWithDpRank>>,
}

impl WorkerEligibilitySnapshot {
    pub fn new(workers: impl IntoIterator<Item = WorkerWithDpRank>) -> Self {
        let workers: Arc<FxHashSet<_>> = Arc::new(workers.into_iter().collect());
        Self {
            structural: Arc::clone(&workers),
            available: workers,
        }
    }

    pub fn with_availability(
        structural: FxHashSet<WorkerWithDpRank>,
        mut available: FxHashSet<WorkerWithDpRank>,
    ) -> Self {
        available.retain(|worker| structural.contains(worker));
        Self {
            structural: Arc::new(structural),
            available: Arc::new(available),
        }
    }

    /// Whether routing constraints permit this worker right now.
    pub fn allows(&self, worker: WorkerWithDpRank) -> bool {
        self.available.contains(&worker)
    }

    /// Whether routing constraints permit this worker independent of
    /// transient overload state.
    pub fn structurally_allows(&self, worker: WorkerWithDpRank) -> bool {
        self.structural.contains(&worker)
    }

    pub fn has_available_worker(&self) -> bool {
        !self.available.is_empty()
    }

    pub fn has_structural_worker(&self) -> bool {
        !self.structural.is_empty()
    }
}

/// Read-only request facts exposed to admission strategies.
///
/// Only [`AdmissionId`] is universal. A strategy may ignore any other fact or
/// return [`AdmissionDecision::Bypass`] when optional context does not apply.
/// The actor-owned scheduling request is intentionally not exposed.
#[derive(Clone)]
pub struct AdmissionRequest<'a> {
    id: AdmissionId,
    session_id: Option<&'a str>,
    context_tokens: usize,
    progress: RequestProgress,
    worker_eligibility: WorkerEligibility,
}

impl<'a> AdmissionRequest<'a> {
    /// Constructs a request with progress fixed at `context_tokens`.
    ///
    /// Live progress is only supplied by the scheduler-owned admission path.
    pub fn new(
        id: AdmissionId,
        session_id: Option<&'a str>,
        context_tokens: usize,
        worker_eligibility: WorkerEligibility,
    ) -> Self {
        let (progress, _) = RequestProgress::new(context_tokens);
        Self::with_progress(id, session_id, context_tokens, progress, worker_eligibility)
    }

    pub(crate) fn with_progress(
        id: AdmissionId,
        session_id: Option<&'a str>,
        context_tokens: usize,
        progress: RequestProgress,
        worker_eligibility: WorkerEligibility,
    ) -> Self {
        Self {
            id,
            session_id,
            context_tokens,
            progress,
            worker_eligibility,
        }
    }

    pub fn id(&self) -> AdmissionId {
        self.id
    }

    pub fn session_id(&self) -> Option<&'a str> {
        self.session_id
    }

    /// Full tokenized request context, not uncached prefill work.
    pub fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    /// Live logical context for this request.
    pub fn progress(&self) -> &RequestProgress {
        &self.progress
    }

    pub fn worker_eligibility(&self) -> &WorkerEligibility {
        &self.worker_eligibility
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionDecision {
    /// Continue through normal scheduling without a strategy lifecycle.
    Bypass,
    Ready(WorkerPlacement),
    Defer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionEvent {
    /// The backend accepted the request after the router selected and reserved
    /// its worker.
    Dispatched {
        id: AdmissionId,
        worker: WorkerWithDpRank,
    },
    /// The response stream ended normally.
    Completed {
        id: AdmissionId,
        context_tokens: usize,
    },
    /// The request ended without committing a new logical context.
    Aborted { id: AdmissionId },
    /// The host is giving the strategy an opportunity to reconsider deferred work.
    Reconcile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionAction {
    MakeReady {
        id: AdmissionId,
        placement: WorkerPlacement,
    },
}

/// Policy-class admission behavior.
///
/// The host calls [`Self::admit`] exactly once for each tracked scheduling
/// request, using a unique ID. Query-only selection bypasses admission. A
/// bypassed request receives no lifecycle events. A ready request may receive
/// one `Dispatched` event and every tracked request receives exactly one
/// terminal `Completed` or `Aborted` event while the host remains alive. A
/// deferred request receives no `Dispatched` event until the first valid
/// `MakeReady` action is accepted. Duplicate or unknown actions are ignored.
/// While any request is deferred, `Reconcile` is delivered at least once per
/// configured queue recheck interval and may also be delivered after lifecycle
/// or capacity changes. Host shutdown drops the strategy and its requests
/// together, so no terminal events are delivered after shutdown begins.
pub trait PolicyClassAdmissionStrategy: Send {
    fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision;

    fn on_event(&mut self, _event: AdmissionEvent) -> Vec<AdmissionAction> {
        Vec::new()
    }

    /// Maximum time requested between reconciliation opportunities. A returned
    /// interval must be nonzero.
    fn reconcile_interval(&self) -> Option<Duration> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum QueueAdmissionConfig {
    SessionAware {
        #[serde(default)]
        pause_threshold: Option<f64>,
        #[serde(default)]
        pause_target: Option<f64>,
        #[serde(default)]
        resume_timeout_seconds: Option<f64>,
        #[serde(default)]
        session_retention_seconds: Option<f64>,
        #[serde(default)]
        scheduler_interval_seconds: Option<f64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReadyStrategy;

    impl PolicyClassAdmissionStrategy for ReadyStrategy {
        fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
            assert_eq!(request.id(), AdmissionId::new(7));
            assert_eq!(request.session_id(), Some("session"));
            assert_eq!(request.context_tokens(), 42);
            let worker = WorkerWithDpRank::new(3, 0);
            let eligibility = request.worker_eligibility().snapshot();
            assert!(eligibility.allows(worker));
            assert!(eligibility.structurally_allows(worker));
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }
    }

    #[test]
    fn strategy_contract_is_object_safe() {
        let mut strategy: Box<dyn PolicyClassAdmissionStrategy> = Box::new(ReadyStrategy);
        let worker = WorkerWithDpRank::new(3, 0);
        let eligibility = WorkerEligibility::new(move || WorkerEligibilitySnapshot::new([worker]));
        assert_eq!(
            strategy.admit(AdmissionRequest::new(
                AdmissionId::new(7),
                Some("session"),
                42,
                eligibility,
            )),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());
    }

    #[test]
    fn request_progress_is_monotonic() {
        let (progress, updater) = RequestProgress::new(42);

        updater.update_context_tokens(55);
        updater.update_context_tokens(50);

        assert_eq!(progress.context_tokens(), 55);
    }

    #[test]
    fn worker_eligibility_distinguishes_structure_from_availability() {
        let available = WorkerWithDpRank::new(1, 0);
        let overloaded = WorkerWithDpRank::new(2, 0);
        let snapshot = WorkerEligibilitySnapshot::with_availability(
            FxHashSet::from_iter([available, overloaded]),
            FxHashSet::from_iter([available]),
        );

        assert!(snapshot.allows(available));
        assert!(!snapshot.allows(overloaded));
        assert!(snapshot.structurally_allows(overloaded));
        assert!(snapshot.has_available_worker());
        assert!(snapshot.has_structural_worker());
    }
}
