// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use serde_yaml::Mapping;

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

/// Live worker availability for one request.
///
/// The host combines request constraints and configured admission conditions.
/// Strategies may retain this handle and sample current state when needed.
#[derive(Clone)]
pub struct WorkerAvailability {
    snapshot: Arc<dyn Fn() -> WorkerAvailabilitySnapshot + Send + Sync>,
}

impl WorkerAvailability {
    pub fn new(snapshot: impl Fn() -> WorkerAvailabilitySnapshot + Send + Sync + 'static) -> Self {
        Self {
            snapshot: Arc::new(snapshot),
        }
    }

    pub fn snapshot(&self) -> WorkerAvailabilitySnapshot {
        (self.snapshot)()
    }
}

/// One consistent view of where a request is legal and can dispatch now.
#[derive(Clone)]
pub struct WorkerAvailabilitySnapshot {
    eligible: Arc<HashSet<WorkerWithDpRank>>,
    available: Arc<HashSet<WorkerWithDpRank>>,
}

impl WorkerAvailabilitySnapshot {
    pub fn new(workers: impl IntoIterator<Item = WorkerWithDpRank>) -> Self {
        let workers: Arc<HashSet<_>> = Arc::new(workers.into_iter().collect());
        Self {
            eligible: Arc::clone(&workers),
            available: workers,
        }
    }

    pub fn from_sets(
        eligible: HashSet<WorkerWithDpRank>,
        mut available: HashSet<WorkerWithDpRank>,
    ) -> Self {
        available.retain(|worker| eligible.contains(worker));
        Self {
            eligible: Arc::new(eligible),
            available: Arc::new(available),
        }
    }

    pub fn is_available(&self, worker: WorkerWithDpRank) -> bool {
        self.available.contains(&worker)
    }

    pub fn is_eligible(&self, worker: WorkerWithDpRank) -> bool {
        self.eligible.contains(&worker)
    }

    pub fn has_available_worker(&self) -> bool {
        !self.available.is_empty()
    }

    pub fn has_eligible_worker(&self) -> bool {
        !self.eligible.is_empty()
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
    worker_availability: WorkerAvailability,
}

impl<'a> AdmissionRequest<'a> {
    pub fn new(
        id: AdmissionId,
        session_id: Option<&'a str>,
        context_tokens: usize,
        worker_availability: WorkerAvailability,
    ) -> Self {
        Self {
            id,
            session_id,
            context_tokens,
            worker_availability,
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

    pub fn worker_availability(&self) -> &WorkerAvailability {
        &self.worker_availability
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
    /// Strategy state permits dispatch when the requested placement is available.
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
/// bypassed request receives no lifecycle events. A ready
/// request may receive one `Dispatched` event and every tracked request
/// receives exactly one terminal `Completed` or `Aborted` event while the host
/// remains alive. A deferred request receives no `Dispatched` event until the
/// first valid `MakeReady` action is accepted. Duplicate or unknown actions
/// are ignored. While any request is deferred, `Reconcile` is delivered at
/// least once per configured queue recheck interval and may also be delivered
/// after lifecycle or capacity changes. Host shutdown drops the strategy and
/// its requests together, so no terminal events are delivered after shutdown
/// begins.
pub trait PolicyClassAdmissionStrategy: Send {
    fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision;

    fn on_event(&mut self, _event: AdmissionEvent) -> Vec<AdmissionAction> {
        Vec::new()
    }

    /// Maximum time requested between reconciliation opportunities.
    fn reconcile_interval(&self) -> Option<Duration> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QueueAdmissionConfig {
    #[serde(rename = "type")]
    pub strategy: String,
    #[serde(flatten)]
    pub options: Mapping,
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
            let availability = request.worker_availability().snapshot();
            assert!(availability.is_available(worker));
            assert!(availability.is_eligible(worker));
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }
    }

    #[test]
    fn strategy_contract_is_object_safe() {
        let mut strategy: Box<dyn PolicyClassAdmissionStrategy> = Box::new(ReadyStrategy);
        let worker = WorkerWithDpRank::new(3, 0);
        let availability =
            WorkerAvailability::new(move || WorkerAvailabilitySnapshot::new([worker]));
        assert_eq!(
            strategy.admit(AdmissionRequest::new(
                AdmissionId::new(7),
                Some("session"),
                42,
                availability,
            )),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());
    }

    #[test]
    fn worker_availability_distinguishes_eligibility_from_current_availability() {
        let available = WorkerWithDpRank::new(1, 0);
        let overloaded = WorkerWithDpRank::new(2, 0);
        let snapshot = WorkerAvailabilitySnapshot::from_sets(
            HashSet::from([available, overloaded]),
            HashSet::from([available]),
        );

        assert!(snapshot.is_available(available));
        assert!(!snapshot.is_available(overloaded));
        assert!(snapshot.is_eligible(overloaded));
        assert!(snapshot.has_available_worker());
        assert!(snapshot.has_eligible_worker());
    }
}
