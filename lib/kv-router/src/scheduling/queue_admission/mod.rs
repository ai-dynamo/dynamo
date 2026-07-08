// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use serde::Deserialize;
use serde_yaml::Mapping;

use crate::protocols::WorkerWithDpRank;

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

/// Read-only request data exposed to admission strategies.
///
/// Accessors are added only when a strategy demonstrates a generic need for
/// them; the actor-owned scheduling request is intentionally not exposed.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionRequest<'a> {
    id: AdmissionId,
    _borrowed: PhantomData<&'a ()>,
}

impl AdmissionRequest<'_> {
    pub fn new(id: AdmissionId) -> Self {
        Self {
            id,
            _borrowed: PhantomData,
        }
    }

    pub fn id(&self) -> AdmissionId {
        self.id
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
    Ready(WorkerPlacement),
    Defer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionEvent {
    /// The router selected and reserved a worker for the request.
    Dispatched {
        id: AdmissionId,
        worker: WorkerWithDpRank,
    },
    /// The request left the admission host, including cancellation at any stage.
    Finished { id: AdmissionId },
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
/// The host calls [`Self::admit`] exactly once with a unique ID. A ready
/// request may receive one `Dispatched` event and every admitted request
/// receives exactly one terminal `Finished` event while the host remains
/// alive. A deferred request receives no `Dispatched` event until the first
/// valid `MakeReady` action is accepted. Duplicate or unknown actions are
/// ignored. While any request is deferred, `Reconcile` is delivered at least
/// once per configured queue recheck interval and may also be delivered after
/// lifecycle or capacity changes. Host shutdown drops the strategy and its
/// requests together, so no terminal events are delivered after shutdown
/// begins.
pub trait PolicyClassAdmissionStrategy: Send {
    fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision;

    fn on_event(&mut self, _event: AdmissionEvent) -> Vec<AdmissionAction> {
        Vec::new()
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
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }
    }

    #[test]
    fn strategy_contract_is_object_safe() {
        let mut strategy: Box<dyn PolicyClassAdmissionStrategy> = Box::new(ReadyStrategy);
        assert_eq!(
            strategy.admit(AdmissionRequest::new(AdmissionId::new(7))),
            AdmissionDecision::Ready(WorkerPlacement::Any)
        );
        assert!(strategy.on_event(AdmissionEvent::Reconcile).is_empty());
    }
}
