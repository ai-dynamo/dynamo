// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::protocols::WorkerWithDpRank;
use crate::scheduling::types::SessionContext;

/// Scheduler-assigned identity for one request managed by a [`PolicyQueuePolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolicyQueueId(u64);

impl PolicyQueueId {
    /// Create an identity for testing a queue policy outside the scheduler host.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

/// One worker/rank visible to queue admission for this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyQueueWorker {
    worker: WorkerWithDpRank,
    capacity_tokens: Option<usize>,
    available: bool,
    eligible: bool,
}

impl PolicyQueueWorker {
    /// Create a worker snapshot for testing a queue policy outside the scheduler host.
    pub fn new(
        worker: WorkerWithDpRank,
        capacity_tokens: Option<usize>,
        available: bool,
        eligible: bool,
    ) -> Self {
        Self {
            worker,
            capacity_tokens,
            available,
            eligible,
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

    /// Return whether the worker is currently eligible, including transient overload state.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Return whether this request can use the worker.
    ///
    /// Reconciliation events describe the partition rather than one request,
    /// so their workers are always eligible.
    pub fn is_eligible(&self) -> bool {
        self.eligible
    }

    pub(crate) fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    pub(crate) fn set_eligible(&mut self, eligible: bool) {
        self.eligible = eligible;
    }
}

/// Read-only request facts supplied to a queue admission policy.
pub struct PolicyQueueRequest<'a> {
    id: PolicyQueueId,
    request_id: &'a str,
    context_tokens: usize,
    session_context: Option<&'a SessionContext>,
    workers: &'a [PolicyQueueWorker],
}

impl<'a> PolicyQueueRequest<'a> {
    /// Create a request view for testing a queue policy outside the scheduler host.
    pub fn new(
        id: PolicyQueueId,
        request_id: &'a str,
        context_tokens: usize,
        session_context: Option<&'a SessionContext>,
        workers: &'a [PolicyQueueWorker],
    ) -> Self {
        Self {
            id,
            request_id,
            context_tokens,
            session_context,
            workers,
        }
    }

    pub fn id(&self) -> PolicyQueueId {
        self.id
    }

    pub fn request_id(&self) -> &str {
        self.request_id
    }

    pub fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context
    }

    /// Return the routing partition's current workers.
    ///
    /// Transient overload and hard availability are reported by
    /// [`PolicyQueueWorker::is_available`]. Request eligibility is reported by
    /// [`PolicyQueueWorker::is_eligible`]. The worker picker validates eligibility
    /// again before final selection.
    pub fn workers(&self) -> &[PolicyQueueWorker] {
        self.workers
    }
}

/// Initial queue admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolicyQueueDecision {
    /// Do not track this request in the custom queue policy.
    Bypass,
    /// Allow normal queue ordering and worker selection.
    Ready,
    /// Keep the request in host-owned deferred storage until the policy wakes it.
    Defer,
}

/// Lifecycle input delivered serially by the scheduler queue actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolicyQueueEvent<'a> {
    Dispatched {
        id: PolicyQueueId,
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
        workers: &'a [PolicyQueueWorker],
    },
}

/// Optional admission policy hosted by [`super::policy_queue::PolicyQueue`].
///
/// Dynamo keeps ownership of requests and deferred storage. Implementations retain only
/// policy state and append IDs of previously deferred requests that should become ready.
pub trait PolicyQueuePolicy: Send {
    fn admit(&mut self, request: PolicyQueueRequest<'_>) -> PolicyQueueDecision;

    fn on_event(&mut self, _event: PolicyQueueEvent<'_>, _ready: &mut Vec<PolicyQueueId>) {}

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
}
