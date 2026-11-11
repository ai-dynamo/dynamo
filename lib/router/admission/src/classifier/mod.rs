// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public request-classifier contract and startup configuration.

mod inputs;
mod progress;
pub mod runtime;
mod view;
pub use inputs::RequestClassifierContext;
pub use progress::{RequestProgress, RequestProgressUpdater};
pub use runtime::{ClassificationError, RequestClassifierRuntime, RequestLifecycle};
pub use view::{NoWorkers, WorkerCacheView, WorkerLoadView, WorkerView};

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_tokens::SequenceHash;
use tokio::time::Instant;

use dynamo_router_protocol::columns::WorkerInputs;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_session_affinity::WorkerAffinityTarget;
use dynamo_slot_tracker::busy::BusyThresholds;

use crate::profile::PolicyProfile;
use crate::queue::QueueSnapshot;

#[derive(Debug)]
pub struct ClassifyRequest {
    classification_id: u64,
    request_id: Option<String>,
    policy_class: Option<String>,
    overrides: ClassificationOverrides,
    ingress_at: Instant,
    input_tokens: usize,
    initial_cached_tokens: usize,
    sequence_hashes: Option<Vec<SequenceHash>>,
    progress: RequestProgress,
    session_context: Option<SessionContext>,
    pinned_worker: Option<WorkerWithDpRank>,
    busy_thresholds: Option<ClassifyBusyThresholds>,
}

/// The router's policy classes and live router-wide thresholds, as seen by one
/// classification.
#[derive(Debug)]
struct ClassifyBusyThresholds {
    profile: Arc<PolicyProfile>,
    reject: BusyThresholds,
}

#[derive(Clone, Debug, Default)]
pub struct ClassificationOverrides {
    pub policy_class: Option<String>,
    pub due_at: Option<Instant>,
    pub scheduling_cost_tokens: Option<usize>,
    pub worker_selection_target: Option<Option<WorkerAffinityTarget>>,
}

impl ClassifyRequest {
    #[cfg(test)]
    pub fn new(input_tokens: usize, initial_cached_tokens: usize) -> Self {
        Self::with_timing(input_tokens, initial_cached_tokens, Instant::now())
    }

    /// A classification input for plugin tests and benchmarks: `input_tokens` with no cached
    /// prefix, an optional request id, and an optional session.
    #[cfg(any(test, feature = "testing", feature = "bench"))]
    pub fn for_test(
        input_tokens: usize,
        request_id: Option<&str>,
        session_context: Option<SessionContext>,
    ) -> Self {
        let mut request = Self::with_timing(input_tokens, 0, Instant::now());
        request.request_id = request_id.map(str::to_owned);
        request.session_context = session_context;
        request
    }

    pub fn with_timing(
        input_tokens: usize,
        initial_cached_tokens: usize,
        ingress_at: Instant,
    ) -> Self {
        Self {
            classification_id: 0,
            request_id: None,
            policy_class: None,
            overrides: ClassificationOverrides::default(),
            ingress_at,
            input_tokens,
            initial_cached_tokens,
            sequence_hashes: None,
            progress: RequestProgress::new(input_tokens).0,
            session_context: None,
            pinned_worker: None,
            busy_thresholds: None,
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_initial_policy_class(mut self, policy_class: impl Into<String>) -> Self {
        self.policy_class = Some(policy_class.into());
        self
    }

    pub fn with_session_context(mut self, session_context: SessionContext) -> Self {
        self.session_context = Some(session_context);
        self
    }

    pub fn with_pinned_worker(mut self, pinned_worker: Option<WorkerWithDpRank>) -> Self {
        self.pinned_worker = pinned_worker;
        self
    }

    /// Copy tracking hashes into storage owned by this classification attempt.
    pub fn with_sequence_hashes(mut self, hashes: Option<&[SequenceHash]>) -> Self {
        self.sequence_hashes = hashes.map(<[SequenceHash]>::to_vec);
        self
    }

    pub fn with_busy_thresholds(
        mut self,
        profile: Arc<PolicyProfile>,
        reject: BusyThresholds,
    ) -> Self {
        self.busy_thresholds = Some(ClassifyBusyThresholds { profile, reject });
        self
    }

    /// Busy thresholds of the policy class this request resolves to now, including a class
    /// set with [`set_policy_class`](Self::set_policy_class). The queue holds the request
    /// while every eligible worker is past them. A name that matches no configured class
    /// reads the default class's thresholds; the request fails before it is queued unless
    /// its class is then set to a configured one.
    pub fn queue_thresholds(&self) -> BusyThresholds {
        self.busy_thresholds
            .as_ref()
            .map(|thresholds| {
                let profile = &thresholds.profile;
                profile
                    .resolve_class_index(self.policy_class())
                    .map_or(profile.default_class(), |index| profile.class(index))
                    .busy_thresholds
            })
            .unwrap_or_default()
    }

    /// Busy thresholds of the policy class named `policy_class`, or `None` when the router
    /// does not know the class.
    pub fn queue_thresholds_for(&self, policy_class: &str) -> Option<BusyThresholds> {
        let profile = &self.busy_thresholds.as_ref()?.profile;
        profile
            .resolve_class_index_strict(policy_class)
            .map(|index| profile.class(index).busy_thresholds)
    }

    /// Router-wide busy thresholds, read when classification started. Selection skips a
    /// worker past them and rejects the request when none remains.
    pub fn reject_thresholds(&self) -> BusyThresholds {
        self.busy_thresholds
            .as_ref()
            .map(|thresholds| thresholds.reject)
            .unwrap_or_default()
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.overrides
            .policy_class
            .as_deref()
            .or(self.policy_class.as_deref())
    }

    pub fn set_policy_class(&mut self, policy_class: impl Into<String>) {
        self.overrides.policy_class = Some(policy_class.into());
    }

    /// Input sequence length in tokens, the size the scheduler uses.
    pub fn isl_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Return the request's ordered active-tracking sequence hashes.
    ///
    /// This is an owned snapshot of the host's existing scheduling input, taken
    /// before classification. Each entry represents a complete tracking block;
    /// use the factory's [`RequestClassifierContext::block_size`] for its token
    /// unit. A partial trailing block is not covered. `None` preserves missing
    /// tracking data; `Some(&[])` preserves a present, empty sequence.
    ///
    /// Values retain the host's public/keyed hashing and reuse configuration.
    /// With KV reuse disabled they are randomized tracking identities, not
    /// reusable content hashes. Compare only within the same tracking domain
    /// and configuration. These are not physical block IDs, cache-hit evidence,
    /// or a guarantee that requests share resident KV.
    ///
    /// The slice lives with this request, including while its classification
    /// future is pending. It is not extended by generation progress. Reading it
    /// neither computes hashes nor changes scheduling cost or worker selection.
    /// Building the snapshot copies eight bytes per supplied hash; retaining
    /// multiple classification inputs retains each copy independently.
    pub fn sequence_hashes(&self) -> Option<&[SequenceHash]> {
        self.sequence_hashes.as_deref()
    }

    /// Live context high-water mark, initialized from `input_tokens()` and raised
    /// by host observations of prompt plus generated tokens. This is not physical
    /// KV occupancy. The host alone updates this request's counter.
    pub fn progress(&self) -> &RequestProgress {
        &self.progress
    }

    /// Return the original router ingress time on the monotonic clock.
    pub fn ingress_at(&self) -> Instant {
        self.ingress_at
    }

    pub fn due_at(&self) -> Option<Instant> {
        self.overrides.due_at
    }

    pub fn set_due_at(&mut self, due_at: Instant) {
        self.overrides.due_at = Some(due_at);
    }

    pub fn scheduling_cost_tokens(&self) -> usize {
        self.overrides.scheduling_cost_tokens.unwrap_or_else(|| {
            QueueSnapshot::new(self.input_tokens, self.initial_cached_tokens).scheduling_cost_tokens
        })
    }

    pub fn set_scheduling_cost_tokens(&mut self, scheduling_cost_tokens: usize) {
        self.overrides.scheduling_cost_tokens = Some(scheduling_cost_tokens);
    }

    /// Prefer a worker/rank for this request, replacing its soft affinity target.
    /// Hard pins and caller eligibility constraints remain authoritative. A custom
    /// selector can fall back if the target is ineligible; `Sent` reports the result.
    pub fn set_worker_selection_target(&mut self, worker: WorkerWithDpRank) {
        self.overrides.worker_selection_target = Some(Some(worker.into()));
    }

    /// Clear this request's soft affinity preference without changing hard pins.
    pub fn clear_worker_selection_target(&mut self) {
        self.overrides.worker_selection_target = Some(None);
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    /// Exact worker/rank required by the caller. A classifier's soft selection
    /// target cannot override this pin.
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    /// Only the explicit overrides feed the queue: cache eligibility is
    /// recomputed from the current workers at enqueue, because worker state
    /// may have changed while the classification was pending.
    pub fn into_queue_inputs(self) -> ClassificationOverrides {
        self.overrides
    }
}

/// Error returned by [`RequestClassifier::classify`].
pub type ClassifierError = dyn Error + Send + Sync + 'static;

/// Cause delivered to the classifier when a request aborts. Produced by the
/// router or the worker path, not by the classifier.
pub type AbortCause = dyn Error + Send + Sync + 'static;

/// Request lifecycle events, delivered to [`RequestClassifier::on_event`] one
/// at a time in lifecycle order. A terminal event can arrive without a prior
/// `Sent`: an attempt that recorded a worker via selection but was never
/// dispatched still ends with `Completed`.
///
/// Delivery is asynchronous, but a re-registered request id is not classified
/// until the previous lifecycle's terminal event has been delivered, so a
/// plugin keying state by request id observes each lifecycle's `classify` and
/// events in order and never receives a stale terminal event after the id's
/// next `classify`.
#[derive(Debug)]
#[non_exhaustive]
pub enum ClassifyEvent {
    Sent {
        request_id: String,
        worker: WorkerWithDpRank,
    },
    Responding {
        request_id: String,
        worker: WorkerWithDpRank,
    },
    Completed {
        request_id: String,
        worker: WorkerWithDpRank,
        context_tokens: Option<usize>,
    },
    Aborted {
        request_id: String,
        worker: Option<WorkerWithDpRank>,
        error: Option<Arc<AbortCause>>,
    },
}

/// User-provided request classifier.
///
/// [`Self::classify`] is synchronous and runs under the router-wide classifier lock: it reads
/// the request and the worker rows, writes its overrides, and returns. It must not wait. A
/// policy that has to hold a request until something else happens (a turn finishes, a worker
/// gets room, a session resumes) returns a [`DispatchGate`] from [`Self::dispatch_gate`]; the
/// router queues the request at once and asks the gate before each dispatch, so the queue
/// keeps the order, the class budget, the expiry, and the wakeups.
///
/// [`Self::on_event`] is `async fn`. A dedicated router task delivers events one at a time in
/// lifecycle order and awaits each callback, so `on_event` may await freely. A slow `on_event`
/// delays classification of new requests because both share the classifier lock. Terminal
/// events arrive asynchronously, shortly after the request ends; events still queued at router
/// shutdown are dropped.
#[async_trait]
pub trait RequestClassifier: Send + 'static {
    /// Declare the worker-signal groups the `workers` view passed to [`Self::classify`]
    /// carries. The host reads this once at install and builds rows only when it is not
    /// [`WorkerInputs::NONE`].
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// `workers` covers every worker rank eligible for this request under its pin, allowed
    /// workers, availability, and routing constraints, in unspecified order, with the groups
    /// [`Self::required_worker_inputs`] declares (`CACHE` and `LOAD`; other groups are
    /// selection's and read as `None` here); it is empty when that is
    /// [`WorkerInputs::NONE`]. Overloaded workers are included. Values come from the row
    /// builder worker selection uses, so each group means the same thing here as in a filter
    /// or scorer.
    ///
    /// The view is a snapshot taken for this call and is lent only for its duration: the host
    /// frees the rows when `classify` returns. Worker selection takes its own snapshot when it
    /// later picks a worker, so it can pick a different worker.
    ///
    /// Return `Ok(request)` to queue the request, with any overrides set on it, or an error to
    /// reject it.
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: &dyn WorkerView,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        Ok(request)
    }

    async fn on_event(&mut self, _event: ClassifyEvent) {}

    /// The classifier's dispatch gate, read once at install. With a gate installed, the router
    /// asks it about each queue head before dispatch and keeps a held head queued. A request
    /// with a session context always waits in its session's lane, gate or not, so a hold blocks
    /// only that session; a held lane is asked again when the gate's recheck handle fires or its
    /// hold deadline passes, not on every worker free. `None` dispatches every head the host's
    /// own capacity checks allow.
    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        None
    }
}

/// One queue head the dispatch gate is asked about.
#[derive(Debug, Clone, Copy)]
pub struct DispatchHead<'a> {
    request_id: Option<&'a str>,
    session_context: Option<&'a SessionContext>,
    isl_tokens: usize,
    pinned_worker: Option<WorkerWithDpRank>,
    policy_class: &'a str,
    queued_at: Instant,
}

impl<'a> DispatchHead<'a> {
    /// Builds a head view. Hosts build one per queue head; plugin tests build them directly.
    pub fn new(
        request_id: Option<&'a str>,
        session_context: Option<&'a SessionContext>,
        isl_tokens: usize,
        pinned_worker: Option<WorkerWithDpRank>,
        policy_class: &'a str,
        queued_at: Instant,
    ) -> Self {
        Self {
            request_id,
            session_context,
            isl_tokens,
            pinned_worker,
            policy_class,
            queued_at,
        }
    }

    pub fn request_id(&self) -> Option<&'a str> {
        self.request_id
    }

    pub fn session_context(&self) -> Option<&'a SessionContext> {
        self.session_context
    }

    /// Input sequence length in tokens, the size the scheduler uses.
    pub fn isl_tokens(&self) -> usize {
        self.isl_tokens
    }

    /// Exact worker/rank required by the caller. A `Go` target cannot override this pin.
    pub fn pinned_worker(&self) -> Option<WorkerWithDpRank> {
        self.pinned_worker
    }

    /// Name of the policy class whose queue holds this head.
    pub fn policy_class(&self) -> &'a str {
        self.policy_class
    }

    /// When the head entered the router queue, after classification.
    pub fn queued_at(&self) -> Instant {
        self.queued_at
    }
}

/// Answer of [`DispatchGate::check`] for one queue head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Dispatch now. `target` replaces the request's soft affinity target unless the request
    /// carries a hard pin; a custom selector may still pick another eligible worker, and
    /// `Sent` reports the result.
    Go { target: Option<WorkerWithDpRank> },
    /// Keep the head queued. The host asks again when [`DispatchRecheck::notify`] is called,
    /// when the worker set changes, and no later than `until` when it is set. A worker freeing
    /// capacity does not re-ask a held head: the gate knows what it waits for and must `notify`
    /// when that happens. An `until` that is not in the future arms no timer: the head waits for
    /// a `notify`.
    Hold { until: Option<Instant> },
}

/// Wakes the router queue to ask the gate about its held heads again. Cheap to clone and to
/// call from any thread; calls coalesce into one recheck.
#[derive(Clone)]
pub struct DispatchRecheck(Arc<dyn Fn() + Send + Sync>);

impl DispatchRecheck {
    /// Wraps the wake function the host (or a test) wants `notify` to call.
    pub fn new(wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self(wake)
    }

    pub fn notify(&self) {
        (self.0)();
    }
}

impl std::fmt::Debug for DispatchRecheck {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DispatchRecheck")
    }
}

/// Per-head admission decision consulted by the router queue at dispatch time.
///
/// The gate is the place for a policy that must hold a request after it is queued: a session
/// admission controller, a wait for a preferred worker, a per-request fit check. The queue
/// keeps the request, its order, its class budget, and its expiry; the gate only answers
/// whether the head at the front of a lane may go now.
///
/// [`Self::check`] runs on the scheduler's actor thread, inside its dispatch loop, and may be
/// asked more than once about the same head within one drain. It must not block and must not
/// change what later answers depend on: treat it as a read. The host's own capacity checks run
/// after a `Go`, so a `Go` is a permission, not a dispatch. [`Self::dispatched`] runs once, when
/// the host takes the head out of the queue to admit it; admission can still fail, and the
/// request's lifecycle then ends with [`ClassifyEvent::Aborted`] as for any other failure.
pub trait DispatchGate: Send + Sync + 'static {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision;

    /// The host took `head` out of the queue for admission with `target`, the `Go` target
    /// after the request's hard pin, if any, replaced it.
    fn dispatched(&self, _head: &DispatchHead<'_>, _target: Option<WorkerWithDpRank>) {}

    /// Handed the recheck handle once, when the gate is installed. Keep it to wake the queue
    /// when a hold's reason goes away.
    fn attach(&self, _recheck: DispatchRecheck) {}
}
