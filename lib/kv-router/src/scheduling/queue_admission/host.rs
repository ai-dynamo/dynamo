// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host side of the queue-admission layer.
//!
//! This layer sits directly under the scheduler queue actor and above policy
//! classes. It owns the [`QueueAdmissionPolicy`], the payloads that policy
//! deferred, and the lifecycle bookkeeping for the requests it manages. It
//! never selects a policy class, reads or mutates a class queue, participates
//! in deficit round robin, or applies a class SLO.
//!
//! Every request it hands back to the actor — bypassed, made ready, or woken
//! from deferral — enters the same policy-class admission path, and nothing
//! below this layer can tell those three apart.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use rustc_hash::FxHashMap;

use super::{
    QueueAdmissionDecision, QueueAdmissionEvent, QueueAdmissionId, QueueAdmissionPolicy,
    QueueAdmissionRequest, QueueAdmissionWorkerSnapshot,
};
use crate::protocols::{WorkerId, WorkerWithDpRank};
use crate::scheduling::types::SessionContext;

/// The terminal outcome the response path observed for one tracked request.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AdmissionRequestOutcome {
    Completed { context_tokens: Option<usize> },
    Aborted,
}

/// What the admission layer decided for one arriving request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdmissionHandoff {
    /// Pass the request into policy-class admission now. `Bypass` carries no
    /// identity because the policy does not manage the request; `Ready` carries
    /// the identity its lifecycle events are reported under.
    Admit(Option<QueueAdmissionId>),
    /// Park the request under this identity until the policy wakes it. It does
    /// not reach policy-class selection, class limits, class statistics, a
    /// class queue, or deficit round robin before then.
    Defer(QueueAdmissionId),
}

impl AdmissionHandoff {
    /// The identity this request's lifecycle is reported under, if the policy
    /// manages it.
    pub(crate) fn admission_id(self) -> Option<QueueAdmissionId> {
        match self {
            Self::Admit(id) => id,
            Self::Defer(id) => Some(id),
        }
    }
}

/// Host-owned state for one [`QueueAdmissionPolicy`].
///
/// `T` is the scheduler's opaque request payload. The layer only stores and
/// returns it; it never inspects scheduling fields.
pub(crate) struct QueueAdmissionLayer<T> {
    policy: Box<dyn QueueAdmissionPolicy>,
    /// Unordered, non-runnable holding storage for deferred payloads. It is not
    /// a second queue: nothing here is ordered, counted, or dispatchable.
    deferred: FxHashMap<QueueAdmissionId, T>,
    /// Reused buffer for the IDs a policy releases, so a wake-up costs no
    /// allocation on a steady-state path.
    ready_scratch: Vec<QueueAdmissionId>,
    next_id: u64,
    /// Every request that reached this layer, including bypassed work. Owning
    /// lifecycle state is what makes a duplicate request ID detectable and a
    /// dropped stream releasable.
    lifecycle_request_ids: HashSet<String>,
    /// The subset the policy still manages, and therefore the only requests
    /// that receive a terminal lifecycle event.
    managed_request_ids: HashSet<String>,
    /// Worker each dispatched request was booked on, bypassed work included.
    /// A request appears here only once it has been dispatched, and its absence
    /// is what makes a worker-qualified terminal event a non-match.
    bookings: HashMap<String, WorkerWithDpRank>,
}

impl<T> QueueAdmissionLayer<T> {
    pub(crate) fn new(policy: Box<dyn QueueAdmissionPolicy>) -> Self {
        Self {
            policy,
            deferred: FxHashMap::default(),
            ready_scratch: Vec::new(),
            next_id: 0,
            lifecycle_request_ids: HashSet::new(),
            managed_request_ids: HashSet::new(),
            bookings: HashMap::new(),
        }
    }

    /// The policy's maximum reconciliation interval, ignoring a zero duration.
    pub(crate) fn reconcile_interval(&self) -> Option<Duration> {
        self.policy
            .reconcile_interval()
            .filter(|interval| !interval.is_zero())
    }

    /// Start tracking one request ID, or report that it is already active.
    pub(crate) fn begin_lifecycle(&mut self, request_id: &str) -> bool {
        self.lifecycle_request_ids.insert(request_id.to_owned())
    }

    /// Ask the policy what to do with one arriving request.
    ///
    /// The caller has already registered the request with
    /// [`Self::begin_lifecycle`]. Only a managed decision records the request
    /// as one this policy owns.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decide(
        &mut self,
        request_id: &str,
        context_tokens: usize,
        session_context: Option<&SessionContext>,
        worker_snapshot: &QueueAdmissionWorkerSnapshot,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<&HashSet<WorkerId>>,
        has_hard_constraints: bool,
        is_worker_eligible: &dyn Fn(WorkerWithDpRank) -> bool,
    ) -> AdmissionHandoff {
        let id = QueueAdmissionId::new(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let decision = self
            .policy
            .admit(QueueAdmissionRequest::new_with_eligibility(
                id,
                request_id,
                context_tokens,
                session_context,
                worker_snapshot,
                pinned_worker,
                allowed_worker_ids,
                has_hard_constraints,
                is_worker_eligible,
            ));
        if matches!(decision, QueueAdmissionDecision::Bypass) {
            return AdmissionHandoff::Admit(None);
        }

        let inserted = self.managed_request_ids.insert(request_id.to_owned());
        debug_assert!(inserted, "duplicate managed admission request ID");
        match decision {
            QueueAdmissionDecision::Defer => AdmissionHandoff::Defer(id),
            _ => AdmissionHandoff::Admit(Some(id)),
        }
    }

    /// Park one payload until the policy wakes it.
    pub(crate) fn defer(&mut self, id: QueueAdmissionId, payload: T) {
        let replaced = self.deferred.insert(id, payload);
        debug_assert!(replaced.is_none(), "duplicate queue admission ID");
    }

    /// Deliver one lifecycle event and collect every payload the policy released.
    ///
    /// A woken payload carries no class, deadline, or accounting, so the caller
    /// puts it through the ordinary policy-class admission path.
    pub(crate) fn on_event(&mut self, event: QueueAdmissionEvent<'_>, woken: &mut Vec<T>) {
        let mut ready = std::mem::take(&mut self.ready_scratch);
        ready.clear();
        self.policy.on_event(event, &mut ready);
        for id in ready.drain(..) {
            match self.deferred.remove(&id) {
                Some(payload) => woken.push(payload),
                None => tracing::debug!(
                    queue_admission_id = id.get(),
                    "Ignoring unknown queue wake-up"
                ),
            }
        }
        self.ready_scratch = ready;
    }

    /// Record the worker one request was booked on, and report the dispatch to
    /// the policy when the policy manages that request.
    ///
    /// The booking is host state and is recorded for every tracked request that
    /// dispatches, bypassed work included, even though the policy never hears
    /// about the bypassed ones. Together with [`Self::finish`] refusing a
    /// worker-qualified event that has no booking, it is what makes a
    /// worker-conditional terminal event safe: without it, a late event from an
    /// abandoned attempt matches on request ID alone and clears the lifecycle of
    /// a newer attempt that reused the ID.
    pub(crate) fn dispatched(
        &mut self,
        request_id: &str,
        id: Option<QueueAdmissionId>,
        worker: WorkerWithDpRank,
        woken: &mut Vec<T>,
    ) {
        if !self.lifecycle_request_ids.contains(request_id) {
            return;
        }
        let previous = self.bookings.insert(request_id.to_owned(), worker);
        debug_assert!(previous.is_none(), "duplicate admission booking request ID");
        if let Some(id) = id {
            self.on_event(QueueAdmissionEvent::Dispatched { id, worker }, woken);
        }
    }

    /// Release one request's lifecycle state and report its terminal abort.
    ///
    /// Lifecycle state belongs to every request this layer saw, including one
    /// the policy bypassed, so it is released unconditionally. Only work the
    /// policy still manages produces an event, and it produces exactly one.
    pub(crate) fn abort(&mut self, request_id: &str, woken: &mut Vec<T>) {
        self.lifecycle_request_ids.remove(request_id);
        self.bookings.remove(request_id);
        if self.managed_request_ids.remove(request_id) {
            self.on_event(QueueAdmissionEvent::Aborted { request_id }, woken);
        }
    }

    /// Report one terminal outcome observed on the response path.
    ///
    /// Returns `None` when this layer does not own `request_id`, when managed
    /// work has no booking yet, or when a worker-qualified event does not match
    /// the booking. Otherwise it returns the booked worker, which is absent only
    /// for an unqualified event on work that has not been dispatched.
    ///
    /// A worker-qualified event requires an existing booking on exactly that
    /// worker. Treating a missing booking as a match would let a late event from
    /// an abandoned attempt clear the lifecycle of a newer attempt that reused
    /// the ID and has not been dispatched yet.
    pub(crate) fn finish(
        &mut self,
        request_id: &str,
        expected_worker: Option<WorkerWithDpRank>,
        outcome: AdmissionRequestOutcome,
        woken: &mut Vec<T>,
    ) -> Option<Option<WorkerWithDpRank>> {
        if !self.lifecycle_request_ids.contains(request_id) {
            return None;
        }
        let managed = self.managed_request_ids.contains(request_id);
        let worker = self.bookings.get(request_id).copied();
        if managed && worker.is_none() {
            return None;
        }
        if expected_worker.is_some() && worker != expected_worker {
            return None;
        }

        self.lifecycle_request_ids.remove(request_id);
        self.bookings.remove(request_id);
        if !self.managed_request_ids.remove(request_id) {
            return Some(worker);
        }
        let event = match outcome {
            AdmissionRequestOutcome::Completed { context_tokens } => {
                QueueAdmissionEvent::Completed {
                    request_id,
                    context_tokens,
                }
            }
            AdmissionRequestOutcome::Aborted => QueueAdmissionEvent::Aborted { request_id },
        };
        self.on_event(event, woken);
        Some(worker)
    }

    /// Drop deferred payloads the host abandoned, so a later wake-up cannot
    /// resurrect work whose caller is already gone.
    pub(crate) fn retain_deferred(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.deferred.retain(|_, payload| keep(payload));
    }

    /// Take every remaining deferred payload, for shutdown.
    pub(crate) fn into_deferred(self) -> impl Iterator<Item = T> {
        self.deferred.into_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defers every request and releases everything it holds on the first
    /// terminal event, so one abort can drive a wake chain.
    #[derive(Default)]
    struct DeferUntilTerminal {
        deferred: Vec<QueueAdmissionId>,
        dispatched: Vec<WorkerWithDpRank>,
        aborts: usize,
        completions: usize,
    }

    impl QueueAdmissionPolicy for DeferUntilTerminal {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            if request.request_id().starts_with("bypass-") {
                return QueueAdmissionDecision::Bypass;
            }
            if request.request_id().starts_with("ready-") {
                return QueueAdmissionDecision::Ready;
            }
            self.deferred.push(request.id());
            QueueAdmissionDecision::Defer
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
            match event {
                QueueAdmissionEvent::Dispatched { worker, .. } => self.dispatched.push(worker),
                QueueAdmissionEvent::Aborted { .. } => {
                    self.aborts += 1;
                    ready.append(&mut self.deferred);
                }
                QueueAdmissionEvent::Completed { .. } => {
                    self.completions += 1;
                    ready.append(&mut self.deferred);
                }
                _ => {}
            }
        }
    }

    fn snapshot() -> QueueAdmissionWorkerSnapshot {
        QueueAdmissionWorkerSnapshot::new(1, Vec::new())
    }

    fn decide<T>(
        layer: &mut QueueAdmissionLayer<T>,
        request_id: &str,
        snapshot: &QueueAdmissionWorkerSnapshot,
    ) -> AdmissionHandoff {
        assert!(layer.begin_lifecycle(request_id));
        layer.decide(request_id, 32, None, snapshot, None, None, false, &|_| true)
    }

    #[test]
    fn bypass_hands_the_request_over_without_managing_it() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();

        assert_eq!(
            decide(&mut layer, "bypass-request", &snapshot),
            AdmissionHandoff::Admit(None),
            "a bypassed request carries no admission identity"
        );
        assert_eq!(layer.deferred.len(), 0);

        // Lifecycle state still belongs to the host, but the policy owns
        // nothing, so an abort reports no terminal event.
        let mut woken = Vec::new();
        layer.abort("bypass-request", &mut woken);
        assert!(woken.is_empty());
        assert!(!layer.lifecycle_request_ids.contains("bypass-request"));
    }

    #[test]
    fn ready_hands_the_request_over_under_a_managed_identity() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();

        let AdmissionHandoff::Admit(Some(_)) = decide(&mut layer, "ready-request", &snapshot)
        else {
            panic!("a ready request is handed over under a managed identity");
        };
        assert_eq!(layer.deferred.len(), 0);
        assert!(layer.managed_request_ids.contains("ready-request"));
    }

    #[test]
    fn deferred_payload_is_retained_until_the_policy_wakes_it() {
        let mut layer = QueueAdmissionLayer::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();

        let AdmissionHandoff::Defer(id) = decide(&mut layer, "defer-request", &snapshot) else {
            panic!("the policy deferred this request");
        };
        layer.defer(id, "payload");
        assert_eq!(layer.deferred.len(), 1);

        // Only a terminal event releases it, and it comes back as the same
        // opaque payload the host parked.
        let mut woken = Vec::new();
        layer.on_event(
            QueueAdmissionEvent::Reconcile {
                snapshot: &snapshot,
            },
            &mut woken,
        );
        assert!(woken.is_empty());

        layer.abort("defer-request", &mut woken);
        assert_eq!(woken, ["payload"]);
        assert_eq!(layer.deferred.len(), 0);
    }

    #[test]
    fn a_terminal_event_is_reported_exactly_once_for_managed_work() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();
        let worker = WorkerWithDpRank::new(3, 0);

        let AdmissionHandoff::Admit(Some(id)) = decide(&mut layer, "ready-request", &snapshot)
        else {
            panic!("a ready request is handed over under a managed identity");
        };
        let mut woken = Vec::new();
        layer.dispatched("ready-request", Some(id), worker, &mut woken);

        assert_eq!(
            layer.finish(
                "ready-request",
                Some(worker),
                AdmissionRequestOutcome::Completed {
                    context_tokens: Some(48),
                },
                &mut woken,
            ),
            Some(Some(worker))
        );
        // The second report finds no lifecycle state, so the policy sees one
        // terminal event rather than two.
        assert_eq!(
            layer.finish(
                "ready-request",
                Some(worker),
                AdmissionRequestOutcome::Aborted,
                &mut woken,
            ),
            None
        );
        layer.abort("ready-request", &mut woken);
        assert!(woken.is_empty());
    }

    #[test]
    fn a_mismatched_expected_worker_is_not_a_terminal_event() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();
        let booked = WorkerWithDpRank::new(1, 0);

        let AdmissionHandoff::Admit(Some(id)) = decide(&mut layer, "ready-request", &snapshot)
        else {
            panic!("a ready request is handed over under a managed identity");
        };
        let mut woken = Vec::new();
        layer.dispatched("ready-request", Some(id), booked, &mut woken);

        assert_eq!(
            layer.finish(
                "ready-request",
                Some(WorkerWithDpRank::new(2, 0)),
                AdmissionRequestOutcome::Aborted,
                &mut woken,
            ),
            None
        );
        assert_eq!(
            layer.finish(
                "ready-request",
                Some(booked),
                AdmissionRequestOutcome::Aborted,
                &mut woken,
            ),
            Some(Some(booked))
        );
    }

    #[test]
    fn a_bypassed_request_records_a_booking_without_telling_the_policy() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();
        let stale = WorkerWithDpRank::new(9, 0);
        let booked = WorkerWithDpRank::new(4, 0);

        // This attempt reuses an ID an earlier attempt already released, so a
        // terminal event naming the earlier attempt's worker may still be in
        // flight while this one is only queued.
        assert_eq!(
            decide(&mut layer, "bypass-request", &snapshot),
            AdmissionHandoff::Admit(None)
        );
        let mut woken = Vec::new();

        // No booking yet, so a worker-qualified event cannot match. Treating the
        // absent booking as a match would clear this attempt's lifecycle before
        // it ever ran.
        assert_eq!(
            layer.finish(
                "bypass-request",
                Some(stale),
                AdmissionRequestOutcome::Completed {
                    context_tokens: None,
                },
                &mut woken,
            ),
            None
        );
        assert!(
            layer.lifecycle_request_ids.contains("bypass-request"),
            "a stale worker-qualified event must leave the newer attempt intact"
        );

        layer.dispatched("bypass-request", None, booked, &mut woken);

        // The policy does not manage bypassed work, so it hears nothing.
        assert!(woken.is_empty());
        assert_eq!(layer.bookings.get("bypass-request"), Some(&booked));

        // The booking keeps refusing the stale worker...
        assert_eq!(
            layer.finish(
                "bypass-request",
                Some(stale),
                AdmissionRequestOutcome::Aborted,
                &mut woken,
            ),
            None
        );
        assert!(layer.lifecycle_request_ids.contains("bypass-request"));

        // ...while this attempt's own terminal event completes it.
        assert_eq!(
            layer.finish(
                "bypass-request",
                Some(booked),
                AdmissionRequestOutcome::Completed {
                    context_tokens: None,
                },
                &mut woken,
            ),
            Some(Some(booked))
        );
        assert!(!layer.lifecycle_request_ids.contains("bypass-request"));
        assert!(woken.is_empty());
    }

    #[test]
    fn an_unqualified_terminal_event_still_completes_work_that_never_dispatched() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();

        assert_eq!(
            decide(&mut layer, "bypass-request", &snapshot),
            AdmissionHandoff::Admit(None)
        );

        // An event that names no worker makes no claim about which attempt it
        // belongs to, so it keeps its existing meaning: it releases the request.
        let mut woken = Vec::new();
        assert_eq!(
            layer.finish(
                "bypass-request",
                None,
                AdmissionRequestOutcome::Aborted,
                &mut woken,
            ),
            Some(None)
        );
        assert!(!layer.lifecycle_request_ids.contains("bypass-request"));
        assert!(woken.is_empty());
    }

    #[test]
    fn a_duplicate_request_id_is_refused_while_the_first_is_active() {
        let mut layer = QueueAdmissionLayer::<&str>::new(Box::new(DeferUntilTerminal::default()));

        assert!(layer.begin_lifecycle("request"));
        assert!(!layer.begin_lifecycle("request"));

        let mut woken = Vec::new();
        layer.abort("request", &mut woken);
        assert!(layer.begin_lifecycle("request"));
    }

    #[test]
    fn retaining_drops_abandoned_deferred_payloads() {
        let mut layer = QueueAdmissionLayer::new(Box::new(DeferUntilTerminal::default()));
        let snapshot = snapshot();

        for request_id in ["defer-keep", "defer-drop"] {
            let AdmissionHandoff::Defer(id) = decide(&mut layer, request_id, &snapshot) else {
                panic!("the policy deferred this request");
            };
            layer.defer(id, request_id);
        }
        assert_eq!(layer.deferred.len(), 2);

        layer.retain_deferred(|payload| *payload != "defer-drop");

        // The dropped payload is gone even though the policy still lists its ID,
        // and the unknown wake-up is ignored rather than resurrected.
        let mut woken = Vec::new();
        layer.abort("defer-keep", &mut woken);
        assert_eq!(woken, ["defer-keep"]);
        assert_eq!(layer.deferred.len(), 0);
    }
}
