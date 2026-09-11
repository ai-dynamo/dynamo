// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission control and the shutdown stages built on it.
//!
//! [`RequestTracker`] is the admission gate and in-flight counter: one packed
//! atomic word holding "are we accepting" plus the count, so admitting a
//! request costs one CAS and finishing one costs one decrement.
//!
//! The stage functions below are the shutdown sequence, one implementation
//! each. [`crate::worker::Worker`] drives them in order on SIGTERM. They take
//! a [`CancellationToken`] and a [`ShutdownBudget`] that may be unbounded,
//! because the worker Admin API's reversible `drain`/`resume` — landing
//! separately — needs to abandon a stage in flight and has no deadline. The
//! terminal path passes a token nothing cancels and a bounded budget.
//!
//! [`DiscoveryRegistration`] keeps `register` alongside `unregister` for the
//! same reason: the reversible path re-registers on resume.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::error::{DynamoError, ErrorType};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::disagg::DisaggregationMode;
use crate::shutdown::{
    DRAIN_HEARTBEAT_INTERVAL, KvTransferFallback, ShutdownBudget, Stage, StageOutcome, StageReason,
    kv_transfer_fallback_override,
};
use crate::worker::EngineKind;

const QUIESCENCE_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[async_trait]
pub trait DiscoveryRegistration: Send + Sync {
    async fn unregister(&self) -> Result<()>;
    async fn register(&self) -> Result<()>;
}

#[async_trait]
impl DiscoveryRegistration for Endpoint {
    async fn unregister(&self) -> Result<()> {
        self.unregister_endpoint_instance().await
    }

    async fn register(&self) -> Result<()> {
        self.register_endpoint_instance().await
    }
}

#[async_trait]
pub trait QuiescenceCheck: Send + Sync {
    async fn is_quiescent(&self) -> Result<Option<bool>>;

    /// Declared policy for when `is_quiescent` never reports a value.
    fn kv_transfer_fallback(&self) -> KvTransferFallback {
        KvTransferFallback::Undeclared
    }
}

#[async_trait]
impl QuiescenceCheck for EngineKind {
    async fn is_quiescent(&self) -> Result<Option<bool>> {
        Ok(EngineKind::is_quiescent(self).await?)
    }

    fn kv_transfer_fallback(&self) -> KvTransferFallback {
        EngineKind::kv_transfer_fallback(self)
    }
}

/// Shared request-admission and in-flight tracker.
///
/// Admission and the in-flight count share one atomic word, so closing
/// admission is linearizable with request acquisition.
#[derive(Debug)]
pub struct RequestTracker {
    state: AtomicU64,
    changed: Notify,
}

const ACCEPTING_BIT: u64 = 1 << 63;
const INFLIGHT_MASK: u64 = !ACCEPTING_BIT;

impl RequestTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(ACCEPTING_BIT),
            changed: Notify::new(),
        })
    }

    pub fn try_acquire(self: &Arc<Self>) -> Result<RequestGuard> {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            if current & ACCEPTING_BIT == 0 {
                return Err(worker_draining_error(
                    "worker is not accepting new requests",
                ));
            }
            assert!(
                current & INFLIGHT_MASK < INFLIGHT_MASK,
                "request tracker overflow"
            );
            match self.state.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }

        Ok(RequestGuard {
            tracker: Arc::clone(self),
        })
    }

    pub fn stop_accepting(&self) {
        self.state.fetch_and(INFLIGHT_MASK, Ordering::AcqRel);
        self.changed.notify_waiters();
    }

    pub fn inflight(&self) -> u64 {
        self.state.load(Ordering::Acquire) & INFLIGHT_MASK
    }

    fn release(&self) {
        let previous = self.state.fetch_sub(1, Ordering::AcqRel);
        let previous_inflight = previous & INFLIGHT_MASK;
        debug_assert!(previous_inflight > 0, "request tracker underflow");
        if previous_inflight == 1 {
            self.changed.notify_waiters();
        }
    }
}

fn worker_draining_error(message: &'static str) -> anyhow::Error {
    DynamoError::builder()
        .error_type(ErrorType::WorkerDraining)
        .message(message)
        .build()
        .into()
}

pub struct RequestGuard {
    tracker: Arc<RequestTracker>,
}

/// Releasing the guard is what decrements the in-flight count, so the shutdown
/// barrier cannot clear while a request is still running. RAII rather than an
/// explicit call: a request can end by completing, erroring, or having its
/// response stream dropped, and only `Drop` covers all three.
impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.tracker.release();
    }
}

// ---------------------------------------------------------------------------
// Shutdown stages
//
// One implementation per stage, shared by both entry points. Each takes a
// `CancellationToken` so the reversible Admin drain can abandon a stage in
// flight (`resume`), and a `ShutdownBudget` so the terminal SIGTERM path can
// bound it — `ShutdownBudget::unbounded()` makes a stage wait indefinitely,
// which is what a pre-delete drain wants.
//
// Every stage reports a `StageOutcome` rather than `()`, so "we waited 30s"
// and "we waited 30s and gave up" are distinguishable in logs and metrics.
// ---------------------------------------------------------------------------

/// Remove this worker from discovery so routers stop selecting it.
///
/// Bounded by the remaining total, like every other stage. "A worker that
/// cannot unregister must still proceed with the rest of shutdown" only holds
/// if `unregister()` *returns*: it is a CR write against the API server on the
/// kube backend and a JetStream delete on the KV-store backend, and neither has
/// a bound this worker controls. Left unbudgeted, a partitioned API server
/// wedged shutdown here — force-exiting at 70 with the engine never cleaned up
/// on the signal path, and hanging forever holding the GPUs on the paths that
/// never arm the watchdog.
///
/// `Stage::Unregister`'s cap is already `Duration::MAX`, so this costs no new
/// knob: the remaining total is the only bound that applies.
pub async fn stage_unregister(
    discovery: &dyn DiscoveryRegistration,
    budget: &ShutdownBudget,
) -> StageOutcome {
    let started = Instant::now();
    let (reason, detail) =
        match maybe_timeout(budget.allowance(Stage::Unregister), discovery.unregister()).await {
            Ok(Ok(())) => (StageReason::Completed, None),
            Ok(Err(error)) => {
                tracing::warn!(%error, "discovery unregister failed");
                (StageReason::Skipped, Some(error.to_string()))
            }
            Err(()) => {
                tracing::warn!(
                    "discovery unregister exceeded the remaining shutdown budget; \
                 continuing so engine cleanup still runs"
                );
                (StageReason::TimedOut, None)
            }
        };
    let outcome =
        StageOutcome::new(Stage::Unregister, reason, started.elapsed(), budget).with_detail(detail);
    outcome.log();
    outcome
}

/// Keep serving while routers observe the unregister.
///
/// A fixed wait, not a condition: already-routed requests are still arriving,
/// and there is no signal that says every router has caught up.
pub async fn stage_router_grace(
    requested: Option<Duration>,
    budget: &ShutdownBudget,
    cancel: &CancellationToken,
) -> StageOutcome {
    let started = Instant::now();
    // `requested` lets a caller override the configured grace; the budget
    // still caps it, so an over-long grace cannot starve the stages after it.
    let requested = requested.unwrap_or_else(|| budget.stage_max(Stage::RouterGrace));
    let grace = match budget.allowance(Stage::RouterGrace) {
        // Unbounded budgets still get a bounded grace: this stage is a fixed
        // sleep, so "wait forever" is never the intent.
        None => requested.min(budget.stage_max(Stage::RouterGrace)),
        Some(allowance) => requested.min(allowance),
    };
    let reason = if grace.is_zero() {
        StageReason::Skipped
    } else {
        tokio::select! {
            _ = cancel.cancelled() => StageReason::Cancelled,
            _ = tokio::time::sleep(grace) => StageReason::Completed,
        }
    };
    let outcome = StageOutcome::new(Stage::RouterGrace, reason, started.elapsed(), budget);
    outcome.log();
    outcome
}

/// Close admission. Synchronous and always completes: it flips a bit, and
/// requests already admitted keep running.
pub fn stage_stop_admission(tracker: &RequestTracker, budget: &ShutdownBudget) -> StageOutcome {
    let started = Instant::now();
    tracker.stop_accepting();
    let outcome = StageOutcome::new(
        Stage::StopAdmission,
        StageReason::Completed,
        started.elapsed(),
        budget,
    );
    outcome.log();
    outcome
}

/// Wait for admitted requests to finish.
///
/// Event-driven rather than polled: `RequestTracker` notifies on every
/// release, so a clean drain returns as soon as the last request completes.
pub async fn stage_await_inflight(
    tracker: &RequestTracker,
    budget: &ShutdownBudget,
    cancel: &CancellationToken,
) -> StageOutcome {
    let started = Instant::now();
    let allowance = budget.allowance(Stage::Inflight);

    let wait = async {
        loop {
            // Register for the notification before re-reading the count, so a
            // release between the two cannot be missed.
            let changed = tracker.changed.notified();
            if tracker.inflight() == 0 {
                return;
            }
            changed.await;
        }
    };

    let reason = tokio::select! {
        _ = cancel.cancelled() => StageReason::Cancelled,
        result = maybe_timeout(allowance, wait) => match result {
            Ok(()) => StageReason::Completed,
            Err(()) => StageReason::TimedOut,
        },
    };

    let outcome = StageOutcome::new(Stage::Inflight, reason, started.elapsed(), budget);
    if reason == StageReason::TimedOut {
        tracing::warn!(
            inflight = tracker.inflight(),
            "shutdown: requests still in flight when the stage expired"
        );
    }
    outcome.log();
    outcome
}

/// Wait for prefill KV transfers that outlive the request stream.
///
/// Prefill-only: an aggregated or decode worker has no transfer a peer could
/// still be pulling from, so the stage is skipped outright.
///
/// An engine that returns `None` cannot report transfer state. Today that
/// means waiting out the whole budget and reporting `Unsupported`, which is
/// honest but is a fixed delay — engines should implement the predicate, and a
/// declared fallback policy is the follow-up.
pub async fn stage_kv_quiescence(
    quiescence: &dyn QuiescenceCheck,
    mode: DisaggregationMode,
    budget: &ShutdownBudget,
    cancel: &CancellationToken,
) -> StageOutcome {
    let started = Instant::now();
    if !mode.is_prefill() {
        let outcome = StageOutcome::new(
            Stage::KvTransfer,
            StageReason::Skipped,
            started.elapsed(),
            budget,
        );
        outcome.log();
        return outcome;
    }

    // Operator override wins over the engine's declaration: someone who knows
    // their deployment can overrule a conservative engine default.
    // Precedence: programmatic config > environment > engine declaration.
    let policy = budget
        .kv_fallback_override()
        .or_else(kv_transfer_fallback_override)
        .unwrap_or_else(|| quiescence.kv_transfer_fallback());
    if policy == KvTransferFallback::Skip {
        tracing::info!(
            policy = policy.as_str(),
            "shutdown: skipping the KV-transfer stage by declared policy"
        );
        let outcome = StageOutcome::new(
            Stage::KvTransfer,
            StageReason::Skipped,
            started.elapsed(),
            budget,
        );
        outcome.log();
        return outcome;
    }

    let allowance = budget.allowance(Stage::KvTransfer);

    // Tracks whether the engine ever reported a value, so a timeout can say
    // whether we waited on a real signal or on nothing at all.
    let mut ever_reported = false;
    // Surfaced on the outcome so a stalled drain is explainable: the Admin
    // status endpoint reports it as `last_error`.
    let mut last_error: Option<String> = None;
    let poll = async {
        let mut announced = false;
        let mut last_heartbeat = Instant::now();
        loop {
            match quiescence.is_quiescent().await {
                Ok(Some(true)) => return true,
                Ok(Some(false)) => ever_reported = true,
                Ok(None) => {}
                Err(error) => {
                    tracing::debug!(%error, "is_quiescent raised; treating as not quiescent");
                    last_error = Some(error.to_string());
                }
            }
            if !announced {
                announced = true;
                tracing::info!(
                    timeout_s = allowance.map(|d| d.as_secs_f64()).unwrap_or(f64::INFINITY),
                    "shutdown: waiting for prefill KV transfers to quiesce"
                );
            }
            // A drain can legitimately run for minutes; without this it would
            // be indistinguishable from a hang.
            if last_heartbeat.elapsed() >= DRAIN_HEARTBEAT_INTERVAL {
                last_heartbeat = Instant::now();
                tracing::info!(
                    elapsed_s = started.elapsed().as_secs_f64(),
                    "shutdown: still waiting for KV quiescence"
                );
            }
            tokio::time::sleep(QUIESCENCE_POLL_INTERVAL).await;
        }
    };

    let reason = tokio::select! {
        _ = cancel.cancelled() => StageReason::Cancelled,
        result = maybe_timeout(allowance, poll) => match result {
            Ok(true) => StageReason::Completed,
            // `poll` only returns on quiescence, so a non-timeout exit is
            // unreachable; treat defensively as completed.
            Ok(false) => StageReason::Completed,
            Err(()) if ever_reported => StageReason::TimedOut,
            // A zero allowance grants exactly one poll, and a real
            // `is_quiescent` — a PyO3 call, an RPC — is `Pending` on that poll,
            // so `ever_reported` stays false through no fault of the engine.
            // Reporting `Unsupported` there told authors to implement a
            // predicate they had already implemented, and put the wrong
            // `reason` on `dynamo_component_shutdown_stage_seconds`, which
            // could then not distinguish "no introspection" from "no budget".
            Err(()) if allowance.is_some_and(|limit| limit.is_zero()) => StageReason::Skipped,
            Err(()) => StageReason::Unsupported,
        },
    };

    let outcome = StageOutcome::new(Stage::KvTransfer, reason, started.elapsed(), budget)
        .with_detail(last_error);
    if reason == StageReason::Unsupported {
        // Keyed on the *policy*, not the reason. An engine that declared
        // `WaitFullBudget` made this choice deliberately and does not need to
        // be told to declare it; only `Undeclared` is an open question. Both
        // wait — the difference is whether anyone decided that.
        if policy == KvTransferFallback::Undeclared {
            tracing::warn!(
                "shutdown: engine never reported KV-transfer state and declares no \
                 fallback, so the full budget was spent before releasing GPU memory. \
                 Implement LLMEngine::is_quiescent to exit as soon as transfers \
                 finish, or declare kv_transfer_fallback to record the choice."
            );
        } else {
            tracing::info!(
                policy = policy.as_str(),
                "shutdown: engine cannot report KV-transfer state; waited the full \
                 budget per its declared policy"
            );
        }
    }
    outcome.log();
    outcome
}

/// Run `fut` under `allowance`, or unbounded when `None`.
/// `Err(())` means the bound expired.
async fn maybe_timeout<T>(
    allowance: Option<Duration>,
    fut: impl Future<Output = T>,
) -> Result<T, ()> {
    match allowance {
        None => Ok(fut.await),
        Some(limit) => tokio::time::timeout(limit, fut).await.map_err(|_| ()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_tracker_rejects_after_drain_starts() {
        let tracker = RequestTracker::new();
        let guard = tracker.try_acquire().unwrap();
        assert_eq!(tracker.inflight(), 1);

        tracker.stop_accepting();
        assert!(tracker.try_acquire().is_err());
        assert_eq!(tracker.inflight(), 1);

        drop(guard);
        assert_eq!(tracker.inflight(), 0);
    }

    #[test]
    fn request_tracker_uses_worker_draining_error() {
        let tracker = RequestTracker::new();
        tracker.stop_accepting();

        let error = match tracker.try_acquire() {
            Ok(_) => panic!("draining worker must reject new requests"),
            Err(error) => error,
        };
        let error = error
            .downcast_ref::<DynamoError>()
            .expect("drain rejection should preserve its Dynamo error type");
        assert_eq!(error.error_type(), ErrorType::WorkerDraining);
    }

    #[test]
    fn concurrent_admission_is_counted_or_rejected_when_drain_closes() {
        const CONTENDERS: usize = 16;

        let tracker = RequestTracker::new();
        let start = Arc::new(std::sync::Barrier::new(CONTENDERS + 1));
        let admitted = Arc::new(std::sync::Mutex::new(Vec::new()));

        std::thread::scope(|scope| {
            for _ in 0..CONTENDERS {
                let tracker = Arc::clone(&tracker);
                let start = Arc::clone(&start);
                let admitted = Arc::clone(&admitted);
                scope.spawn(move || {
                    start.wait();
                    if let Ok(guard) = tracker.try_acquire() {
                        admitted.lock().unwrap().push(guard);
                    }
                });
            }

            start.wait();
            tracker.stop_accepting();
        });

        let guards = admitted.lock().unwrap();
        assert_eq!(tracker.inflight(), guards.len() as u64);
        assert!(tracker.try_acquire().is_err());
        drop(guards);
        admitted.lock().unwrap().clear();
        assert_eq!(tracker.inflight(), 0);
    }

    struct FlakyDiscovery {
        fail: bool,
    }

    #[async_trait]
    impl DiscoveryRegistration for FlakyDiscovery {
        async fn unregister(&self) -> Result<()> {
            if self.fail {
                anyhow::bail!("discovery is unreachable");
            }
            Ok(())
        }
        async fn register(&self) -> Result<()> {
            Ok(())
        }
    }

    /// The first stage: routers must stop selecting this worker.
    #[tokio::test(start_paused = true)]
    async fn stage_unregister_reports_completion() {
        let outcome = stage_unregister(
            &FlakyDiscovery { fail: false },
            &ShutdownBudget::unbounded(),
        )
        .await;
        assert_eq!(outcome.stage, Stage::Unregister);
        assert_eq!(outcome.reason, StageReason::Completed);
        assert!(outcome.detail.is_none());
    }

    struct HangingDiscovery;

    #[async_trait]
    impl DiscoveryRegistration for HangingDiscovery {
        async fn unregister(&self) -> Result<()> {
            std::future::pending::<()>().await;
            unreachable!("pending never resolves")
        }
        async fn register(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Regression: this stage used to ignore the budget entirely. A discovery
    /// backend that never answers — an unreachable API server, a NATS client
    /// stuck reconnecting — wedged the whole sequence here, so `engine.cleanup()`
    /// never ran and the GPUs were never released. It must give up and let the
    /// rest of shutdown proceed.
    #[tokio::test(start_paused = true)]
    async fn stage_unregister_gives_up_when_discovery_never_answers() {
        let budget = ShutdownBudget::starting_now(Duration::from_secs(10));
        let outcome = stage_unregister(&HangingDiscovery, &budget).await;
        assert_eq!(outcome.reason, StageReason::TimedOut);
    }

    /// A worker that cannot reach discovery must still shut down — reporting
    /// the failure rather than aborting the sequence, since the alternative is
    /// a worker that never releases its GPU.
    #[tokio::test(start_paused = true)]
    async fn stage_unregister_survives_a_discovery_failure() {
        let outcome =
            stage_unregister(&FlakyDiscovery { fail: true }, &ShutdownBudget::unbounded()).await;
        assert_eq!(outcome.reason, StageReason::Skipped);
        assert!(
            outcome.detail.is_some(),
            "the failure must be reported, not swallowed"
        );
    }

    /// Closing admission is synchronous and always succeeds; what matters is
    /// that it actually closes the gate.
    #[tokio::test(start_paused = true)]
    async fn stage_stop_admission_closes_the_gate() {
        let tracker = RequestTracker::new();
        assert!(tracker.try_acquire().is_ok());

        let outcome = stage_stop_admission(&tracker, &ShutdownBudget::unbounded());
        assert_eq!(outcome.stage, Stage::StopAdmission);
        assert_eq!(outcome.reason, StageReason::Completed);
        assert!(
            tracker.try_acquire().is_err(),
            "admission must be closed once the stage reports completion"
        );
    }

    /// An explicit grace overrides the configured cap.
    #[tokio::test(start_paused = true)]
    async fn stage_router_grace_honours_an_explicit_request() {
        let start = tokio::time::Instant::now();
        let outcome = stage_router_grace(
            Some(Duration::from_secs(3)),
            &ShutdownBudget::unbounded(),
            &CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.reason, StageReason::Completed);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    /// ...but the budget still caps it, so an over-long grace cannot starve
    /// the stages that follow.
    #[tokio::test(start_paused = true)]
    async fn stage_router_grace_is_capped_by_the_budget() {
        let budget = ShutdownBudget::starting_now(Duration::from_secs(2));
        let start = tokio::time::Instant::now();
        stage_router_grace(
            Some(Duration::from_secs(3600)),
            &budget,
            &CancellationToken::new(),
        )
        .await;
        assert!(
            start.elapsed() <= Duration::from_secs(2),
            "an over-long grace must be capped, took {:?}",
            start.elapsed()
        );
    }

    /// A zero grace is skipped outright rather than slept for zero.
    #[tokio::test(start_paused = true)]
    async fn stage_router_grace_skips_when_zero() {
        let outcome = stage_router_grace(
            Some(Duration::ZERO),
            &ShutdownBudget::unbounded(),
            &CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.reason, StageReason::Skipped);
    }

    /// Give a spawned task room to reach its await point before asserting on
    /// whether it finished.
    async fn yield_to_background_tasks() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    /// The ordering fix for #13286: the request-plane barrier must run before
    /// anything frees engine memory.
    ///
    /// Before this, `engine.cleanup()` ran while requests were still
    /// executing, so a decode peer could read KV memory the prefill worker had
    /// already released.
    #[tokio::test(start_paused = true)]
    async fn inflight_barrier_completes_before_cleanup_runs() {
        let tracker = RequestTracker::new();

        // One admitted request, still running.
        let guard = tracker.try_acquire().expect("admission is open");
        assert_eq!(tracker.inflight(), 1);

        let budget = ShutdownBudget::unbounded();
        let cancel = CancellationToken::new();
        let waiter = {
            let tracker = Arc::clone(&tracker);
            let cancel = cancel.clone();
            tokio::spawn(async move { stage_await_inflight(&tracker, &budget, &cancel).await })
        };

        yield_to_background_tasks().await;
        assert!(
            !waiter.is_finished(),
            "the barrier must not clear while a request is still in flight"
        );

        // Request finishes.
        drop(guard);
        let outcome = waiter.await.expect("barrier task panicked");

        assert_eq!(outcome.reason, StageReason::Completed);
        assert_eq!(tracker.inflight(), 0);
    }

    /// A bounded budget must not let the barrier hang shutdown forever.
    #[tokio::test(start_paused = true)]
    async fn inflight_barrier_gives_up_when_the_budget_expires() {
        let tracker = RequestTracker::new();

        // Never released: stands in for a request that will not finish.
        let _guard = tracker.try_acquire().expect("admission is open");

        let budget = ShutdownBudget::starting_now(Duration::from_secs(3600));
        let cancel = CancellationToken::new();

        let outcome = tokio::time::timeout(
            Duration::from_secs(7200),
            stage_await_inflight(&tracker, &budget, &cancel),
        )
        .await
        .expect("the barrier must be bounded; it hung past the outer guard");

        assert_eq!(outcome.reason, StageReason::TimedOut);
        assert_eq!(tracker.inflight(), 1, "the stuck request is still counted");
    }

    /// The reversible path must be able to abandon the barrier: `resume`
    /// cancels a drain in flight.
    #[tokio::test(start_paused = true)]
    async fn inflight_barrier_is_cancellable() {
        let tracker = RequestTracker::new();
        let _guard = tracker.try_acquire().expect("admission is open");

        let budget = ShutdownBudget::unbounded();
        let cancel = CancellationToken::new();
        let waiter = {
            let tracker = Arc::clone(&tracker);
            let cancel = cancel.clone();
            tokio::spawn(async move { stage_await_inflight(&tracker, &budget, &cancel).await })
        };

        yield_to_background_tasks().await;
        assert!(!waiter.is_finished());

        cancel.cancel();
        let outcome = waiter.await.expect("barrier task panicked");
        assert_eq!(outcome.reason, StageReason::Cancelled);
    }
}
