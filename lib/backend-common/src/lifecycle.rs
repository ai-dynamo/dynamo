// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-owned drain/resume lifecycle control.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::engine_routes::{
    EngineRouteCallback, EngineRouteMethod, EngineRouteRegistration, EngineRouteRegistry,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use parking_lot::RwLock;
use serde::Serialize;
use tokio::sync::{Mutex, Notify, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::disagg::DisaggregationMode;
use crate::worker::EngineKind;

const QUIESCENCE_POLL_INTERVAL: Duration = Duration::from_millis(250);

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WorkerLifecycleState {
    Serving = 0,
    Draining = 1,
    Drained = 2,
    Stopping = 3,
}

impl WorkerLifecycleState {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Serving,
            1 => Self::Draining,
            2 => Self::Drained,
            3 => Self::Stopping,
            _ => unreachable!("invalid worker lifecycle state {value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct WorkerLifecycleStatus {
    state: WorkerLifecycleState,
    inflight_requests: u64,
    discovery_registered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

#[async_trait]
trait DiscoveryRegistration: Send + Sync {
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
trait QuiescenceCheck: Send + Sync {
    async fn is_quiescent(&self) -> Result<Option<bool>>;
}

#[async_trait]
impl QuiescenceCheck for EngineKind {
    async fn is_quiescent(&self) -> Result<Option<bool>> {
        Ok(EngineKind::is_quiescent(self).await?)
    }
}

/// Shared request-admission and in-flight tracker.
///
/// Admission and the in-flight count share one atomic word, so closing
/// admission is linearizable with request acquisition.
#[derive(Debug)]
pub(crate) struct RequestTracker {
    state: AtomicU64,
    changed: Notify,
}

const ACCEPTING_BIT: u64 = 1 << 63;
const INFLIGHT_MASK: u64 = !ACCEPTING_BIT;

impl RequestTracker {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(ACCEPTING_BIT),
            changed: Notify::new(),
        })
    }

    pub(crate) fn try_acquire(self: &Arc<Self>) -> Result<RequestGuard> {
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

    pub(crate) fn stop_accepting(&self) {
        self.state.fetch_and(INFLIGHT_MASK, Ordering::AcqRel);
        self.changed.notify_waiters();
    }

    fn start_accepting(&self) {
        self.state.fetch_or(ACCEPTING_BIT, Ordering::AcqRel);
        self.changed.notify_waiters();
    }

    fn is_accepting(&self) -> bool {
        self.state.load(Ordering::Acquire) & ACCEPTING_BIT != 0
    }

    pub(crate) fn inflight(&self) -> u64 {
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

pub(crate) struct RequestGuard {
    tracker: Arc<RequestTracker>,
}

struct DrainTask {
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

pub(crate) struct AdminRouteRegistration {
    _routes: Vec<EngineRouteRegistration>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.tracker.release();
    }
}

/// Coordinates the Admin API, request admission, discovery, and SIGTERM.
pub(crate) struct WorkerLifecycleController {
    state: AtomicU8,
    discovery_registered: AtomicBool,
    last_error: RwLock<Option<String>>,
    operation_lock: Mutex<()>,
    drain_task: Mutex<Option<DrainTask>>,
    drain_generation: AtomicU64,
    tracker: Arc<RequestTracker>,
    discovery: Arc<dyn DiscoveryRegistration>,
    quiescence: Arc<dyn QuiescenceCheck>,
    mode: DisaggregationMode,
    discovery_grace_period: Duration,
}

impl WorkerLifecycleController {
    pub(crate) fn new(
        tracker: Arc<RequestTracker>,
        endpoint: Endpoint,
        engine: EngineKind,
        mode: DisaggregationMode,
        discovery_grace_period: Duration,
    ) -> Arc<Self> {
        Self::with_dependencies(
            tracker,
            Arc::new(endpoint),
            Arc::new(engine),
            mode,
            discovery_grace_period,
        )
    }

    fn with_dependencies(
        tracker: Arc<RequestTracker>,
        discovery: Arc<dyn DiscoveryRegistration>,
        quiescence: Arc<dyn QuiescenceCheck>,
        mode: DisaggregationMode,
        discovery_grace_period: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(WorkerLifecycleState::Serving as u8),
            discovery_registered: AtomicBool::new(true),
            last_error: RwLock::new(None),
            operation_lock: Mutex::new(()),
            drain_task: Mutex::new(None),
            drain_generation: AtomicU64::new(0),
            tracker,
            discovery,
            quiescence,
            mode,
            discovery_grace_period,
        })
    }

    fn state(&self) -> WorkerLifecycleState {
        WorkerLifecycleState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub(crate) fn status(&self) -> WorkerLifecycleStatus {
        let state = self.state();
        WorkerLifecycleStatus {
            state,
            inflight_requests: self.tracker.inflight(),
            discovery_registered: self.discovery_registered.load(Ordering::Acquire),
            last_error: self.last_error.read().clone(),
        }
    }

    pub(crate) fn register_admin_routes(
        self: &Arc<Self>,
        registry: &EngineRouteRegistry,
    ) -> Result<AdminRouteRegistration> {
        let routes = registry.try_register_scoped_methods(vec![
            (
                "drain",
                EngineRouteMethod::Post,
                lifecycle_callback(Arc::clone(self), LifecycleAction::Drain),
            ),
            (
                "resume",
                EngineRouteMethod::Post,
                lifecycle_callback(Arc::clone(self), LifecycleAction::Resume),
            ),
            (
                "status",
                EngineRouteMethod::Get,
                lifecycle_callback(Arc::clone(self), LifecycleAction::Status),
            ),
        ])?;
        tracing::info!("registered worker Admin API routes under /engine");
        Ok(AdminRouteRegistration { _routes: routes })
    }

    pub(crate) async fn drain(self: &Arc<Self>) -> Result<WorkerLifecycleStatus> {
        let operation = self.operation_lock.lock().await;
        let started = match self.state() {
            WorkerLifecycleState::Serving => {
                self.last_error.write().take();
                self.state
                    .store(WorkerLifecycleState::Draining as u8, Ordering::Release);
                let generation = self.drain_generation.fetch_add(1, Ordering::AcqRel) + 1;
                let (started_tx, started_rx) = oneshot::channel();
                let cancel = CancellationToken::new();
                let controller = Arc::clone(self);
                let task_cancel = cancel.clone();
                let handle = tokio::spawn(async move {
                    controller
                        .run_drain_operation(generation, started_tx, task_cancel)
                        .await;
                });
                let previous = self
                    .drain_task
                    .lock()
                    .await
                    .replace(DrainTask { cancel, handle });
                debug_assert!(
                    previous
                        .as_ref()
                        .is_none_or(|task| task.handle.is_finished()),
                    "serving worker must not have active drain work"
                );
                Some(started_rx)
            }
            WorkerLifecycleState::Draining => None,
            WorkerLifecycleState::Drained => return Ok(self.status()),
            WorkerLifecycleState::Stopping => bail!("worker shutdown is already in progress"),
        };
        drop(operation);

        if let Some(started) = started {
            started
                .await
                .context("drain operation ended before discovery was updated")??;
        }
        Ok(self.status())
    }

    /// Own the drain after the HTTP handler starts it. The task is detached
    /// from the request so cancelling the client cannot strand `Draining`.
    async fn run_drain_operation(
        self: Arc<Self>,
        generation: u64,
        started: oneshot::Sender<Result<()>>,
        cancel: CancellationToken,
    ) {
        let mut started = Some(started);
        {
            let _operation = tokio::select! {
                _ = cancel.cancelled() => return,
                operation = self.operation_lock.lock() => operation,
            };
            if self.state() != WorkerLifecycleState::Draining
                || self.drain_generation.load(Ordering::Acquire) != generation
            {
                if let Some(started) = started.take() {
                    let _ = started.send(Err(anyhow!(
                        "drain operation was superseded before discovery was updated"
                    )));
                }
                return;
            }

            if self.discovery_registered.load(Ordering::Acquire) {
                if let Err(error) = self.discovery.unregister().await {
                    let message = format!("failed to unregister worker from discovery: {error}");
                    *self.last_error.write() = Some(message);
                    self.drain_generation.fetch_add(1, Ordering::AcqRel);
                    self.state
                        .store(WorkerLifecycleState::Serving as u8, Ordering::Release);
                    self.tracker.start_accepting();
                    if let Some(started) = started.take() {
                        let _ = started
                            .send(Err(error).context("failed to unregister worker from discovery"));
                    }
                    return;
                }
                self.discovery_registered.store(false, Ordering::Release);
            }

            if let Some(started) = started.take() {
                let _ = started.send(Ok(()));
            }
        }

        // Keep accepting requests already selected by frontends until their
        // discovery views have had time to observe the unregister.
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(self.discovery_grace_period) => {}
        }
        {
            let _operation = tokio::select! {
                _ = cancel.cancelled() => return,
                operation = self.operation_lock.lock() => operation,
            };
            if self.state() != WorkerLifecycleState::Draining
                || self.drain_generation.load(Ordering::Acquire) != generation
            {
                return;
            }
            self.tracker.stop_accepting();
        }

        self.monitor_drain(generation, cancel).await;
    }

    pub(crate) async fn resume(self: &Arc<Self>) -> Result<WorkerLifecycleStatus> {
        let (completed_tx, completed_rx) = oneshot::channel();
        let controller = Arc::clone(self);
        tokio::spawn(async move {
            let result = controller.run_resume_operation().await;
            let _ = completed_tx.send(result);
        });
        completed_rx
            .await
            .context("resume operation ended before publishing its result")?
    }

    /// Resume is controller-owned so cancellation of the Admin request cannot
    /// leave admission and discovery in different states.
    async fn run_resume_operation(self: Arc<Self>) -> Result<WorkerLifecycleStatus> {
        let _operation = self.operation_lock.lock().await;
        let previous_state = match self.state() {
            WorkerLifecycleState::Serving => return Ok(self.status()),
            WorkerLifecycleState::Stopping => bail!("worker shutdown is already in progress"),
            state @ (WorkerLifecycleState::Draining | WorkerLifecycleState::Drained) => state,
        };

        let generation = self.drain_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.state
            .store(WorkerLifecycleState::Draining as u8, Ordering::Release);
        self.cancel_and_join_drain_task().await;

        // A worker must be able to accept requests before discovery can make
        // it visible. Roll this back if registration fails.
        self.tracker.start_accepting();

        if !self.discovery_registered.load(Ordering::Acquire) {
            if let Err(error) = self.discovery.register().await {
                let message = format!("failed to re-register worker in discovery: {error}");
                *self.last_error.write() = Some(message);
                self.tracker.stop_accepting();
                self.state.store(previous_state as u8, Ordering::Release);
                if previous_state == WorkerLifecycleState::Draining {
                    self.start_monitor_task(generation).await;
                }
                return Err(error).context("failed to re-register worker in discovery");
            }
            self.discovery_registered.store(true, Ordering::Release);
        }

        self.last_error.write().take();
        self.state
            .store(WorkerLifecycleState::Serving as u8, Ordering::Release);
        Ok(self.status())
    }

    /// Move into the irreversible SIGTERM path using the same discovery-first
    /// ordering as Admin drain.
    pub(crate) async fn begin_shutdown(&self) {
        let needs_convergence_grace = {
            let _operation = self.operation_lock.lock().await;
            self.drain_generation.fetch_add(1, Ordering::AcqRel);
            self.state
                .store(WorkerLifecycleState::Stopping as u8, Ordering::Release);
            self.cancel_and_join_drain_task().await;

            if self.discovery_registered.load(Ordering::Acquire) {
                if let Err(error) = self.discovery.unregister().await {
                    tracing::warn!(%error, "discovery unregister failed during shutdown");
                    *self.last_error.write() = Some(error.to_string());
                } else {
                    self.discovery_registered.store(false, Ordering::Release);
                }
            }
            self.tracker.is_accepting()
        };

        if needs_convergence_grace {
            tokio::time::sleep(self.discovery_grace_period).await;
        }
        self.tracker.stop_accepting();
    }

    async fn cancel_and_join_drain_task(&self) {
        let Some(task) = self.drain_task.lock().await.take() else {
            return;
        };
        task.cancel.cancel();
        if let Err(error) = task.handle.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "drain task failed while being joined");
        }
    }

    async fn start_monitor_task(self: &Arc<Self>, generation: u64) {
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let controller = Arc::clone(self);
        let handle = tokio::spawn(async move {
            controller.monitor_drain(generation, task_cancel).await;
        });
        let previous = self
            .drain_task
            .lock()
            .await
            .replace(DrainTask { cancel, handle });
        debug_assert!(previous.is_none(), "drain task slot must be empty");
    }

    async fn monitor_drain(self: Arc<Self>, generation: u64, cancel: CancellationToken) {
        loop {
            if cancel.is_cancelled() {
                return;
            }
            if self.state() != WorkerLifecycleState::Draining
                || self.drain_generation.load(Ordering::Acquire) != generation
            {
                return;
            }

            if self.tracker.inflight() == 0 {
                let quiescence = if self.mode.is_prefill() {
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        result = self.quiescence.is_quiescent() => result,
                    }
                } else {
                    Ok(Some(true))
                };

                // Publish Drained under the same operation fence used by
                // resume. A stale monitor may finish an engine
                // check, but it can never publish after its generation ends.
                let _operation = tokio::select! {
                    _ = cancel.cancelled() => return,
                    operation = self.operation_lock.lock() => operation,
                };
                if self.state() != WorkerLifecycleState::Draining
                    || self.drain_generation.load(Ordering::Acquire) != generation
                {
                    return;
                }

                match quiescence {
                    Ok(Some(true)) => {
                        self.state
                            .store(WorkerLifecycleState::Drained as u8, Ordering::Release);
                        tracing::info!("worker drained and is safe to delete");
                        return;
                    }
                    Ok(Some(false)) | Ok(None) => {}
                    Err(error) => {
                        *self.last_error.write() = Some(error.to_string());
                    }
                }
            }

            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = self.tracker.changed.notified() => {}
                _ = tokio::time::sleep(QUIESCENCE_POLL_INTERVAL) => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
enum LifecycleAction {
    Drain,
    Resume,
    Status,
}

fn lifecycle_callback(
    controller: Arc<WorkerLifecycleController>,
    action: LifecycleAction,
) -> EngineRouteCallback {
    Arc::new(move |body| {
        let controller = Arc::clone(&controller);
        Box::pin(async move {
            if !body.is_object() {
                bail!("worker Admin API request body must be a JSON object");
            }
            let status = match action {
                LifecycleAction::Drain => controller.drain().await?,
                LifecycleAction::Resume => controller.resume().await?,
                LifecycleAction::Status => controller.status(),
            };
            serde_json::to_value(status).context("serialize worker lifecycle status")
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Semaphore;

    struct MockDiscovery {
        unregister_started: Semaphore,
        unregister_gate: Semaphore,
        register_started: Semaphore,
        register_gate: Semaphore,
        register_fails: AtomicBool,
    }

    impl MockDiscovery {
        fn new(unregister_permits: usize) -> Arc<Self> {
            Self::with_register_permits(unregister_permits, 1)
        }

        fn with_register_permits(unregister_permits: usize, register_permits: usize) -> Arc<Self> {
            Arc::new(Self {
                unregister_started: Semaphore::new(0),
                unregister_gate: Semaphore::new(unregister_permits),
                register_started: Semaphore::new(0),
                register_gate: Semaphore::new(register_permits),
                register_fails: AtomicBool::new(false),
            })
        }
    }

    #[async_trait]
    impl DiscoveryRegistration for MockDiscovery {
        async fn unregister(&self) -> Result<()> {
            self.unregister_started.add_permits(1);
            self.unregister_gate
                .acquire()
                .await
                .expect("unregister gate should stay open")
                .forget();
            Ok(())
        }

        async fn register(&self) -> Result<()> {
            self.register_started.add_permits(1);
            self.register_gate
                .acquire()
                .await
                .expect("register gate should stay open")
                .forget();
            if self.register_fails.load(Ordering::Acquire) {
                bail!("injected register failure");
            }
            Ok(())
        }
    }

    struct MockQuiescence {
        result: Option<bool>,
        check_started: Semaphore,
        check_gate: Semaphore,
    }

    impl MockQuiescence {
        fn new(result: Option<bool>, check_permits: usize) -> Arc<Self> {
            Arc::new(Self {
                result,
                check_started: Semaphore::new(0),
                check_gate: Semaphore::new(check_permits),
            })
        }
    }

    #[async_trait]
    impl QuiescenceCheck for MockQuiescence {
        async fn is_quiescent(&self) -> Result<Option<bool>> {
            self.check_started.add_permits(1);
            self.check_gate
                .acquire()
                .await
                .expect("quiescence gate should stay open")
                .forget();
            Ok(self.result)
        }
    }

    fn test_controller(
        discovery: Arc<MockDiscovery>,
        quiescence: Arc<MockQuiescence>,
        mode: DisaggregationMode,
        grace_period: Duration,
    ) -> (Arc<WorkerLifecycleController>, Arc<RequestTracker>) {
        let tracker = RequestTracker::new();
        let controller = WorkerLifecycleController::with_dependencies(
            Arc::clone(&tracker),
            discovery,
            quiescence,
            mode,
            grace_period,
        );
        (controller, tracker)
    }

    async fn yield_to_background_tasks() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    async fn wait_for_state(
        controller: &WorkerLifecycleController,
        expected: WorkerLifecycleState,
    ) {
        for _ in 0..20 {
            if controller.status().state == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "worker did not reach {expected:?}; current state is {:?}",
            controller.status().state
        );
    }

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
    fn request_tracker_accepts_again_after_resume() {
        let tracker = RequestTracker::new();
        tracker.stop_accepting();
        tracker.start_accepting();
        let guard = tracker.try_acquire().unwrap();
        assert_eq!(tracker.inflight(), 1);
        drop(guard);
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

    #[test]
    fn second_lifecycle_controller_cannot_replace_admin_routes() {
        let registry = EngineRouteRegistry::new();
        let (first, _) = test_controller(
            MockDiscovery::new(1),
            MockQuiescence::new(Some(true), 1),
            DisaggregationMode::Aggregated,
            Duration::ZERO,
        );
        let (second, _) = test_controller(
            MockDiscovery::new(1),
            MockQuiescence::new(Some(true), 1),
            DisaggregationMode::Aggregated,
            Duration::ZERO,
        );

        let first_routes = first.register_admin_routes(&registry).unwrap();
        let error = match second.register_admin_routes(&registry) {
            Ok(_) => panic!("a second lifecycle controller must not replace Admin API routes"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("already registered"));
        assert!(registry.get("drain").is_some());
        assert!(registry.get("resume").is_some());
        assert!(registry.get("status").is_some());

        drop(first_routes);
        assert!(registry.routes().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn drain_keeps_admission_open_during_discovery_grace_period() {
        let discovery = MockDiscovery::new(1);
        let quiescence = MockQuiescence::new(Some(true), 1);
        let grace_period = Duration::from_secs(5);
        let (controller, tracker) = test_controller(
            discovery,
            quiescence,
            DisaggregationMode::Aggregated,
            grace_period,
        );

        let status = controller.drain().await.unwrap();
        assert_eq!(status.state, WorkerLifecycleState::Draining);
        let admitted_during_convergence = tracker
            .try_acquire()
            .expect("late frontend selections should be admitted during convergence");

        yield_to_background_tasks().await;
        tokio::time::advance(grace_period).await;
        yield_to_background_tasks().await;
        assert!(tracker.try_acquire().is_err());

        drop(admitted_during_convergence);
        tokio::time::advance(QUIESCENCE_POLL_INTERVAL).await;
        wait_for_state(&controller, WorkerLifecycleState::Drained).await;
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_drain_caller_does_not_strand_the_worker() {
        let discovery = MockDiscovery::new(0);
        let quiescence = MockQuiescence::new(Some(true), 1);
        let (controller, _) = test_controller(
            Arc::clone(&discovery),
            quiescence,
            DisaggregationMode::Aggregated,
            Duration::ZERO,
        );

        let first_controller = Arc::clone(&controller);
        let first_call = tokio::spawn(async move { first_controller.drain().await });
        discovery
            .unregister_started
            .acquire()
            .await
            .unwrap()
            .forget();
        first_call.abort();
        let _ = first_call.await;

        let retry_controller = Arc::clone(&controller);
        let retry = tokio::spawn(async move { retry_controller.drain().await });
        discovery.unregister_gate.add_permits(1);
        retry.await.unwrap().unwrap();
        wait_for_state(&controller, WorkerLifecycleState::Drained).await;
    }

    #[tokio::test(start_paused = true)]
    async fn stale_monitor_cannot_publish_after_resume() {
        let discovery = MockDiscovery::new(1);
        let quiescence = MockQuiescence::new(Some(true), 0);
        let (controller, _) = test_controller(
            discovery,
            Arc::clone(&quiescence),
            DisaggregationMode::Prefill,
            Duration::ZERO,
        );

        controller.drain().await.unwrap();
        quiescence.check_started.acquire().await.unwrap().forget();

        let resumed = controller.resume().await.unwrap();
        assert_eq!(resumed.state, WorkerLifecycleState::Serving);
        quiescence.check_gate.add_permits(1);
        yield_to_background_tasks().await;
        assert_eq!(
            quiescence.check_gate.available_permits(),
            1,
            "resume must cancel and join the blocked quiescence check"
        );

        let status = controller.status();
        assert_eq!(status.state, WorkerLifecycleState::Serving);
    }

    #[tokio::test(start_paused = true)]
    async fn resume_opens_admission_before_registration_and_survives_caller_cancel() {
        let discovery = MockDiscovery::with_register_permits(1, 0);
        let quiescence = MockQuiescence::new(Some(true), 0);
        let (controller, tracker) = test_controller(
            Arc::clone(&discovery),
            Arc::clone(&quiescence),
            DisaggregationMode::Prefill,
            Duration::ZERO,
        );

        controller.drain().await.unwrap();
        quiescence.check_started.acquire().await.unwrap().forget();

        let resume_controller = Arc::clone(&controller);
        let resume_call = tokio::spawn(async move { resume_controller.resume().await });
        discovery.register_started.acquire().await.unwrap().forget();

        let admitted = tracker
            .try_acquire()
            .expect("admission must be open before discovery registration publishes");
        drop(admitted);

        resume_call.abort();
        let _ = resume_call.await;
        discovery.register_gate.add_permits(1);
        wait_for_state(&controller, WorkerLifecycleState::Serving).await;
        assert!(controller.status().discovery_registered);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_resume_registration_rolls_back_admission_and_monitoring() {
        let discovery = MockDiscovery::new(1);
        discovery.register_fails.store(true, Ordering::Release);
        let quiescence = MockQuiescence::new(Some(true), 0);
        let (controller, tracker) = test_controller(
            discovery,
            Arc::clone(&quiescence),
            DisaggregationMode::Prefill,
            Duration::ZERO,
        );

        controller.drain().await.unwrap();
        quiescence.check_started.acquire().await.unwrap().forget();
        assert!(controller.resume().await.is_err());

        assert_eq!(controller.status().state, WorkerLifecycleState::Draining);
        assert!(tracker.try_acquire().is_err());
        quiescence.check_started.acquire().await.unwrap().forget();
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_unregisters_before_closing_admission() {
        let discovery = MockDiscovery::new(0);
        let quiescence = MockQuiescence::new(Some(true), 1);
        let grace_period = Duration::from_secs(5);
        let (controller, tracker) = test_controller(
            Arc::clone(&discovery),
            quiescence,
            DisaggregationMode::Aggregated,
            grace_period,
        );

        let shutdown_controller = Arc::clone(&controller);
        let shutdown = tokio::spawn(async move { shutdown_controller.begin_shutdown().await });
        discovery
            .unregister_started
            .acquire()
            .await
            .unwrap()
            .forget();
        let during_unregister = tracker
            .try_acquire()
            .expect("SIGTERM must keep admission open while unregistering");
        drop(during_unregister);

        discovery.unregister_gate.add_permits(1);
        yield_to_background_tasks().await;
        let during_convergence = tracker
            .try_acquire()
            .expect("SIGTERM must keep admission open during discovery convergence");
        drop(during_convergence);

        tokio::time::advance(grace_period).await;
        shutdown.await.unwrap();
        assert!(tracker.try_acquire().is_err());
    }
}
