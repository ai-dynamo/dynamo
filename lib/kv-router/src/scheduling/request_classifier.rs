// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use async_trait::async_trait;
use futures_util::FutureExt;
use parking_lot::Mutex;
use tokio::runtime::Handle;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio_util::sync::CancellationToken;

use super::types::{KvSchedulerError, SessionContext};
use crate::protocols::WorkerWithDpRank;

static NEXT_CLASSIFICATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct ClassifyRequest {
    classification_id: u64,
    request_id: Option<String>,
    policy_class: Option<String>,
    overrides: ClassificationOverrides,
    ingress_at: Instant,
    caller_deadline: Option<Instant>,
    input_tokens: usize,
    initial_cached_tokens: usize,
    session_context: Option<SessionContext>,
}

#[derive(Clone, Debug, Default)]
struct ClassificationOverrides {
    policy_class: Option<String>,
    due_at: Option<Instant>,
    scheduling_cost_tokens: Option<usize>,
}

impl ClassifyRequest {
    #[cfg(test)]
    pub(crate) fn new(input_tokens: usize, initial_cached_tokens: usize) -> Self {
        Self::with_timing(input_tokens, initial_cached_tokens, Instant::now(), None)
    }

    pub(crate) fn with_timing(
        input_tokens: usize,
        initial_cached_tokens: usize,
        ingress_at: Instant,
        caller_deadline: Option<Instant>,
    ) -> Self {
        Self {
            classification_id: 0,
            request_id: None,
            policy_class: None,
            overrides: ClassificationOverrides::default(),
            ingress_at,
            caller_deadline,
            input_tokens,
            initial_cached_tokens: initial_cached_tokens.min(input_tokens),
            session_context: None,
        }
    }

    pub(crate) fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub(crate) fn with_initial_policy_class(mut self, policy_class: impl Into<String>) -> Self {
        self.policy_class = Some(policy_class.into());
        self
    }

    pub(crate) fn with_session_context(mut self, session_context: SessionContext) -> Self {
        self.session_context = Some(session_context);
        self
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

    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Return the original router ingress time on the monotonic clock.
    pub fn ingress_at(&self) -> Instant {
        self.ingress_at
    }

    /// Return the caller's authoritative deadline, when one was supplied.
    pub fn caller_deadline(&self) -> Option<Instant> {
        self.caller_deadline
    }

    pub fn due_at(&self) -> Option<Instant> {
        self.overrides.due_at
    }

    pub fn set_due_at(&mut self, due_at: Instant) {
        self.overrides.due_at = Some(due_at);
    }

    pub fn scheduling_cost_tokens(&self) -> usize {
        self.overrides.scheduling_cost_tokens.unwrap_or_else(|| {
            self.input_tokens
                .saturating_sub(self.initial_cached_tokens)
                .max(1)
        })
    }

    pub fn set_scheduling_cost_tokens(&mut self, scheduling_cost_tokens: usize) {
        self.overrides.scheduling_cost_tokens = Some(scheduling_cost_tokens);
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    pub(crate) fn into_queue_inputs(
        self,
    ) -> (Option<String>, Option<Instant>, Option<usize>, usize) {
        (
            self.overrides.policy_class,
            self.overrides.due_at,
            self.overrides.scheduling_cost_tokens,
            self.initial_cached_tokens,
        )
    }
}

pub type ClassifierError = dyn Error + Send + Sync + 'static;

#[derive(Debug)]
#[non_exhaustive]
pub enum ClassifyEvent<'a> {
    Sent {
        request_id: &'a str,
        worker: WorkerWithDpRank,
    },
    Responding {
        request_id: &'a str,
        worker: WorkerWithDpRank,
    },
    Completed {
        request_id: &'a str,
        worker: WorkerWithDpRank,
        context_tokens: Option<usize>,
    },
    Aborted {
        request_id: &'a str,
        worker: Option<WorkerWithDpRank>,
        error: Option<&'a ClassifierError>,
    },
}

/// An independently pollable classifier invocation.
///
/// The future is `'static` so it owns its continuation instead of borrowing the
/// classifier mutex while pending. This lets other requests and events proceed.
pub type ClassifyFuture =
    Pin<Box<dyn Future<Output = Result<ClassifyRequest, Box<ClassifierError>>> + Send + 'static>>;

/// User-provided request classifier.
///
/// [`Self::classify`] deliberately avoids `async fn` and returns a `'static`
/// future which the router polls on its own runtime, so a classifier may
/// await, sleep, or park a request without blocking a thread and without
/// standing up a runtime of its own. `async fn` cannot express this: its
/// future borrows `&mut self` for the whole pending wait, which would hold the
/// router-wide classifier lock and block every other request and every event.
///
/// Because the returned future is `'static`, state it touches must be owned or
/// shared rather than borrowed from `self`:
///
/// ```ignore
/// struct Pauser {
///     state: Arc<Mutex<ProgramState>>,
/// }
///
/// impl RequestClassifier for Pauser {
///     fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
///         // Clone the handle into the future; `self` cannot be borrowed.
///         let state = Arc::clone(&self.state);
///         Box::pin(async move {
///             state.lock().await.wait_for_slot().await;
///             Ok(request)
///         })
///     }
/// }
/// ```
///
/// `classify`'s synchronous prologue runs under the router-wide classifier
/// lock: build the future and return promptly, then wait inside it. Blocking
/// before the future is returned stalls classification and event delivery.
///
/// [`Self::on_event`] is `async fn`. A dedicated router task delivers events
/// one at a time in lifecycle order and awaits each callback, so `on_event`
/// may await freely. A pending `classify` future never blocks delivery, but a
/// slow `on_event` delays classification of new requests because both share
/// the classifier lock. Terminal events arrive asynchronously, shortly after
/// the request ends; events still queued at router shutdown are dropped.
#[async_trait]
pub trait RequestClassifier: Send + 'static {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        Box::pin(async move { Ok(request) })
    }

    async fn on_event(&mut self, _event: ClassifyEvent<'_>) {}
}

/// Owned lifecycle event queued between a request task and the delivery task.
enum OwnedEvent {
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
        error: Option<Arc<ClassifierError>>,
    },
}

impl OwnedEvent {
    fn request_id(&self) -> &str {
        match self {
            Self::Sent { request_id, .. }
            | Self::Responding { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Aborted { request_id, .. } => request_id,
        }
    }

    fn as_event(&self) -> ClassifyEvent<'_> {
        match self {
            Self::Sent { request_id, worker } => ClassifyEvent::Sent {
                request_id,
                worker: *worker,
            },
            Self::Responding { request_id, worker } => ClassifyEvent::Responding {
                request_id,
                worker: *worker,
            },
            Self::Completed {
                request_id,
                worker,
                context_tokens,
            } => ClassifyEvent::Completed {
                request_id,
                worker: *worker,
                context_tokens: *context_tokens,
            },
            Self::Aborted {
                request_id,
                worker,
                error,
            } => ClassifyEvent::Aborted {
                request_id,
                worker: *worker,
                error: error.as_deref(),
            },
        }
    }
}

pub(crate) struct RequestClassifierRuntime {
    classifier: Arc<AsyncMutex<Box<dyn RequestClassifier>>>,
    live_requests: Mutex<HashMap<String, Option<ClassificationOverrides>>>,
    events: mpsc::UnboundedSender<OwnedEvent>,
    // Bounded by live requests: at most one queued event per lifecycle phase.
    pending_events: Mutex<Option<mpsc::UnboundedReceiver<OwnedEvent>>>,
    delivery: OnceLock<()>,
    shutdown: CancellationToken,
}

impl RequestClassifierRuntime {
    pub(crate) fn new(
        classifier: Box<dyn RequestClassifier>,
        shutdown: CancellationToken,
    ) -> Arc<Self> {
        let (events, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            classifier: Arc::new(AsyncMutex::new(classifier)),
            live_requests: Mutex::new(HashMap::new()),
            events,
            pending_events: Mutex::new(Some(receiver)),
            delivery: OnceLock::new(),
            shutdown,
        })
    }

    /// Start the event-delivery task once a runtime is available. Events sent
    /// before the first classify or lifecycle registration stay queued.
    fn ensure_delivery(&self) {
        if self.delivery.get().is_some() {
            return;
        }
        let Ok(handle) = Handle::try_current() else {
            return;
        };
        self.delivery.get_or_init(|| {
            let Some(mut events) = self.pending_events.lock().take() else {
                return;
            };
            let classifier = Arc::clone(&self.classifier);
            let shutdown = self.shutdown.clone();
            handle.spawn(async move {
                loop {
                    let event = tokio::select! {
                        biased;
                        _ = shutdown.cancelled() => break,
                        event = events.recv() => match event {
                            Some(event) => event,
                            None => break,
                        },
                    };
                    let mut classifier = classifier.lock().await;
                    if let Err(panic) = AssertUnwindSafe(classifier.on_event(event.as_event()))
                        .catch_unwind()
                        .await
                    {
                        tracing::error!(
                            panic = %panic_message(panic),
                            "Request classifier panicked while processing a lifecycle event"
                        );
                    }
                }
            });
        });
    }

    pub(crate) async fn classify_with(
        &self,
        make_request: impl FnOnce() -> ClassifyRequest,
    ) -> Result<ClassifyRequest, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }
        self.ensure_delivery();

        let mut request = make_request();
        if let Some(overrides) = request.request_id().and_then(|request_id| {
            self.live_requests
                .lock()
                .get(request_id)
                .and_then(Clone::clone)
        }) {
            request.overrides = overrides;
            return Ok(request);
        }

        let deadline = request.caller_deadline();
        let classification_id = NEXT_CLASSIFICATION_ID.fetch_add(1, Ordering::Relaxed);
        request.classification_id = classification_id;
        let classification = {
            let mut classifier = self.classifier.lock().await;
            catch_unwind(AssertUnwindSafe(|| classifier.classify(request))).map_err(|panic| {
                KvSchedulerError::RequestClassifierPanicked(panic_message(panic))
            })?
        };
        let classification = AssertUnwindSafe(classification).catch_unwind();

        let expiry = async move {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending().await,
            }
        };
        let result = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
            _ = expiry => return Err(KvSchedulerError::DueTimeExpired),
            result = classification => result,
        };
        let classified = result
            .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic)))?
            .map_err(|error| KvSchedulerError::RequestClassifierFailed(Arc::from(error)))?;

        if classified.classification_id != classification_id {
            return Err(KvSchedulerError::RequestClassifierReplacedRequest);
        }
        if let Some(request_id) = classified.request_id()
            && let Some(cached) = self.live_requests.lock().get_mut(request_id)
        {
            *cached = Some(classified.overrides.clone());
        }
        Ok(classified)
    }

    pub(crate) fn begin_request(
        self: &Arc<Self>,
        request_id: &str,
    ) -> Result<RequestLifecycle, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }
        self.ensure_delivery();
        match self.live_requests.lock().entry(request_id.to_owned()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(KvSchedulerError::DuplicateClassificationRequestId(
                    request_id.to_owned(),
                ));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(None);
            }
        }
        Ok(RequestLifecycle {
            runtime: Arc::clone(self),
            request_id: request_id.to_owned(),
            worker: None,
            context_tokens: None,
            phase: LifecyclePhase::Registered,
        })
    }

    pub(crate) fn has_request(&self, request_id: &str) -> bool {
        self.live_requests.lock().contains_key(request_id)
    }

    fn send_event(&self, event: OwnedEvent) {
        let _ = self.events.send(event);
    }

    fn finish_request(&self, event: OwnedEvent) {
        if self
            .live_requests
            .lock()
            .remove(event.request_id())
            .is_none()
        {
            return;
        }
        self.send_event(event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecyclePhase {
    Registered,
    Sent,
    Responding,
    Terminal,
}

#[doc(hidden)]
pub struct RequestLifecycle {
    runtime: Arc<RequestClassifierRuntime>,
    request_id: String,
    worker: Option<WorkerWithDpRank>,
    context_tokens: Option<usize>,
    phase: LifecyclePhase,
}

impl std::fmt::Debug for RequestLifecycle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycle")
            .field("request_id", &self.request_id)
            .field("worker", &self.worker)
            .field("phase", &self.phase)
            .finish()
    }
}

#[doc(hidden)]
impl RequestLifecycle {
    pub fn selected(&mut self, worker: WorkerWithDpRank) {
        if self.phase == LifecyclePhase::Registered {
            self.worker = Some(worker);
        }
    }

    pub fn sent(&mut self, worker: WorkerWithDpRank) {
        if self.phase != LifecyclePhase::Registered {
            return;
        }
        self.worker = Some(worker);
        self.phase = LifecyclePhase::Sent;
        self.runtime.send_event(OwnedEvent::Sent {
            request_id: self.request_id.clone(),
            worker,
        });
    }

    pub fn responding(&mut self) {
        if self.phase != LifecyclePhase::Sent {
            return;
        }
        let Some(worker) = self.worker else {
            return;
        };
        self.phase = LifecyclePhase::Responding;
        self.runtime.send_event(OwnedEvent::Responding {
            request_id: self.request_id.clone(),
            worker,
        });
    }

    pub fn observe_output_tokens(&mut self, output_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .unwrap_or_default()
                .saturating_add(output_tokens),
        );
    }

    pub fn observe_context_tokens(&mut self, context_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .map_or(context_tokens, |current| current.max(context_tokens)),
        );
    }

    pub fn prepare_retry(&mut self) {
        if self.phase != LifecyclePhase::Terminal {
            self.phase = LifecyclePhase::Registered;
        }
    }

    pub fn complete(&mut self) {
        if self.phase == LifecyclePhase::Terminal {
            return;
        }
        let Some(worker) = self.worker else {
            self.abort(None);
            return;
        };
        self.phase = LifecyclePhase::Terminal;
        self.runtime.finish_request(OwnedEvent::Completed {
            request_id: std::mem::take(&mut self.request_id),
            worker,
            context_tokens: self.context_tokens,
        });
    }

    pub fn abort(&mut self, error: Option<Arc<ClassifierError>>) {
        if self.phase == LifecyclePhase::Terminal {
            return;
        }
        self.phase = LifecyclePhase::Terminal;
        self.runtime.finish_request(OwnedEvent::Aborted {
            request_id: std::mem::take(&mut self.request_id),
            worker: self.worker,
            error,
        });
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        self.abort(None);
    }
}

fn panic_message(panic: Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "request classifier panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::{Notify, mpsc};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::protocols::WorkerWithDpRank;
    use crate::scheduling::KvSchedulerError;

    struct PassThrough;

    impl RequestClassifier for PassThrough {}

    struct SynchronousPanicOnce {
        calls: Arc<AtomicUsize>,
    }

    impl RequestClassifier for SynchronousPanicOnce {
        fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                panic!("synchronous classifier panic");
            }
            Box::pin(async move { Ok(request) })
        }
    }

    #[tokio::test]
    async fn synchronous_panic_fails_one_request_and_retains_the_classifier() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(SynchronousPanicOnce {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );

        let error = runtime
            .classify_with(|| ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierPanicked(message)
                if message == "synchronous classifier panic"
        ));
        runtime
            .classify_with(|| ClassifyRequest::new(2, 0))
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    struct FuturePanicOnce {
        calls: Arc<AtomicUsize>,
    }

    impl RequestClassifier for FuturePanicOnce {
        fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
            let should_panic = self.calls.fetch_add(1, Ordering::Relaxed) == 0;
            Box::pin(async move {
                assert!(!should_panic, "classifier future panic");
                Ok(request)
            })
        }
    }

    #[tokio::test]
    async fn future_panic_fails_one_request_and_retains_the_classifier() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(FuturePanicOnce {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );

        let error = runtime
            .classify_with(|| ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierPanicked(message)
                if message == "classifier future panic"
        ));
        runtime
            .classify_with(|| ClassifyRequest::new(2, 0))
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[derive(Debug, thiserror::Error)]
    #[error("classifier rejected request")]
    struct TestClassifierError;

    struct FailingClassifier;

    impl RequestClassifier for FailingClassifier {
        fn classify(&mut self, _request: ClassifyRequest) -> ClassifyFuture {
            Box::pin(async { Err(Box::new(TestClassifierError) as Box<ClassifierError>) })
        }
    }

    #[tokio::test]
    async fn classifier_error_is_preserved_as_scheduler_error() {
        let runtime =
            RequestClassifierRuntime::new(Box::new(FailingClassifier), CancellationToken::new());

        let error = runtime
            .classify_with(|| ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierFailed(source)
                if source.to_string() == "classifier rejected request"
        ));
    }

    struct ReplacingClassifier;

    impl RequestClassifier for ReplacingClassifier {
        fn classify(&mut self, _request: ClassifyRequest) -> ClassifyFuture {
            Box::pin(async { Ok(ClassifyRequest::new(2, 0)) })
        }
    }

    #[tokio::test]
    async fn classifier_cannot_replace_the_logical_request() {
        let runtime =
            RequestClassifierRuntime::new(Box::new(ReplacingClassifier), CancellationToken::new());

        let error = runtime
            .classify_with(|| ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierReplacedRequest
        ));
    }

    #[test]
    fn duplicate_live_request_id_is_rejected_until_lifecycle_ends() {
        let runtime =
            RequestClassifierRuntime::new(Box::new(PassThrough), CancellationToken::new());
        let lifecycle = runtime.begin_request("request-1").unwrap();

        let error = runtime.begin_request("request-1").unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::DuplicateClassificationRequestId(request_id)
                if request_id == "request-1"
        ));

        drop(lifecycle);
        runtime.begin_request("request-1").unwrap();
    }

    #[tokio::test]
    async fn default_classifier_returns_the_same_request_without_overrides() {
        let request = ClassifyRequest::new(128, 32)
            .with_request_id("request-1")
            .with_initial_policy_class("latency");

        let result = PassThrough.classify(request).await.unwrap();

        assert_eq!(result.request_id(), Some("request-1"));
        assert_eq!(result.policy_class(), Some("latency"));
        assert_eq!(result.scheduling_cost_tokens(), 96);
        assert_eq!(result.into_queue_inputs(), (None, None, None, 32));
    }

    struct EventReleasedClassifier {
        entered: Arc<Notify>,
        released: Arc<Notify>,
    }

    #[async_trait]
    impl RequestClassifier for EventReleasedClassifier {
        fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
            let entered = Arc::clone(&self.entered);
            let released = Arc::clone(&self.released);
            Box::pin(async move {
                entered.notify_one();
                released.notified().await;
                Ok(request)
            })
        }

        async fn on_event(&mut self, _event: ClassifyEvent<'_>) {
            self.released.notify_one();
        }
    }

    #[tokio::test]
    async fn pending_classification_does_not_block_event_delivery() {
        let entered = Arc::new(Notify::new());
        let released = Arc::new(Notify::new());
        let runtime = RequestClassifierRuntime::new(
            Box::new(EventReleasedClassifier {
                entered: Arc::clone(&entered),
                released,
            }),
            CancellationToken::new(),
        );
        let mut lifecycle = runtime.begin_request("event-source").unwrap();
        let worker = WorkerWithDpRank::new(7, 0);

        let pending_runtime = Arc::clone(&runtime);
        let pending = tokio::spawn(async move {
            pending_runtime
                .classify_with(|| ClassifyRequest::new(1, 1))
                .await
        });
        entered.notified().await;
        lifecycle.sent(worker);

        pending.await.unwrap().unwrap();
        lifecycle.abort(None);
    }

    struct PendingClassifier {
        entered: Arc<AtomicUsize>,
        events: mpsc::UnboundedSender<String>,
    }

    #[async_trait]
    impl RequestClassifier for PendingClassifier {
        fn classify(&mut self, _request: ClassifyRequest) -> ClassifyFuture {
            self.entered.fetch_add(1, Ordering::Relaxed);
            Box::pin(std::future::pending())
        }

        async fn on_event(&mut self, event: ClassifyEvent<'_>) {
            if let ClassifyEvent::Aborted { request_id, .. } = event {
                self.events.send(request_id.to_owned()).unwrap();
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_and_deadline_abort_pending_classification() {
        let entered = Arc::new(AtomicUsize::new(0));
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(PendingClassifier {
                entered: Arc::clone(&entered),
                events: event_tx,
            }),
            CancellationToken::new(),
        );

        let cancelled_runtime = Arc::clone(&runtime);
        let cancelled = tokio::spawn(async move {
            let _lifecycle = cancelled_runtime.begin_request("cancelled").unwrap();
            cancelled_runtime
                .classify_with(|| ClassifyRequest::new(1, 0).with_request_id("cancelled"))
                .await
        });
        while entered.load(Ordering::Relaxed) < 1 {
            tokio::task::yield_now().await;
        }
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        assert_eq!(event_rx.recv().await.as_deref(), Some("cancelled"));

        let deadline_runtime = Arc::clone(&runtime);
        let deadline = tokio::spawn(async move {
            let _lifecycle = deadline_runtime.begin_request("deadline").unwrap();
            deadline_runtime
                .classify_with(|| {
                    ClassifyRequest::with_timing(
                        1,
                        0,
                        Instant::now(),
                        Some(Instant::now() + std::time::Duration::from_secs(1)),
                    )
                    .with_request_id("deadline")
                })
                .await
        });
        while entered.load(Ordering::Relaxed) < 2 {
            tokio::task::yield_now().await;
        }
        tokio::time::advance(std::time::Duration::from_secs(1)).await;
        assert!(matches!(
            deadline.await.unwrap(),
            Err(KvSchedulerError::DueTimeExpired)
        ));
        assert_eq!(event_rx.recv().await.as_deref(), Some("deadline"));
        assert!(event_rx.try_recv().is_err());
    }

    struct AwaitingEventClassifier {
        events: mpsc::UnboundedSender<String>,
    }

    #[async_trait]
    impl RequestClassifier for AwaitingEventClassifier {
        async fn on_event(&mut self, event: ClassifyEvent<'_>) {
            let ClassifyEvent::Aborted { request_id, .. } = event else {
                return;
            };
            // Await inside the callback: delivery must tolerate suspension.
            tokio::task::yield_now().await;
            self.events.send(request_id.to_owned()).unwrap();
        }
    }

    #[tokio::test]
    async fn off_runtime_drop_still_delivers_terminal_event() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(AwaitingEventClassifier { events: event_tx }),
            CancellationToken::new(),
        );
        let lifecycle = runtime.begin_request("off-runtime-drop").unwrap();
        runtime
            .classify_with(|| ClassifyRequest::new(1, 0).with_request_id("off-runtime-drop"))
            .await
            .unwrap();

        std::thread::spawn(move || drop(lifecycle)).join().unwrap();
        assert_eq!(event_rx.recv().await.as_deref(), Some("off-runtime-drop"));
    }

    struct CountingClassifier {
        calls: Arc<AtomicUsize>,
    }

    impl RequestClassifier for CountingClassifier {
        fn classify(&mut self, mut request: ClassifyRequest) -> ClassifyFuture {
            self.calls.fetch_add(1, Ordering::Relaxed);
            request.set_scheduling_cost_tokens(7);
            Box::pin(async move { Ok(request) })
        }
    }

    #[tokio::test]
    async fn retry_reuses_classification_overrides_without_reinvoking_plugin() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );
        let mut lifecycle = runtime.begin_request("request-1").unwrap();

        let first = runtime
            .classify_with(|| ClassifyRequest::new(10, 10).with_request_id("request-1"))
            .await
            .unwrap();
        let retry = runtime
            .classify_with(|| ClassifyRequest::new(20, 20).with_request_id("request-1"))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(first.input_tokens(), 10);
        assert_eq!(retry.input_tokens(), 20);
        assert_eq!(retry.scheduling_cost_tokens(), 7);
        lifecycle.abort(None);
    }

    #[derive(Debug, PartialEq, Eq)]
    enum RecordedEvent {
        Sent(String, WorkerWithDpRank),
        Completed(String, WorkerWithDpRank, Option<usize>),
        Aborted(String, Option<WorkerWithDpRank>),
    }

    struct RecordingClassifier {
        events: mpsc::UnboundedSender<RecordedEvent>,
    }

    #[async_trait]
    impl RequestClassifier for RecordingClassifier {
        async fn on_event(&mut self, event: ClassifyEvent<'_>) {
            let event = match event {
                ClassifyEvent::Sent { request_id, worker } => {
                    Some(RecordedEvent::Sent(request_id.to_owned(), worker))
                }
                ClassifyEvent::Completed {
                    request_id,
                    worker,
                    context_tokens,
                } => Some(RecordedEvent::Completed(
                    request_id.to_owned(),
                    worker,
                    context_tokens,
                )),
                ClassifyEvent::Aborted {
                    request_id, worker, ..
                } => Some(RecordedEvent::Aborted(request_id.to_owned(), worker)),
                ClassifyEvent::Responding { .. } => None,
            };
            if let Some(event) = event {
                let _ = self.events.send(event);
            }
        }
    }

    #[tokio::test]
    async fn lifecycle_delivers_one_terminal_event() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(RecordingClassifier { events: event_tx }),
            CancellationToken::new(),
        );
        let worker = WorkerWithDpRank::new(7, 2);
        let mut lifecycle = runtime.begin_request("request-1").unwrap();
        runtime
            .classify_with(|| ClassifyRequest::new(40, 0).with_request_id("request-1"))
            .await
            .unwrap();

        lifecycle.sent(worker);
        lifecycle.observe_context_tokens(40);
        lifecycle.observe_output_tokens(2);
        lifecycle.complete();
        lifecycle.abort(None);

        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Sent("request-1".to_string(), worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Completed(
                "request-1".to_string(),
                worker,
                Some(42)
            ))
        );
        assert!(event_rx.try_recv().is_err());
    }
}
