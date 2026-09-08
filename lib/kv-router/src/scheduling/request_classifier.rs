// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use futures_util::FutureExt;
use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc};
use tokio::task::JoinHandle;
// Deadline math across the scheduler uses `tokio::time::Instant` so
// paused-clock tests stay coherent.
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::policy_queue::QueueSnapshot;
use super::types::{KvSchedulerError, SessionContext};
use crate::protocols::WorkerWithDpRank;

static NEXT_CLASSIFICATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_LIFECYCLE_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct ClassifyRequest {
    classification_id: u64,
    request_id: Option<String>,
    policy_class: Option<String>,
    overrides: ClassificationOverrides,
    ingress_at: Instant,
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
        Self::with_timing(input_tokens, initial_cached_tokens, Instant::now())
    }

    pub(crate) fn with_timing(
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

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    /// Only the explicit overrides feed the queue: cache eligibility is
    /// recomputed from the current workers at enqueue, because worker state
    /// may have changed while the classification was pending.
    pub(crate) fn into_queue_inputs(self) -> (Option<String>, Option<Instant>, Option<usize>) {
        (
            self.overrides.policy_class,
            self.overrides.due_at,
            self.overrides.scheduling_cost_tokens,
        )
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

/// An independently pollable classifier invocation. See [`RequestClassifier`]
/// for why this is a `'static` future rather than `async fn`.
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

    async fn on_event(&mut self, _event: ClassifyEvent) {}
}

/// One live lifecycle's bookkeeping. `generation` fences `classify_with`
/// against a lifecycle that released this request id and a new one that
/// re-registered it mid-classification: only the lifecycle a classification
/// started under may receive its overrides or enter Order.
struct LiveRequest {
    generation: u64,
    overrides: Option<ClassificationOverrides>,
}

/// Terminal events enqueued for delivery but not yet delivered, per request
/// id. `classify_with` waits on this before invoking the plugin for a reused
/// id, so the plugin never observes a lifecycle's `classify` ahead of the
/// previous lifecycle's terminal event for the same id.
#[derive(Default)]
struct PendingTerminals {
    counts: Mutex<HashMap<String, usize>>,
    delivered: Notify,
}

impl PendingTerminals {
    fn reserve(&self, request_id: &str) {
        *self.counts.lock().entry(request_id.to_owned()).or_insert(0) += 1;
    }

    fn release(&self, request_id: &str) {
        {
            let mut counts = self.counts.lock();
            if let Some(count) = counts.get_mut(request_id) {
                *count -= 1;
                if *count == 0 {
                    counts.remove(request_id);
                }
            }
        }
        self.delivered.notify_waiters();
    }

    fn is_pending(&self, request_id: &str) -> bool {
        self.counts.lock().contains_key(request_id)
    }
}

pub(crate) struct RequestClassifierRuntime {
    // Box: the install seam is object-safe and `Mutex::new` needs `Sized`;
    // Arc: the delivery task holds its own handle to the classifier.
    // Panics in `classify` and `on_event` are caught and the same instance
    // keeps serving: one request's failure must not disable classification
    // router-wide, and the plugin owns its own state across an unwind.
    classifier: Arc<AsyncMutex<Box<dyn RequestClassifier>>>,
    live_requests: Mutex<HashMap<String, LiveRequest>>,
    // Shared with the delivery task, which releases each terminal event once
    // `on_event` has returned for it.
    pending_terminals: Arc<PendingTerminals>,
    // Deliberately unbounded and lossless: dropping a lifecycle event (above
    // all a terminal one) silently corrupts plugin bookkeeping, and senders —
    // including `Drop` — must not await. The cost is unbounded growth while
    // `on_event` is stuck, backstopped by shutdown cancelling delivery and
    // `Drop` aborting the task.
    events: mpsc::UnboundedSender<ClassifyEvent>,
    shutdown: CancellationToken,
    // Aborted on drop as a backstop for a shutdown token that never fires
    // while `on_event` is stuck.
    delivery: JoinHandle<()>,
}

impl RequestClassifierRuntime {
    /// Create the runtime and spawn its event-delivery task. Must be called
    /// from within a Tokio runtime.
    pub(crate) fn new(
        classifier: Box<dyn RequestClassifier>,
        shutdown: CancellationToken,
    ) -> Arc<Self> {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let classifier = Arc::new(AsyncMutex::new(classifier));
        let pending_terminals = Arc::new(PendingTerminals::default());
        let delivery_classifier = Arc::clone(&classifier);
        let delivery_pending = Arc::clone(&pending_terminals);
        let delivery_shutdown = shutdown.clone();
        let delivery = tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    biased;
                    _ = delivery_shutdown.cancelled() => break,
                    event = receiver.recv() => match event {
                        Some(event) => event,
                        None => break,
                    },
                };
                let terminal_request_id = terminal_request_id(&event);
                // Shutdown must also interrupt a stuck `on_event`, not just
                // fire between events: dropping this branch releases the
                // classifier lock so callers queued on it can observe
                // shutdown instead of hanging.
                let delivered = tokio::select! {
                    biased;
                    _ = delivery_shutdown.cancelled() => break,
                    delivered = async {
                        let mut classifier = delivery_classifier.lock().await;
                        AssertUnwindSafe(classifier.on_event(event))
                            .catch_unwind()
                            .await
                    } => delivered,
                };
                if let Err(panic) = delivered {
                    tracing::error!(
                        panic = %panic_message(panic),
                        "Request classifier panicked while processing a lifecycle event"
                    );
                }
                // Released only after `on_event` returned (or unwound): the
                // plugin has observed the terminal event, so the id's next
                // lifecycle may now be classified.
                if let Some(request_id) = terminal_request_id {
                    delivery_pending.release(&request_id);
                }
            }
        });
        Arc::new(Self {
            classifier,
            live_requests: Mutex::new(HashMap::new()),
            pending_terminals,
            events,
            shutdown,
            delivery,
        })
    }

    pub(crate) async fn classify_with(
        &self,
        mut request: ClassifyRequest,
    ) -> Result<ClassifyRequest, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        if let Some(overrides) = request.request_id().and_then(|request_id| {
            self.live_requests
                .lock()
                .get(request_id)
                .and_then(|live| live.overrides.clone())
        }) {
            request.overrides = overrides;
            return Ok(request);
        }

        let classification_id = NEXT_CLASSIFICATION_ID.fetch_add(1, Ordering::Relaxed);
        request.classification_id = classification_id;
        let (classification, generation) = {
            let mut classifier = self.lock_classifier_for(request.request_id()).await?;
            // Re-check registration under the classifier lock: terminal events
            // are delivered under this same lock after the id leaves the live
            // set, so an id seen live here cannot have had its Aborted
            // delivered yet — the plugin never observes classify-after-abort.
            let generation = match request.request_id() {
                Some(request_id) => match self.live_requests.lock().get(request_id) {
                    Some(live) => Some(live.generation),
                    // The lifecycle ended while this caller waited for the
                    // lock: only a still-live request may enter Order.
                    None => {
                        return Err(KvSchedulerError::ClassificationLifecycleEnded(
                            request_id.to_owned(),
                        ));
                    }
                },
                None => None,
            };
            let classification = catch_unwind(AssertUnwindSafe(|| classifier.classify(request)))
                .map_err(|panic| {
                    KvSchedulerError::RequestClassifierPanicked(panic_message(panic))
                })?;
            (classification, generation)
        };
        let classification = AssertUnwindSafe(classification).catch_unwind();

        let result = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
            result = classification => result,
        };
        let classified = result
            .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic)))?
            .map_err(|error| KvSchedulerError::RequestClassifierFailed(Arc::from(error)))?;

        if classified.classification_id != classification_id {
            return Err(KvSchedulerError::InvalidClassificationMetadata(
                "classifier replaced the logical request".to_string(),
            ));
        }
        // Only the lifecycle the classification started under may enter Order:
        // the id may have been released, and re-registered, while the plugin
        // ran. Write the overrides back onto that lifecycle; otherwise reject,
        // so a request whose lifecycle ended can neither reserve capacity nor
        // leak stale overrides onto the id's next lifecycle.
        if let (Some(generation), Some(request_id)) = (generation, classified.request_id()) {
            match self.live_requests.lock().get_mut(request_id) {
                Some(live) if live.generation == generation => {
                    live.overrides = Some(classified.overrides.clone());
                }
                _ => {
                    return Err(KvSchedulerError::ClassificationLifecycleEnded(
                        request_id.to_owned(),
                    ));
                }
            }
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
        match self.live_requests.lock().entry(request_id.to_owned()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(KvSchedulerError::DuplicateClassificationRequestId(
                    request_id.to_owned(),
                ));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(LiveRequest {
                    generation: NEXT_LIFECYCLE_GENERATION.fetch_add(1, Ordering::Relaxed),
                    overrides: None,
                });
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

    /// Take the classifier lock for a `classify` call. A reused id must not
    /// reach the plugin ahead of the previous lifecycle's terminal event, so
    /// wait for that id's pending terminals outside the lock (delivery needs
    /// it), then re-check under it: a terminal enqueued in between cannot be
    /// delivered while the lock is held, so release it and wait again.
    async fn lock_classifier_for(
        &self,
        request_id: Option<&str>,
    ) -> Result<tokio::sync::MutexGuard<'_, Box<dyn RequestClassifier>>, KvSchedulerError> {
        loop {
            if let Some(request_id) = request_id {
                self.await_terminal_delivery(request_id).await?;
            }
            let classifier = self.classifier.lock().await;
            if request_id.is_some_and(|request_id| self.pending_terminals.is_pending(request_id)) {
                continue;
            }
            return Ok(classifier);
        }
    }

    /// Wait until no terminal event for `request_id` is queued undelivered.
    async fn await_terminal_delivery(&self, request_id: &str) -> Result<(), KvSchedulerError> {
        loop {
            // Register before checking so a release between the check and the
            // await still wakes this waiter.
            let delivered = self.pending_terminals.delivered.notified();
            if !self.pending_terminals.is_pending(request_id) {
                return Ok(());
            }
            tokio::select! {
                biased;
                _ = self.shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
                _ = delivered => {}
            }
        }
    }

    fn send_event(&self, event: ClassifyEvent) {
        let _ = self.events.send(event);
    }

    /// Remove the request from the live set and, if it was live, enqueue its
    /// terminal event before releasing the lock (the unbounded send cannot
    /// block). `begin_request` re-registers a reused id under the same lock,
    /// so the terminal event is always ordered ahead of any event from the
    /// id's next lifecycle, and the pending-terminal reservation taken here
    /// holds that lifecycle's `classify` until the event has been delivered.
    fn finish_request_and_send(
        &self,
        request_id: String,
        event: impl FnOnce(String) -> ClassifyEvent,
    ) {
        let mut live_requests = self.live_requests.lock();
        if live_requests.remove(&request_id).is_none() {
            return;
        }
        // Reserve before sending so delivery can never release first. A send
        // only fails once delivery has ended (shutdown or Drop), and waiters
        // observe shutdown rather than the reservation.
        self.pending_terminals.reserve(&request_id);
        let _ = self.events.send(event(request_id));
    }
}

impl Drop for RequestClassifierRuntime {
    fn drop(&mut self) {
        self.delivery.abort();
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
        self.runtime.send_event(ClassifyEvent::Sent {
            request_id: self.request_id.clone(),
            worker,
        });
    }

    pub fn responding(&mut self) {
        if self.phase != LifecyclePhase::Sent {
            return;
        }
        let Some(worker) = self.worker else {
            debug_assert!(false, "phase Sent implies a recorded worker");
            return;
        };
        self.phase = LifecyclePhase::Responding;
        self.runtime.send_event(ClassifyEvent::Responding {
            request_id: self.request_id.clone(),
            worker,
        });
    }

    /// Add generated output tokens on top of the current context total.
    ///
    /// Order matters: [`Self::observe_context_tokens`] floors the same total,
    /// so report a context before its outputs or the floor erases them.
    pub fn observe_output_tokens(&mut self, output_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .unwrap_or_default()
                .saturating_add(output_tokens),
        );
    }

    /// Raise the context total to at least `context_tokens` (an
    /// engine-reported absolute count).
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
        let context_tokens = self.context_tokens;
        self.runtime
            .finish_request_and_send(std::mem::take(&mut self.request_id), |request_id| {
                ClassifyEvent::Completed {
                    request_id,
                    worker,
                    context_tokens,
                }
            });
    }

    pub fn abort(&mut self, error: Option<Arc<AbortCause>>) {
        if self.phase == LifecyclePhase::Terminal {
            return;
        }
        self.phase = LifecyclePhase::Terminal;
        let worker = self.worker;
        self.runtime
            .finish_request_and_send(std::mem::take(&mut self.request_id), |request_id| {
                ClassifyEvent::Aborted {
                    request_id,
                    worker,
                    error,
                }
            });
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        self.abort(None);
    }
}

fn terminal_request_id(event: &ClassifyEvent) -> Option<String> {
    match event {
        ClassifyEvent::Completed { request_id, .. } | ClassifyEvent::Aborted { request_id, .. } => {
            Some(request_id.clone())
        }
        ClassifyEvent::Sent { .. } | ClassifyEvent::Responding { .. } => None,
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
            .classify_with(ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierPanicked(message)
                if message == "synchronous classifier panic"
        ));
        runtime
            .classify_with(ClassifyRequest::new(2, 0))
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
            .classify_with(ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::RequestClassifierPanicked(message)
                if message == "classifier future panic"
        ));
        runtime
            .classify_with(ClassifyRequest::new(2, 0))
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
            .classify_with(ClassifyRequest::new(1, 0))
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
            .classify_with(ClassifyRequest::new(1, 0))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::InvalidClassificationMetadata(message)
                if message == "classifier replaced the logical request"
        ));
    }

    #[tokio::test]
    async fn duplicate_live_request_id_is_rejected_until_lifecycle_ends() {
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
        assert_eq!(result.into_queue_inputs(), (None, None, None));
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

        async fn on_event(&mut self, _event: ClassifyEvent) {
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
                .classify_with(ClassifyRequest::new(1, 1))
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

        async fn on_event(&mut self, event: ClassifyEvent) {
            if let ClassifyEvent::Aborted { request_id, .. } = event {
                self.events.send(request_id).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn cancellation_aborts_pending_classification() {
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
                .classify_with(ClassifyRequest::new(1, 0).with_request_id("cancelled"))
                .await
        });
        while entered.load(Ordering::Relaxed) < 1 {
            tokio::task::yield_now().await;
        }
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());
        assert_eq!(event_rx.recv().await.as_deref(), Some("cancelled"));
        assert!(event_rx.try_recv().is_err());
    }

    struct AwaitingEventClassifier {
        events: mpsc::UnboundedSender<String>,
    }

    #[async_trait]
    impl RequestClassifier for AwaitingEventClassifier {
        async fn on_event(&mut self, event: ClassifyEvent) {
            let ClassifyEvent::Aborted { request_id, .. } = event else {
                return;
            };
            // Await inside the callback: delivery must tolerate suspension.
            tokio::task::yield_now().await;
            self.events.send(request_id).unwrap();
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
            .classify_with(ClassifyRequest::new(1, 0).with_request_id("off-runtime-drop"))
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
            .classify_with(ClassifyRequest::new(10, 10).with_request_id("request-1"))
            .await
            .unwrap();
        let retry = runtime
            .classify_with(ClassifyRequest::new(20, 20).with_request_id("request-1"))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(first.input_tokens(), 10);
        assert_eq!(retry.input_tokens(), 20);
        assert_eq!(retry.scheduling_cost_tokens(), 7);
        lifecycle.abort(None);
    }

    // `LocalScheduler::classify_request` routes ids with no registered
    // lifecycle to the default path before reaching the runtime, so at this
    // level an unregistered id is indistinguishable from one whose lifecycle
    // ended: neither may enter Order, and the plugin is never invoked.
    #[tokio::test]
    async fn unregistered_request_id_is_rejected_without_calling_the_plugin() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );

        let result = runtime
            .classify_with(ClassifyRequest::new(4, 0).with_request_id("ghost"))
            .await;

        assert!(matches!(
            result,
            Err(KvSchedulerError::ClassificationLifecycleEnded(_))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    struct GatedClassifier {
        entered: Arc<Notify>,
        release: Arc<Notify>,
        calls: Arc<AtomicUsize>,
    }

    impl RequestClassifier for GatedClassifier {
        fn classify(&mut self, mut request: ClassifyRequest) -> ClassifyFuture {
            let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            let entered = Arc::clone(&self.entered);
            let release = Arc::clone(&self.release);
            Box::pin(async move {
                entered.notify_one();
                release.notified().await;
                request.set_scheduling_cost_tokens(call);
                Ok(request)
            })
        }
    }

    #[tokio::test]
    async fn stale_classification_does_not_cache_onto_a_reused_request_id() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(GatedClassifier {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );

        let lifecycle = runtime.begin_request("reused").unwrap();
        let stale_runtime = Arc::clone(&runtime);
        let stale = tokio::spawn(async move {
            stale_runtime
                .classify_with(ClassifyRequest::new(1, 0).with_request_id("reused"))
                .await
        });
        entered.notified().await;

        // End the first lifecycle and re-register the id while its
        // classification is still in flight.
        drop(lifecycle);
        let _reused = runtime.begin_request("reused").unwrap();
        release.notify_one();
        // The stale lifecycle may not enter Order.
        assert!(matches!(
            stale.await.unwrap(),
            Err(KvSchedulerError::ClassificationLifecycleEnded(_))
        ));

        // The stale result must not be cached onto the new lifecycle: its
        // classification reaches the plugin instead of reusing overrides.
        release.notify_one();
        let second = runtime
            .classify_with(ClassifyRequest::new(1, 0).with_request_id("reused"))
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(second.scheduling_cost_tokens(), 2);
    }

    #[tokio::test]
    async fn lifecycle_ended_while_waiting_for_the_classifier_lock_is_rejected() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );
        let lifecycle = runtime.begin_request("ended").unwrap();

        // Hold the classifier lock so the classification parks ahead of the
        // plugin call, then end the lifecycle while it waits.
        let guard = runtime.classifier.lock().await;
        let ended_runtime = Arc::clone(&runtime);
        let ended = tokio::spawn(async move {
            ended_runtime
                .classify_with(ClassifyRequest::new(1, 0).with_request_id("ended"))
                .await
        });
        tokio::task::yield_now().await;
        drop(lifecycle);
        drop(guard);

        assert!(matches!(
            ended.await.unwrap(),
            Err(KvSchedulerError::ClassificationLifecycleEnded(_))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn lifecycle_ended_during_classification_is_rejected() {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let runtime = RequestClassifierRuntime::new(
            Box::new(GatedClassifier {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            CancellationToken::new(),
        );
        let lifecycle = runtime.begin_request("ended").unwrap();
        let ended_runtime = Arc::clone(&runtime);
        let ended = tokio::spawn(async move {
            ended_runtime
                .classify_with(ClassifyRequest::new(1, 0).with_request_id("ended"))
                .await
        });
        entered.notified().await;

        // End the lifecycle while the plugin still holds the request, without
        // re-registering the id.
        drop(lifecycle);
        release.notify_one();

        assert!(matches!(
            ended.await.unwrap(),
            Err(KvSchedulerError::ClassificationLifecycleEnded(_))
        ));
        assert!(!runtime.has_request("ended"));
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
        async fn on_event(&mut self, event: ClassifyEvent) {
            let event = match event {
                ClassifyEvent::Sent { request_id, worker } => {
                    Some(RecordedEvent::Sent(request_id, worker))
                }
                ClassifyEvent::Completed {
                    request_id,
                    worker,
                    context_tokens,
                } => Some(RecordedEvent::Completed(request_id, worker, context_tokens)),
                ClassifyEvent::Aborted {
                    request_id, worker, ..
                } => Some(RecordedEvent::Aborted(request_id, worker)),
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
            .classify_with(ClassifyRequest::new(40, 0).with_request_id("request-1"))
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

    #[tokio::test]
    async fn terminal_event_is_ordered_before_events_from_a_reused_request_id() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(RecordingClassifier { events: event_tx }),
            CancellationToken::new(),
        );
        let worker = WorkerWithDpRank::new(1, 0);

        for _ in 0..4096 {
            let mut lifecycle = runtime.begin_request("reused").unwrap();
            lifecycle.sent(worker);

            // Race a re-registration of the client-controlled id against
            // `complete`, grabbing it the instant it is released.
            let attempts = Arc::new(AtomicUsize::new(0));
            let reuse_attempts = Arc::clone(&attempts);
            let reuse_runtime = Arc::clone(&runtime);
            let reuse = std::thread::spawn(move || {
                let mut reused = loop {
                    reuse_attempts.fetch_add(1, Ordering::Relaxed);
                    match reuse_runtime.begin_request("reused") {
                        Ok(lifecycle) => break lifecycle,
                        Err(_) => std::hint::spin_loop(),
                    }
                };
                reused.sent(worker);
                reused
            });
            while attempts.load(Ordering::Relaxed) == 0 {
                std::thread::yield_now();
            }
            lifecycle.complete();
            let reused = reuse.join().unwrap();

            assert_eq!(
                event_rx.recv().await,
                Some(RecordedEvent::Sent("reused".to_string(), worker))
            );
            assert_eq!(
                event_rx.recv().await,
                Some(RecordedEvent::Completed("reused".to_string(), worker, None)),
                "terminal event for the released id must precede the reused id's Sent"
            );
            assert_eq!(
                event_rx.recv().await,
                Some(RecordedEvent::Sent("reused".to_string(), worker))
            );
            drop(reused);
            assert_eq!(
                event_rx.recv().await,
                Some(RecordedEvent::Aborted("reused".to_string(), Some(worker)))
            );
        }
    }

    /// Records what the plugin observes, in order, and holds the `Sent`
    /// callback open so a terminal event can queue behind it undelivered.
    struct OrderingClassifier {
        observed: mpsc::UnboundedSender<String>,
        entered_sent: Arc<Notify>,
        release_sent: Arc<Notify>,
    }

    #[async_trait]
    impl RequestClassifier for OrderingClassifier {
        fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
            let _ = self
                .observed
                .send(format!("classify:{}", request.request_id().unwrap_or("")));
            Box::pin(async move { Ok(request) })
        }

        async fn on_event(&mut self, event: ClassifyEvent) {
            match event {
                ClassifyEvent::Sent { request_id, .. } => {
                    self.entered_sent.notify_one();
                    self.release_sent.notified().await;
                    let _ = self.observed.send(format!("sent:{request_id}"));
                }
                ClassifyEvent::Completed { request_id, .. } => {
                    let _ = self.observed.send(format!("completed:{request_id}"));
                }
                ClassifyEvent::Aborted { request_id, .. } => {
                    let _ = self.observed.send(format!("aborted:{request_id}"));
                }
                ClassifyEvent::Responding { .. } => {}
            }
        }
    }

    #[tokio::test]
    async fn reused_id_is_not_classified_before_the_prior_terminal_is_delivered() {
        let (observed_tx, mut observed) = mpsc::unbounded_channel();
        let entered_sent = Arc::new(Notify::new());
        let release_sent = Arc::new(Notify::new());
        let runtime = RequestClassifierRuntime::new(
            Box::new(OrderingClassifier {
                observed: observed_tx,
                entered_sent: Arc::clone(&entered_sent),
                release_sent: Arc::clone(&release_sent),
            }),
            CancellationToken::new(),
        );
        let worker = WorkerWithDpRank::new(1, 0);

        // Park delivery inside the first lifecycle's `Sent` callback, then end
        // that lifecycle: its `Completed` is queued but undelivered.
        let mut first = runtime.begin_request("reused").unwrap();
        first.sent(worker);
        entered_sent.notified().await;
        first.complete();

        // Re-register the id and start classifying it while the terminal
        // event is still queued. The classification is parked before delivery
        // resumes, so without the gate the lock's FIFO hand-off would run it
        // ahead of the queued `Completed`.
        let _second = runtime.begin_request("reused").unwrap();
        let classify_runtime = Arc::clone(&runtime);
        let classify = tokio::spawn(async move {
            classify_runtime
                .classify_with(ClassifyRequest::new(1, 0).with_request_id("reused"))
                .await
        });
        tokio::task::yield_now().await;

        release_sent.notify_one();
        classify.await.unwrap().unwrap();

        let mut order = Vec::new();
        for _ in 0..3 {
            order.push(observed.recv().await.unwrap());
        }
        assert_eq!(
            order,
            vec![
                "sent:reused".to_string(),
                "completed:reused".to_string(),
                "classify:reused".to_string(),
            ],
            "the prior lifecycle's terminal event must reach the plugin before the reused id's classify"
        );
    }

    struct StuckOnEventClassifier {
        entered: Arc<Notify>,
    }

    #[async_trait]
    impl RequestClassifier for StuckOnEventClassifier {
        async fn on_event(&mut self, _event: ClassifyEvent) {
            self.entered.notify_one();
            std::future::pending::<()>().await;
        }
    }

    #[tokio::test]
    async fn shutdown_interrupts_stuck_on_event_and_releases_classifier_lock() {
        let entered = Arc::new(Notify::new());
        let shutdown = CancellationToken::new();
        let runtime = RequestClassifierRuntime::new(
            Box::new(StuckOnEventClassifier {
                entered: Arc::clone(&entered),
            }),
            shutdown.clone(),
        );
        let mut lifecycle = runtime.begin_request("stuck").unwrap();
        lifecycle.sent(WorkerWithDpRank::new(1, 0));
        // The callback now holds the classifier lock and never returns.
        entered.notified().await;

        shutdown.cancel();
        let _guard =
            tokio::time::timeout(std::time::Duration::from_secs(5), runtime.classifier.lock())
                .await
                .expect("shutdown did not release the classifier lock held by on_event");
    }
}
