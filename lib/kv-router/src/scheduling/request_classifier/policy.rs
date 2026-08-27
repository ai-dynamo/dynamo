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
use std::time::Instant;

use futures_util::FutureExt;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::super::types::{KvSchedulerError, SessionContext};
use crate::protocols::WorkerWithDpRank;

static NEXT_CLASSIFICATION_ID: AtomicU64 = AtomicU64::new(1);

/// An owned request view passed to a classifier before router queueing.
///
/// The router keeps the request payload and identity private. A classifier can
/// inspect the request facts exposed here, override scheduling metadata, and
/// return the same value immediately or from a pending future.
#[derive(Debug)]
pub struct ClassifyRequest {
    classification_id: u64,
    request_id: Option<String>,
    policy_class: Option<String>,
    policy_class_override: Option<String>,
    due_at_override: Option<Instant>,
    input_tokens: usize,
    initial_cached_tokens: usize,
    default_scheduling_cost_tokens: usize,
    scheduling_cost_tokens_override: Option<usize>,
    session_context: Option<SessionContext>,
}

#[derive(Clone)]
struct ClassificationOverrides {
    policy_class: Option<String>,
    due_at: Option<Instant>,
    scheduling_cost_tokens: Option<usize>,
}

impl ClassifyRequest {
    pub(crate) fn new(input_tokens: usize, initial_cached_tokens: usize) -> Self {
        let initial_cached_tokens = initial_cached_tokens.min(input_tokens);
        Self {
            classification_id: 0,
            request_id: None,
            policy_class: None,
            policy_class_override: None,
            due_at_override: None,
            input_tokens,
            initial_cached_tokens,
            default_scheduling_cost_tokens: input_tokens
                .saturating_sub(initial_cached_tokens)
                .max(1),
            scheduling_cost_tokens_override: None,
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

    /// Return the request ID when this scheduling request carries one.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Return the effective policy class.
    pub fn policy_class(&self) -> Option<&str> {
        self.policy_class_override
            .as_deref()
            .or(self.policy_class.as_deref())
    }

    /// Override the policy class used by the router-owned queue.
    pub fn set_policy_class(&mut self, policy_class: impl Into<String>) {
        self.policy_class_override = Some(policy_class.into());
    }

    /// Return the model input context size after preprocessing.
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Return the effective due time, if one was supplied by the classifier.
    pub fn due_at(&self) -> Option<Instant> {
        self.due_at_override
    }

    /// Set the due time enforced by the router-owned queue.
    pub fn set_due_at(&mut self, due_at: Instant) {
        self.due_at_override = Some(due_at);
    }

    /// Return the effective DRR scheduling cost in tokens.
    pub fn scheduling_cost_tokens(&self) -> usize {
        self.scheduling_cost_tokens_override
            .unwrap_or(self.default_scheduling_cost_tokens)
    }

    /// Override the DRR scheduling cost in tokens.
    pub fn set_scheduling_cost_tokens(&mut self, scheduling_cost_tokens: usize) {
        self.scheduling_cost_tokens_override = Some(scheduling_cost_tokens);
    }

    /// Return the optional session metadata already carried by the request.
    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    pub(crate) fn into_queue_inputs(
        self,
    ) -> (Option<String>, Option<Instant>, Option<usize>, usize) {
        (
            self.policy_class_override,
            self.due_at_override,
            self.scheduling_cost_tokens_override,
            self.initial_cached_tokens,
        )
    }

    fn begin_classification(&mut self) -> u64 {
        let classification_id = NEXT_CLASSIFICATION_ID.fetch_add(1, Ordering::Relaxed);
        self.classification_id = classification_id;
        classification_id
    }

    fn classification_overrides(&self) -> ClassificationOverrides {
        ClassificationOverrides {
            policy_class: self.policy_class_override.clone(),
            due_at: self.due_at_override,
            scheduling_cost_tokens: self.scheduling_cost_tokens_override,
        }
    }

    fn apply_classification_overrides(&mut self, overrides: ClassificationOverrides) {
        self.policy_class_override = overrides.policy_class;
        self.due_at_override = overrides.due_at;
        self.scheduling_cost_tokens_override = overrides.scheduling_cost_tokens;
    }
}

/// Error detail returned by a classifier or attached to an aborted request.
pub type ClassifyError = dyn Error + Send + Sync + 'static;

/// Request lifecycle notifications delivered to [`RequestClassifier::on_event`].
#[derive(Debug)]
#[non_exhaustive]
pub enum ClassifyEvent<'a> {
    Sent {
        request_id: &'a str,
        worker: WorkerWithDpRank,
    },
    Received {
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
        error: Option<&'a ClassifyError>,
    },
}

/// An independently pollable classifier invocation.
///
/// The future owns its continuation. A pending invocation therefore does not
/// retain a borrow or lock on the classifier and cannot block another request
/// or lifecycle notification.
pub type ClassifyFuture =
    Pin<Box<dyn Future<Output = Result<ClassifyRequest, Box<ClassifyError>>> + Send + 'static>>;

/// User-provided request classifier.
pub trait RequestClassifier: Send + 'static {
    /// Return the same logical request immediately or after a classifier-specific wait.
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        Box::pin(async move { Ok(request) })
    }

    /// Observe one lifecycle transition without releasing or dispatching work directly.
    ///
    /// The router invokes this callback synchronously. Implementations should update
    /// shared state, notify pending futures, and return promptly.
    fn on_event(&mut self, _event: ClassifyEvent<'_>) {}
}

pub(crate) struct RequestClassifierRuntime {
    classifier: Mutex<Box<dyn RequestClassifier>>,
    live_requests: Mutex<HashMap<String, Option<ClassificationOverrides>>>,
    shutdown: CancellationToken,
}

impl RequestClassifierRuntime {
    pub(crate) fn new(
        classifier: Box<dyn RequestClassifier>,
        shutdown: CancellationToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            classifier: Mutex::new(classifier),
            live_requests: Mutex::new(HashMap::new()),
            shutdown,
        })
    }

    pub(crate) async fn classify_with(
        &self,
        request_id: Option<&str>,
        make_request: impl FnOnce() -> ClassifyRequest,
    ) -> Result<ClassifyRequest, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        let cached_overrides = request_id.and_then(|request_id| {
            self.live_requests
                .lock()
                .get(request_id)
                .and_then(Clone::clone)
        });
        if let Some(overrides) = cached_overrides {
            let mut request = make_request();
            request.apply_classification_overrides(overrides);
            return Ok(request);
        }

        let mut request = make_request();
        let classification_id = request.begin_classification();
        let classification = catch_unwind(AssertUnwindSafe(|| {
            self.classifier.lock().classify(request)
        }))
        .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic)))?;
        let classification = AssertUnwindSafe(classification).catch_unwind();

        let classified_request = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => return Err(KvSchedulerError::SubscriberShutdown),
            result = classification => result
                .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic)))?
                .map_err(KvSchedulerError::RequestClassifierFailed)?,
        };

        if classified_request.classification_id != classification_id {
            return Err(KvSchedulerError::RequestClassifierReplacedRequest);
        }

        if let Some(request_id) = classified_request.request_id()
            && let Some(cached) = self.live_requests.lock().get_mut(request_id)
        {
            *cached = Some(classified_request.classification_overrides());
        }

        Ok(classified_request)
    }

    pub(crate) fn begin_request(
        self: &Arc<Self>,
        request_id: &str,
    ) -> Result<RequestLifecycle, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        let mut live_requests = self.live_requests.lock();
        match live_requests.entry(request_id.to_owned()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(KvSchedulerError::DuplicateClassificationRequestId(
                    request_id.to_owned(),
                ));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(None);
            }
        }
        drop(live_requests);

        Ok(RequestLifecycle {
            runtime: Arc::clone(self),
            request_id: request_id.to_owned(),
            worker: None,
            last_worker: None,
            context_tokens: None,
            phase: LifecyclePhase::Registered,
        })
    }

    fn deliver_event(&self, event: ClassifyEvent<'_>) {
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| {
            self.classifier.lock().on_event(event);
        })) {
            tracing::warn!(
                panic = %panic_message(panic),
                "Request classifier panicked while processing a lifecycle event"
            );
        }
    }

    fn finish_request(&self, request_id: &str, event: ClassifyEvent<'_>) {
        if !self.live_requests.lock().contains_key(request_id) {
            return;
        }
        self.deliver_event(event);
        self.live_requests.lock().remove(request_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifecyclePhase {
    Registered,
    Sent,
    Received,
    Responding,
    Terminal,
}

/// Router-owned lifecycle state transferred from scheduling to response handling.
#[doc(hidden)]
pub struct RequestLifecycle {
    runtime: Arc<RequestClassifierRuntime>,
    request_id: String,
    worker: Option<WorkerWithDpRank>,
    last_worker: Option<WorkerWithDpRank>,
    context_tokens: Option<usize>,
    phase: LifecyclePhase,
}

impl std::fmt::Debug for RequestLifecycle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycle")
            .field("request_id", &self.request_id)
            .field("worker", &self.worker)
            .field("last_worker", &self.last_worker)
            .field("context_tokens", &self.context_tokens)
            .field("phase", &self.phase)
            .finish()
    }
}

impl RequestLifecycle {
    #[doc(hidden)]
    pub fn selected(&mut self, worker: WorkerWithDpRank) {
        if self.phase == LifecyclePhase::Registered {
            self.worker = Some(worker);
        }
    }

    #[doc(hidden)]
    pub fn sent(&mut self, worker: WorkerWithDpRank) {
        if self.phase != LifecyclePhase::Registered {
            return;
        }
        self.worker = Some(worker);
        self.phase = LifecyclePhase::Sent;
        self.runtime.deliver_event(ClassifyEvent::Sent {
            request_id: &self.request_id,
            worker,
        });
    }

    #[doc(hidden)]
    pub fn received(&mut self) {
        if self.phase != LifecyclePhase::Sent {
            return;
        }
        let Some(worker) = self.worker else {
            return;
        };
        self.phase = LifecyclePhase::Received;
        self.runtime.deliver_event(ClassifyEvent::Received {
            request_id: &self.request_id,
            worker,
        });
    }

    #[doc(hidden)]
    pub fn responding(&mut self) {
        if !matches!(self.phase, LifecyclePhase::Sent | LifecyclePhase::Received) {
            return;
        }
        let Some(worker) = self.worker else {
            return;
        };
        self.phase = LifecyclePhase::Responding;
        self.runtime.deliver_event(ClassifyEvent::Responding {
            request_id: &self.request_id,
            worker,
        });
    }

    #[doc(hidden)]
    pub fn observe_input_tokens(&mut self, input_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .map_or(input_tokens, |current| current.max(input_tokens)),
        );
    }

    #[doc(hidden)]
    pub fn observe_output_tokens(&mut self, output_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .unwrap_or_default()
                .saturating_add(output_tokens),
        );
    }

    #[doc(hidden)]
    pub fn observe_context_tokens(&mut self, context_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .map_or(context_tokens, |current| current.max(context_tokens)),
        );
    }

    /// Reset attempt-local worker state without terminating the logical request.
    #[doc(hidden)]
    pub fn prepare_retry(&mut self) {
        if self.phase != LifecyclePhase::Terminal {
            self.last_worker = self.worker.or(self.last_worker);
            self.worker = None;
            self.phase = LifecyclePhase::Registered;
        }
    }

    #[doc(hidden)]
    pub fn complete(&mut self) {
        if self.phase == LifecyclePhase::Terminal {
            return;
        }
        let Some(worker) = self.worker else {
            self.abort(None);
            return;
        };
        self.phase = LifecyclePhase::Terminal;
        self.runtime.finish_request(
            &self.request_id,
            ClassifyEvent::Completed {
                request_id: &self.request_id,
                worker,
                context_tokens: self.context_tokens,
            },
        );
    }

    #[doc(hidden)]
    pub fn abort(&mut self, error: Option<&ClassifyError>) {
        if self.phase == LifecyclePhase::Terminal {
            return;
        }
        self.phase = LifecyclePhase::Terminal;
        self.runtime.finish_request(
            &self.request_id,
            ClassifyEvent::Aborted {
                request_id: &self.request_id,
                worker: self.worker.or(self.last_worker),
                error,
            },
        );
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::{Notify, mpsc};

    use super::*;

    struct PassThrough;

    impl RequestClassifier for PassThrough {}

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

        fn on_event(&mut self, _event: ClassifyEvent<'_>) {
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
                .classify_with(None, || ClassifyRequest::new(1, 1))
                .await
        });
        entered.notified().await;
        lifecycle.sent(worker);

        pending.await.unwrap().unwrap();
        lifecycle.abort(None);
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
            .classify_with(Some("request-1"), || {
                ClassifyRequest::new(10, 10).with_request_id("request-1")
            })
            .await
            .unwrap();
        let retry = runtime
            .classify_with(Some("request-1"), || {
                ClassifyRequest::new(20, 20).with_request_id("request-1")
            })
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

    impl RequestClassifier for RecordingClassifier {
        fn on_event(&mut self, event: ClassifyEvent<'_>) {
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
                ClassifyEvent::Received { .. } | ClassifyEvent::Responding { .. } => None,
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

        lifecycle.sent(worker);
        lifecycle.observe_input_tokens(40);
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
