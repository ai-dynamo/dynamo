// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::ops::BitOr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::Instant;

use futures_util::FutureExt;
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use super::super::types::{KvSchedulerError, SessionContext};
use crate::protocols::WorkerWithDpRank;

const DEFAULT_MAX_PENDING_CLASSIFICATIONS: usize = 1_024;
static NEXT_CLASSIFICATION_ID: AtomicU64 = AtomicU64::new(1);

/// Optional inputs materialized for a classifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassifyInputs(u8);

impl ClassifyInputs {
    /// Request no per-worker inputs.
    pub const NONE: Self = Self(0);
    /// Request per-worker KV-cache overlap inputs.
    pub const CACHE: Self = Self(1 << 0);
    /// Request configured per-worker KV capacity.
    pub const CAPACITY: Self = Self(1 << 1);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for ClassifyInputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Request-specific cache inputs for one structurally eligible worker/DP rank.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassifyCacheInput {
    pub(crate) effective_overlap_blocks: f64,
    pub(crate) device_overlap_blocks: usize,
    pub(crate) host_overlap_blocks: usize,
    pub(crate) disk_overlap_blocks: usize,
    pub(crate) cached_tokens: usize,
}

impl ClassifyCacheInput {
    pub fn effective_overlap_blocks(&self) -> f64 {
        self.effective_overlap_blocks
    }

    pub fn device_overlap_blocks(&self) -> usize {
        self.device_overlap_blocks
    }

    pub fn host_overlap_blocks(&self) -> usize {
        self.host_overlap_blocks
    }

    pub fn disk_overlap_blocks(&self) -> usize {
        self.disk_overlap_blocks
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }
}

/// Configured KV capacity for one structurally eligible worker/DP rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassifyCapacityInput {
    pub(crate) device_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
}

impl ClassifyCapacityInput {
    pub fn device_tokens(&self) -> Option<u64> {
        self.device_tokens
    }

    pub fn total_tokens(&self) -> Option<u64> {
        self.total_tokens
    }
}

/// Typed classifier inputs for one structurally eligible worker/DP rank.
#[derive(Clone, Copy, Debug)]
pub struct ClassifyWorker {
    pub(crate) worker: WorkerWithDpRank,
    pub(crate) cache: Option<ClassifyCacheInput>,
    pub(crate) capacity: Option<ClassifyCapacityInput>,
}

impl ClassifyWorker {
    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    pub fn cache(&self) -> Option<&ClassifyCacheInput> {
        self.cache.as_ref()
    }

    pub fn capacity(&self) -> Option<&ClassifyCapacityInput> {
        self.capacity.as_ref()
    }
}

/// An owned request view passed to a classifier before normal scheduling.
///
/// The router constructs this value and keeps its identity and payload private.
/// A classifier can inspect request facts, override scheduling metadata, and return
/// the same value immediately or from a pending future.
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
    workers: Option<Vec<ClassifyWorker>>,
    was_pending: bool,
    classification_permit: Option<OwnedSemaphorePermit>,
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
            workers: None,
            was_pending: false,
            classification_permit: None,
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

    pub(crate) fn with_workers(mut self, workers: Vec<ClassifyWorker>) -> Self {
        self.workers = Some(workers);
        self
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn policy_class(&self) -> Option<&str> {
        self.policy_class_override
            .as_deref()
            .or(self.policy_class.as_deref())
    }

    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    pub fn set_policy_class(&mut self, policy_class: impl Into<String>) {
        self.policy_class_override = Some(policy_class.into());
    }

    pub fn set_due_at(&mut self, due_at: Instant) {
        self.due_at_override = Some(due_at);
    }

    pub fn scheduling_cost_tokens(&self) -> usize {
        self.scheduling_cost_tokens_override
            .unwrap_or(self.default_scheduling_cost_tokens)
    }

    pub fn set_scheduling_cost_tokens(&mut self, scheduling_cost_tokens: usize) {
        self.scheduling_cost_tokens_override = Some(scheduling_cost_tokens);
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    /// Return the structurally eligible workers when the classifier declared optional inputs.
    pub fn workers(&self) -> Option<&[ClassifyWorker]> {
        self.workers.as_deref()
    }

    pub(crate) fn into_queue_inputs(
        self,
    ) -> (
        Option<String>,
        Option<Instant>,
        Option<usize>,
        Option<OwnedSemaphorePermit>,
    ) {
        (
            self.policy_class_override,
            self.due_at_override,
            self.scheduling_cost_tokens_override,
            self.classification_permit,
        )
    }

    pub(crate) fn initial_cached_tokens(&self) -> usize {
        self.initial_cached_tokens
    }

    pub(crate) fn was_pending(&self) -> bool {
        self.was_pending
    }

    pub(crate) fn mark_pending(&mut self) {
        self.was_pending = true;
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

/// Error detail attached to an aborted request.
///
/// The response path supplies its underlying error through this dependency-neutral view.
/// Policies that know the concrete error type may downcast it with [`Error::downcast_ref`].
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
/// The future owns its continuation so the router can poll each invocation
/// independently without retaining a lock on the classifier object.
pub type ClassifyFuture = Pin<Box<dyn Future<Output = ClassifyRequest> + Send + 'static>>;

/// User-provided request classifier.
pub trait RequestClassifier: Send + 'static {
    /// Declare the optional inputs needed by this classifier.
    fn required_inputs(&self) -> ClassifyInputs {
        ClassifyInputs::NONE
    }

    /// Return the same logical request immediately or after a classifier-specific wait.
    /// Advisory query-only worker selections do not invoke the classifier.
    fn classify(&self, request: ClassifyRequest) -> ClassifyFuture {
        Box::pin(async move { request })
    }

    /// Observe one lifecycle transition without directly releasing or dispatching work.
    /// Implementations must return promptly; the router invokes this synchronously.
    fn on_event(&mut self, _event: ClassifyEvent<'_>) {}
}

pub(crate) struct RequestClassifierRuntime {
    classifier: Mutex<Box<dyn RequestClassifier>>,
    required_inputs: ClassifyInputs,
    pending_classifications: Arc<Semaphore>,
    max_pending_classifications: usize,
    live_requests: Mutex<HashMap<String, Option<ClassificationOverrides>>>,
    shutdown: CancellationToken,
}

impl RequestClassifierRuntime {
    pub(crate) fn new(
        classifier: Box<dyn RequestClassifier>,
        shutdown: CancellationToken,
    ) -> Arc<Self> {
        let required_inputs = classifier.required_inputs();
        Arc::new(Self {
            classifier: Mutex::new(classifier),
            required_inputs,
            pending_classifications: Arc::new(Semaphore::new(DEFAULT_MAX_PENDING_CLASSIFICATIONS)),
            max_pending_classifications: DEFAULT_MAX_PENDING_CLASSIFICATIONS,
            live_requests: Mutex::new(HashMap::new()),
            shutdown,
        })
    }

    pub(crate) async fn classify_with(
        &self,
        request_id: Option<&str>,
        make_request: impl FnOnce(ClassifyInputs) -> ClassifyRequest,
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
            let mut request = make_request(ClassifyInputs::NONE);
            request.apply_classification_overrides(overrides);
            if self.shutdown.is_cancelled() {
                return Err(KvSchedulerError::SubscriberShutdown);
            }
            return Ok(request);
        }

        let _permit = self.acquire_classification_permit()?;
        let mut request = make_request(self.required_inputs);
        request.classification_permit = Some(_permit);
        let classification_id = request.begin_classification();

        let classification = catch_unwind(AssertUnwindSafe(|| {
            self.classifier.lock().classify(request)
        }))
        .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic)))?;
        let mut classification = AssertUnwindSafe(classification).catch_unwind();
        let mut was_pending = false;

        let mut classified_request = tokio::select! {
            biased;
            _ = self.shutdown.cancelled() => Err(KvSchedulerError::SubscriberShutdown),
            result = std::future::poll_fn(|context| {
                match Pin::new(&mut classification).poll(context) {
                    Poll::Ready(result) => Poll::Ready(result),
                    Poll::Pending => {
                        was_pending = true;
                        Poll::Pending
                    }
                }
            }) => result
                .map_err(|panic| KvSchedulerError::RequestClassifierPanicked(panic_message(panic))),
        }?;

        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }
        if classified_request.classification_id != classification_id {
            return Err(KvSchedulerError::RequestClassifierReplacedRequest);
        }
        if was_pending {
            classified_request.mark_pending();
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
        {
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
        }
        Ok(RequestLifecycle {
            runtime: Arc::clone(self),
            request_id: request_id.to_owned(),
            worker: None,
            last_worker: None,
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

    fn acquire_classification_permit(&self) -> Result<OwnedSemaphorePermit, KvSchedulerError> {
        if self.shutdown.is_cancelled() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }
        Arc::clone(&self.pending_classifications)
            .try_acquire_owned()
            .map_err(|_| KvSchedulerError::ClassifyPendingLimit {
                limit: self.max_pending_classifications,
            })
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
    phase: LifecyclePhase,
}

impl std::fmt::Debug for RequestLifecycle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycle")
            .field("request_id", &self.request_id)
            .field("worker", &self.worker)
            .field("last_worker", &self.last_worker)
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
    pub fn complete(&mut self, context_tokens: Option<usize>) {
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
                context_tokens,
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

        let result = PassThrough.classify(request).await;

        assert_eq!(result.request_id(), Some("request-1"));
        assert_eq!(result.policy_class(), Some("latency"));
        assert_eq!(result.scheduling_cost_tokens(), 96);
        let (policy_class, due_at, scheduling_cost, permit) = result.into_queue_inputs();
        assert_eq!((policy_class, due_at, scheduling_cost), (None, None, None));
        assert!(permit.is_none());
    }

    struct PendingClassifier {
        entered: Arc<Notify>,
        released: Arc<Notify>,
    }

    impl RequestClassifier for PendingClassifier {
        fn classify(&self, mut request: ClassifyRequest) -> ClassifyFuture {
            let entered = Arc::clone(&self.entered);
            let released = Arc::clone(&self.released);
            Box::pin(async move {
                entered.notify_one();
                released.notified().await;
                request.set_scheduling_cost_tokens(7);
                request
            })
        }
    }

    #[tokio::test]
    async fn runtime_bounds_independently_pollable_classifications() {
        let entered = Arc::new(Notify::new());
        let released = Arc::new(Notify::new());
        let shutdown = CancellationToken::new();
        let runtime = Arc::new(RequestClassifierRuntime {
            classifier: Mutex::new(Box::new(PendingClassifier {
                entered: Arc::clone(&entered),
                released: Arc::clone(&released),
            })),
            required_inputs: ClassifyInputs::NONE,
            pending_classifications: Arc::new(Semaphore::new(1)),
            max_pending_classifications: 1,
            live_requests: Mutex::new(HashMap::new()),
            shutdown: shutdown.clone(),
        });

        let first_runtime = Arc::clone(&runtime);
        let first = tokio::spawn(async move {
            first_runtime
                .classify_with(None, |_| ClassifyRequest::new(1, 0))
                .await
        });
        entered.notified().await;

        let second = runtime
            .classify_with(None, |_| ClassifyRequest::new(1, 0))
            .await;
        assert!(matches!(
            second,
            Err(KvSchedulerError::ClassifyPendingLimit { limit: 1 })
        ));

        released.notify_one();
        let result = first.await.unwrap().unwrap();
        assert_eq!(result.scheduling_cost_tokens(), 7);
        assert!(result.was_pending());
        assert!(matches!(
            runtime.acquire_classification_permit(),
            Err(KvSchedulerError::ClassifyPendingLimit { limit: 1 })
        ));
        drop(result);
        assert!(runtime.acquire_classification_permit().is_ok());
        shutdown.cancel();
    }

    struct CountingClassifier {
        calls: Arc<AtomicUsize>,
    }

    impl RequestClassifier for CountingClassifier {
        fn classify(&self, mut request: ClassifyRequest) -> ClassifyFuture {
            self.calls.fetch_add(1, Ordering::Relaxed);
            request.set_scheduling_cost_tokens(7);
            Box::pin(async move { request })
        }
    }

    #[tokio::test]
    async fn migration_retry_reuses_only_classification_overrides() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );
        let mut lifecycle = runtime.begin_request("request-1").unwrap();

        let first = runtime
            .classify_with(Some("request-1"), |_| {
                ClassifyRequest::new(10, 0).with_request_id("request-1")
            })
            .await
            .unwrap();
        let retry = runtime
            .classify_with(Some("request-1"), |_| {
                ClassifyRequest::new(20, 0).with_request_id("request-1")
            })
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(first.input_tokens(), 10);
        assert_eq!(retry.input_tokens(), 20);
        assert_eq!(retry.scheduling_cost_tokens(), 7);
        lifecycle.abort(None);
    }

    #[tokio::test]
    async fn cached_migration_retry_stops_after_shutdown() {
        let shutdown = CancellationToken::new();
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::new(AtomicUsize::new(0)),
            }),
            shutdown.clone(),
        );
        let mut lifecycle = runtime.begin_request("request-1").unwrap();

        runtime
            .classify_with(Some("request-1"), |_| {
                ClassifyRequest::new(10, 0).with_request_id("request-1")
            })
            .await
            .unwrap();
        shutdown.cancel();

        let retry = runtime
            .classify_with(Some("request-1"), |_| {
                ClassifyRequest::new(20, 0).with_request_id("request-1")
            })
            .await;
        assert!(matches!(retry, Err(KvSchedulerError::SubscriberShutdown)));
        lifecycle.abort(None);
    }

    #[derive(Debug, PartialEq, Eq)]
    enum RecordedEvent {
        Sent(String, WorkerWithDpRank),
        Received(String, WorkerWithDpRank),
        Responding(String, WorkerWithDpRank),
        Completed(String, WorkerWithDpRank, Option<usize>),
        Aborted(String, Option<WorkerWithDpRank>, Option<String>),
    }

    struct RecordingClassifier {
        events: mpsc::UnboundedSender<RecordedEvent>,
    }

    impl RequestClassifier for RecordingClassifier {
        fn on_event(&mut self, event: ClassifyEvent<'_>) {
            let event = match event {
                ClassifyEvent::Sent { request_id, worker } => {
                    RecordedEvent::Sent(request_id.to_owned(), worker)
                }
                ClassifyEvent::Received { request_id, worker } => {
                    RecordedEvent::Received(request_id.to_owned(), worker)
                }
                ClassifyEvent::Responding { request_id, worker } => {
                    RecordedEvent::Responding(request_id.to_owned(), worker)
                }
                ClassifyEvent::Completed {
                    request_id,
                    worker,
                    context_tokens,
                } => RecordedEvent::Completed(request_id.to_owned(), worker, context_tokens),
                ClassifyEvent::Aborted {
                    request_id,
                    worker,
                    error,
                } => RecordedEvent::Aborted(
                    request_id.to_owned(),
                    worker,
                    error.map(ToString::to_string),
                ),
            };
            let _ = self.events.send(event);
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("backend failed")]
    struct TestClassifyError;

    #[tokio::test]
    async fn lifecycle_delivers_response_path_transitions() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(RecordingClassifier { events: event_tx }),
            CancellationToken::new(),
        );
        let worker = WorkerWithDpRank::new(7, 2);
        let mut lifecycle = runtime.begin_request("request-1").unwrap();

        lifecycle.selected(worker);
        lifecycle.sent(worker);
        lifecycle.received();
        lifecycle.responding();
        lifecycle.complete(Some(42));

        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Sent("request-1".to_string(), worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Received("request-1".to_string(), worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Responding("request-1".to_string(), worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Completed(
                "request-1".to_string(),
                worker,
                Some(42)
            ))
        );
    }

    #[tokio::test]
    async fn lifecycle_drop_aborts_and_releases_request_id() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(RecordingClassifier { events: event_tx }),
            CancellationToken::new(),
        );

        let lifecycle = runtime.begin_request("request-1").unwrap();
        assert!(matches!(
            runtime.begin_request("request-1"),
            Err(KvSchedulerError::DuplicateClassificationRequestId(request_id))
                if request_id == "request-1"
        ));
        drop(lifecycle);
        assert_eq!(
            event_rx.try_recv(),
            Ok(RecordedEvent::Aborted("request-1".to_string(), None, None))
        );

        let worker = WorkerWithDpRank::new(3, 1);
        let mut retry = runtime.begin_request("request-1").unwrap();
        retry.selected(worker);
        retry.abort(Some(&TestClassifyError));
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Aborted(
                "request-1".to_string(),
                Some(worker),
                Some("backend failed".to_string())
            ))
        );
    }

    #[tokio::test]
    async fn lifecycle_retry_retargets_without_terminal_event() {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let runtime = RequestClassifierRuntime::new(
            Box::new(RecordingClassifier { events: event_tx }),
            CancellationToken::new(),
        );
        let first_worker = WorkerWithDpRank::new(3, 0);
        let second_worker = WorkerWithDpRank::new(7, 1);
        let mut lifecycle = runtime.begin_request("request-1").unwrap();

        lifecycle.sent(first_worker);
        lifecycle.responding();
        lifecycle.prepare_retry();
        lifecycle.sent(second_worker);
        lifecycle.complete(Some(42));

        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Sent("request-1".to_string(), first_worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Responding(
                "request-1".to_string(),
                first_worker
            ))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Sent("request-1".to_string(), second_worker))
        );
        assert_eq!(
            event_rx.recv().await,
            Some(RecordedEvent::Completed(
                "request-1".to_string(),
                second_worker,
                Some(42)
            ))
        );
        assert!(event_rx.try_recv().is_err());
    }
}
