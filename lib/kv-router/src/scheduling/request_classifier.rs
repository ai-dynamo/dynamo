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
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tokio::task::JoinHandle;
// Deadline math across the scheduler uses `tokio::time::Instant` so
// paused-clock tests stay coherent.
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::policy_queue::QueueSnapshot;
use super::types::{KvSchedulerError, SessionContext};
use super::{RequestProgress, RequestProgressUpdater};
use crate::protocols::{WorkerAffinityTarget, WorkerWithDpRank};

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
    progress: RequestProgress,
    session_context: Option<SessionContext>,
}

#[derive(Clone, Debug, Default)]
struct ClassificationOverrides {
    policy_class: Option<String>,
    due_at: Option<Instant>,
    scheduling_cost_tokens: Option<usize>,
    /// Outer `None` leaves session affinity untouched; `Some(None)` clears it;
    /// `Some(Some(target))` replaces it for this request.
    worker_selection_target: Option<Option<WorkerAffinityTarget>>,
}

pub(crate) struct ClassificationQueueInputs {
    pub(crate) policy_class: Option<String>,
    pub(crate) due_at: Option<Instant>,
    pub(crate) scheduling_cost_tokens: Option<usize>,
    pub(crate) worker_selection_target: Option<Option<WorkerAffinityTarget>>,
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
        let (progress, _) = RequestProgress::new(input_tokens);
        Self {
            classification_id: 0,
            request_id: None,
            policy_class: None,
            overrides: ClassificationOverrides::default(),
            ingress_at,
            input_tokens,
            initial_cached_tokens,
            progress,
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

    /// Return lock-free access to the latest logical context observed while
    /// this request is live.
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

    /// Ask Place to prefer one worker and data-parallel rank for this request.
    ///
    /// The router keeps final selection authority: caller constraints and worker eligibility are
    /// still enforced, and a custom worker-selection policy may fall back when the target is not
    /// eligible. The target replaces any session-affinity target for this request only; a
    /// pinned request (an explicit caller pin, a routing plan, or a disaggregation hint)
    /// ignores it.
    pub fn set_worker_selection_target(&mut self, worker: WorkerWithDpRank) {
        self.overrides.worker_selection_target = Some(Some(worker.into()));
    }

    /// Remove the session-affinity target from this request before Place.
    ///
    /// Required caller routing constraints remain in force. This is primarily useful with soft
    /// session affinity when a classifier deliberately repacks work across workers.
    pub fn clear_worker_selection_target(&mut self) {
        self.overrides.worker_selection_target = Some(None);
    }

    pub fn session_context(&self) -> Option<&SessionContext> {
        self.session_context.as_ref()
    }

    /// Only the explicit overrides feed the queue: cache eligibility is
    /// recomputed from the current workers at enqueue, because worker state
    /// may have changed while the classification was pending.
    pub(crate) fn into_queue_inputs(self) -> ClassificationQueueInputs {
        ClassificationQueueInputs {
            policy_class: self.overrides.policy_class,
            due_at: self.overrides.due_at,
            scheduling_cost_tokens: self.overrides.scheduling_cost_tokens,
            worker_selection_target: self.overrides.worker_selection_target,
        }
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

/// One live lifecycle's bookkeeping. `generation` fences the overrides
/// write-back in `classify_with` against a lifecycle that released this
/// request id and a new one that re-registered it mid-classification.
struct LiveRequest {
    generation: u64,
    overrides: Option<ClassificationOverrides>,
    progress: RequestProgress,
    progress_updater: RequestProgressUpdater,
}

/// One live worker rank visible to a request-classifier plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestClassifierWorker {
    worker: WorkerWithDpRank,
    total_kv_blocks: Option<u64>,
}

impl RequestClassifierWorker {
    pub fn new(worker: WorkerWithDpRank, total_kv_blocks: Option<u64>) -> Self {
        Self {
            worker,
            total_kv_blocks,
        }
    }

    pub fn worker(&self) -> WorkerWithDpRank {
        self.worker
    }

    /// Total KV blocks of [`RequestClassifierContext::block_size`] tokens on this rank, from the
    /// worker's published runtime config. `None` when the worker did not publish a value.
    pub fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// Cached host inputs supplied when constructing one classifier instance.
#[derive(Clone)]
pub struct RequestClassifierContext {
    block_size: u32,
    workers: Arc<dyn Fn() -> Vec<RequestClassifierWorker> + Send + Sync>,
}

impl RequestClassifierContext {
    pub fn new(
        block_size: u32,
        workers: impl Fn() -> Vec<RequestClassifierWorker> + Send + Sync + 'static,
    ) -> Self {
        Self {
            block_size,
            workers: Arc::new(workers),
        }
    }

    /// Tokens per KV block for this router's hash domain.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Return a non-blocking snapshot from the host's existing discovery watcher.
    /// Ranks appear as workers register and leave when they deregister.
    pub fn workers(&self) -> Vec<RequestClassifierWorker> {
        (self.workers)()
    }
}

impl std::fmt::Debug for RequestClassifierContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestClassifierContext")
            .field("block_size", &self.block_size)
            .finish_non_exhaustive()
    }
}

/// Factory resolved once from the linked catalog and invoked for each routed model.
pub type RequestClassifierFactory =
    Arc<dyn Fn(RequestClassifierContext) -> Box<dyn RequestClassifier> + Send + Sync>;

pub(crate) struct RequestClassifierRuntime {
    // Box: the install seam is object-safe and `Mutex::new` needs `Sized`;
    // Arc: the delivery task holds its own handle to the classifier.
    // Panics in `classify` and `on_event` are caught and the same instance
    // keeps serving: one request's failure must not disable classification
    // router-wide, and the plugin owns its own state across an unwind.
    classifier: Arc<AsyncMutex<Box<dyn RequestClassifier>>>,
    live_requests: Mutex<HashMap<String, LiveRequest>>,
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
        let delivery_classifier = Arc::clone(&classifier);
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
            }
        });
        Arc::new(Self {
            classifier,
            live_requests: Mutex::new(HashMap::new()),
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

        let input_tokens = request.input_tokens;
        let live_request = request.request_id().and_then(|request_id| {
            let mut live_requests = self.live_requests.lock();
            let live_request = live_requests.get_mut(request_id)?;
            live_request
                .progress_updater
                .update_context_tokens(input_tokens);
            Some((
                live_request.progress.clone(),
                live_request.overrides.clone(),
            ))
        });
        if let Some((progress, overrides)) = live_request {
            request.progress = progress;
            if let Some(overrides) = overrides {
                // Cached overrides replay without re-consulting the plugin, so a retry can
                // carry a worker-selection target whose worker has since deregistered.
                // Availability filtering excludes dead workers, degrading to fallback
                // selection rather than misrouting.
                request.overrides = overrides;
                return Ok(request);
            }
        }

        let classification_id = NEXT_CLASSIFICATION_ID.fetch_add(1, Ordering::Relaxed);
        request.classification_id = classification_id;
        let (classification, generation) = {
            let mut classifier = self.classifier.lock().await;
            // Re-check registration under the classifier lock: terminal events
            // are delivered under this same lock after the id leaves the live
            // set, so an id seen live here cannot have had its Aborted
            // delivered yet — the plugin never observes classify-after-abort.
            let generation = match request.request_id() {
                Some(request_id) => match self.live_requests.lock().get(request_id) {
                    Some(live) => Some(live.generation),
                    // The lifecycle ended while this caller waited for the
                    // lock; skip the plugin and take the default inputs.
                    None => return Ok(request),
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
        // Write back only onto the lifecycle the classification started under:
        // the id may have been released and re-registered while the plugin ran,
        // and stale overrides must not leak onto the new lifecycle.
        if let Some(request_id) = classified.request_id()
            && let Some(generation) = generation
            && let Some(live) = self.live_requests.lock().get_mut(request_id)
            && live.generation == generation
        {
            live.overrides = Some(classified.overrides.clone());
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
        let progress_updater = match self.live_requests.lock().entry(request_id.to_owned()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(KvSchedulerError::DuplicateClassificationRequestId(
                    request_id.to_owned(),
                ));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let (progress, progress_updater) = RequestProgress::new(0);
                entry.insert(LiveRequest {
                    generation: NEXT_LIFECYCLE_GENERATION.fetch_add(1, Ordering::Relaxed),
                    overrides: None,
                    progress,
                    progress_updater: progress_updater.clone(),
                });
                progress_updater
            }
        };
        Ok(RequestLifecycle {
            runtime: Arc::clone(self),
            request_id: request_id.to_owned(),
            worker: None,
            context_tokens: None,
            progress_updater,
            phase: LifecyclePhase::Registered,
        })
    }

    pub(crate) fn has_request(&self, request_id: &str) -> bool {
        self.live_requests.lock().contains_key(request_id)
    }

    fn send_event(&self, event: ClassifyEvent) {
        let _ = self.events.send(event);
    }

    /// Remove the request from the live set and, if it was live, enqueue its
    /// terminal event before releasing the lock (the unbounded send cannot
    /// block). `begin_request` re-registers a reused id under the same lock,
    /// so the terminal event is always ordered ahead of any event from the
    /// id's next lifecycle.
    fn finish_request_and_send(
        &self,
        request_id: String,
        event: impl FnOnce(String) -> ClassifyEvent,
    ) {
        let mut live_requests = self.live_requests.lock();
        if live_requests.remove(&request_id).is_none() {
            return;
        }
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
    progress_updater: RequestProgressUpdater,
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
        let context_tokens = self
            .context_tokens
            .unwrap_or_default()
            .saturating_add(output_tokens);
        self.context_tokens = Some(context_tokens);
        self.progress_updater.update_context_tokens(context_tokens);
    }

    /// Raise the context total to at least `context_tokens` (an
    /// engine-reported absolute count).
    pub fn observe_context_tokens(&mut self, context_tokens: usize) {
        self.context_tokens = Some(
            self.context_tokens
                .map_or(context_tokens, |current| current.max(context_tokens)),
        );
        self.progress_updater.update_context_tokens(context_tokens);
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
        let inputs = result.into_queue_inputs();
        assert_eq!(inputs.policy_class, None);
        assert_eq!(inputs.due_at, None);
        assert_eq!(inputs.scheduling_cost_tokens, None);
        assert_eq!(inputs.worker_selection_target, None);
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
        assert_eq!(first.progress().context_tokens(), 10);
        lifecycle.observe_context_tokens(10);
        lifecycle.observe_output_tokens(5);
        assert_eq!(first.progress().context_tokens(), 15);
        let retry = runtime
            .classify_with(ClassifyRequest::new(12, 12).with_request_id("request-1"))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(first.input_tokens(), 10);
        assert_eq!(retry.input_tokens(), 12);
        assert_eq!(retry.scheduling_cost_tokens(), 7);
        assert_eq!(retry.progress().context_tokens(), 15);
        lifecycle.observe_context_tokens(20);
        assert_eq!(first.progress().context_tokens(), 20);
        assert_eq!(retry.progress().context_tokens(), 20);
        lifecycle.abort(None);
    }

    #[tokio::test]
    async fn unregistered_request_id_skips_the_plugin() {
        let calls = Arc::new(AtomicUsize::new(0));
        let runtime = RequestClassifierRuntime::new(
            Box::new(CountingClassifier {
                calls: Arc::clone(&calls),
            }),
            CancellationToken::new(),
        );

        let result = runtime
            .classify_with(ClassifyRequest::new(4, 0).with_request_id("ghost"))
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        let inputs = result.into_queue_inputs();
        assert_eq!(inputs.policy_class, None);
        assert_eq!(inputs.due_at, None);
        assert_eq!(inputs.scheduling_cost_tokens, None);
        assert_eq!(inputs.worker_selection_target, None);
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
        stale.await.unwrap().unwrap();

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
