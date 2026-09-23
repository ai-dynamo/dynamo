// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod capacity;
mod scheduler;

pub use capacity::{WorkerCapacityProvider, WorkerCapacitySnapshot};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::plugins::request_classifier::{
    ClassifierError, ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
    RequestClassifierContext, RequestClassifierFactory, RequestClassifierParameters,
    RequestClassifierProviderError, RequestClassifierRegistryError, RequestProgress,
};
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::Notify;

use self::scheduler::{RequestRegistration, State, WaitStatus};
use super::{ConfigError, ThunderAgentConfig};

fn capacity_provider(context: RequestClassifierContext) -> Arc<dyn WorkerCapacityProvider> {
    let block_size = u64::from(context.block_size());
    Arc::new(move || {
        let workers = context.workers();
        let live_workers = workers.iter().map(|worker| worker.worker());
        let capacities = workers.iter().filter_map(|worker| {
            let total_kv_blocks = worker.total_kv_blocks()?;
            let capacity = total_kv_blocks.saturating_mul(block_size);
            Some((
                worker.worker(),
                usize::try_from(capacity).unwrap_or(usize::MAX),
            ))
        });
        Arc::new(WorkerCapacitySnapshot::new(capacities).with_live_workers(live_workers))
    })
}

fn classifier_provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let config: ThunderAgentConfig = parameters.deserialize()?;
    config
        .validate()
        .map_err(|error| RequestClassifierProviderError::new(error.to_string()))?;

    Ok(Arc::new(move |context| {
        let capacity_provider = capacity_provider(context);
        Box::new(
            ThunderAgentClassifier::new(config.clone(), capacity_provider)
                .expect("ThunderAgent configuration was validated during catalog resolution"),
        )
    }))
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(
        super::THUNDERAGENT_CLASSIFIER_TYPE,
        Arc::new(classifier_provider),
    )
}

#[derive(Debug, Error)]
pub(crate) enum ThunderAgentError {
    #[error("session-aware classification requires a request ID")]
    MissingRequestId,

    #[error("request {0:?} is already active in the ThunderAgent classifier")]
    DuplicateRequestId(String),

    #[error("request {0:?} ended while classification was pending")]
    RequestEnded(String),

    #[error("ThunderAgent is already tracking its configured limit of {limit} requests")]
    RequestLimitExceeded { limit: usize },

    #[error("ThunderAgent is already tracking its configured limit of {limit} programs")]
    ProgramLimitExceeded { limit: usize },
}

struct Inner {
    state: Mutex<State>,
    capacity_provider: Arc<dyn WorkerCapacityProvider>,
    scheduler_started: AtomicBool,
}

impl Inner {
    fn register(
        &self,
        request_id: String,
        session_id: String,
        input_tokens: usize,
        progress: RequestProgress,
        session_final: bool,
    ) -> Result<Arc<Notify>, ThunderAgentError> {
        let capacities = self.capacity_provider.snapshot();
        self.state.lock().register(
            RequestRegistration::new(
                request_id,
                session_id,
                input_tokens,
                progress,
                session_final,
            ),
            &capacities,
            Instant::now(),
        )
    }

    fn start_scheduler(self: &Arc<Self>) {
        if self
            .scheduler_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let interval = self.state.lock().scheduler_interval();
        let inner = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(inner) = inner.upgrade() else {
                    break;
                };
                inner.reconcile();
            }
        });
    }

    fn reconcile(&self) {
        if !self.state.lock().needs_reconcile() {
            return;
        }
        let capacities = self.capacity_provider.snapshot();
        let debug_enabled = tracing::enabled!(target: "thunderagent", tracing::Level::DEBUG);
        let warn_enabled = tracing::enabled!(target: "thunderagent", tracing::Level::WARN);
        let (outcome, telemetry) = {
            let mut state = self.state.lock();
            let outcome = state.reconcile(&capacities, Instant::now());
            if !outcome.changed || !(debug_enabled || warn_enabled && outcome.forced_resumes > 0) {
                return;
            }
            (outcome, state.telemetry())
        };
        if outcome.forced_resumes > 0 {
            tracing::warn!(
                target: "thunderagent",
                forced_resumes = outcome.forced_resumes,
                waiting_requests = telemetry.waiting_requests,
                "ThunderAgent deferral timeout forced program resumes"
            );
        }
        tracing::debug!(
            target: "thunderagent",
            programs = telemetry.programs,
            active_programs = telemetry.active_programs,
            paused_programs = telemetry.paused_programs,
            marked_for_pause = telemetry.marked_for_pause,
            waiting_requests = telemetry.waiting_requests,
            tracked_requests = telemetry.tracked_requests,
            "ThunderAgent scheduler state changed"
        );
    }

    fn on_event(&self, event: ClassifyEvent) {
        let request_id = match &event {
            ClassifyEvent::Sent { request_id, .. }
            | ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => request_id,
            _ => return,
        };
        let mut state = self.state.lock();
        if !state.requests.contains_key(request_id) {
            return;
        }
        let capacities = self.capacity_provider.snapshot();
        state.on_event(event, &capacities, Instant::now());
    }

    fn cancel_request(&self, request_id: &str, notify: &Arc<Notify>) {
        let mut state = self.state.lock();
        if state.wait_status(request_id, notify) == WaitStatus::Missing {
            return;
        }
        let capacities = self.capacity_provider.snapshot();
        state.cancel_request(request_id, &capacities, Instant::now());
    }
}

struct PendingClassification {
    inner: Arc<Inner>,
    request_id: String,
    notify: Arc<Notify>,
    armed: bool,
}

impl PendingClassification {
    fn new(inner: Arc<Inner>, request_id: String, notify: Arc<Notify>) -> Self {
        Self {
            inner,
            request_id,
            notify,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingClassification {
    fn drop(&mut self) {
        if self.armed {
            self.inner.cancel_request(&self.request_id, &self.notify);
        }
    }
}

async fn await_release<T>(
    mut pending: PendingClassification,
    value: T,
) -> Result<(T, Option<dynamo_kv_router::protocols::WorkerWithDpRank>), ThunderAgentError> {
    pending.inner.start_scheduler();
    let inner = Arc::clone(&pending.inner);
    let request_id = pending.request_id.clone();
    loop {
        let notify = Arc::clone(&pending.notify);
        let notified = notify.notified();
        let status = inner.state.lock().wait_status(&request_id, &notify);
        match status {
            WaitStatus::Released(worker) => {
                pending.disarm();
                return Ok((value, worker));
            }
            WaitStatus::Missing => {
                pending.disarm();
                return Err(ThunderAgentError::RequestEnded(request_id));
            }
            WaitStatus::Waiting => {}
        }
        notified.await;
    }
}

/// Program-aware flow control implemented on Dynamo's request-classifier seam.
pub struct ThunderAgentClassifier {
    inner: Arc<Inner>,
}

impl ThunderAgentClassifier {
    pub fn new(
        config: ThunderAgentConfig,
        capacity_provider: Arc<dyn WorkerCapacityProvider>,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        tracing::info!(
            target: "thunderagent",
            scheduler_interval_seconds = config.scheduler_interval_seconds,
            resume_timeout_seconds = config.resume_timeout_seconds,
            max_tracked_requests = config.max_tracked_requests,
            "ThunderAgent admission enabled"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::new(config)),
                capacity_provider,
                scheduler_started: AtomicBool::new(false),
            }),
        })
    }
}

#[async_trait]
impl RequestClassifier for ThunderAgentClassifier {
    fn classify(&mut self, request: ClassifyRequest) -> ClassifyFuture {
        let Some(session) = request.session_context() else {
            return Box::pin(async move { Ok(request) });
        };
        let Some(request_id) = request.request_id().map(str::to_owned) else {
            return Box::pin(async move {
                Err(Box::new(ThunderAgentError::MissingRequestId) as Box<ClassifierError>)
            });
        };
        let session_id = session.session_id().to_owned();
        let session_final = session.session_final() == Some(true);
        let input_tokens = request.input_tokens();
        let progress = request.progress().clone();
        let notify = match self.inner.register(
            request_id.clone(),
            session_id,
            input_tokens,
            progress,
            session_final,
        ) {
            Ok(notify) => notify,
            Err(error) => {
                return Box::pin(async move { Err(Box::new(error) as Box<ClassifierError>) });
            }
        };

        let inner = Arc::clone(&self.inner);
        let pending = PendingClassification::new(inner, request_id, notify);
        Box::pin(async move {
            await_release(pending, request)
                .await
                .map(|(mut request, worker)| {
                    if let Some(worker) = worker {
                        request.set_worker_selection_target(worker);
                    }
                    request
                })
                .map_err(|error| Box::new(error) as Box<ClassifierError>)
        })
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        self.inner.on_event(event);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::plugins::request_classifier::RequestClassifierWorker;
    use dynamo_kv_router::protocols::WorkerWithDpRank;

    use super::scheduler::ProgramLifecycle;
    use super::*;

    #[test]
    fn derives_capacity_and_liveness_from_the_host_context() {
        let worker_with_capacity = WorkerWithDpRank::new(1, 0);
        let worker_without_capacity = WorkerWithDpRank::new(2, 0);
        let context = RequestClassifierContext::new(16, move || {
            vec![
                RequestClassifierWorker::new(worker_with_capacity, Some(10)),
                RequestClassifierWorker::new(worker_without_capacity, None),
            ]
        });

        let snapshot = capacity_provider(context).snapshot();
        assert_eq!(
            snapshot.iter().collect::<Vec<_>>(),
            [(worker_with_capacity, 160)]
        );
        assert!(snapshot.is_live(worker_with_capacity));
        assert!(snapshot.is_live(worker_without_capacity));
    }

    fn config() -> ThunderAgentConfig {
        ThunderAgentConfig {
            scheduler_interval_seconds: 0.005,
            resume_timeout_seconds: 1.0,
            session_retention_seconds: 1.0,
            buffer_per_program: 0,
            ..Default::default()
        }
    }

    fn capacities(values: &[(u64, usize)]) -> Arc<WorkerCapacitySnapshot> {
        Arc::new(WorkerCapacitySnapshot::new(values.iter().map(
            |&(worker, capacity)| (WorkerWithDpRank::new(worker, 0), capacity),
        )))
    }

    fn classifier(values: &[(u64, usize)]) -> ThunderAgentClassifier {
        let snapshot = capacities(values);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        ThunderAgentClassifier::new(config(), provider).unwrap()
    }

    #[test]
    fn timeout_logs_are_aggregated_outside_the_lock_and_unchanged_ticks_are_silent() {
        #[derive(Debug, PartialEq)]
        struct Event {
            level: tracing::Level,
            forced_resumes: Option<u64>,
        }
        impl tracing::field::Visit for Event {
            fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                if field.name() == "forced_resumes" {
                    self.forced_resumes = Some(value);
                }
            }
        }
        struct Subscriber {
            inner: std::sync::Weak<Inner>,
            max_level: tracing::Level,
            events: Arc<Mutex<Vec<Event>>>,
        }
        impl tracing::Subscriber for Subscriber {
            fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
                *metadata.level() <= self.max_level
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                assert!(self.inner.upgrade().unwrap().state.try_lock().is_some());
                let mut captured = Event {
                    level: *event.metadata().level(),
                    forced_resumes: None,
                };
                event.record(&mut captured);
                self.events.lock().push(captured);
            }
        }

        for max_level in [tracing::Level::INFO, tracing::Level::DEBUG] {
            let classifier = classifier(&[(1, 100)]);
            let capacity = capacities(&[(1, 100)]);
            let now = Instant::now() - Duration::from_secs(2);
            for id in ["first", "second"] {
                classifier
                    .inner
                    .state
                    .lock()
                    .register(
                        RequestRegistration::new(
                            id.into(),
                            id.into(),
                            120,
                            RequestProgress::new(120).0,
                            false,
                        ),
                        &capacity,
                        now,
                    )
                    .unwrap();
            }
            let events = Arc::new(Mutex::new(Vec::new()));
            tracing::subscriber::with_default(
                Subscriber {
                    inner: Arc::downgrade(&classifier.inner),
                    max_level,
                    events: Arc::clone(&events),
                },
                || {
                    classifier.inner.reconcile();
                    classifier.inner.reconcile();
                },
            );
            let mut expected = vec![Event {
                level: tracing::Level::WARN,
                forced_resumes: Some(2),
            }];
            if max_level == tracing::Level::DEBUG {
                expected.push(Event {
                    level: tracing::Level::DEBUG,
                    forced_resumes: None,
                });
            }
            assert_eq!(*events.lock(), expected);
        }
    }

    #[test]
    fn marked_count_stays_current_after_completion_and_rollback() {
        let mut state = State::new(config());
        let capacity = capacities(&[(1, 100)]);
        let now = Instant::now();
        let (progress, updater) = RequestProgress::new(60);
        state
            .register(
                RequestRegistration::new("first".into(), "first".into(), 60, progress, false),
                &capacity,
                now,
            )
            .unwrap();
        state
            .register(
                RequestRegistration::new(
                    "second".into(),
                    "second".into(),
                    30,
                    RequestProgress::new(30).0,
                    false,
                ),
                &capacity,
                now,
            )
            .unwrap();
        updater.update_context_tokens(120);
        state.reconcile(&capacity, now);
        assert_eq!(state.telemetry().marked_for_pause, 2);
        state.cancel_request("first", &capacity, now);
        assert_eq!(state.telemetry().marked_for_pause, 1);
        state.on_event(
            ClassifyEvent::Completed {
                request_id: "second".into(),
                worker: WorkerWithDpRank::new(1, 0),
                context_tokens: Some(30),
            },
            &capacity,
            now,
        );
        assert_eq!(state.telemetry().marked_for_pause, 0);
        assert_eq!(state.telemetry().paused_programs, 1);
    }

    fn status(state: &State, request_id: &str) -> WaitStatus {
        state.wait_status(request_id, &state.requests[request_id].notify)
    }

    fn register(
        classifier: &ThunderAgentClassifier,
        request_id: &str,
        session_id: &str,
        tokens: usize,
        session_final: bool,
    ) {
        let (progress, _) = RequestProgress::new(tokens);
        classifier
            .inner
            .register(
                request_id.to_owned(),
                session_id.to_owned(),
                tokens,
                progress,
                session_final,
            )
            .unwrap();
    }

    async fn release(classifier: &ThunderAgentClassifier, request_id: &str) {
        await_release(pending(classifier, request_id), ())
            .await
            .unwrap();
    }

    fn pending(classifier: &ThunderAgentClassifier, request_id: &str) -> PendingClassification {
        let notify = classifier
            .inner
            .state
            .lock()
            .requests
            .get(request_id)
            .map(|request| Arc::clone(&request.notify))
            .unwrap();
        PendingClassification::new(Arc::clone(&classifier.inner), request_id.to_owned(), notify)
    }

    async fn sent(classifier: &mut ThunderAgentClassifier, request_id: &str, worker: u64) {
        classifier
            .on_event(ClassifyEvent::Sent {
                request_id: request_id.to_owned(),
                worker: WorkerWithDpRank::new(worker, 0),
            })
            .await;
    }

    async fn completed(classifier: &mut ThunderAgentClassifier, request_id: &str, tokens: usize) {
        classifier
            .on_event(ClassifyEvent::Completed {
                request_id: request_id.to_owned(),
                worker: WorkerWithDpRank::new(1, 0),
                context_tokens: Some(tokens),
            })
            .await;
    }

    #[test]
    fn implements_request_classifier() {
        fn assert_classifier<T: RequestClassifier>() {}
        assert_classifier::<ThunderAgentClassifier>();
    }

    #[tokio::test]
    async fn dropping_an_old_classification_preserves_a_reused_request_id() {
        let mut classifier = classifier(&[(1, 1_000)]);
        register(&classifier, "reused", "old-session", 100, false);
        let old = await_release(pending(&classifier, "reused"), ());
        classifier.inner.on_event(ClassifyEvent::Aborted {
            request_id: "reused".into(),
            worker: None,
            error: None,
        });
        register(&classifier, "reused", "new-session", 100, false);
        let replacement = pending(&classifier, "reused");

        drop(old);

        assert!(
            classifier
                .inner
                .state
                .lock()
                .requests
                .contains_key("reused")
        );
        await_release(replacement, ()).await.unwrap();
        completed(&mut classifier, "reused", 100).await;
    }

    #[tokio::test]
    async fn an_old_waiter_cannot_read_a_reused_request_id() {
        use std::future::{Future, poll_fn};
        use std::task::Poll;

        let classifier = classifier(&[(1, 1_000)]);
        register(&classifier, "reused", "old-session", 1_500, false);
        let mut old = Box::pin(await_release(pending(&classifier, "reused"), ()));
        assert!(poll_fn(|cx| Poll::Ready(old.as_mut().poll(cx).is_pending())).await);
        classifier.inner.on_event(ClassifyEvent::Aborted {
            request_id: "reused".into(),
            worker: None,
            error: None,
        });
        register(&classifier, "reused", "new-session", 100, false);

        assert!(matches!(old.await, Err(ThunderAgentError::RequestEnded(id)) if id == "reused"));
        assert!(
            classifier
                .inner
                .state
                .lock()
                .requests
                .contains_key("reused")
        );
    }

    #[tokio::test]
    async fn a_reused_request_id_does_not_revive_a_canceled_queue_entry() {
        for replacement_session in ["session-a", "session-b"] {
            let mut classifier = classifier(&[(1, 1_000)]);
            register(&classifier, "first-a", "session-a", 100, false);
            register(&classifier, "second-a", "session-a", 100, false);
            register(&classifier, "reused", "session-a", 100, false);
            register(&classifier, "tail-a", "session-a", 100, false);
            drop(pending(&classifier, "reused"));
            register(&classifier, "first-b", "session-b", 100, false);
            register(&classifier, "reused", replacement_session, 100, false);

            completed(&mut classifier, "first-a", 100).await;
            completed(&mut classifier, "second-a", 100).await;
            assert!(matches!(
                status(&classifier.inner.state.lock(), "tail-a"),
                WaitStatus::Released(_)
            ));
            assert_eq!(
                status(&classifier.inner.state.lock(), "reused"),
                WaitStatus::Waiting
            );

            completed(&mut classifier, "tail-a", 100).await;
            completed(&mut classifier, "first-b", 100).await;
            assert!(matches!(
                status(&classifier.inner.state.lock(), "reused"),
                WaitStatus::Released(_)
            ));
            completed(&mut classifier, "reused", 100).await;
            assert!(classifier.inner.state.lock().requests.is_empty());
        }
    }

    #[test]
    fn ignored_and_untracked_events_do_not_read_capacity() {
        use std::sync::atomic::AtomicUsize;

        let reads = Arc::new(AtomicUsize::new(0));
        let snapshot = capacities(&[(1, 1_000)]);
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let reads = Arc::clone(&reads);
            Arc::new(move || {
                reads.fetch_add(1, Ordering::Relaxed);
                Arc::clone(&snapshot)
            })
        };
        let classifier = ThunderAgentClassifier::new(config(), provider).unwrap();
        register(&classifier, "tracked", "session", 100, false);
        reads.store(0, Ordering::Relaxed);
        let worker = WorkerWithDpRank::new(1, 0);
        for event in [
            ClassifyEvent::Responding {
                request_id: "tracked".into(),
                worker,
            },
            ClassifyEvent::Responding {
                request_id: "sessionless".into(),
                worker,
            },
            ClassifyEvent::Sent {
                request_id: "sessionless".into(),
                worker,
            },
            ClassifyEvent::Completed {
                request_id: "sessionless".into(),
                worker,
                context_tokens: Some(100),
            },
            ClassifyEvent::Aborted {
                request_id: "sessionless".into(),
                worker: None,
                error: None,
            },
        ] {
            classifier.inner.on_event(event);
        }
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert!(
            classifier
                .inner
                .state
                .lock()
                .requests
                .contains_key("tracked")
        );
    }

    /// Run with `cargo test -p dynamo-custom-policy-builtin --release benchmark_ignored_events -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual CPU benchmark"]
    fn benchmark_ignored_events() {
        use std::hint::black_box;

        for programs in [256, 10_000] {
            let classifier = classifier(&[(1, 1_000_000)]);
            register(&classifier, "tracked", "session", 100, false);
            {
                let mut state = classifier.inner.state.lock();
                for i in 1..programs {
                    let mut program = scheduler::Program::new(100);
                    program.assigned_worker = Some(WorkerWithDpRank::new(1, 0));
                    state.programs.insert(format!("session-{i}"), program);
                }
            }
            for responding in [true, false] {
                let mut samples = Vec::new();
                for iteration in 0..35 {
                    let start = Instant::now();
                    for _ in 0..100 {
                        let worker = WorkerWithDpRank::new(1, 0);
                        let event = if responding {
                            ClassifyEvent::Responding {
                                request_id: "tracked".into(),
                                worker,
                            }
                        } else {
                            ClassifyEvent::Completed {
                                request_id: "sessionless".into(),
                                worker,
                                context_tokens: None,
                            }
                        };
                        classifier.inner.on_event(black_box(event));
                    }
                    if iteration >= 5 {
                        samples.push(start.elapsed().as_nanos() / 100);
                    }
                }
                samples.sort_unstable();
                println!(
                    "programs={programs} responding={responding} median_ns={} p95_ns={}",
                    samples[samples.len() / 2],
                    samples[samples.len() * 95 / 100]
                );
            }
        }
    }

    #[tokio::test]
    async fn serializes_requests_for_one_session() {
        let mut classifier = classifier(&[(1, 1_000)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-a", 100, false);

        let second = tokio::spawn(await_release(pending(&classifier, "request-2"), ()));
        tokio::task::yield_now().await;
        assert!(!second.is_finished());

        completed(&mut classifier, "request-1", 150).await;
        second.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn selects_a_worker_then_reconciles_the_sent_event() {
        let mut classifier = classifier(&[(1, 300), (2, 1_000)]);
        register(&classifier, "request-1", "session-a", 250, false);
        release(&classifier, "request-1").await;
        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(WorkerWithDpRank::new(1, 0))
        );
        sent(&mut classifier, "request-1", 2).await;
        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(WorkerWithDpRank::new(2, 0))
        );
    }

    #[tokio::test]
    async fn final_session_frees_capacity_at_admission_without_restoring_on_abort() {
        let snapshot = capacities(&[(1, 250)]);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                scheduler_interval_seconds: 1.0,
                ..config()
            },
            provider,
        )
        .unwrap();
        register(&classifier, "request-1", "session-a", 200, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        completed(&mut classifier, "request-1", 200).await;

        register(&classifier, "request-2", "session-b", 100, false);
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );

        register(&classifier, "request-3", "session-a", 1, true);
        release(&classifier, "request-3").await;
        assert!(
            !classifier
                .inner
                .state
                .lock()
                .programs
                .contains_key("session-a")
        );
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );

        let capacity = capacities(&[(1, 250)]);
        classifier
            .inner
            .state
            .lock()
            .reconcile(&capacity, Instant::now());
        release(&classifier, "request-2").await;

        classifier
            .on_event(ClassifyEvent::Aborted {
                request_id: "request-3".to_owned(),
                worker: Some(WorkerWithDpRank::new(1, 0)),
                error: None,
            })
            .await;
        assert!(
            !classifier
                .inner
                .state
                .lock()
                .programs
                .contains_key("session-a")
        );
    }

    #[tokio::test]
    async fn final_session_waits_for_inflight_turn_then_removes_program() {
        let mut classifier = classifier(&[(1, 1_000)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-a", 1, true);

        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );
        assert!(
            classifier
                .inner
                .state
                .lock()
                .programs
                .contains_key("session-a")
        );

        completed(&mut classifier, "request-1", 150).await;
        release(&classifier, "request-2").await;
        assert!(
            !classifier
                .inner
                .state
                .lock()
                .programs
                .contains_key("session-a")
        );
    }

    #[tokio::test]
    async fn capacity_growth_resumes_a_pending_program() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 250)])));
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let current = Arc::clone(&current);
            Arc::new(move || Arc::clone(&current.lock()))
        };
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                buffer_per_program: 100,
                ..config()
            },
            provider,
        )
        .unwrap();

        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-b", 100, false);
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );

        *current.lock() = capacities(&[(1, 500)]);
        release(&classifier, "request-2").await;
    }

    #[tokio::test]
    async fn pressure_pauses_an_acting_program_and_capacity_resumes_it() {
        let current = Arc::new(Mutex::new(capacities(&[(1, 500)])));
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let current = Arc::clone(&current);
            Arc::new(move || Arc::clone(&current.lock()))
        };
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                buffer_per_program: 100,
                scheduler_interval_seconds: 0.05,
                ..config()
            },
            provider,
        )
        .unwrap();

        register(&classifier, "request-1", "session-a", 400, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        completed(&mut classifier, "request-1", 400).await;
        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].lifecycle,
            ProgramLifecycle::Active
        );
        tokio::time::timeout(Duration::from_millis(200), async {
            loop {
                if classifier.inner.state.lock().programs["session-a"].lifecycle
                    == ProgramLifecycle::Paused
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();

        *current.lock() = capacities(&[(1, 1_000)]);
        register(&classifier, "request-2", "session-a", 400, false);
        tokio::time::timeout(
            Duration::from_millis(200),
            release(&classifier, "request-2"),
        )
        .await
        .unwrap();
        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].lifecycle,
            ProgramLifecycle::Active
        );
    }

    #[tokio::test]
    async fn sent_event_reconciles_the_actual_worker() {
        let mut classifier = classifier(&[(1, 1_000), (2, 1_000)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 2).await;

        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(WorkerWithDpRank::new(2, 0))
        );
    }

    #[tokio::test]
    async fn live_worker_without_a_model_card_keeps_its_session_assignment() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);
        let current = Arc::new(Mutex::new(Arc::new(
            WorkerCapacitySnapshot::new([(worker_1, 1_000), (worker_2, 1_000)])
                .with_live_workers([worker_1, worker_2]),
        )));
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let current = Arc::clone(&current);
            Arc::new(move || Arc::clone(&current.lock()))
        };
        let mut classifier = ThunderAgentClassifier::new(config(), provider).unwrap();
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        completed(&mut classifier, "request-1", 100).await;

        *current.lock() = Arc::new(
            WorkerCapacitySnapshot::new([(worker_2, 1_000)])
                .with_live_workers([worker_1, worker_2]),
        );
        register(&classifier, "request-2", "session-a", 100, false);
        release(&classifier, "request-2").await;

        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(worker_1)
        );

        completed(&mut classifier, "request-2", 100).await;
        *current.lock() = Arc::new(
            WorkerCapacitySnapshot::new([(worker_2, 1_000)]).with_live_workers([worker_2]),
        );
        register(&classifier, "request-3", "session-a", 100, false);
        release(&classifier, "request-3").await;
        sent(&mut classifier, "request-3", 2).await;
        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(worker_2)
        );
    }

    #[tokio::test]
    async fn stale_capacity_card_is_not_used_for_assignment() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);
        let snapshot = Arc::new(
            WorkerCapacitySnapshot::new([(worker_1, 1_000), (worker_2, 1_000)])
                .with_live_workers([worker_2]),
        );
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let mut classifier = ThunderAgentClassifier::new(config(), provider).unwrap();

        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 2).await;

        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            Some(worker_2)
        );
    }

    #[tokio::test]
    async fn authoritative_empty_liveness_clears_the_last_assignment() {
        let worker = WorkerWithDpRank::new(1, 0);
        let current = Arc::new(Mutex::new(Arc::new(
            WorkerCapacitySnapshot::new([(worker, 1_000)]).with_live_workers([worker]),
        )));
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let current = Arc::clone(&current);
            Arc::new(move || Arc::clone(&current.lock()))
        };
        let mut classifier = ThunderAgentClassifier::new(config(), provider).unwrap();
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        completed(&mut classifier, "request-1", 100).await;

        *current.lock() = Arc::new(
            WorkerCapacitySnapshot::default()
                .with_live_workers(std::iter::empty::<WorkerWithDpRank>()),
        );
        register(&classifier, "request-2", "session-a", 100, false);
        release(&classifier, "request-2").await;

        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].assigned_worker,
            None
        );
    }

    #[tokio::test]
    async fn missing_capacity_respects_pause_until_timeout() {
        let worker = WorkerWithDpRank::new(1, 0);
        let current = Arc::new(Mutex::new(Arc::new(
            WorkerCapacitySnapshot::new([(worker, 250)]).with_live_workers([worker]),
        )));
        let provider: Arc<dyn WorkerCapacityProvider> = {
            let current = Arc::clone(&current);
            Arc::new(move || Arc::clone(&current.lock()))
        };
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                buffer_per_program: 100,
                resume_timeout_seconds: 0.02,
                ..config()
            },
            provider,
        )
        .unwrap();
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-b", 100, false);
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );

        *current.lock() = Arc::new(WorkerCapacitySnapshot::default().with_live_workers([worker]));
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-2"),
            WaitStatus::Waiting
        );

        tokio::time::timeout(
            Duration::from_millis(100),
            release(&classifier, "request-2"),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn retained_session_expires_before_its_next_request() {
        let snapshot = capacities(&[(1, 1_000)]);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                session_retention_seconds: 0.005,
                ..config()
            },
            provider,
        )
        .unwrap();
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        completed(&mut classifier, "request-1", 100).await;
        tokio::time::sleep(Duration::from_millis(10)).await;

        register(&classifier, "request-2", "session-a", 100, false);
        release(&classifier, "request-2").await;

        assert_eq!(
            classifier.inner.state.lock().programs["session-a"].step_count,
            1
        );
    }

    #[test]
    fn tracked_request_limit_is_enforced_before_allocating_state() {
        let snapshot = capacities(&[(1, 1_000)]);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                max_tracked_requests: 1,
                ..config()
            },
            provider,
        )
        .unwrap();
        register(&classifier, "request-1", "session-a", 100, false);

        let result = classifier.inner.register(
            "request-2".into(),
            "session-b".into(),
            100,
            RequestProgress::new(100).0,
            false,
        );
        assert!(matches!(
            result,
            Err(ThunderAgentError::RequestLimitExceeded { limit: 1 })
        ));
        assert_eq!(classifier.inner.state.lock().requests.len(), 1);
    }

    #[tokio::test]
    async fn retained_programs_are_bounded_by_the_tracking_limit() {
        let snapshot = capacities(&[(1, 1_000)]);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let mut classifier = ThunderAgentClassifier::new(
            ThunderAgentConfig {
                max_tracked_requests: 2,
                ..config()
            },
            provider,
        )
        .unwrap();

        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        completed(&mut classifier, "request-1", 100).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        register(&classifier, "request-2", "session-b", 100, false);
        release(&classifier, "request-2").await;
        completed(&mut classifier, "request-2", 100).await;

        register(&classifier, "request-3", "session-c", 100, false);
        release(&classifier, "request-3").await;

        let state = classifier.inner.state.lock();
        assert_eq!(state.programs.len(), 2);
        assert!(!state.programs.contains_key("session-a"));
        assert!(state.programs.contains_key("session-b"));
        assert!(state.programs.contains_key("session-c"));
    }

    #[tokio::test]
    async fn arrival_tombstones_are_compacted_amortized() {
        let mut classifier = classifier(&[(1, 1_000)]);
        for sequence in 0..2_000 {
            let request_id = format!("request-{sequence}");
            let session_id = format!("session-{sequence}");
            register(&classifier, &request_id, &session_id, 1, true);
            assert!(matches!(
                status(&classifier.inner.state.lock(), &request_id),
                WaitStatus::Released(_)
            ));
            completed(&mut classifier, &request_id, 1).await;
        }

        let state = classifier.inner.state.lock();
        assert!(state.arrival_order.len() <= 256);
        assert!(state.requests.is_empty());
        assert!(state.programs.is_empty());
    }

    #[test]
    fn releases_a_large_backlog_without_per_release_linear_removal() {
        const REQUESTS: usize = 5_000;

        let classifier = classifier(&[(1, 1)]);
        for sequence in 0..REQUESTS {
            register(
                &classifier,
                &format!("request-{sequence}"),
                &format!("session-{sequence}"),
                100,
                false,
            );
        }

        let expanded = capacities(&[(1, 1_000_000)]);
        let mut state = classifier.inner.state.lock();
        state.reconcile(&expanded, Instant::now());

        assert_eq!(
            (0..REQUESTS)
                .filter(|sequence| {
                    matches!(
                        status(&state, &format!("request-{sequence}")),
                        WaitStatus::Released(_)
                    )
                })
                .count(),
            REQUESTS
        );
        assert!(state.arrival_order.len() <= 256);
    }

    #[tokio::test]
    async fn aborted_request_rolls_back_new_program() {
        let mut classifier = classifier(&[(1, 1_000)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        classifier
            .on_event(ClassifyEvent::Aborted {
                request_id: "request-1".to_owned(),
                worker: None,
                error: None,
            })
            .await;

        let state = classifier.inner.state.lock();
        assert!(!state.programs.contains_key("session-a"));
        assert!(!state.requests.contains_key("request-1"));
    }

    #[tokio::test]
    async fn dropping_pending_classification_rolls_back_its_program() {
        let mut classifier = classifier(&[(1, 100)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-b", 100, false);

        let result = tokio::time::timeout(
            Duration::from_millis(20),
            await_release(pending(&classifier, "request-2"), ()),
        )
        .await;
        assert!(result.is_err());

        let state = classifier.inner.state.lock();
        assert!(!state.programs.contains_key("session-b"));
        assert!(!state.requests.contains_key("request-2"));
    }

    #[tokio::test]
    async fn dropping_an_unpolled_classification_rolls_back_its_program() {
        let mut classifier = classifier(&[(1, 100)]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
        sent(&mut classifier, "request-1", 1).await;
        register(&classifier, "request-2", "session-b", 100, false);

        let future = await_release(pending(&classifier, "request-2"), ());
        drop(future);

        let state = classifier.inner.state.lock();
        assert!(!state.programs.contains_key("session-b"));
        assert!(!state.requests.contains_key("request-2"));
    }

    #[tokio::test]
    async fn timeout_forces_release_on_the_least_loaded_worker() {
        let config = ThunderAgentConfig {
            scheduler_interval_seconds: 0.005,
            resume_timeout_seconds: 0.02,
            session_retention_seconds: 1.0,
            buffer_per_program: 0,
            ..Default::default()
        };
        let snapshot = capacities(&[(1, 100)]);
        let provider: Arc<dyn WorkerCapacityProvider> = Arc::new(move || Arc::clone(&snapshot));
        let classifier = ThunderAgentClassifier::new(config, provider).unwrap();
        register(&classifier, "request-1", "session-a", 200, false);

        tokio::time::timeout(
            Duration::from_millis(200),
            release(&classifier, "request-1"),
        )
        .await
        .unwrap();
        assert_eq!(
            status(&classifier.inner.state.lock(), "request-1"),
            WaitStatus::Released(Some(WorkerWithDpRank::new(1, 0)))
        );
    }

    #[tokio::test]
    async fn cold_start_without_model_cards_flows_through() {
        let classifier = classifier(&[]);
        register(&classifier, "request-1", "session-a", 100, false);
        release(&classifier, "request-1").await;
    }
}
