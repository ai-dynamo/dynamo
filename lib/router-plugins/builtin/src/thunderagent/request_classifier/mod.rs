// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ThunderAgent on the router's dispatch gate.
//!
//! The classifier records each session request and returns at once; the gate decides, head by
//! head, whether a queued turn may go and where; lifecycle events settle the ledger. The router
//! queue keeps the waiting turns, in per-session lanes, so a held session blocks only itself.

mod capacity;
mod scheduler;

pub use capacity::{WorkerCapacityProvider, WorkerCapacitySnapshot};

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::time::Instant;

use async_trait::async_trait;
use dynamo_kv_router::plugins::prelude::*;
use parking_lot::Mutex;
use thiserror::Error;

use self::scheduler::{RequestRegistration, State};
use super::{ConfigError, ThunderAgentConfig};

/// How long one capacity snapshot serves gate checks before it is rebuilt from the host's
/// worker view. Checks run once per queue head per dispatch pass; rebuilding per check would
/// allocate per head.
const CAPACITY_SNAPSHOT_TTL: Duration = Duration::from_millis(100);

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

    #[error("ThunderAgent is already tracking its configured limit of {limit} requests")]
    RequestLimitExceeded { limit: usize },
}

struct Inner {
    state: Mutex<State>,
    capacity_provider: Arc<dyn WorkerCapacityProvider>,
    capacity_cache: Mutex<Option<(Instant, Arc<WorkerCapacitySnapshot>)>>,
    recheck: OnceLock<DispatchRecheck>,
}

impl Inner {
    fn capacities(&self) -> Arc<WorkerCapacitySnapshot> {
        let now = Instant::now();
        let mut cache = self.capacity_cache.lock();
        if let Some((taken_at, snapshot)) = cache.as_ref()
            && now.duration_since(*taken_at) < CAPACITY_SNAPSHOT_TTL
        {
            return Arc::clone(snapshot);
        }
        let snapshot = self.capacity_provider.snapshot();
        *cache = Some((now, Arc::clone(&snapshot)));
        snapshot
    }

    /// Wakes the router queue so held heads are asked again.
    fn notify(&self) {
        if let Some(recheck) = self.recheck.get() {
            recheck.notify();
        }
    }

    fn on_event(&self, event: ClassifyEvent) {
        let request_id = match &event {
            ClassifyEvent::Sent { request_id, .. }
            | ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => request_id,
            _ => return,
        };
        let debug_enabled = tracing::enabled!(target: "thunderagent", tracing::Level::DEBUG);
        // The snapshot is taken before the ledger lock, so a cache miss, O(workers), does
        // not hold up the actor's `check`.
        let capacities = self.capacities();
        let telemetry = {
            let mut state = self.state.lock();
            if !state.requests.contains_key(request_id) {
                return;
            }
            if !state.on_event(event, &capacities, Instant::now()) {
                return;
            }
            debug_enabled.then(|| state.telemetry())
        };
        self.notify();
        if let Some(telemetry) = telemetry {
            tracing::debug!(
                target: "thunderagent",
                programs = telemetry.programs,
                active_programs = telemetry.active_programs,
                paused_programs = telemetry.paused_programs,
                marked_for_pause = telemetry.marked_for_pause,
                in_flight = telemetry.in_flight,
                tracked_requests = telemetry.tracked_requests,
                "ThunderAgent ledger changed"
            );
        }
    }
}

/// Program-aware admission on Dynamo's request-classifier and dispatch-gate seams.
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
            resume_timeout_seconds = config.resume_timeout_seconds,
            max_tracked_requests = config.max_tracked_requests,
            "ThunderAgent admission enabled"
        );
        Ok(Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::new(config)),
                capacity_provider,
                capacity_cache: Mutex::new(None),
                recheck: OnceLock::new(),
            }),
        })
    }
}

#[async_trait]
impl RequestClassifier for ThunderAgentClassifier {
    fn classify(
        &mut self,
        request: ClassifyRequest,
        _workers: &dyn WorkerView,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let Some(session) = request.session_context() else {
            return Ok(request);
        };
        let Some(request_id) = request.request_id().map(str::to_owned) else {
            return Err(Box::new(ThunderAgentError::MissingRequestId) as Box<ClassifierError>);
        };
        let registration = RequestRegistration::new(
            request_id,
            session.session_id().to_owned(),
            request.isl_tokens(),
            request.progress().clone(),
            session.session_final() == Some(true),
        )
        .with_pinned_worker(request.pinned_worker());
        self.inner
            .state
            .lock()
            .register(registration, Instant::now())
            .map_err(|error| Box::new(error) as Box<ClassifierError>)?;
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        self.inner.on_event(event);
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        Some(Arc::new(ThunderAgentGate {
            inner: Arc::clone(&self.inner),
        }))
    }
}

struct ThunderAgentGate {
    inner: Arc<Inner>,
}

impl DispatchGate for ThunderAgentGate {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
        // `classify` registers only requests with a session, so nothing here knows a
        // sessionless head: it goes without the snapshot or the ledger lock.
        let (Some(request_id), Some(_)) = (head.request_id(), head.session_context()) else {
            return DispatchDecision::Go { target: None };
        };
        let capacities = self.inner.capacities();
        self.inner
            .state
            .lock()
            .check(request_id, &capacities, Instant::now())
    }

    fn dispatched(&self, head: &DispatchHead<'_>, target: Option<WorkerWithDpRank>) {
        let (Some(request_id), Some(_)) = (head.request_id(), head.session_context()) else {
            return;
        };
        let capacities = self.inner.capacities();
        let changed =
            self.inner
                .state
                .lock()
                .dispatched(request_id, target, &capacities, Instant::now());
        if changed {
            self.inner.notify();
        }
    }

    fn attach(&self, recheck: DispatchRecheck) {
        let _ = self.inner.recheck.set(recheck);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_kv_router::plugins::prelude::*;

    use super::*;

    fn config() -> ThunderAgentConfig {
        ThunderAgentConfig {
            buffer_per_program: 100,
            ..Default::default()
        }
    }

    fn capacities(values: &[(u64, usize)]) -> Arc<WorkerCapacitySnapshot> {
        let workers: Vec<_> = values
            .iter()
            .map(|&(id, _)| WorkerWithDpRank::new(id, 0))
            .collect();
        Arc::new(
            WorkerCapacitySnapshot::new(
                values
                    .iter()
                    .map(|&(id, capacity)| (WorkerWithDpRank::new(id, 0), capacity)),
            )
            .with_live_workers(workers),
        )
    }

    fn classifier(values: &[(u64, usize)]) -> ThunderAgentClassifier {
        let snapshot = capacities(values);
        ThunderAgentClassifier::new(config(), Arc::new(move || Arc::clone(&snapshot))).unwrap()
    }

    fn session_request(request_id: &str, session_id: &str, tokens: usize) -> ClassifyRequest {
        ClassifyRequest::for_test(
            tokens,
            Some(request_id),
            Some(SessionContext::new(session_id.into(), None, None, None)),
        )
    }

    fn head<'a>(
        request_id: &'a str,
        session: &'a SessionContext,
        tokens: usize,
    ) -> DispatchHead<'a> {
        DispatchHead::new(
            Some(request_id),
            Some(session),
            tokens,
            None,
            "default",
            Instant::now(),
        )
    }

    fn attach_counter(gate: &dyn DispatchGate) -> Arc<AtomicUsize> {
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        gate.attach(DispatchRecheck::new(Arc::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        })));
        wakes
    }

    #[test]
    fn derives_capacity_and_liveness_from_the_host_context() {
        let context = RequestClassifierContext::new(16, move || {
            vec![
                WorkerInfo::new(
                    WorkerWithDpRank::new(1, 0),
                    WorkerCapacity::new(None, Some(10)),
                ),
                WorkerInfo::new(WorkerWithDpRank::new(2, 0), WorkerCapacity::default()),
            ]
        });
        let snapshot = capacity_provider(context).snapshot();
        assert_eq!(snapshot.capacity(WorkerWithDpRank::new(1, 0)), Some(160));
        assert!(snapshot.is_live(WorkerWithDpRank::new(2, 0)));
        assert!(!snapshot.is_live(WorkerWithDpRank::new(3, 0)));
    }

    #[tokio::test]
    async fn classify_returns_at_once_and_the_gate_decides() {
        // Room for one 200-token turn plus its buffer, not two.
        let mut classifier = classifier(&[(1, 500)]);
        let gate = classifier
            .dispatch_gate()
            .expect("ThunderAgent exposes a gate");
        let wakes = attach_counter(gate.as_ref());

        // A request without a session passes through untouched.
        let plain = ClassifyRequest::for_test(10, Some("plain"), None);
        assert!(classifier.classify(plain, &NoWorkers).is_ok());

        let session_a = SessionContext::new("a".into(), None, None, None);
        let session_b = SessionContext::new("b".into(), None, None, None);
        assert!(
            classifier
                .classify(session_request("a1", "a", 200), &NoWorkers)
                .is_ok()
        );
        assert!(
            classifier
                .classify(session_request("b1", "b", 200), &NoWorkers)
                .is_ok()
        );

        // a fits; b must wait for room, with a forced-dispatch time.
        assert_eq!(
            gate.check(&head("a1", &session_a, 200)),
            DispatchDecision::Go {
                target: Some(WorkerWithDpRank::new(1, 0))
            }
        );
        gate.dispatched(
            &head("a1", &session_a, 200),
            Some(WorkerWithDpRank::new(1, 0)),
        );
        assert!(matches!(
            gate.check(&head("b1", &session_b, 200)),
            DispatchDecision::Hold { until: Some(_) }
        ));
        // Heads the ledger does not know go through.
        assert_eq!(
            gate.check(&DispatchHead::new(
                Some("unknown"),
                None,
                1,
                None,
                "default",
                Instant::now()
            )),
            DispatchDecision::Go { target: None }
        );

        // a's turn ends at 50 context tokens: the ledger changes and the queue is woken for b.
        let before = wakes.load(Ordering::Relaxed);
        classifier
            .on_event(ClassifyEvent::Sent {
                request_id: "a1".into(),
                worker: WorkerWithDpRank::new(1, 0),
            })
            .await;
        classifier
            .on_event(ClassifyEvent::Completed {
                request_id: "a1".into(),
                worker: WorkerWithDpRank::new(1, 0),
                context_tokens: Some(50),
            })
            .await;
        assert!(wakes.load(Ordering::Relaxed) > before);
        assert_eq!(
            gate.check(&head("b1", &session_b, 200)),
            DispatchDecision::Go {
                target: Some(WorkerWithDpRank::new(1, 0))
            }
        );
    }

    #[tokio::test]
    async fn classification_requires_a_request_id_for_session_traffic() {
        let mut classifier = classifier(&[(1, 400)]);
        let request = ClassifyRequest::for_test(
            10,
            None,
            Some(SessionContext::new("a".into(), None, None, None)),
        );
        let error = classifier.classify(request, &NoWorkers).unwrap_err();
        assert!(error.to_string().contains("request ID"));
    }
}
