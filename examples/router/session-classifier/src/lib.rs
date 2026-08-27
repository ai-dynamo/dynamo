// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A classifier that permits one active request per session in one router process.
//!
//! Requests for different sessions proceed independently. A request for an
//! active session waits until the active request completes or aborts.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use dynamo_kv_router::scheduling::{
    ClassifyEvent, ClassifyFuture, ClassifyRequest, RequestClassifier,
};
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Permits at most one active request for each session ID.
#[derive(Clone, Default)]
pub struct SessionClassifier {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    sessions: HashMap<String, SessionState>,
    reservations: HashMap<String, Reservation>,
}

struct SessionState {
    semaphore: Arc<Semaphore>,
    users: usize,
}

struct SessionLease {
    inner: Weak<Inner>,
    session_id: String,
    semaphore: Arc<Semaphore>,
}

struct Reservation {
    _lease: SessionLease,
    _permit: OwnedSemaphorePermit,
}

impl Inner {
    fn lease(self: &Arc<Self>, session_id: String) -> SessionLease {
        let mut state = self.state.lock();
        let session = state
            .sessions
            .entry(session_id.clone())
            .or_insert_with(|| SessionState {
                semaphore: Arc::new(Semaphore::new(1)),
                users: 0,
            });
        session.users += 1;
        SessionLease {
            inner: Arc::downgrade(self),
            session_id,
            semaphore: Arc::clone(&session.semaphore),
        }
    }

    fn release(&self, request_id: &str) {
        let reservation = self.state.lock().reservations.remove(request_id);
        drop(reservation);
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut state = inner.state.lock();
        let Some(session) = state.sessions.get_mut(&self.session_id) else {
            return;
        };
        debug_assert!(Arc::ptr_eq(&session.semaphore, &self.semaphore));
        session.users -= 1;
        if session.users == 0 {
            state.sessions.remove(&self.session_id);
        }
    }
}

impl RequestClassifier for SessionClassifier {
    fn classify(&self, request: ClassifyRequest) -> ClassifyFuture {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let (Some(request_id), Some(session)) =
                (request.request_id(), request.session_context())
            else {
                return request;
            };
            let request_id = request_id.to_owned();
            let lease = inner.lease(session.session_id().to_owned());
            let Ok(permit) = Arc::clone(&lease.semaphore).acquire_owned().await else {
                return request;
            };

            let replaced = inner.state.lock().reservations.insert(
                request_id,
                Reservation {
                    _lease: lease,
                    _permit: permit,
                },
            );
            debug_assert!(replaced.is_none(), "request IDs are unique while live");
            request
        })
    }

    fn on_event(&mut self, event: ClassifyEvent<'_>) {
        let request_id = match event {
            ClassifyEvent::Completed { request_id, .. }
            | ClassifyEvent::Aborted { request_id, .. } => request_id,
            _ => return,
        };
        self.inner.release(request_id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{
        LocalScheduler, OverlapSignals, PolicyProfile, RequestLifecycle, ScheduleMode,
        ScheduleRequest, SchedulingResponse, SessionContext,
    };
    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, DefaultWorkerSelector, NoopSequencePublisher, RouterQueuePolicy,
    };
    use tokio::sync::watch;
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[derive(Clone, PartialEq)]
    struct TestWorkerConfig;

    impl WorkerConfigLike for TestWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            Some(64)
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(64)
        }

        fn taints(&self) -> &HashSet<String> {
            static EMPTY: std::sync::LazyLock<HashSet<String>> =
                std::sync::LazyLock::new(HashSet::new);
            &EMPTY
        }
    }

    type TestScheduler = LocalScheduler<NoopSequencePublisher, TestWorkerConfig>;

    fn request(request_id: &str, session_id: &str) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: request_id.to_owned(),
            },
            token_seq: None,
            block_hashes: None,
            isl_tokens: 8,
            lora_name: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: Some(SessionContext::new(
                session_id.to_owned(),
                None,
                None,
                None,
                None,
            )),
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
        }
    }

    fn scheduler() -> (Arc<TestScheduler>, Arc<Inner>, CancellationToken) {
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            64,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "session-classifier-test",
        ));
        let (_worker_configs, worker_configs) =
            watch::channel(HashMap::from([(0, TestWorkerConfig)]));
        let cancel = CancellationToken::new();
        let classifier = SessionClassifier::default();
        let classifier_state = Arc::clone(&classifier.inner);
        let scheduler = LocalScheduler::new_with_policy_profile_and_request_classifier(
            slots,
            worker_configs,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            64,
            DefaultWorkerSelector::new(None, "session-classifier-test"),
            None,
            None,
            None,
            None,
            None,
            Duration::from_secs(60),
            true,
            cancel.clone(),
            "session-classifier-test",
            false,
            Box::new(classifier),
        )
        .unwrap();
        (Arc::new(scheduler), classifier_state, cancel)
    }

    async fn schedule(
        scheduler: &TestScheduler,
        request_id: &str,
        session_id: &str,
    ) -> (SchedulingResponse, RequestLifecycle) {
        let mut lifecycle = scheduler
            .begin_request_lifecycle(request_id)
            .unwrap()
            .expect("classifier is enabled");
        let response = scheduler
            .schedule_request(request(request_id, session_id))
            .await
            .unwrap();
        lifecycle.selected(response.best_worker);
        (response, lifecycle)
    }

    fn complete(response: &SchedulingResponse, lifecycle: &mut RequestLifecycle) {
        lifecycle.sent(response.best_worker);
        lifecycle.responding();
        lifecycle.complete(Some(8));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn serializes_each_session_without_blocking_other_sessions() {
        let (scheduler, state, cancel) = scheduler();
        let (first, mut first_lifecycle) = schedule(&scheduler, "request-1", "session-a").await;

        let second_scheduler = Arc::clone(&scheduler);
        let second =
            tokio::spawn(
                async move { schedule(&second_scheduler, "request-2", "session-a").await },
            );
        while state
            .state
            .lock()
            .sessions
            .get("session-a")
            .is_none_or(|session| session.users < 2)
        {
            tokio::task::yield_now().await;
        }
        assert!(!second.is_finished());

        let (_other, mut other_lifecycle) = schedule(&scheduler, "request-3", "session-b").await;
        other_lifecycle.abort(None);
        scheduler.free("request-3").await.unwrap();

        complete(&first, &mut first_lifecycle);
        scheduler.free("request-1").await.unwrap();
        let (_second, mut second_lifecycle) = second.await.unwrap();
        second_lifecycle.abort(None);
        scheduler.free("request-2").await.unwrap();

        let state = state.state.lock();
        assert!(state.reservations.is_empty());
        assert!(state.sessions.is_empty());
        cancel.cancel();
    }
}
