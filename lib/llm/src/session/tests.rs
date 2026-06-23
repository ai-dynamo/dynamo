// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use dynamo_runtime::{
    pipeline::{ManyOut, ResponseStream, context::Controller},
    protocols::annotated::Annotated,
};
use futures::{FutureExt, StreamExt, stream};
use parking_lot::Mutex;
use tokio::{sync::Semaphore, time::Instant};

use super::{
    coordinator::{REAPER_INTERVAL, SessionTestHooks},
    *,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SessionAction, SessionControl},
        preprocessor::RoutingHints,
        timing::RequestPhase,
    },
};

struct FakeLifecycle {
    open_succeeds: AtomicBool,
    close_results: Mutex<VecDeque<bool>>,
    close_default: AtomicBool,
    block_open: AtomicBool,
    open_calls: AtomicUsize,
    close_calls: AtomicUsize,
    open_started: Arc<Semaphore>,
    open_release: Arc<Semaphore>,
    close_started: Arc<Semaphore>,
}

impl FakeLifecycle {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            open_succeeds: AtomicBool::new(true),
            close_results: Mutex::new(VecDeque::new()),
            close_default: AtomicBool::new(true),
            block_open: AtomicBool::new(false),
            open_calls: AtomicUsize::new(0),
            close_calls: AtomicUsize::new(0),
            open_started: Arc::new(Semaphore::new(0)),
            open_release: Arc::new(Semaphore::new(0)),
            close_started: Arc::new(Semaphore::new(0)),
        })
    }

    fn close_sequence(&self, results: impl IntoIterator<Item = bool>) {
        *self.close_results.lock() = results.into_iter().collect();
    }
}

#[async_trait]
impl SessionLifecycleBackend for FakeLifecycle {
    async fn open(
        &self,
        _session_id: &str,
        _timeout: Duration,
        _target: SessionTarget,
        _context_id: &str,
    ) -> Result<(), LifecycleError> {
        self.open_calls.fetch_add(1, Ordering::Relaxed);
        if self.block_open.load(Ordering::Relaxed) {
            self.open_started.add_permits(1);
            self.open_release
                .acquire()
                .await
                .expect("open release semaphore")
                .forget();
        }
        if self.open_succeeds.load(Ordering::Relaxed) {
            Ok(())
        } else {
            Err(LifecycleError::new("open failed"))
        }
    }

    async fn close(
        &self,
        _session_id: &str,
        _target: SessionTarget,
        _context_id: &str,
    ) -> Result<(), LifecycleError> {
        self.close_calls.fetch_add(1, Ordering::Relaxed);
        self.close_started.add_permits(1);
        let succeeds = self
            .close_results
            .lock()
            .pop_front()
            .unwrap_or_else(|| self.close_default.load(Ordering::Relaxed));
        if succeeds {
            Ok(())
        } else {
            Err(LifecycleError::new("close failed"))
        }
    }
}

fn hooks() -> SessionTestHooks {
    SessionTestHooks {
        retry_armed: Arc::new(Semaphore::new(0)),
        reaper_armed: Arc::new(Semaphore::new(0)),
        coalesce_waiting: Arc::new(Semaphore::new(0)),
        retry_stopped: Arc::new(Semaphore::new(0)),
        reaper_stopped: Arc::new(Semaphore::new(0)),
    }
}

fn coordinator(lifecycle: Arc<FakeLifecycle>, hooks: SessionTestHooks) -> SessionCoordinator {
    SessionCoordinator::with_lifecycle_and_hooks(lifecycle, hooks)
}

fn control(action: Option<SessionAction>) -> SessionControl {
    SessionControl {
        session_id: "session-1".to_string(),
        action,
        timeout: 10,
    }
}

fn target(worker_id: u64) -> SessionTarget {
    SessionTarget {
        worker_id,
        dp_rank: Some(0),
    }
}

fn binding(kind: SessionKind) -> SessionBinding {
    SessionBinding {
        target: target(1),
        kind,
        timeout: Duration::from_secs(10),
        expires_at: Instant::now() + Duration::from_secs(10),
    }
}

fn empty_stream() -> ManyOut<LlmResponse> {
    ResponseStream::new(Box::pin(stream::empty()), Arc::new(Controller::default()))
}

#[derive(Clone, Copy)]
enum SeedState {
    Absent,
    Opening(SessionKind),
    Bound(SessionKind, usize),
    Closing,
}

fn seed(coordinator: &SessionCoordinator, state: SeedState) {
    let state = match state {
        SeedState::Absent => return,
        SeedState::Opening(kind) => SessionState::Opening(OpeningState {
            revision: 1,
            kind,
            requested_target: None,
            target: Some(target(1)),
            timeout: Duration::from_secs(10),
            notify: Arc::new(tokio::sync::Notify::new()),
        }),
        SeedState::Bound(kind, active_leases) => SessionState::Bound(BoundState {
            revision: 1,
            binding: binding(kind),
            active_leases,
        }),
        SeedState::Closing => SessionState::Closing(ClosingState {
            revision: 1,
            binding: binding(SessionKind::EngineBacked),
            remove_at: Instant::now() + Duration::from_secs(40),
            retry_started: true,
        }),
    };
    coordinator
        .inner
        .sessions
        .insert("session-1".to_string(), state);
}

#[derive(Clone, Copy)]
enum Expected {
    Ready,
    Error,
    Waiting,
}

#[tokio::test]
async fn transition_contract_table() {
    struct Case {
        name: &'static str,
        state: SeedState,
        action: Option<SessionAction>,
        expected: Expected,
    }

    let cases = [
        Case {
            name: "absent open",
            state: SeedState::Absent,
            action: Some(SessionAction::Open),
            expected: Expected::Ready,
        },
        Case {
            name: "absent bind",
            state: SeedState::Absent,
            action: Some(SessionAction::Bind),
            expected: Expected::Ready,
        },
        Case {
            name: "absent continue",
            state: SeedState::Absent,
            action: None,
            expected: Expected::Error,
        },
        Case {
            name: "absent close",
            state: SeedState::Absent,
            action: Some(SessionAction::Close),
            expected: Expected::Error,
        },
        Case {
            name: "opening matching open",
            state: SeedState::Opening(SessionKind::EngineBacked),
            action: Some(SessionAction::Open),
            expected: Expected::Waiting,
        },
        Case {
            name: "opening conflicting bind",
            state: SeedState::Opening(SessionKind::EngineBacked),
            action: Some(SessionAction::Bind),
            expected: Expected::Error,
        },
        Case {
            name: "opening continue",
            state: SeedState::Opening(SessionKind::EngineBacked),
            action: None,
            expected: Expected::Error,
        },
        Case {
            name: "opening close",
            state: SeedState::Opening(SessionKind::EngineBacked),
            action: Some(SessionAction::Close),
            expected: Expected::Error,
        },
        Case {
            name: "bound continue",
            state: SeedState::Bound(SessionKind::EngineBacked, 0),
            action: None,
            expected: Expected::Ready,
        },
        Case {
            name: "bound matching open",
            state: SeedState::Bound(SessionKind::EngineBacked, 0),
            action: Some(SessionAction::Open),
            expected: Expected::Ready,
        },
        Case {
            name: "bound conflicting bind",
            state: SeedState::Bound(SessionKind::EngineBacked, 0),
            action: Some(SessionAction::Bind),
            expected: Expected::Error,
        },
        Case {
            name: "bound close",
            state: SeedState::Bound(SessionKind::EngineBacked, 0),
            action: Some(SessionAction::Close),
            expected: Expected::Ready,
        },
        Case {
            name: "bound close with lease",
            state: SeedState::Bound(SessionKind::EngineBacked, 1),
            action: Some(SessionAction::Close),
            expected: Expected::Error,
        },
        Case {
            name: "closing continue",
            state: SeedState::Closing,
            action: None,
            expected: Expected::Error,
        },
        Case {
            name: "closing open",
            state: SeedState::Closing,
            action: Some(SessionAction::Open),
            expected: Expected::Error,
        },
        Case {
            name: "closing bind",
            state: SeedState::Closing,
            action: Some(SessionAction::Bind),
            expected: Expected::Error,
        },
        Case {
            name: "closing duplicate close",
            state: SeedState::Closing,
            action: Some(SessionAction::Close),
            expected: Expected::Error,
        },
    ];

    for case in cases {
        let lifecycle = FakeLifecycle::new();
        let coordinator = coordinator(lifecycle, hooks());
        seed(&coordinator, case.state);
        let request = control(case.action);
        let mut future = Box::pin(coordinator.begin(&request, None, false));
        match case.expected {
            Expected::Ready => assert!(
                future.as_mut().now_or_never().unwrap().is_ok(),
                "{}",
                case.name
            ),
            Expected::Error => assert!(
                future.as_mut().now_or_never().unwrap().is_err(),
                "{}",
                case.name
            ),
            Expected::Waiting => {
                assert!(future.as_mut().now_or_never().is_none(), "{}", case.name)
            }
        }
    }
}

#[tokio::test]
async fn duplicate_ensure_contract_is_parameterized() {
    struct Case {
        name: &'static str,
        kind: SessionKind,
        action: SessionAction,
        requested: Option<SessionTarget>,
        timeout: u64,
        succeeds: bool,
    }
    let cases = [
        Case {
            name: "matching open",
            kind: SessionKind::EngineBacked,
            action: SessionAction::Open,
            requested: Some(target(1)),
            timeout: 10,
            succeeds: true,
        },
        Case {
            name: "matching open without rank",
            kind: SessionKind::EngineBacked,
            action: SessionAction::Open,
            requested: Some(SessionTarget {
                worker_id: 1,
                dp_rank: None,
            }),
            timeout: 10,
            succeeds: true,
        },
        Case {
            name: "open target conflict",
            kind: SessionKind::EngineBacked,
            action: SessionAction::Open,
            requested: Some(target(2)),
            timeout: 10,
            succeeds: false,
        },
        Case {
            name: "open timeout conflict",
            kind: SessionKind::EngineBacked,
            action: SessionAction::Open,
            requested: Some(target(1)),
            timeout: 11,
            succeeds: false,
        },
        Case {
            name: "matching bind",
            kind: SessionKind::RouterOnly,
            action: SessionAction::Bind,
            requested: Some(target(1)),
            timeout: 10,
            succeeds: true,
        },
        Case {
            name: "bind kind conflict",
            kind: SessionKind::EngineBacked,
            action: SessionAction::Bind,
            requested: Some(target(1)),
            timeout: 10,
            succeeds: false,
        },
    ];

    for case in cases {
        let coordinator = coordinator(FakeLifecycle::new(), hooks());
        seed(&coordinator, SeedState::Bound(case.kind, 0));
        let request = SessionControl {
            session_id: "session-1".to_string(),
            action: Some(case.action),
            timeout: case.timeout,
        };
        assert_eq!(
            coordinator
                .begin(&request, case.requested, false)
                .await
                .is_ok(),
            case.succeeds,
            "{}",
            case.name
        );
    }
}

#[tokio::test]
async fn concurrent_open_coalesces_backend_setup() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.block_open.store(true, Ordering::Relaxed);
    let hooks = hooks();
    let coordinator = coordinator(lifecycle.clone(), hooks.clone());
    let first_coordinator = coordinator.clone();
    let first = tokio::spawn(async move {
        let mut operation = first_coordinator
            .begin(&control(Some(SessionAction::Open)), None, false)
            .await
            .unwrap();
        operation.selected(target(1), "first").await.unwrap();
        operation.into_stream(empty_stream()).unwrap()
    });

    lifecycle.open_started.acquire().await.unwrap().forget();
    let second_coordinator = coordinator.clone();
    let second = tokio::spawn(async move {
        second_coordinator
            .begin(&control(Some(SessionAction::Open)), None, false)
            .await
            .unwrap()
    });
    hooks.coalesce_waiting.acquire().await.unwrap().forget();
    lifecycle.open_release.add_permits(1);

    let first_stream = first.await.unwrap();
    let second_operation = second.await.unwrap();
    assert_eq!(lifecycle.open_calls.load(Ordering::Relaxed), 1);
    drop(first_stream);
    drop(second_operation);
}

#[tokio::test(start_paused = true)]
async fn active_lease_prevents_expiry() {
    let hooks = hooks();
    let coordinator = coordinator(FakeLifecycle::new(), hooks.clone());
    coordinator.inner.sessions.insert(
        "session-1".to_string(),
        SessionState::Bound(BoundState {
            revision: 1,
            binding: SessionBinding {
                expires_at: Instant::now() + Duration::from_secs(1),
                ..binding(SessionKind::RouterOnly)
            },
            active_leases: 1,
        }),
    );
    hooks.reaper_armed.acquire().await.unwrap().forget();
    tokio::time::advance(REAPER_INTERVAL + Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert!(
        coordinator
            .inner
            .sessions
            .get("session-1")
            .is_some_and(|state| matches!(
                state.value(),
                SessionState::Bound(bound) if bound.active_leases == 1
            ))
    );
}

#[tokio::test(start_paused = true)]
async fn stream_cleanup_releases_lease_after_eof_error_or_drop() {
    enum Completion {
        Eof,
        Error,
        Drop,
    }

    for completion in [Completion::Eof, Completion::Error, Completion::Drop] {
        let coordinator = coordinator(FakeLifecycle::new(), hooks());
        let mut initial_binding = binding(SessionKind::RouterOnly);
        initial_binding.expires_at = Instant::now() + Duration::from_secs(1);
        coordinator.inner.sessions.insert(
            "session-1".to_string(),
            SessionState::Bound(BoundState {
                revision: 1,
                binding: initial_binding,
                active_leases: 0,
            }),
        );
        let mut operation = coordinator
            .begin(&control(None), None, false)
            .await
            .unwrap();
        operation.selected(target(1), "continue").await.unwrap();
        let inner = match completion {
            Completion::Error => ResponseStream::new(
                Box::pin(stream::iter(vec![Annotated::from_error("stream failed")])),
                Arc::new(Controller::default()),
            ),
            Completion::Eof | Completion::Drop => empty_stream(),
        };
        let mut stream = operation.into_stream(inner).unwrap();
        match completion {
            Completion::Eof => assert!(stream.next().await.is_none()),
            Completion::Error => {
                assert!(stream.next().await.unwrap().is_error());
                assert!(stream.next().await.is_none());
            }
            Completion::Drop => drop(stream),
        }

        let Some(state) = coordinator.inner.sessions.get("session-1") else {
            panic!("session should remain bound after continuation cleanup");
        };
        let SessionState::Bound(bound) = state.value() else {
            panic!("session should remain bound after continuation cleanup");
        };
        assert_eq!(bound.active_leases, 0);
        assert!(bound.binding.expires_at >= Instant::now() + Duration::from_secs(10));
    }
}

#[tokio::test]
async fn pre_dispatch_drop_rolls_back_opening_state() {
    let lifecycle = FakeLifecycle::new();
    let coordinator = coordinator(lifecycle.clone(), hooks());
    let operation = coordinator
        .begin(&control(Some(SessionAction::Bind)), None, false)
        .await
        .unwrap();
    drop(operation);
    assert!(coordinator.inner.sessions.is_empty());

    let mut operation = coordinator
        .begin(&control(Some(SessionAction::Open)), None, false)
        .await
        .unwrap();
    operation.selected(target(1), "open").await.unwrap();
    drop(operation);
    lifecycle.close_started.acquire().await.unwrap().forget();
    tokio::task::yield_now().await;
    assert!(coordinator.inner.sessions.is_empty());
}

#[tokio::test(start_paused = true)]
async fn engine_expiry_enters_closing() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.close_default.store(false, Ordering::Relaxed);
    let hooks = hooks();
    let coordinator = coordinator(lifecycle.clone(), hooks.clone());
    hooks.reaper_armed.acquire().await.unwrap().forget();
    coordinator.inner.sessions.insert(
        "session-1".to_string(),
        SessionState::Bound(BoundState {
            revision: 1,
            binding: SessionBinding {
                expires_at: Instant::now() + Duration::from_secs(1),
                ..binding(SessionKind::EngineBacked)
            },
            active_leases: 0,
        }),
    );
    tokio::time::advance(REAPER_INTERVAL).await;
    lifecycle.close_started.acquire().await.unwrap().forget();
    hooks.retry_armed.acquire().await.unwrap().forget();
    assert!(
        coordinator
            .inner
            .sessions
            .get("session-1")
            .is_some_and(|state| matches!(state.value(), SessionState::Closing(_)))
    );
    drop(coordinator);
    hooks.reaper_stopped.acquire().await.unwrap().forget();
    hooks.retry_stopped.acquire().await.unwrap().forget();
}

#[tokio::test(start_paused = true)]
async fn lazy_engine_expiry_starts_close_and_remains_unroutable() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.close_default.store(false, Ordering::Relaxed);
    let hooks = hooks();
    let coordinator = coordinator(lifecycle.clone(), hooks.clone());
    hooks.reaper_armed.acquire().await.unwrap().forget();
    coordinator.inner.sessions.insert(
        "session-1".to_string(),
        SessionState::Bound(BoundState {
            revision: 1,
            binding: SessionBinding {
                expires_at: Instant::now(),
                ..binding(SessionKind::EngineBacked)
            },
            active_leases: 0,
        }),
    );

    let result = coordinator.begin(&control(None), None, false).await;
    assert!(result.is_err());
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("expired and is closing")
    );
    lifecycle.close_started.acquire().await.unwrap().forget();
    hooks.retry_armed.acquire().await.unwrap().forget();

    assert!(
        coordinator
            .inner
            .sessions
            .get("session-1")
            .is_some_and(|state| matches!(state.value(), SessionState::Closing(_)))
    );
    assert!(
        coordinator
            .begin(&control(None), None, false)
            .await
            .is_err()
    );
    assert_eq!(lifecycle.close_calls.load(Ordering::Relaxed), 1);

    drop(coordinator);
    hooks.reaper_stopped.acquire().await.unwrap().forget();
    hooks.retry_stopped.acquire().await.unwrap().forget();
}

#[tokio::test]
async fn validation_contract_table() {
    enum Case {
        ZeroTimeout,
        DirectWithoutWorker,
        RankWithoutWorker,
        TargetMismatch,
    }
    for case in [
        Case::ZeroTimeout,
        Case::DirectWithoutWorker,
        Case::RankWithoutWorker,
        Case::TargetMismatch,
    ] {
        let coordinator = coordinator(FakeLifecycle::new(), hooks());
        let result = match case {
            Case::ZeroTimeout => {
                let mut request = control(Some(SessionAction::Bind));
                request.timeout = 0;
                coordinator.begin(&request, None, false).await.map(|_| ())
            }
            Case::DirectWithoutWorker => coordinator
                .begin(&control(Some(SessionAction::Bind)), None, true)
                .await
                .map(|_| ()),
            Case::RankWithoutWorker => {
                let request = PreprocessedRequest::builder()
                    .model("model".to_string())
                    .token_ids(vec![1])
                    .stop_conditions(Default::default())
                    .sampling_options(Default::default())
                    .output_options(Default::default())
                    .routing(Some(RoutingHints {
                        session_control: Some(control(Some(SessionAction::Bind))),
                        dp_rank: Some(1),
                        ..Default::default()
                    }))
                    .build()
                    .unwrap();
                explicit_target(&request, RequestPhase::Aggregated).map(|_| ())
            }
            Case::TargetMismatch => {
                seed(&coordinator, SeedState::Bound(SessionKind::RouterOnly, 0));
                coordinator
                    .begin(
                        &control(None),
                        Some(SessionTarget {
                            worker_id: 2,
                            dp_rank: Some(0),
                        }),
                        false,
                    )
                    .await
                    .map(|_| ())
            }
        };
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn query_only_validation_does_not_mutate_session_or_call_lifecycle() {
    let lifecycle = FakeLifecycle::new();
    let coordinator = coordinator(lifecycle.clone(), hooks());

    assert_eq!(
        coordinator
            .query_target(&control(Some(SessionAction::Open)), None)
            .unwrap(),
        None
    );
    assert!(coordinator.inner.sessions.is_empty());

    seed(&coordinator, SeedState::Bound(SessionKind::EngineBacked, 0));
    let (expires_at, active_leases) = {
        let Some(state) = coordinator.inner.sessions.get("session-1") else {
            panic!("expected bound session");
        };
        let SessionState::Bound(bound) = state.value() else {
            panic!("expected bound session");
        };
        (bound.binding.expires_at, bound.active_leases)
    };
    assert_eq!(
        coordinator.query_target(&control(None), None).unwrap(),
        Some(target(1))
    );
    assert_eq!(
        coordinator
            .query_target(&control(Some(SessionAction::Close)), None)
            .unwrap(),
        Some(target(1))
    );

    let Some(state) = coordinator.inner.sessions.get("session-1") else {
        panic!("query-only request changed session state");
    };
    let SessionState::Bound(bound) = state.value() else {
        panic!("query-only request changed session state");
    };
    assert_eq!(bound.binding.expires_at, expires_at);
    assert_eq!(bound.active_leases, active_leases);
    assert_eq!(lifecycle.open_calls.load(Ordering::Relaxed), 0);
    assert_eq!(lifecycle.close_calls.load(Ordering::Relaxed), 0);
}

async fn close_stream(coordinator: &SessionCoordinator) -> ManyOut<LlmResponse> {
    seed(coordinator, SeedState::Bound(SessionKind::EngineBacked, 0));
    let mut operation = coordinator
        .begin(&control(Some(SessionAction::Close)), None, false)
        .await
        .unwrap();
    operation.selected(target(1), "close").await.unwrap();
    operation.into_stream(empty_stream()).unwrap()
}

#[tokio::test(start_paused = true)]
async fn close_retry_keeps_tombstone_until_confirmed_success() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.close_sequence([false, true]);
    let hooks = hooks();
    let coordinator = coordinator(lifecycle.clone(), hooks.clone());
    hooks.reaper_armed.acquire().await.unwrap().forget();
    let stream = close_stream(&coordinator).await;
    drop(stream);
    lifecycle.close_started.acquire().await.unwrap().forget();
    hooks.retry_armed.acquire().await.unwrap().forget();

    assert!(
        coordinator
            .inner
            .sessions
            .get("session-1")
            .is_some_and(|state| matches!(state.value(), SessionState::Closing(_)))
    );
    assert!(
        coordinator
            .begin(&control(None), None, false)
            .await
            .is_err()
    );

    tokio::time::advance(Duration::from_secs(1)).await;
    lifecycle.close_started.acquire().await.unwrap().forget();
    tokio::task::yield_now().await;
    assert!(coordinator.inner.sessions.get("session-1").is_none());
    hooks.retry_stopped.acquire().await.unwrap().forget();
    drop(coordinator);
    hooks.reaper_stopped.acquire().await.unwrap().forget();
}

#[tokio::test(start_paused = true)]
async fn always_failing_close_expires_tombstone() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.close_default.store(false, Ordering::Relaxed);
    let hooks = hooks();
    let coordinator = coordinator(lifecycle.clone(), hooks.clone());
    hooks.reaper_armed.acquire().await.unwrap().forget();
    let stream = close_stream(&coordinator).await;
    drop(stream);
    lifecycle.close_started.acquire().await.unwrap().forget();
    hooks.retry_armed.acquire().await.unwrap().forget();

    let mut observed_calls = lifecycle.close_calls.load(Ordering::Relaxed);
    for _ in 0..45 {
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        if coordinator.inner.sessions.get("session-1").is_none() {
            break;
        }
        let calls = lifecycle.close_calls.load(Ordering::Relaxed);
        if calls > observed_calls {
            hooks.retry_armed.acquire().await.unwrap().forget();
            observed_calls = calls;
        }
    }
    assert!(coordinator.inner.sessions.get("session-1").is_none());
    hooks.retry_stopped.acquire().await.unwrap().forget();
    drop(coordinator);
    hooks.reaper_stopped.acquire().await.unwrap().forget();
}

#[tokio::test]
async fn failed_open_returns_to_absent() {
    let lifecycle = FakeLifecycle::new();
    lifecycle.open_succeeds.store(false, Ordering::Relaxed);
    let coordinator = coordinator(lifecycle.clone(), hooks());
    let mut operation = coordinator
        .begin(&control(Some(SessionAction::Open)), None, false)
        .await
        .unwrap();
    assert!(operation.selected(target(1), "open").await.is_err());
    assert!(coordinator.inner.sessions.get("session-1").is_none());
    assert_eq!(lifecycle.close_calls.load(Ordering::Relaxed), 0);
}
