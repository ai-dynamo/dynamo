// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use dynamo_runtime::{
    discovery::{
        ClaimCloseOutcome, ClaimEvent, ClaimOutcome, ClaimPayload, ClaimPayloadFuture, Discovery,
        DiscoveryInstance, DiscoveryQuery, DiscoverySpec, DiscoveryStream,
    },
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, ResponseStream, context::Controller},
    protocols::maybe_error::MaybeError,
};
use futures::{StreamExt, stream};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use super::coordinator::AffinityAcquire;
use super::{AffinityCoordinator, AffinityTarget, LlmResponse, affinity_id, explicit_target};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        llm_backend::LLMEngineOutput,
        preprocessor::RoutingHints,
        timing::RequestPhase,
    },
    types::Annotated,
};

fn session_id() -> SessionAffinityId {
    SessionAffinityId::new("session-1")
}

fn target(worker_id: u64, dp_rank: Option<u32>) -> AffinityTarget {
    AffinityTarget { worker_id, dp_rank }
}

fn coordinator() -> AffinityCoordinator {
    AffinityCoordinator::new(Duration::from_secs(10)).unwrap()
}

struct ClaimTestDiscovery {
    claims: Mutex<HashMap<String, ClaimPayload>>,
    create_calls: AtomicUsize,
    close_calls: AtomicUsize,
    events: Mutex<Option<broadcast::Sender<ClaimEvent>>>,
}

impl ClaimTestDiscovery {
    fn new(event_capacity: usize) -> Arc<Self> {
        let (events, _) = broadcast::channel(event_capacity);
        Arc::new(Self {
            claims: Mutex::new(HashMap::new()),
            create_calls: AtomicUsize::new(0),
            close_calls: AtomicUsize::new(0),
            events: Mutex::new(Some(events)),
        })
    }

    fn emit(&self, event: ClaimEvent) {
        if let Some(events) = self.events.lock().unwrap().as_ref() {
            let _ = events.send(event);
        }
    }

    fn disconnect(&self) {
        self.events.lock().unwrap().take();
    }
}

#[async_trait]
impl Discovery for ClaimTestDiscovery {
    fn instance_id(&self) -> u64 {
        1
    }

    async fn register_internal(&self, _spec: DiscoverySpec) -> anyhow::Result<DiscoveryInstance> {
        anyhow::bail!("not used by claim tests")
    }

    async fn unregister(&self, _instance: DiscoveryInstance) -> anyhow::Result<()> {
        Ok(())
    }

    async fn list(&self, _query: DiscoveryQuery) -> anyhow::Result<Vec<DiscoveryInstance>> {
        Ok(Vec::new())
    }

    async fn list_and_watch(
        &self,
        _query: DiscoveryQuery,
        _cancel_token: Option<CancellationToken>,
    ) -> anyhow::Result<DiscoveryStream> {
        Ok(Box::pin(stream::pending()))
    }

    async fn create_or_get_claim(
        &self,
        key: &str,
        proposed_payload: &mut ClaimPayloadFuture<'_>,
    ) -> anyhow::Result<ClaimOutcome> {
        self.create_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(payload) = self.claims.lock().unwrap().get(key).cloned() {
            return Ok(ClaimOutcome::Existing(payload));
        }

        let proposed = proposed_payload.as_mut().await?;
        let mut claims = self.claims.lock().unwrap();
        if let Some(payload) = claims.get(key).cloned() {
            return Ok(ClaimOutcome::Existing(payload));
        }
        claims.insert(key.to_string(), proposed.clone());
        Ok(ClaimOutcome::Created(proposed))
    }

    async fn close_claim(&self, key: &str) -> anyhow::Result<ClaimCloseOutcome> {
        self.close_calls.fetch_add(1, Ordering::Relaxed);
        if self.claims.lock().unwrap().remove(key).is_some() {
            self.emit(ClaimEvent::Delete(key.to_string()));
        }
        Ok(ClaimCloseOutcome::Closed)
    }

    fn subscribe_claim_events(&self) -> Option<broadcast::Receiver<ClaimEvent>> {
        self.events
            .lock()
            .unwrap()
            .as_ref()
            .map(broadcast::Sender::subscribe)
    }
}

fn distributed_coordinator(discovery: Arc<dyn Discovery>) -> AffinityCoordinator {
    AffinityCoordinator::new_distributed(
        Duration::from_secs(10),
        "ns/component/endpoint".to_string(),
        discovery,
    )
    .unwrap()
}

fn target_payload(target: AffinityTarget) -> ClaimPayloadFuture<'static> {
    Box::pin(async move { Ok(serde_json::to_value(target)?) })
}

async fn resolve_local(
    coordinator: &AffinityCoordinator,
    session_id: &SessionAffinityId,
    selected: AffinityTarget,
) -> super::ResolvedAffinity {
    coordinator
        .acquire(session_id)
        .await
        .unwrap()
        .resolve(target_payload(selected))
        .await
        .unwrap()
}

async fn wait_for_cached_target(
    coordinator: &AffinityCoordinator,
    session_id: &SessionAffinityId,
    expected: Option<AffinityTarget>,
) {
    for _ in 0..100 {
        if coordinator.query_target(session_id).unwrap() == expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(coordinator.query_target(session_id).unwrap(), expected);
}

fn response_stream(items: usize) -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    let items = (0..items).map(|_| Annotated::from_data(LLMEngineOutput::default()));
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
}

fn error_response_stream() -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    ResponseStream::new(
        Box::pin(stream::iter([Annotated::from_error("backend failed")])),
        Arc::new(Controller::default()),
    )
}

fn cancelled_response_stream() -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    let controller = Controller::new("cancelled-stream".to_string());
    controller.stop();
    ResponseStream::new(Box::pin(stream::empty()), Arc::new(controller))
}

async fn assert_binding_expires_after_refreshed_ttl(coordinator: &AffinityCoordinator) {
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&session_id()).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id()).unwrap(), None);
}

fn request_with_routing(routing: RoutingHints) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("test".to_string())
        .token_ids(vec![1])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .routing(Some(routing))
        .build()
        .unwrap()
}

#[test]
fn session_affinity_explicit_targets_are_phase_local_and_preserve_rank_zero() {
    let request = request_with_routing(RoutingHints {
        backend_instance_id: Some(1),
        prefill_worker_id: Some(2),
        decode_worker_id: Some(3),
        dp_rank: Some(0),
        prefill_dp_rank: Some(4),
        ..Default::default()
    });

    assert_eq!(
        explicit_target(&request, RequestPhase::Prefill).unwrap(),
        Some(target(2, Some(4)))
    );
    assert_eq!(
        explicit_target(&request, RequestPhase::Decode).unwrap(),
        Some(target(3, Some(0)))
    );
    assert_eq!(
        explicit_target(&request, RequestPhase::Aggregated).unwrap(),
        Some(target(3, Some(0)))
    );

    let decode_only = request_with_routing(RoutingHints {
        decode_worker_id: Some(3),
        ..Default::default()
    });
    assert_eq!(
        explicit_target(&decode_only, RequestPhase::Aggregated).unwrap(),
        Some(target(3, None))
    );

    let rank_without_worker = request_with_routing(RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    assert!(explicit_target(&rank_without_worker, RequestPhase::Decode).is_err());
}

#[test]
fn session_affinity_context_type_errors_are_preserved() {
    let mut request = Context::new(request_with_routing(RoutingHints::default()));
    request.insert(SESSION_AFFINITY_CONTEXT_KEY, "wrong type".to_string());

    let error = affinity_id(&request).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid session affinity context")
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initialization_is_atomic() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id()).await.unwrap();
    assert!(matches!(&first, AffinityAcquire::Initialize(_)));

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id()).await });
    coordinator.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first = first
        .resolve(target_payload(target(7, Some(0))))
        .await
        .unwrap();
    let second = waiter
        .await
        .unwrap()
        .unwrap()
        .resolve(target_payload(target(8, Some(0))))
        .await
        .unwrap();
    assert_eq!(second.target(), target(7, Some(0)));
    drop(first);
    drop(second);
}

#[tokio::test]
async fn distributed_existing_winner_skips_proposal_and_cache_hits_skip_discovery() {
    let discovery = ClaimTestDiscovery::new(16);
    let first = distributed_coordinator(discovery.clone());
    let second = distributed_coordinator(discovery.clone());
    let session_id = session_id();

    let created = first
        .acquire(&session_id)
        .await
        .unwrap()
        .resolve(target_payload(target(7, Some(0))))
        .await
        .unwrap();
    assert_eq!(created.target(), target(7, Some(0)));
    assert!(created.was_created());
    drop(created);

    let proposal_polled = Arc::new(AtomicBool::new(false));
    let polled = proposal_polled.clone();
    let proposed: ClaimPayloadFuture<'_> = Box::pin(async move {
        polled.store(true, Ordering::Relaxed);
        Ok(serde_json::to_value(target(8, Some(0)))?)
    });
    let winner = second
        .acquire(&session_id)
        .await
        .unwrap()
        .resolve(proposed)
        .await
        .unwrap();
    assert_eq!(winner.target(), target(7, Some(0)));
    assert!(!winner.was_created());
    assert!(!proposal_polled.load(Ordering::Relaxed));
    drop(winner);
    assert_eq!(discovery.create_calls.load(Ordering::Relaxed), 2);

    let cached_proposal_polled = Arc::new(AtomicBool::new(false));
    let polled = cached_proposal_polled.clone();
    let proposed: ClaimPayloadFuture<'_> = Box::pin(async move {
        polled.store(true, Ordering::Relaxed);
        Ok(serde_json::to_value(target(9, Some(0)))?)
    });
    let cached = second
        .acquire(&session_id)
        .await
        .unwrap()
        .resolve(proposed)
        .await
        .unwrap();
    assert_eq!(cached.target(), target(7, Some(0)));
    assert!(!cached_proposal_polled.load(Ordering::Relaxed));
    assert_eq!(discovery.create_calls.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn distributed_delete_evicts_one_entry_and_duplicate_is_harmless() {
    let discovery = ClaimTestDiscovery::new(16);
    let first = distributed_coordinator(discovery.clone());
    let second = distributed_coordinator(discovery.clone());
    let first_session = SessionAffinityId::new("first-session");
    let second_session = SessionAffinityId::new("second-session");

    for (coordinator, session, worker_id) in
        [(&first, &first_session, 7), (&second, &second_session, 8)]
    {
        drop(
            coordinator
                .acquire(session)
                .await
                .unwrap()
                .resolve(target_payload(target(worker_id, Some(0))))
                .await
                .unwrap(),
        );
    }

    let first_key = first.claim_key_for_test(&first_session);
    discovery.emit(ClaimEvent::Delete(first_key.clone()));
    wait_for_cached_target(&first, &first_session, None).await;
    assert_eq!(
        second.query_target(&second_session).unwrap(),
        Some(target(8, Some(0)))
    );
    discovery.emit(ClaimEvent::Delete(first_key));
    wait_for_cached_target(&first, &first_session, None).await;
}

#[tokio::test]
async fn distributed_subscriber_lag_clears_all_entries() {
    let discovery = ClaimTestDiscovery::new(1);
    let coordinator = distributed_coordinator(discovery.clone());
    let session_id = session_id();
    drop(
        coordinator
            .acquire(&session_id)
            .await
            .unwrap()
            .resolve(target_payload(target(8, Some(0))))
            .await
            .unwrap(),
    );

    for index in 0..16 {
        discovery.emit(ClaimEvent::Delete(format!("unrelated/{index}")));
    }
    wait_for_cached_target(&coordinator, &session_id, None).await;
}

#[tokio::test]
async fn distributed_reset_and_disconnect_clear_entries() {
    let discovery = ClaimTestDiscovery::new(16);
    let coordinator = distributed_coordinator(discovery.clone());
    let session_id = session_id();

    drop(
        coordinator
            .acquire(&session_id)
            .await
            .unwrap()
            .resolve(target_payload(target(8, Some(0))))
            .await
            .unwrap(),
    );
    discovery.emit(ClaimEvent::Reset);
    wait_for_cached_target(&coordinator, &session_id, None).await;

    drop(
        coordinator
            .acquire(&session_id)
            .await
            .unwrap()
            .resolve(target_payload(target(99, Some(0))))
            .await
            .unwrap(),
    );
    discovery.disconnect();
    wait_for_cached_target(&coordinator, &session_id, None).await;
}

#[tokio::test]
async fn terminal_close_evicts_synchronously() {
    let discovery = ClaimTestDiscovery::new(16);
    let coordinator = distributed_coordinator(discovery.clone());
    let session_id = session_id();
    let resolved = coordinator
        .acquire(&session_id)
        .await
        .unwrap()
        .resolve(target_payload(target(7, Some(0))))
        .await
        .unwrap();
    let key = coordinator.claim_key_for_test(&session_id);
    let mut stream = resolved.into_stream(response_stream(0), true);
    assert!(stream.next().await.is_none());
    assert_eq!(coordinator.query_target(&session_id).unwrap(), None);

    for _ in 0..100 {
        if discovery.close_calls.load(Ordering::Relaxed) == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(discovery.close_calls.load(Ordering::Relaxed), 1);
    assert!(!discovery.claims.lock().unwrap().contains_key(&key));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id()).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id()).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, AffinityAcquire::Initialize(_)));
    drop(next);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id()).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id()).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let context = Arc::new(Controller::default());
    let waiter_context = context.clone();
    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move {
        waiter_coordinator
            .acquire_with_context(&session_id(), waiter_context.as_ref())
            .await
    });
    coordinator.wait_for_initializing_waiter().await;
    context.stop();

    let Err(error) = waiter.await.unwrap() else {
        panic!("cancelled waiter must return an error");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::Cancelled],
        &[]
    ));
    drop(first);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_existing_binding_overrides_explicit_proposals() {
    let coordinator = coordinator();
    drop(resolve_local(&coordinator, &session_id(), target(7, None)).await);

    for proposal in [target(8, None), target(7, Some(0)), target(7, None)] {
        let resolved = coordinator
            .acquire(&session_id())
            .await
            .unwrap()
            .resolve(target_payload(proposal))
            .await
            .unwrap();
        assert_eq!(resolved.target(), target(7, None));
    }
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let resolved = resolve_local(&coordinator, &session_id(), target(7, Some(0))).await;
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = resolved.into_stream(response_stream(1), false);
    assert!(stream.next().await.is_some());
    drop(stream);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_empty_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let resolved = resolve_local(&coordinator, &session_id(), target(7, Some(0))).await;
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = resolved.into_stream(response_stream(0), false);
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_cancelled_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    drop(resolve_local(&coordinator, &session_id(), target(7, Some(0))).await);

    tokio::time::advance(Duration::from_secs(9)).await;
    let resolved = resolve_local(&coordinator, &session_id(), target(8, Some(0))).await;
    assert_eq!(resolved.target(), target(7, Some(0)));
    let mut stream = resolved.into_stream(cancelled_response_stream(), false);
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_committed_binding_survives_cancelled_stream_until_ttl() {
    let coordinator = coordinator();
    let resolved = resolve_local(&coordinator, &session_id(), target(7, Some(0))).await;
    let mut stream = resolved.into_stream(cancelled_response_stream(), false);
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_error_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let resolved = resolve_local(&coordinator, &session_id(), target(7, Some(0))).await;
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = resolved.into_stream(error_response_stream(), false);
    assert!(stream.next().await.unwrap().is_err());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_eof_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let resolved = resolve_local(&coordinator, &session_id(), target(7, Some(0))).await;
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = resolved.into_stream(response_stream(1), false);
    while stream.next().await.is_some() {}

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    drop(resolve_local(&coordinator, &session_id(), target(7, Some(0))).await);

    tokio::time::advance(Duration::from_secs(9)).await;
    let resolved = resolve_local(&coordinator, &session_id(), target(8, Some(0))).await;
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        coordinator.query_target(&session_id()).unwrap(),
        Some(target(7, Some(0)))
    );
    drop(resolved);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let coordinator = coordinator();
    assert_eq!(coordinator.query_target(&session_id()).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);

    let initializing = coordinator.acquire(&session_id()).await.unwrap();
    assert_eq!(coordinator.query_target(&session_id()).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
    drop(initializing);
    assert_eq!(coordinator.entry_count(), 0);

    drop(resolve_local(&coordinator, &session_id(), target(7, Some(0))).await);
    assert_eq!(
        coordinator.query_target(&session_id()).unwrap(),
        Some(target(7, Some(0)))
    );
    coordinator.expire_for_test(&session_id());
    assert_eq!(coordinator.query_target(&session_id()).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let coordinator = coordinator();
    let cancellation = coordinator.cancellation_token();
    drop(resolve_local(&coordinator, &session_id(), target(7, Some(0))).await);

    coordinator.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(coordinator.entry_count(), 0);

    drop(coordinator);
    cancellation.cancelled().await;
}

#[test]
fn session_affinity_rejects_invalid_ttl_before_starting_reaper() {
    for ttl in [
        Duration::ZERO,
        Duration::from_secs(super::MAX_SESSION_AFFINITY_TTL_SECS + 1),
    ] {
        let Err(error) = AffinityCoordinator::new(ttl) else {
            panic!("invalid TTL must fail coordinator construction");
        };
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::InvalidArgument],
            &[]
        ));
        assert!(error.to_string().contains("session affinity TTL"));
    }
}

#[tokio::test(start_paused = true)]
async fn session_affinity_enforces_id_and_entry_limits() {
    let coordinator = AffinityCoordinator::with_test_limits(1, 8);
    let oversized = SessionAffinityId::new("123456789");
    let Err(error) = coordinator.acquire(&oversized).await else {
        panic!("oversized session ID must fail");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));
    assert_eq!(coordinator.entry_count(), 0);

    let first_id = SessionAffinityId::new("first");
    let first = coordinator.acquire(&first_id).await.unwrap();
    let second_id = SessionAffinityId::new("second");
    let Err(error) = coordinator.acquire(&second_id).await else {
        panic!("entry limit must reject a second session");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::ResourceExhausted],
        &[]
    ));

    drop(first);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&second_id).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}
