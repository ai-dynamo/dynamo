// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, ResponseStream, context::Controller},
    protocols::maybe_error::MaybeError,
};
use futures::{StreamExt, stream};

use super::{
    AffinityAcquire, AffinityCoordinator, AffinityTarget, LlmResponse, affinity_id,
    coordinator::ReplicaApplyOutcome, explicit_target,
};
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
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
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
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first_lease = first.commit(target(7, Some(0))).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let AffinityAcquire::Bound {
        target: second_target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter must acquire the committed binding");
    };
    assert_eq!(second_target, target(7, Some(0)));
    drop(first_lease);
    drop(second_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, AffinityAcquire::Initialize(_)));
    drop(next);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let context = Arc::new(Controller::default());
    let waiter_context = context.clone();
    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move {
        waiter_coordinator
            .acquire_with_context(&session_id(), None, waiter_context.as_ref())
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
async fn session_affinity_validates_worker_and_rank_contract() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) = coordinator
        .acquire(&session_id(), Some(target(7, None)))
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, None)).unwrap());

    assert!(
        coordinator
            .acquire(&session_id(), Some(target(8, None)))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(target(7, Some(0))))
            .await
            .is_err()
    );
    assert!(
        coordinator
            .acquire(&session_id(), Some(target(7, None)))
            .await
            .is_ok()
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_operation_invalidates_binding() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(operation.target(), Some(target(7, Some(0))));
    operation.invalidate();

    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = lease.into_stream(response_stream(1));
    assert!(stream.next().await.is_some());
    drop(stream);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_empty_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = lease.into_stream(response_stream(0));
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_cancelled_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let AffinityAcquire::Bound {
        target: bound_target,
        lease,
    } = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("continuation must acquire the existing binding");
    };
    assert_eq!(bound_target, target(7, Some(0)));
    let mut stream = lease.into_stream(cancelled_response_stream());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_committed_binding_survives_cancelled_stream_until_ttl() {
    let coordinator = coordinator();
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let mut stream = operation
        .into_stream(
            target(7, Some(0)),
            RequestPhase::Aggregated,
            cancelled_response_stream(),
        )
        .unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_error_stream_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = lease.into_stream(error_response_stream());
    assert!(stream.next().await.unwrap().is_err());
    assert!(stream.next().await.is_none());

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_eof_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = lease.into_stream(response_stream(1));
    while stream.next().await.is_some() {}

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let AffinityAcquire::Bound { lease, .. } =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let coordinator = coordinator();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);

    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
    drop(initializing);
    assert_eq!(coordinator.entry_count(), 0);

    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    coordinator.expire_for_test(&session_id());
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_reaper_removes_idle_entries_and_stops_on_drop() {
    let coordinator = coordinator();
    let cancellation = coordinator.cancellation_token();
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(target(7, Some(0))).unwrap());

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
    let Err(error) = coordinator.acquire(&oversized, None).await else {
        panic!("oversized session ID must fail");
    };
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[ErrorType::InvalidArgument],
        &[]
    ));
    assert_eq!(coordinator.entry_count(), 0);

    let first_id = SessionAffinityId::new("first");
    let first = coordinator.acquire(&first_id, None).await.unwrap();
    let second_id = SessionAffinityId::new("second");
    let Err(error) = coordinator.acquire(&second_id, None).await else {
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
        coordinator.acquire(&second_id, None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_applies_first_live_binding_wins() {
    let coordinator = coordinator();
    let local_target = target(7, Some(0));
    let conflicting_target = target(8, Some(1));

    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", local_target),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(local_target)
    );
    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", conflicting_target),
        ReplicaApplyOutcome::IgnoredConflict
    );

    coordinator.expire_for_test(&SessionAffinityId::new("replicated"));
    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", conflicting_target),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("replicated"), None)
            .unwrap(),
        Some(conflicting_target)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_duplicate_refreshes_local_ttl() {
    let coordinator = coordinator();
    let replicated_id = SessionAffinityId::new("replicated");
    let replicated_target = target(7, Some(0));

    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.apply_replica_update_for_test("replicated", replicated_target),
        ReplicaApplyOutcome::Refreshed
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(
        coordinator.query_target(&replicated_id, None).unwrap(),
        Some(replicated_target)
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&replicated_id, None).unwrap(),
        None
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_ignores_initializing_sessions() {
    let coordinator = coordinator();
    let initializing = coordinator.acquire(&session_id(), None).await.unwrap();

    assert_eq!(
        coordinator.apply_replica_update_for_test(session_id().as_str(), target(7, Some(0))),
        ReplicaApplyOutcome::IgnoredInitializing
    );
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    drop(initializing);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_replica_enforces_id_and_entry_limits() {
    let coordinator = AffinityCoordinator::with_test_limits(1, 8);

    assert_eq!(
        coordinator.apply_replica_update_for_test("123456789", target(7, Some(0))),
        ReplicaApplyOutcome::RejectedSessionId
    );
    assert_eq!(
        coordinator.apply_replica_update_for_test("first", target(7, Some(0))),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator.apply_replica_update_for_test("second", target(8, Some(0))),
        ReplicaApplyOutcome::RejectedCapacity
    );
    assert_eq!(coordinator.entry_count(), 1);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_publishes_after_dispatch_and_lease_completion() {
    let coordinator = coordinator();
    let mut updates = coordinator.enable_test_replica(99, 4);
    let selected_target = target(7, Some(0));
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let stream = operation
        .into_stream(
            selected_target,
            RequestPhase::Aggregated,
            response_stream(1),
        )
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, session_id().as_str());
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.router_id, 99);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = coordinator();
    let mut updates = origin.enable_test_replica(99, 4);
    let replica = coordinator();
    let replicated_target = target(7, Some(0));
    let operation = origin.acquire(&session_id(), None).await.unwrap();
    let stream = operation
        .into_stream(
            replicated_target,
            RequestPhase::Aggregated,
            response_stream(1),
        )
        .unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_update_for_test(
            after_dispatch.session_id,
            target(after_dispatch.worker_id, after_dispatch.dp_rank),
        ),
        ReplicaApplyOutcome::Inserted
    );
    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(replica.query_target(&session_id(), None).unwrap(), None);

    drop(stream);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(
        replica.apply_replica_update_for_test(
            after_completion.session_id,
            target(after_completion.worker_id, after_completion.dp_rank),
        ),
        ReplicaApplyOutcome::ReplacedExpired
    );
    assert_eq!(
        replica.query_target(&session_id(), None).unwrap(),
        Some(replicated_target)
    );
}
