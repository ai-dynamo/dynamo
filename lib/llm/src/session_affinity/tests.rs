// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_runtime::pipeline::{ResponseStream, context::Controller};
use futures::{StreamExt, stream};

use super::{AffinityAcquire, AffinityCoordinator, AffinityTarget, LlmResponse, explicit_target};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::SessionAffinityId, llm_backend::LLMEngineOutput, preprocessor::RoutingHints,
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

fn response_stream(items: usize) -> dynamo_runtime::pipeline::ManyOut<LlmResponse> {
    let items = (0..items).map(|_| Annotated::from_data(LLMEngineOutput::default()));
    ResponseStream::new(
        Box::pin(stream::iter(items)),
        Arc::new(Controller::default()),
    )
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
        Some(target(1, Some(0)))
    );

    let rank_without_worker = request_with_routing(RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    assert!(explicit_target(&rank_without_worker, RequestPhase::Decode).is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initialization_is_atomic() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
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
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let AffinityAcquire::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    assert!(matches!(
        waiter.await.unwrap().unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_validates_worker_and_rank_contract() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
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
async fn session_affinity_active_leases_prevent_expiry() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();

    tokio::time::advance(Duration::from_secs(20)).await;
    tokio::task::yield_now().await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );

    drop(lease);
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_drop_refreshes_idle_ttl() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(lease.into_stream(response_stream(1)));

    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(target(7, Some(0)))
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_stream_eof_refreshes_idle_ttl() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    let lease = initializer.commit(target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    let mut stream = lease.into_stream(response_stream(1));
    while stream.next().await.is_some() {}

    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(
        coordinator
            .query_target(&session_id(), None)
            .unwrap()
            .is_some()
    );
    tokio::time::advance(Duration::from_secs(2)).await;
    assert!(
        coordinator
            .query_target(&session_id(), None)
            .unwrap()
            .is_none()
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_failed_bound_attempt_does_not_refresh_ttl() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
    let AffinityAcquire::Initialize(initializer) =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(
        initializer
            .commit(target(7, Some(0)))
            .unwrap()
            .into_stream(response_stream(0)),
    );

    tokio::time::advance(Duration::from_secs(9)).await;
    let AffinityAcquire::Bound { lease, .. } =
        coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("continuation must acquire the binding");
    };
    tokio::time::advance(Duration::from_secs(2)).await;
    drop(lease);

    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        AffinityAcquire::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_query_is_read_only() {
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
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
    let coordinator = AffinityCoordinator::new(Duration::from_secs(10));
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
