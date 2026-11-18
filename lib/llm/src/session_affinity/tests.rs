// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{
    engine::AsyncEngineContext,
    error::ErrorType,
    pipeline::{Context, context::Controller},
};

use super::SessionAffinityMode::{Hard, Soft};
use super::{
    AffinityCoordinator, AffinityTarget, Hold, affinity_id, coordinator::ReplicaApplyOutcome,
    explicit_target, subagent_group_affinity_id, to_table,
};
use crate::{
    preprocessor::PreprocessedRequest,
    protocols::common::{
        extensions::{SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId},
        preprocessor::RoutingHints,
        timing::RequestPhase,
    },
};
use dynamo_session_affinity::host::MAX_SESSION_AFFINITY_ID_BYTES;

fn session_id() -> SessionAffinityId {
    SessionAffinityId::new("session-1")
}

fn target(worker_id: u64, dp_rank: Option<u32>) -> AffinityTarget {
    AffinityTarget { worker_id, dp_rank }
}

fn table_target(
    worker_id: u64,
    dp_rank: Option<u32>,
) -> dynamo_session_affinity::host::AffinityTarget {
    to_table(target(worker_id, dp_rank))
}

fn coordinator() -> AffinityCoordinator {
    AffinityCoordinator::new(Duration::from_secs(10), Hard).unwrap()
}

fn soft_coordinator() -> AffinityCoordinator {
    AffinityCoordinator::new(Duration::from_secs(10), Soft).unwrap()
}

async fn bind(coordinator: &AffinityCoordinator, target: AffinityTarget) {
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(operation, target).unwrap());
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
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    assert!(!waiter.is_finished());

    let first_lease = first.commit(table_target(7, Some(0))).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let Hold::Bound {
        target: second_target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter must acquire the committed binding");
    };
    assert_eq!(second_target, table_target(7, Some(0)));
    drop(first_lease);
    drop(second_lease);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_initializer_cancellation_wakes_waiter() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
        panic!("first request must initialize");
    };

    let waiter_coordinator = coordinator.clone();
    let waiter = tokio::spawn(async move { waiter_coordinator.acquire(&session_id(), None).await });
    coordinator.wait_for_initializing_waiter().await;
    drop(first);

    let next = waiter.await.unwrap().unwrap();
    assert!(matches!(&next, Hold::Initialize(_)));
    drop(next);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_wait_stops_when_request_is_cancelled() {
    let coordinator = coordinator();
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let Hold::Initialize(first) = first else {
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
    let Hold::Initialize(initializer) = coordinator
        .acquire(&session_id(), Some(target(7, None)))
        .await
        .unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, None)).unwrap());

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
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert_eq!(operation.target(), Some(table_target(7, Some(0))));
    operation.invalidate();

    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
    assert_eq!(coordinator.entry_count(), 0);
    assert!(matches!(
        coordinator.acquire(&session_id(), None).await.unwrap(),
        Hold::Initialize(_)
    ));
}

#[tokio::test(start_paused = true)]
async fn session_affinity_committed_binding_survives_a_held_lease_past_the_ttl() {
    let coordinator = coordinator();
    let operation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator.commit(operation, target(7, Some(0))).unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(lease);

    assert_binding_expires_after_refreshed_ttl(&coordinator).await;
}

#[tokio::test(start_paused = true)]
async fn session_affinity_bound_lease_drop_refreshes_idle_ttl() {
    let coordinator = coordinator();
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

    tokio::time::advance(Duration::from_secs(9)).await;
    let Hold::Bound { lease, .. } = coordinator.acquire(&session_id(), None).await.unwrap() else {
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

    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());
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
    let Hold::Initialize(initializer) = coordinator.acquire(&session_id(), None).await.unwrap()
    else {
        panic!("first request must initialize");
    };
    drop(initializer.commit(table_target(7, Some(0))).unwrap());

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
        let Err(error) = AffinityCoordinator::new(ttl, Hard) else {
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
        Hold::Initialize(_)
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
    let lease = coordinator.commit(operation, selected_target).unwrap();

    let after_dispatch = updates.recv().await.unwrap();
    assert_eq!(after_dispatch.session_id, session_id().as_str());
    assert_eq!(after_dispatch.worker_id, selected_target.worker_id);
    assert_eq!(after_dispatch.dp_rank, selected_target.dp_rank);
    assert_eq!(after_dispatch.writer_id, 99);

    drop(lease);
    let after_completion = updates.recv().await.unwrap();
    assert_eq!(after_completion, after_dispatch);
    assert!(updates.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn session_affinity_republishes_the_stored_replica_version() {
    let coordinator = coordinator();
    let mut updates = coordinator.enable_test_replica(99, 1);
    let replicated_target = target(7, Some(0));
    assert_eq!(
        coordinator.apply_versioned_replica_update_for_test(
            session_id().as_str(),
            replicated_target,
            123,
            7,
        ),
        ReplicaApplyOutcome::Inserted
    );

    let Hold::Bound { lease, .. } = coordinator.acquire(&session_id(), None).await.unwrap() else {
        panic!("replicated binding must be acquired");
    };
    drop(lease);

    let update = updates.recv().await.unwrap();
    assert_eq!(update.sequence, 123);
    assert_eq!(update.writer_id, 7);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_worker_only_binding_allows_ranked_dispatch_without_narrowing() {
    let coordinator = coordinator();
    let worker_binding = target(7, None);

    let initialization = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(initialization, worker_binding).unwrap());

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator
        .commit(continuation, target(7, Some(3)))
        .expect("worker-only affinity must allow the scheduler to select a DP rank");

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn soft_worker_only_binding_stays_worker_scoped_after_ranked_dispatch() {
    let coordinator = soft_coordinator();
    let worker_binding = target(7, None);
    bind(&coordinator, worker_binding).await;

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(
        coordinator
            .commit(continuation, target(7, Some(3)))
            .unwrap(),
    );

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(worker_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn soft_ranked_binding_is_not_widened_by_worker_only_dispatch() {
    let coordinator = soft_coordinator();
    let ranked_binding = target(7, Some(2));
    bind(&coordinator, ranked_binding).await;

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(continuation, target(8, None)).unwrap());

    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(ranked_binding)
    );
}

#[tokio::test(start_paused = true)]
async fn session_affinity_ranked_binding_rejects_mismatched_rank_dispatch() {
    let coordinator = coordinator();
    let binding = target(7, Some(2));

    let initialization = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(coordinator.commit(initialization, binding).unwrap());

    let continuation = coordinator.acquire(&session_id(), None).await.unwrap();
    assert!(
        coordinator
            .commit(continuation, target(7, Some(3)))
            .is_err()
    );
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn soft_affinity_rebinds_after_dispatch() {
    let coordinator = soft_coordinator();
    let original = target(7, Some(0));
    let replacement = target(8, Some(1));
    bind(&coordinator, original).await;

    let failed_attempt = coordinator.acquire(&session_id(), None).await.unwrap();
    drop(failed_attempt);
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(original)
    );

    let successful_attempt = coordinator.acquire(&session_id(), None).await.unwrap();
    let lease = coordinator.commit(successful_attempt, replacement).unwrap();
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn concurrent_soft_rebind_uses_observed_version_cas() {
    let coordinator = soft_coordinator();
    let original = target(7, Some(0));
    let winner = target(8, Some(0));
    let stale = target(9, Some(0));
    bind(&coordinator, original).await;
    let first = coordinator.acquire(&session_id(), None).await.unwrap();
    let second = coordinator.acquire(&session_id(), None).await.unwrap();

    let first_lease = coordinator.commit(first, winner).unwrap();
    let second_lease = coordinator.commit(second, stale).unwrap();
    drop(first_lease);
    drop(second_lease);
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(winner)
    );
}

#[tokio::test(start_paused = true)]
async fn replica_applies_only_newer_affinity_versions() {
    let coordinator = coordinator();
    let first = target(7, Some(0));
    let newer = target(8, Some(0));
    let stale = target(9, Some(0));

    assert_eq!(
        coordinator.apply_versioned_replica_update_for_test("ordered", first, 3, 10),
        ReplicaApplyOutcome::Inserted
    );
    assert_eq!(
        coordinator.apply_versioned_replica_update_for_test("ordered", newer, 4, 10),
        ReplicaApplyOutcome::ReplacedNewer
    );
    assert_eq!(
        coordinator.apply_versioned_replica_update_for_test("ordered", stale, 3, 11),
        ReplicaApplyOutcome::IgnoredConflict
    );
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("ordered"), None)
            .unwrap(),
        Some(newer)
    );
}

#[tokio::test(start_paused = true)]
async fn stale_lease_cannot_invalidate_or_refresh_newer_replica_binding() {
    let coordinator = coordinator();
    let original = target(7, Some(0));
    let replacement = target(8, Some(0));
    bind(&coordinator, original).await;
    let stale = coordinator.acquire(&session_id(), None).await.unwrap();
    let replacement_sequence = u64::MAX / 2;
    coordinator.apply_versioned_replica_update_for_test(
        session_id().as_str(),
        replacement,
        replacement_sequence,
        1,
    );
    stale.invalidate();
    assert_eq!(
        coordinator.query_target(&session_id(), None).unwrap(),
        Some(replacement)
    );

    let stale = coordinator.acquire(&session_id(), None).await.unwrap();
    tokio::time::advance(Duration::from_secs(9)).await;
    coordinator.apply_versioned_replica_update_for_test(
        session_id().as_str(),
        replacement,
        replacement_sequence.saturating_add(1),
        1,
    );
    tokio::time::advance(Duration::from_secs(9)).await;
    drop(stale);
    tokio::time::advance(Duration::from_secs(2)).await;
    assert_eq!(coordinator.query_target(&session_id(), None).unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn session_affinity_completion_restores_expired_remote_binding() {
    let origin = coordinator();
    let mut updates = origin.enable_test_replica(99, 4);
    let replica = coordinator();
    let replicated_target = target(7, Some(0));
    let operation = origin.acquire(&session_id(), None).await.unwrap();
    let lease = origin.commit(operation, replicated_target).unwrap();

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

    drop(lease);
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

#[test]
fn a_subagent_group_id_is_namespaced_and_fixed_size() {
    let a = subagent_group_affinity_id("parent-1");
    let b = subagent_group_affinity_id("parent-1");
    let c = subagent_group_affinity_id("parent-2");
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert!(
        a.starts_with("\u{1}sg:"),
        "the key must be un-claimable via a header value"
    );
    let long = subagent_group_affinity_id(&"p".repeat(10_000));
    assert_eq!(long.len(), a.len());
    assert!(long.len() <= MAX_SESSION_AFFINITY_ID_BYTES);
}

#[test]
fn session_affinity_max_entries_comes_from_the_environment() {
    use std::env::VarError;

    assert_eq!(
        super::max_entries_from(Err(VarError::NotPresent)).unwrap(),
        65_536
    );
    assert_eq!(super::max_entries_from(Ok("2".into())).unwrap(), 2);
    for value in ["0", "-1", "abc"] {
        let error = super::max_entries_from(Ok(value.into())).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES must be a positive integer"),
            "{error}"
        );
    }
}

fn coordinator_with(settings: super::PlacementSettings) -> AffinityCoordinator {
    let config = super::affinity_config(Duration::from_secs(10), Hard).unwrap();
    AffinityCoordinator::with_placement(
        dynamo_session_affinity::host::SessionAffinity::with_config(config).unwrap(),
        settings,
    )
}

fn balanced(rebalance_delta: usize) -> AffinityCoordinator {
    coordinator_with(super::PlacementSettings {
        placement: super::SessionPlacement::Balanced,
        rebalance_delta: std::num::NonZeroUsize::new(rebalance_delta),
        active_within: Some(Duration::from_secs(super::DEFAULT_SESSION_ACTIVE_SECS)),
    })
}

fn idle_workers(workers: &[u64]) -> super::WorkerLoads {
    workers
        .iter()
        .map(|worker_id| (*worker_id, (0, 0)))
        .collect()
}

async fn new_session(coordinator: &AffinityCoordinator, id: String) -> Hold {
    let hold = coordinator
        .acquire(&SessionAffinityId::new(id), None)
        .await
        .unwrap();
    assert!(matches!(hold, Hold::Initialize(_)));
    hold
}

#[test]
fn placement_settings_come_from_the_environment() {
    use super::{PlacementSettings, SessionPlacement};
    use std::{collections::HashMap, env::VarError, num::NonZeroUsize};

    fn parse(vars: &[(&str, &str)]) -> Result<PlacementSettings, anyhow::Error> {
        let vars: HashMap<String, String> = vars
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        super::placement_settings_from(|name| vars.get(name).cloned().ok_or(VarError::NotPresent))
    }

    // Unset: normal selection, no re-binding, a 60 s activity window.
    assert_eq!(
        parse(&[]).unwrap(),
        PlacementSettings {
            placement: SessionPlacement::Kv,
            rebalance_delta: None,
            active_within: Some(Duration::from_secs(60)),
        }
    );
    // What the Python frontend exports for its defaults.
    assert_eq!(
        parse(&[
            ("DYN_ROUTER_SESSION_AFFINITY_PLACEMENT", "kv"),
            ("DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA", "0"),
            ("DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS", "60"),
        ])
        .unwrap(),
        parse(&[]).unwrap()
    );
    assert_eq!(
        parse(&[
            ("DYN_ROUTER_SESSION_AFFINITY_PLACEMENT", " balanced "),
            ("DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA", "2"),
            ("DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS", "0"),
        ])
        .unwrap(),
        PlacementSettings {
            placement: SessionPlacement::Balanced,
            rebalance_delta: NonZeroUsize::new(2),
            active_within: None,
        }
    );
    for (name, value) in [
        ("DYN_ROUTER_SESSION_AFFINITY_PLACEMENT", "random"),
        ("DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA", "-1"),
        ("DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS", "1.5"),
    ] {
        let error = parse(&[(name, value)]).unwrap_err();
        assert!(error.to_string().starts_with(name), "{error}");
    }
}

/// Balanced placement counts planned, uncommitted sessions, so a burst spreads
/// evenly before any of it commits; a tie on sessions goes to fewer active requests.
#[tokio::test(start_paused = true)]
async fn balanced_placement_counts_planned_sessions_before_they_commit() {
    use std::collections::HashMap;

    let coordinator = balanced(0);
    let idle = idle_workers(&[1, 2, 3]);
    let mut holds = Vec::new();
    let mut placed: HashMap<u64, usize> = HashMap::new();
    for i in 0..9 {
        let hold = new_session(&coordinator, format!("s{i}")).await;
        let target = coordinator.place_by_live_sessions(&hold, &idle).unwrap();
        assert_eq!(target.dp_rank, None, "selection picks the rank");
        *placed.entry(target.worker_id).or_default() += 1;
        holds.push((hold, target));
    }
    assert_eq!(placed, HashMap::from([(1, 3), (2, 3), (3, 3)]));
    assert_eq!(coordinator.live_sessions(), placed);

    // Committing to a rank of the planned worker keeps the counts.
    for (hold, target) in holds {
        drop(
            coordinator
                .commit(hold, AffinityTarget::new(target.worker_id, Some(0)))
                .unwrap(),
        );
    }
    assert_eq!(coordinator.live_sessions(), placed);
    let late = new_session(&coordinator, "late".into()).await;
    let busy = HashMap::from([(1, (2, 10)), (2, (0, 50)), (3, (1, 0))]);
    assert_eq!(
        coordinator
            .place_by_live_sessions(&late, &busy)
            .unwrap()
            .worker_id,
        2
    );

    // A bound session is never re-placed.
    let bound = coordinator
        .acquire(&SessionAffinityId::new("s0"), None)
        .await
        .unwrap();
    assert!(coordinator.place_by_live_sessions(&bound, &idle).is_none());
}

/// Concurrent new sessions see each other's planned workers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn balanced_placement_spreads_concurrent_new_sessions() {
    use std::collections::HashMap;

    let coordinator = balanced(0);
    let idle = idle_workers(&[1, 2, 3]);
    let tasks = (0..30).map(|i| {
        let coordinator = coordinator.clone();
        let idle = idle.clone();
        tokio::spawn(async move {
            let hold = new_session(&coordinator, format!("burst-{i}")).await;
            let target = coordinator.place_by_live_sessions(&hold, &idle).unwrap();
            (hold, target.worker_id)
        })
    });
    let placed = futures::future::join_all(tasks).await;
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for result in &placed {
        *counts.entry(result.as_ref().unwrap().1).or_default() += 1;
    }
    assert_eq!(counts, HashMap::from([(1, 10), (2, 10), (3, 10)]));
}

/// Live sessions are bindings with a request in flight or released within the
/// activity window; idle bindings past the window do not count.
#[tokio::test(start_paused = true)]
async fn live_sessions_follow_the_activity_window() {
    use std::collections::HashMap;

    let coordinator = coordinator_with(super::PlacementSettings {
        placement: super::SessionPlacement::Balanced,
        active_within: Some(Duration::from_secs(2)),
        ..Default::default()
    });
    let idle = new_session(&coordinator, "idle".into()).await;
    drop(coordinator.commit(idle, target(1, Some(0))).unwrap());
    let busy = new_session(&coordinator, "busy".into()).await;
    let _busy_lease = coordinator.commit(busy, target(2, Some(0))).unwrap();
    assert_eq!(coordinator.live_sessions(), HashMap::from([(1, 1), (2, 1)]));

    tokio::time::advance(Duration::from_secs(3)).await;
    assert_eq!(coordinator.live_sessions(), HashMap::from([(2, 1)]));
    // The idle binding still holds its worker for its own next turn.
    assert_eq!(
        coordinator
            .query_target(&SessionAffinityId::new("idle"), None)
            .unwrap(),
        Some(target(1, Some(0)))
    );
}

/// A bound session moves only when its worker, not counting it, has at least
/// `delta` more live sessions than the least-loaded other eligible worker.
#[tokio::test(start_paused = true)]
async fn rebind_only_past_the_delta() {
    async fn bind_live(
        coordinator: &AffinityCoordinator,
        workers: &[u64],
    ) -> Vec<dynamo_session_affinity::host::AffinityLease> {
        let mut leases = Vec::new();
        for (i, worker) in workers.iter().enumerate() {
            let hold = new_session(coordinator, format!("live-{i}")).await;
            leases.push(coordinator.commit(hold, target(*worker, Some(0))).unwrap());
        }
        leases
    }

    let loads = idle_workers(&[1, 2, 3]);
    // 3 / 1 / 1 live sessions: a session on worker 1 sees 2 others vs 1.
    let off = balanced(0);
    let _off = bind_live(&off, &[1, 1, 1, 2, 3]).await;
    assert!(!off.should_rebind(1, &loads), "0 disables re-binding");

    let on = balanced(1);
    let _on = bind_live(&on, &[1, 1, 1, 2, 3]).await;
    assert!(on.should_rebind(1, &loads));
    assert!(!on.should_rebind(2, &loads), "no other session on worker 2");
    assert!(
        !on.should_rebind(9, &loads),
        "an ineligible worker is left alone"
    );
    assert!(
        !on.should_rebind(1, &idle_workers(&[1])),
        "no other eligible worker"
    );

    let strict = balanced(2);
    let _strict = bind_live(&strict, &[1, 1, 1, 2, 3]).await;
    assert!(!strict.should_rebind(1, &loads));
}
