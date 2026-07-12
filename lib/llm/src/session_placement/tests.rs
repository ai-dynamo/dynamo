// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use super::coordinator::{
    AmbiguousResolution, PlacementAcquire, PlacementDispatch, PlacementDispatchOutcome,
    PlacementInitialization, PlacementLease, SessionPlacement, SessionPlacementConfig,
    SessionPlacementError, TargetGeneration,
};

fn placement<T>(max_entries: usize, max_key_bytes: usize) -> SessionPlacement<T>
where
    T: Send + Sync + 'static,
{
    SessionPlacement::new(SessionPlacementConfig {
        idle_ttl: Duration::from_secs(10),
        initialization_timeout: Some(Duration::from_secs(10)),
        max_entries,
        max_key_bytes,
    })
    .unwrap()
}

fn begin<T>(
    initialization: PlacementInitialization<T>,
    target: T,
    generation: u64,
) -> PlacementDispatch<T>
where
    T: Send + Sync + 'static,
{
    initialization
        .begin_dispatch(target, TargetGeneration::new(generation))
        .unwrap()
}

fn accept<T>(
    initialization: PlacementInitialization<T>,
    target: T,
    generation: u64,
) -> PlacementLease<T>
where
    T: Send + Sync + 'static,
{
    begin(initialization, target, generation)
        .finish(PlacementDispatchOutcome::Accepted)
        .unwrap()
        .unwrap()
}

fn query_string(placement: &SessionPlacement<String>, key: &str) -> Option<String> {
    placement
        .query(key)
        .unwrap()
        .map(|target| target.target().clone())
}

#[tokio::test]
async fn concurrent_miss_initializes_once() {
    let placement = Arc::new(placement(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 7);

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;

    let first_lease = dispatch
        .finish(PlacementDispatchOutcome::Accepted)
        .unwrap()
        .unwrap();
    let second = waiter.await.unwrap().unwrap();
    let PlacementAcquire::Bound {
        target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter should observe the committed placement");
    };
    assert_eq!(target.target(), "cluster-a");
    assert_eq!(target.generation(), TargetGeneration::new(7));
    assert_eq!(placement.entry_count(), 1);

    drop(first_lease);
    drop(second_lease);
}

#[tokio::test]
async fn dropped_reservation_rolls_back_and_wakes_waiter() {
    let placement = Arc::new(placement::<String>(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;
    drop(initialization);

    let next = waiter.await.unwrap().unwrap();
    let PlacementAcquire::Initialize(initialization) = next else {
        panic!("waiter should become the next initializer");
    };
    let lease = accept(initialization, "cluster-b".to_string(), 1);
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-b")
    );
    assert_eq!(placement.entry_count(), 1);
    drop(lease);
}

#[tokio::test(start_paused = true)]
async fn dropped_dispatch_is_quarantined_as_ambiguous() {
    let placement = Arc::new(placement(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 3);

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;
    drop(dispatch);

    assert!(matches!(
        waiter.await.unwrap(),
        Err(SessionPlacementError::DispatchAmbiguous {
            target_generation: 3,
            ..
        })
    ));
    assert!(matches!(
        placement.query("session"),
        Err(SessionPlacementError::DispatchAmbiguous {
            target_generation: 3,
            ..
        })
    ));

    placement.wait_for_reaper().await;
    tokio::time::advance(Duration::from_secs(30)).await;
    tokio::task::yield_now().await;
    assert_eq!(placement.entry_count(), 1);
}

#[tokio::test]
async fn definitely_not_accepted_allows_one_waiter_to_retry() {
    let placement = Arc::new(placement(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 1);

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;
    assert!(
        dispatch
            .finish(PlacementDispatchOutcome::DefinitelyNotAccepted)
            .unwrap()
            .is_none()
    );

    let retry = waiter.await.unwrap().unwrap();
    assert!(matches!(&retry, PlacementAcquire::Initialize(_)));
    drop(retry);
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn dispatch_timeout_blocks_replay_but_late_acceptance_can_commit() {
    let placement = Arc::new(placement(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 9);

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;
    tokio::time::advance(Duration::from_secs(11)).await;
    tokio::task::yield_now().await;

    assert!(matches!(
        waiter.await.unwrap(),
        Err(SessionPlacementError::DispatchAmbiguous {
            target_generation: 9,
            ..
        })
    ));
    let lease = dispatch
        .finish(PlacementDispatchOutcome::Accepted)
        .unwrap()
        .unwrap();
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-a")
    );
    drop(lease);
}

#[tokio::test]
async fn ambiguous_resolution_is_fenced_by_attempt_and_generation() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 4);
    let attempt = dispatch.attempt_id();
    assert_eq!(dispatch.target().target(), "cluster-a");
    drop(dispatch);

    assert_eq!(
        placement
            .resolve_ambiguous(
                "session",
                attempt,
                TargetGeneration::new(5),
                AmbiguousResolution::DefinitelyNotAccepted,
            )
            .err()
            .unwrap(),
        SessionPlacementError::TargetGenerationChanged {
            expected_generation: 5,
            actual_generation: 4,
        }
    );
    assert!(
        placement
            .resolve_ambiguous(
                "session",
                attempt,
                TargetGeneration::new(4),
                AmbiguousResolution::DefinitelyNotAccepted,
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test]
async fn explicit_ambiguity_can_be_resolved_as_accepted() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };
    let dispatch = begin(initialization, "cluster-a".to_string(), 6);
    let attempt = dispatch.attempt_id();

    assert!(
        dispatch
            .finish(PlacementDispatchOutcome::Ambiguous)
            .unwrap()
            .is_none()
    );
    let lease = placement
        .resolve_ambiguous(
            "session",
            attempt,
            TargetGeneration::new(6),
            AmbiguousResolution::Accepted,
        )
        .unwrap()
        .unwrap();

    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-a")
    );
    drop(lease);
}

#[tokio::test]
async fn query_bounds_and_invalidation_are_enforced() {
    let placement = placement(1, 4);
    let acquire = placement.acquire("one").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let mut lease = accept(initialization, "cluster-a".to_string(), 1);

    assert_eq!(
        query_string(&placement, "one").as_deref(),
        Some("cluster-a")
    );
    assert!(matches!(
        placement.acquire("two").await,
        Err(SessionPlacementError::Capacity { max_entries: 1 })
    ));
    assert!(matches!(
        placement.acquire("12345").await,
        Err(SessionPlacementError::KeyTooLong {
            actual_bytes: 5,
            max_bytes: 4
        })
    ));

    lease.invalidate();
    assert!(placement.query("one").unwrap().is_none());
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test]
async fn stale_lease_cannot_remove_a_new_placement() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(first) = first else {
        panic!("first acquire should initialize");
    };
    let mut invalidating_lease = accept(first, "cluster-a".to_string(), 1);
    let second = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Bound {
        lease: mut stale_lease,
        ..
    } = second
    else {
        panic!("second acquire should use the existing placement");
    };

    invalidating_lease.invalidate();
    let replacement = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(replacement) = replacement else {
        panic!("invalidated placement should be initialized again");
    };
    let replacement_lease = accept(replacement, "cluster-b".to_string(), 2);

    stale_lease.invalidate();
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-b")
    );
    assert_eq!(placement.entry_count(), 1);
    drop(replacement_lease);
}

#[tokio::test(start_paused = true)]
async fn stale_reservation_cannot_overwrite_a_replacement() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(first) = first else {
        panic!("first acquire should initialize");
    };

    tokio::time::advance(Duration::from_secs(11)).await;
    let replacement = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(replacement) = replacement else {
        panic!("expired reservation should be replaced");
    };
    let replacement_lease = accept(replacement, "cluster-b".to_string(), 2);

    assert!(matches!(
        first.begin_dispatch("cluster-a".to_string(), TargetGeneration::new(1)),
        Err(SessionPlacementError::InitializationChanged
            | SessionPlacementError::InitializationCancelled)
    ));
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-b")
    );
    drop(replacement_lease);
}

#[tokio::test(start_paused = true)]
async fn expired_reservation_cannot_begin_dispatch() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };

    tokio::time::advance(Duration::from_secs(11)).await;
    assert!(matches!(
        initialization.begin_dispatch("cluster-a".to_string(), TargetGeneration::new(1)),
        Err(SessionPlacementError::InitializationCancelled)
    ));
    assert!(placement.query("session").unwrap().is_none());
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test(start_paused = true)]
async fn active_lease_blocks_expiration_and_release_refreshes_ttl() {
    let placement = placement(10, 32);
    placement.wait_for_reaper().await;
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let lease = accept(initialization, "cluster-a".to_string(), 1);

    tokio::time::advance(Duration::from_secs(20)).await;
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-a")
    );
    drop(lease);
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-a")
    );
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert!(placement.query("session").unwrap().is_none());
}

#[tokio::test(start_paused = true)]
async fn abandoned_lease_does_not_refresh_ttl() {
    let placement = placement(10, 32);
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let mut lease = accept(initialization, "cluster-a".to_string(), 1);

    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(
        query_string(&placement, "session").as_deref(),
        Some("cluster-a")
    );
    lease.abandon();
    assert!(placement.query("session").unwrap().is_none());
}

#[tokio::test]
async fn capacity_path_reaps_expired_entries() {
    let placement = placement(1, 32);
    let acquire = placement.acquire("one").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    drop(accept(initialization, "cluster-a".to_string(), 1));
    placement.expire_for_test("one");

    let acquire = placement.acquire("two").await.unwrap();
    assert!(matches!(&acquire, PlacementAcquire::Initialize(_)));
    assert_eq!(placement.entry_count(), 1);
    drop(acquire);
}

#[tokio::test]
async fn expired_bound_entry_is_reinitialized_before_reaper_runs() {
    let placement = placement(1, 32);
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    drop(accept(initialization, "cluster-a".to_string(), 1));
    placement.expire_for_test("session");

    let replacement = placement.acquire("session").await.unwrap();
    assert!(matches!(&replacement, PlacementAcquire::Initialize(_)));
    assert_eq!(placement.entry_count(), 1);
    drop(replacement);
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test]
async fn target_does_not_need_clone() {
    struct NonCloneTarget(&'static str);

    let placement = placement(1, 32);
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let lease = accept(initialization, NonCloneTarget("cluster-a"), 1);
    let target = placement.query("session").unwrap().unwrap();
    assert_eq!(target.target().0, "cluster-a");
    drop(lease);
}

#[tokio::test]
async fn dropping_last_handle_cancels_reaper() {
    let placement = placement::<String>(10, 32);
    let cancellation = placement.cancellation_token();
    drop(placement);
    cancellation.cancelled().await;
}

#[tokio::test]
async fn reaper_restarts_after_its_task_stops() {
    let placement = placement::<String>(10, 32);
    placement.wait_for_reaper().await;
    placement.stop_reaper_for_test().await;

    assert!(placement.query("session").unwrap().is_none());
    placement.wait_for_reaper().await;
}

#[tokio::test]
async fn invalid_config_is_rejected() {
    for config in [
        SessionPlacementConfig {
            idle_ttl: Duration::ZERO,
            initialization_timeout: Some(Duration::from_secs(1)),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_nanos(1),
            initialization_timeout: Some(Duration::from_secs(1)),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(31_536_001),
            initialization_timeout: Some(Duration::from_secs(1)),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(1),
            initialization_timeout: Some(Duration::ZERO),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(1),
            initialization_timeout: Some(Duration::from_nanos(1)),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(1),
            initialization_timeout: Some(Duration::from_secs(31_536_001)),
            max_entries: 1,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(1),
            initialization_timeout: Some(Duration::from_secs(1)),
            max_entries: 0,
            max_key_bytes: 1,
        },
        SessionPlacementConfig {
            idle_ttl: Duration::from_secs(1),
            initialization_timeout: Some(Duration::from_secs(1)),
            max_entries: 1,
            max_key_bytes: 0,
        },
    ] {
        assert!(matches!(
            SessionPlacement::<String>::new(config),
            Err(SessionPlacementError::InvalidConfig(_))
        ));
    }
}

#[test]
fn placement_requires_a_tokio_runtime() {
    let result = SessionPlacement::<String>::new(SessionPlacementConfig {
        idle_ttl: Duration::from_secs(1),
        initialization_timeout: Some(Duration::from_secs(1)),
        max_entries: 1,
        max_key_bytes: 1,
    });
    assert!(matches!(
        result,
        Err(SessionPlacementError::RuntimeUnavailable)
    ));
}
