// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use super::{PlacementAcquire, SessionPlacement, SessionPlacementConfig, SessionPlacementError};

fn placement(max_entries: usize, max_key_bytes: usize) -> SessionPlacement<String> {
    SessionPlacement::new(SessionPlacementConfig {
        idle_ttl: Duration::from_secs(10),
        initialization_timeout: Some(Duration::from_secs(10)),
        max_entries,
        max_key_bytes,
    })
    .unwrap()
}

#[tokio::test]
async fn concurrent_miss_initializes_once() {
    let placement = Arc::new(placement(10, 32));
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = first else {
        panic!("first acquire should initialize");
    };

    let waiter = {
        let placement = placement.clone();
        tokio::spawn(async move { placement.acquire("session").await })
    };
    placement.wait_for_initializing_waiter().await;

    let first_lease = initialization.commit("cluster-a".to_string()).unwrap();
    let second = waiter.await.unwrap().unwrap();
    let PlacementAcquire::Bound {
        target,
        lease: second_lease,
    } = second
    else {
        panic!("waiter should observe the committed placement");
    };
    assert_eq!(target, "cluster-a");
    assert_eq!(placement.entry_count(), 1);

    drop(first_lease);
    drop(second_lease);
}

#[tokio::test]
async fn dropped_initialization_rolls_back_and_wakes_waiter() {
    let placement = Arc::new(placement(10, 32));
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
    let lease = initialization.commit("cluster-b".to_string()).unwrap();
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-b")
    );
    assert_eq!(placement.entry_count(), 1);
    drop(lease);
}

#[tokio::test]
async fn query_bounds_and_invalidation_are_enforced() {
    let placement = placement(1, 4);
    let acquire = placement.acquire("one").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let mut lease = initialization.commit("cluster-a".to_string()).unwrap();

    assert_eq!(
        placement.query("one").unwrap().as_deref(),
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
    assert_eq!(placement.query("one").unwrap(), None);
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test]
async fn stale_lease_cannot_remove_a_new_placement() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(first) = first else {
        panic!("first acquire should initialize");
    };
    let mut invalidating_lease = first.commit("cluster-a".to_string()).unwrap();
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
    let replacement_lease = replacement.commit("cluster-b".to_string()).unwrap();

    stale_lease.invalidate();
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-b")
    );
    assert_eq!(placement.entry_count(), 1);
    drop(replacement_lease);
}

#[tokio::test(start_paused = true)]
async fn stale_initialization_cannot_overwrite_a_replacement() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(first) = first else {
        panic!("first acquire should initialize");
    };

    tokio::time::advance(Duration::from_secs(11)).await;
    let replacement = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(replacement) = replacement else {
        panic!("expired initialization should be replaced");
    };
    let replacement_lease = replacement.commit("cluster-b".to_string()).unwrap();

    assert!(matches!(
        first.commit("cluster-a".to_string()),
        Err(SessionPlacementError::InitializationChanged
            | SessionPlacementError::InitializationCancelled)
    ));
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-b")
    );
    drop(replacement_lease);
}

#[tokio::test(start_paused = true)]
async fn expired_initialization_cannot_commit() {
    let placement = placement(1, 32);
    let first = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(first) = first else {
        panic!("first acquire should initialize");
    };

    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(
        first.commit("cluster-a".to_string()).err(),
        Some(SessionPlacementError::InitializationCancelled)
    );
    assert_eq!(placement.query("session").unwrap(), None);
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
    let lease = initialization.commit("cluster-a".to_string()).unwrap();

    tokio::time::advance(Duration::from_secs(20)).await;
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-a")
    );
    drop(lease);
    tokio::time::advance(Duration::from_secs(9)).await;
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-a")
    );
    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::task::yield_now().await;
    assert_eq!(placement.query("session").unwrap(), None);
}

#[tokio::test(start_paused = true)]
async fn abandoned_lease_does_not_refresh_ttl() {
    let placement = placement(10, 32);
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    let mut lease = initialization.commit("cluster-a".to_string()).unwrap();

    tokio::time::advance(Duration::from_secs(11)).await;
    assert_eq!(
        placement.query("session").unwrap().as_deref(),
        Some("cluster-a")
    );
    lease.abandon();
    assert_eq!(placement.query("session").unwrap(), None);
}

#[tokio::test]
async fn dropping_last_handle_cancels_reaper() {
    let placement = placement(10, 32);
    let cancellation = placement.cancellation_token();
    drop(placement);
    cancellation.cancelled().await;
}

#[tokio::test]
async fn expired_entry_is_reinitialized_before_reaper_runs() {
    let placement = placement(10, 32);
    let acquire = placement.acquire("session").await.unwrap();
    let PlacementAcquire::Initialize(initialization) = acquire else {
        panic!("first acquire should initialize");
    };
    drop(initialization.commit("cluster-a".to_string()).unwrap());
    placement.expire_for_test("session");

    let replacement = placement.acquire("session").await.unwrap();
    assert!(matches!(&replacement, PlacementAcquire::Initialize(_)));
    assert_eq!(placement.entry_count(), 1);
    drop(replacement);
    assert_eq!(placement.entry_count(), 0);
}

#[tokio::test]
async fn invalid_config_is_rejected() {
    let result = SessionPlacement::<String>::new(SessionPlacementConfig {
        idle_ttl: Duration::ZERO,
        initialization_timeout: Some(Duration::from_secs(1)),
        max_entries: 1,
        max_key_bytes: 1,
    });
    assert!(matches!(
        result,
        Err(SessionPlacementError::InvalidConfig(_))
    ));

    let result = SessionPlacement::<String>::new(SessionPlacementConfig {
        idle_ttl: Duration::from_nanos(1),
        initialization_timeout: Some(Duration::from_secs(1)),
        max_entries: 1,
        max_key_bytes: 1,
    });
    assert!(matches!(
        result,
        Err(SessionPlacementError::InvalidConfig(_))
    ));

    let result = SessionPlacement::<String>::new(SessionPlacementConfig {
        idle_ttl: Duration::from_secs(31_536_001),
        initialization_timeout: Some(Duration::from_secs(1)),
        max_entries: 1,
        max_key_bytes: 1,
    });
    assert!(matches!(
        result,
        Err(SessionPlacementError::InvalidConfig(_))
    ));

    let result = SessionPlacement::<String>::new(SessionPlacementConfig {
        idle_ttl: Duration::from_secs(1),
        initialization_timeout: Some(Duration::ZERO),
        max_entries: 1,
        max_key_bytes: 1,
    });
    assert!(matches!(
        result,
        Err(SessionPlacementError::InvalidConfig(_))
    ));
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
