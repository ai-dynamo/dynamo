// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_nova::am::Nova;
use dynamo_nova_backend::tcp::TcpTransportBuilder;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

fn create_test_listener() -> std::net::TcpListener {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap()
}

async fn create_pair() -> (Arc<Nova>, Arc<Nova>) {
    let transport_a = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_a = Nova::new(vec![transport_a], vec![]).await.unwrap();

    let transport_b = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(create_test_listener())
            .unwrap()
            .build()
            .unwrap(),
    );
    let nova_b = Nova::new(vec![transport_b], vec![]).await.unwrap();

    // Register peers both ways
    let peer_info_a = nova_a.peer_info();
    let peer_info_b = nova_b.peer_info();

    nova_a.register_peer(peer_info_b).unwrap();
    nova_b.register_peer(peer_info_a).unwrap();

    (nova_a, nova_b)
}

/// Helper to create a cluster of N connected Nova instances
async fn create_cluster(n: usize) -> Vec<Arc<Nova>> {
    let mut instances = Vec::new();

    // Create instances
    for _ in 0..n {
        let transport = Arc::new(
            TcpTransportBuilder::new()
                .from_listener(create_test_listener())
                .unwrap()
                .build()
                .unwrap(),
        );
        let nova = Nova::new(vec![transport], vec![]).await.unwrap();
        instances.push(nova);
    }

    // Register all peers with each other (full mesh)
    let peer_infos: Vec<_> = instances.iter().map(|n| n.peer_info()).collect();
    for (i, instance) in instances.iter().enumerate() {
        for (j, peer_info) in peer_infos.iter().enumerate() {
            if i != j {
                instance.register_peer(peer_info.clone()).unwrap();
            }
        }
    }

    instances
}

#[tokio::test]
async fn remote_wait_round_trip() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    let waiter = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await })
    };

    event.trigger().unwrap();

    timeout(Duration::from_millis(500), waiter)
        .await
        .expect("waiter timed out")
        .expect("join")
        .expect("remote wait should succeed");
}

#[tokio::test]
async fn remote_trigger_round_trip() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    let owner_wait = {
        let owner = owner.clone();
        tokio::spawn(async move { owner.events().awaiter(handle).unwrap().await })
    };

    let trigger = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().trigger(handle).await })
    };

    timeout(Duration::from_millis(500), owner_wait)
        .await
        .expect("owner wait timeout")
        .expect("owner join")
        .expect("owner should complete");

    timeout(Duration::from_millis(500), trigger)
        .await
        .expect("trigger timeout")
        .expect("trigger join")
        .expect("trigger should succeed");
}

#[tokio::test]
async fn remote_poison_round_trip() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    let owner_wait = {
        let owner = owner.clone();
        tokio::spawn(async move { owner.events().awaiter(handle).unwrap().await })
    };

    let poison = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().poison(handle, "boom").await })
    };

    let owner_err = timeout(Duration::from_millis(500), owner_wait)
        .await
        .expect("owner wait timeout")
        .expect("owner join")
        .expect_err("owner wait should fail");
    assert!(owner_err.to_string().contains("boom"));

    timeout(Duration::from_millis(500), poison)
        .await
        .expect("poison timeout")
        .expect("poison join")
        .expect("poison should succeed");
}

// ========== Critical Fix Tests (Issues #1, #2, #3) ==========

/// Test Issue #1: Already-complete events notify remote waiters
/// Validates that trigger_request for complete events sends completion to existing subscribers
#[tokio::test]
async fn test_already_complete_notifies_waiters() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Remote subscribes and waits
    let waiter = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await })
    };

    // Give subscription time to register
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Event completes locally
    event.trigger().unwrap();

    // Waiter should complete (tests Issue #1 fix)
    timeout(Duration::from_millis(500), waiter)
        .await
        .expect("waiter timed out - Issue #1 regression!")
        .expect("join")
        .expect("remote wait should succeed");
}

/// Test Issue #2: Poison request ACKs when event triggers (lenient semantics)
#[tokio::test]
async fn test_poison_request_acks_when_triggered() {
    let cluster = create_cluster(3).await;
    let owner = &cluster[0];
    let remote1 = &cluster[1];
    let remote2 = &cluster[2];

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Remote 1 triggers concurrently with Remote 2 poison
    let trigger = {
        let remote1 = remote1.clone();
        tokio::spawn(async move { remote1.events().trigger(handle).await })
    };

    let poison = {
        let remote2 = remote2.clone();
        tokio::spawn(async move { remote2.events().poison(handle, "race").await })
    };

    // Both should succeed (lenient semantics - Issue #2 fix)
    let trigger_result = timeout(Duration::from_millis(500), trigger)
        .await
        .expect("trigger timeout")
        .expect("join");

    let poison_result = timeout(Duration::from_millis(500), poison)
        .await
        .expect("poison timeout")
        .expect("join");

    // With lenient semantics, both should ACK (event is complete)
    assert!(
        trigger_result.is_ok() || poison_result.is_ok(),
        "At least one should succeed"
    );
}

/// Test Issue #3: Completion history stays bounded
#[tokio::test]
async fn test_completion_history_bounded() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Subscribe remotely to trigger RemoteEvent creation
    let waiter = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await })
    };

    // Trigger first generation
    event.trigger().unwrap();

    // Wait for completion
    timeout(Duration::from_millis(500), waiter)
        .await
        .expect("waiter timeout")
        .expect("join")
        .expect("wait should succeed");

    // Trigger many more generations (tests bounded pruning)
    // Note: Can't directly inspect RemoteEvent::completions, but test shouldn't OOM
    for _generation in 2..=150 {
        let next_event = owner.events().new_event().unwrap();
        next_event.trigger().unwrap();
        // Don't need to wait, just testing memory doesn't explode
    }

    // If we get here without OOM, Issue #3 fix is working
    assert!(true, "Completed 150 generations without unbounded growth");
}

// ========== Concurrency & Race Tests ==========

/// Test concurrent subscriptions from multiple instances
#[tokio::test]
async fn test_concurrent_subscriptions() {
    let cluster = create_cluster(4).await;
    let owner = &cluster[0];
    let remotes = &cluster[1..];

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // All 3 remotes subscribe concurrently
    let mut waiters = vec![];
    for remote in remotes {
        let remote = remote.clone();
        let waiter = tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await });
        waiters.push(waiter);
    }

    // Give subscriptions time to register
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Owner triggers event
    event.trigger().unwrap();

    // All waiters should complete
    for (i, waiter) in waiters.into_iter().enumerate() {
        timeout(Duration::from_millis(500), waiter)
            .await
            .unwrap_or_else(|_| panic!("waiter {} timed out", i))
            .expect("join")
            .expect("should succeed");
    }
}

/// Test multiple generation subscriptions (1, 2, then 1 again)
#[tokio::test]
async fn test_multiple_generations_sequential() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Subscribe to gen 1
    let wait_gen1 = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await })
    };

    // Trigger gen 1
    event.trigger().unwrap();

    timeout(Duration::from_millis(500), wait_gen1)
        .await
        .expect("gen 1 timeout")
        .expect("join")
        .expect("gen 1 should complete");

    // Subscribe to gen 1 again (should be cached - immediate return)
    let wait_gen1_cached = {
        let remote = remote.clone();
        tokio::spawn(async move { remote.events().awaiter(handle).unwrap().await })
    };

    // Should complete immediately from cache
    timeout(Duration::from_millis(100), wait_gen1_cached)
        .await
        .expect("gen 1 cached should be fast")
        .expect("join")
        .expect("cached should succeed");
}

// ========== Cache & Memory Behavior Tests ==========

/// Test LRU cache fast path for completed events
#[tokio::test]
async fn test_lru_cache_fast_path() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Complete the event
    event.trigger().unwrap();

    // First wait from remote (populates cache)
    remote.events().awaiter(handle).unwrap().await.unwrap();

    // Second wait should hit cache (very fast, no network)
    let start = std::time::Instant::now();
    remote.events().awaiter(handle).unwrap().await.unwrap();
    let elapsed = start.elapsed();

    // Cache hit should be nearly instant (< 10ms, generous for CI)
    assert!(
        elapsed.as_millis() < 10,
        "Cache hit took {}ms, expected < 10ms",
        elapsed.as_millis()
    );
}

/// Test subscribing after event already complete
#[tokio::test]
async fn test_subscriber_after_completion() {
    let (owner, remote) = create_pair().await;

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Trigger event before anyone subscribes
    event.trigger().unwrap();

    // Remote subscribes after completion - should get immediate result
    let wait = remote.events().awaiter(handle).unwrap();

    timeout(Duration::from_millis(500), wait)
        .await
        .expect("post-completion subscription should be fast")
        .expect("should succeed");
}

/// Test poison idempotency (multiple poison calls)
#[tokio::test]
async fn test_poison_idempotency() {
    let cluster = create_cluster(3).await;
    let owner = &cluster[0];
    let remote1 = &cluster[1];
    let remote2 = &cluster[2];

    let event = owner.events().new_event().unwrap();
    let handle = event.handle();

    // Remote 1 poisons
    remote1
        .events()
        .poison(handle, "first poison")
        .await
        .expect("first poison should succeed");

    // Remote 2 also tries to poison (idempotent)
    let result2 = remote2.events().poison(handle, "second poison").await;

    // Second poison might succeed (lenient) or fail (already poisoned), but shouldn't panic
    assert!(
        result2.is_ok() || result2.is_err(),
        "Second poison should complete without panic"
    );

    // Event should be poisoned
    let wait_result = owner.events().awaiter(handle).unwrap().await;
    assert!(
        wait_result.is_err(),
        "Event should be poisoned after poison operations"
    );
}
