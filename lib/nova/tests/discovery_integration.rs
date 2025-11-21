// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for discovery-driven workflows.
//!
//! These tests verify that the discovery infrastructure is in place and works correctly.

mod common;

use common::MockDiscovery;
use dynamo_discovery::peer::PeerDiscovery;
use dynamo_nova::am::Nova;
use dynamo_nova_backend::tcp::TcpTransportBuilder;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

fn create_test_listener() -> std::net::TcpListener {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap()
}

/// Create a Nova instance with TCP transport and mock discovery.
async fn create_nova_with_discovery(
    discovery: Arc<MockDiscovery>,
) -> (Arc<Nova>, std::net::SocketAddr) {
    let listener = create_test_listener();
    let addr = listener.local_addr().unwrap();

    let transport = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    );

    let nova = Nova::builder()
        .add_transport(transport)
        .add_discovery_backend(discovery)
        .build()
        .await
        .unwrap();

    (nova, addr)
}

/// Test: MockDiscovery integration with Nova
///
/// Verifies that Nova can be created with MockDiscovery and that peers
/// can be discovered after being populated in the discovery backend.
#[tokio::test]
async fn test_mock_discovery_integration() {
    let discovery = Arc::new(MockDiscovery::new());

    // Create two Nova instances with shared discovery
    let (_foo, _) = create_nova_with_discovery(discovery.clone()).await;
    let (bar, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate discovery with bar's info
    discovery.populate(bar.peer_info());

    // Now foo can discover bar via the discovery backend
    let bar_worker_id = bar.instance_id().worker_id();
    let discovered = discovery
        .discover_by_worker_id(bar_worker_id)
        .await
        .expect("Should discover bar");

    assert_eq!(discovered.instance_id(), bar.instance_id());
}

/// Test: Discovery-based peer resolution
///
/// This test verifies that peers can be discovered and registered via discovery.
#[tokio::test]
async fn test_discovery_peer_resolution() {
    let discovery = Arc::new(MockDiscovery::new());

    let (foo, _) = create_nova_with_discovery(discovery.clone()).await;
    let (bar, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate discovery with bar
    discovery.populate(bar.peer_info());

    let bar_worker_id = bar.instance_id().worker_id();

    // Simulate discovery fallback: lookup and register
    let discovered = discovery
        .discover_by_worker_id(bar_worker_id)
        .await
        .expect("Should discover");

    // Register the discovered peer with foo
    foo.register_peer(discovered).expect("Should register");

    // Verify foo now knows about bar's handlers (this would fail if peer wasn't registered)
    // This is the public API way to verify registration worked
    let handlers = timeout(
        Duration::from_secs(2),
        foo.available_handlers(bar.instance_id()),
    )
    .await
    .expect("Should not timeout")
    .expect("Should get handlers");

    // We expect at least the system handlers to be present
    assert!(!handlers.is_empty(), "Should have at least system handlers");
}

/// Test: Active message with manual discovery
///
/// This demonstrates the discovery pattern for active messages:
/// 1. Try to send (fast path fails)
/// 2. Query discovery
/// 3. Register peer
/// 4. Send succeeds
#[tokio::test]
async fn test_active_message_with_manual_discovery() {
    let discovery = Arc::new(MockDiscovery::new());

    let (foo, _) = create_nova_with_discovery(discovery.clone()).await;
    let (bar, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate discovery
    discovery.populate(foo.peer_info());
    discovery.populate(bar.peer_info());

    // Register handler on bar
    bar.register_handler(
        dynamo_nova::am::NovaHandler::unary_handler("test_handler", |ctx| {
            // Simply echo back the payload
            let response = bytes::Bytes::copy_from_slice(&ctx.payload);
            Ok(Some(response))
        })
        .build(),
    )
    .unwrap();

    // Manually discover and register bar (simulating what automatic discovery would do)
    let bar_worker_id = bar.instance_id().worker_id();
    let discovered = discovery
        .discover_by_worker_id(bar_worker_id)
        .await
        .expect("Should discover");
    foo.register_peer(discovered).expect("Should register");

    // Now active message should work
    let test_payload = b"hello";
    let response = timeout(
        Duration::from_secs(2),
        foo.unary("test_handler")
            .unwrap()
            .worker(bar_worker_id)
            .raw_payload(bytes::Bytes::from_static(test_payload))
            .send(),
    )
    .await
    .expect("Should complete within timeout")
    .expect("Should succeed");

    assert_eq!(response.as_ref(), test_payload);
}

/// Test: Event operations with pre-registered peers
///
/// Events require peers to be registered before creating awaiters.
/// This test verifies the pattern works when combined with discovery.
#[tokio::test]
async fn test_events_with_discovery_registered_peers() {
    let discovery = Arc::new(MockDiscovery::new());

    let (foo, _) = create_nova_with_discovery(discovery.clone()).await;
    let (bar, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate discovery
    discovery.populate(foo.peer_info());
    discovery.populate(bar.peer_info());

    // Discover and register peers both ways
    let foo_discovered = discovery
        .discover_by_worker_id(foo.instance_id().worker_id())
        .await
        .expect("Should discover foo");
    let bar_discovered = discovery
        .discover_by_worker_id(bar.instance_id().worker_id())
        .await
        .expect("Should discover bar");

    foo.register_peer(bar_discovered)
        .expect("Should register bar in foo");
    bar.register_peer(foo_discovered)
        .expect("Should register foo in bar");

    // Now event operations should work
    let bar_event = bar.events().new_event().unwrap();
    let bar_handle = bar_event.handle();

    let foo_subscription = foo
        .events()
        .awaiter(bar_handle)
        .expect("Should create awaiter");

    bar_event.trigger().unwrap();

    timeout(Duration::from_secs(2), foo_subscription)
        .await
        .expect("Should complete within timeout")
        .expect("Event should complete");
}

/// Test: Discovery-based peer lookup flow
///
/// This tests the discovery flow: query discovery, then register peer.
#[tokio::test]
async fn test_discovery_lookup_and_register() {
    let discovery = Arc::new(MockDiscovery::new());

    let (foo, _) = create_nova_with_discovery(discovery.clone()).await;
    let (bar, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate discovery
    discovery.populate(bar.peer_info());

    // Get bar's worker_id
    let bar_worker_id = bar.instance_id().worker_id();
    let bar_instance_id = bar.instance_id();

    // Discovery flow:
    // 1. Query discovery
    let discovered = discovery
        .discover_by_worker_id(bar_worker_id)
        .await
        .expect("Discovery should find bar");

    assert_eq!(discovered.instance_id(), bar_instance_id);
    assert_eq!(discovered.worker_id(), bar_worker_id);

    // 2. Register peer with Nova
    foo.register_peer(discovered).expect("Should register");

    // 3. Verify registration worked by checking handlers
    let handlers = foo
        .available_handlers(bar_instance_id)
        .await
        .expect("Should get handlers");
    assert!(!handlers.is_empty());
}

/// Test: Multiple Nova instances sharing discovery
///
/// Verifies that multiple Nova instances can share the same MockDiscovery
/// backend and discover each other.
#[tokio::test]
async fn test_multi_instance_discovery() {
    let discovery = Arc::new(MockDiscovery::new());

    // Create 3 instances
    let (a, _) = create_nova_with_discovery(discovery.clone()).await;
    let (b, _) = create_nova_with_discovery(discovery.clone()).await;
    let (c, _) = create_nova_with_discovery(discovery.clone()).await;

    // Populate all in discovery
    discovery.populate(a.peer_info());
    discovery.populate(b.peer_info());
    discovery.populate(c.peer_info());

    assert_eq!(discovery.len(), 3);

    // Each can discover the others
    for nova in [&a, &b, &c] {
        let worker_id = nova.instance_id().worker_id();
        let found = discovery
            .discover_by_worker_id(worker_id)
            .await
            .expect("Should discover");
        assert_eq!(found.instance_id(), nova.instance_id());
    }
}
