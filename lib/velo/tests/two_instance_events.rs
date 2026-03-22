// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-instance distributed event integration tests.
//!
//! Tests that events created on one Velo instance can be subscribed to, awaited,
//! triggered, and poisoned from another instance.

use std::sync::Arc;
use std::time::Duration;

use velo::*;
use velo_discovery::FilesystemPeerDiscovery;
use velo_transports::tcp::TcpTransportBuilder;

async fn poll_until(timeout: Duration, mut condition: impl FnMut() -> bool) {
    let start = tokio::time::Instant::now();
    let mut interval = Duration::from_millis(5);
    while !condition() {
        if start.elapsed() > timeout {
            panic!("poll_until timed out after {:?}", timeout);
        }
        tokio::time::sleep(interval).await;
        interval = (interval * 2).min(Duration::from_millis(100));
    }
}

fn new_transport() -> Arc<velo_transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn make_pair() -> (Arc<Velo>, Arc<Velo>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let discovery_file = tmp.path().join("peers.json");
    let discovery = Arc::new(FilesystemPeerDiscovery::new(&discovery_file).unwrap());

    let a = Velo::builder()
        .add_transport(new_transport())
        .discovery(discovery.clone() as Arc<dyn PeerDiscovery>)
        .build()
        .await
        .unwrap();

    let b = Velo::builder()
        .add_transport(new_transport())
        .discovery(discovery.clone() as Arc<dyn PeerDiscovery>)
        .build()
        .await
        .unwrap();

    a.register_peer(b.peer_info()).unwrap();
    b.register_peer(a.peer_info()).unwrap();

    // Verify bidirectional connectivity via handshake
    a.available_handlers(b.instance_id()).await.unwrap();
    b.available_handlers(a.instance_id()).await.unwrap();

    (a, b, tmp)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_event_trigger() {
    let (a, _b, _tmp) = make_pair().await;

    let em = a.event_manager();
    let event = em.new_event().unwrap();
    let handle = event.handle();

    let awaiter = em.awaiter(handle).unwrap();
    em.trigger(handle).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
    assert!(result.is_ok(), "Local event trigger should resolve");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_event_poison() {
    let (a, _b, _tmp) = make_pair().await;

    let em = a.event_manager();
    let event = em.new_event().unwrap();
    let handle = event.handle();

    let awaiter = em.awaiter(handle).unwrap();
    em.poison(handle, "test failure").unwrap();

    let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
    assert!(result.is_ok(), "Local event poison should resolve awaiter");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_event_subscribe_and_trigger() {
    let (a, b, _tmp) = make_pair().await;

    // Create event on instance A
    let em_a = a.event_manager();
    let event = em_a.new_event().unwrap();
    let handle = event.handle();

    // Subscribe from instance B
    let em_b = b.event_manager();
    let awaiter = em_b.awaiter(handle).unwrap();

    // Wait for subscription to propagate
    let a_ref = a.clone();
    let b_id = b.instance_id();
    poll_until(Duration::from_secs(5), move || {
        a_ref.has_event_subscriber(handle, b_id)
    })
    .await;

    // Trigger on instance A
    em_a.trigger(handle).unwrap();

    // Instance B's awaiter should resolve
    let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
    assert!(
        result.is_ok(),
        "Remote event trigger should resolve subscriber's awaiter"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_event_poison() {
    let (a, b, _tmp) = make_pair().await;

    // Create event on A
    let em_a = a.event_manager();
    let event = em_a.new_event().unwrap();
    let handle = event.handle();

    // Subscribe from B
    let em_b = b.event_manager();
    let awaiter = em_b.awaiter(handle).unwrap();

    // Wait for subscription to propagate
    let a_ref = a.clone();
    let b_id = b.instance_id();
    poll_until(Duration::from_secs(5), move || {
        a_ref.has_event_subscriber(handle, b_id)
    })
    .await;

    // Poison on A
    em_a.poison(handle, "test error").unwrap();

    // B's awaiter should resolve (with poison)
    let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
    assert!(
        result.is_ok(),
        "Remote event poison should resolve subscriber's awaiter"
    );
}
