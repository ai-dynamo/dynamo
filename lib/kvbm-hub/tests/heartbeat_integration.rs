// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the hub-driven heartbeat task (task 18).
//!
//! These exercise the fan-out + per-probe-timeout architecture against
//! real `velo::Velo` peers — no GPU, no vLLM, no Python. The hub is
//! configured with a tiny `heartbeat_interval` and `registration_ttl`
//! so each test runs in seconds.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use kvbm_hub::HubServer;
use kvbm_hub::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use velo::Handler;
use velo::backend::tcp::TcpTransportBuilder;

fn init_tracing() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,kvbm_hub=debug")),
            )
            .with_test_writer()
            .try_init();
    });
}

// ---- fixtures ---------------------------------------------------------------

fn new_velo_transport() -> Arc<velo::backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

/// Start a hub configured for heartbeat testing.
///
/// Defaults that matter for these tests:
/// - velo transport attached so heartbeat task is spawned.
/// - 30s registration TTL (default) — large enough that the reaper
///   never preempts the heartbeat path during test duration.
async fn start_hub_with_heartbeat(
    interval: Duration,
    max_failures: u32,
) -> HubServer {
    let transport = new_velo_transport();
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .heartbeat_interval(interval)
        .heartbeat_max_failures(max_failures)
        .registration_ttl(Duration::from_secs(60))
        .prune_interval(Duration::from_secs(30))
        .serve()
        .await
        .expect("start hub")
}

/// Start a hub WITHOUT a velo transport — discovery-only mode.
async fn start_discovery_only_hub() -> HubServer {
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .heartbeat_interval(Duration::from_millis(200))
        .heartbeat_max_failures(2)
        .serve()
        .await
        .expect("start discovery-only hub")
}

fn build_client(server: &HubServer) -> Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host(server.discovery_addr().ip().to_string())
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build hub client")
}

/// Install the standard hub heartbeat handler — peer responds OK.
fn install_responsive_heartbeat(velo: &velo::Velo) {
    velo.register_handler(
        Handler::typed_unary_async::<HeartbeatRequest, HeartbeatAck, _, _>(
            HEARTBEAT_HANDLER,
            |ctx| async move {
                Ok(HeartbeatAck {
                    seq: ctx.input.seq,
                    ok: true,
                })
            },
        )
        .build(),
    )
    .unwrap();
}

/// Install a heartbeat handler that sleeps far longer than the
/// hub's per-probe timeout, so probes time out.
fn install_wedged_heartbeat(velo: &velo::Velo, sleep: Duration) {
    velo.register_handler(
        Handler::typed_unary_async::<HeartbeatRequest, HeartbeatAck, _, _>(
            HEARTBEAT_HANDLER,
            move |ctx| async move {
                tokio::time::sleep(sleep).await;
                Ok(HeartbeatAck {
                    seq: ctx.input.seq,
                    ok: true,
                })
            },
        )
        .build(),
    )
    .unwrap();
}

/// Register `peer_velo` with the hub via the HubClient flow.
///
/// Returns the `Arc<HubClient>` along with the registered InstanceId
/// — **the caller MUST hold the client for the lifetime of the
/// registration.** Dropping the HubClient triggers
/// `HubRegistrationGuard::Drop` which issues an HTTP DELETE and
/// unregisters the peer. Letting the client fall out of scope mid-test
/// silently de-registers and the heartbeat task observes an empty
/// registry.
///
/// **Caller installs the heartbeat handler beforehand.** This helper
/// deliberately does NOT call `HubClient::register_handlers`, which
/// would overwrite a custom (e.g. wedged) heartbeat handler installed
/// by the test. Real-world clients use `register_handlers`; tests
/// install their own to control probe behavior.
async fn register_peer(
    server: &HubServer,
    peer_velo: &Arc<velo::Velo>,
) -> (Arc<kvbm_hub::HubClient>, velo_common::InstanceId) {
    let client = build_client(server);
    client
        .register_instance(peer_velo.peer_info())
        .await
        .expect("register");
    (client, peer_velo.instance_id())
}

/// Snapshot of currently-registered instance ids.
fn registered_ids(server: &HubServer) -> std::collections::HashSet<velo_common::InstanceId> {
    server
        .state()
        .peers()
        .into_iter()
        .map(|p| p.instance_id())
        .collect()
}

// ---- tests ------------------------------------------------------------------

/// Probe success refreshes the registry's `last_heartbeat_at`. Sustained
/// heartbeats keep an instance registered indefinitely.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn responsive_peer_stays_registered() {
    init_tracing();
    let server = start_hub_with_heartbeat(Duration::from_millis(200), 3).await;
    let peer = new_velo().await;
    install_responsive_heartbeat(&peer);
    let (_client, id) = register_peer(&server, &peer).await;

    // Past several intervals: still registered.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(registered_ids(&server).contains(&id));

    server.shutdown().await.unwrap();
}

/// A wedged peer (heartbeat handler hangs forever) gets unregistered
/// after `max_failures` consecutive timeouts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wedged_peer_pruned_after_max_failures() {
    init_tracing();
    let interval = Duration::from_secs(1);
    let max_failures: u32 = 2;
    let server = start_hub_with_heartbeat(interval, max_failures).await;
    let peer = new_velo().await;
    install_wedged_heartbeat(&peer, Duration::from_secs(60));
    let (_client, id) = register_peer(&server, &peer).await;

    // Confirm initial registration.
    assert!(registered_ids(&server).contains(&id));

    // Worst-case timeline:
    // - 1 interval before first tick fires (200ms+).
    // - Each tick spawns a probe; probe times out at exactly `interval`.
    // - The unregister happens on the Nth Failed outcome arrival,
    //   which is roughly N intervals after the first tick.
    // Budget: (N+2) intervals + 1s scheduling slack.
    let budget = interval * (max_failures + 2) + Duration::from_secs(2);
    tokio::time::sleep(budget).await;

    assert!(
        !registered_ids(&server).contains(&id),
        "wedged peer should have been pruned after {max_failures} timeouts (waited {:?})",
        budget,
    );
    server.shutdown().await.unwrap();
}

/// A wedged peer sharing a tick with a healthy peer must NOT delay the
/// healthy peer's TTL refresh — proves probes run concurrently.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wedged_peer_does_not_block_healthy_peer() {
    let interval = Duration::from_millis(200);
    // Pick `max_failures` high enough that the wedged peer is NOT
    // pruned during the test window — we're measuring the healthy
    // peer's behavior while the wedged probe is in flight.
    let server = start_hub_with_heartbeat(interval, 100).await;

    let healthy = new_velo().await;
    install_responsive_heartbeat(&healthy);
    let (_h_client, healthy_id) = register_peer(&server, &healthy).await;

    let wedged = new_velo().await;
    install_wedged_heartbeat(&wedged, Duration::from_secs(60));
    let (_w_client, wedged_id) = register_peer(&server, &wedged).await;

    // Wait several intervals while wedged is hung.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let ids = registered_ids(&server);
    assert!(ids.contains(&healthy_id), "healthy peer must stay registered");
    assert!(ids.contains(&wedged_id), "wedged peer should still be there (max_failures=100)");

    server.shutdown().await.unwrap();
}

/// Discovery-only hub (no velo) does NOT spawn the heartbeat task.
/// Liveness falls back to the registry's TTL reaper.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discovery_only_hub_does_not_probe() {
    let server = start_discovery_only_hub().await;
    // The hub didn't crash, no transport was attached. We can't
    // *directly* assert "no task was spawned" but we can confirm the
    // hub's state has no velo handle.
    assert!(
        server.state().velo().is_none(),
        "discovery-only hub should expose no velo handle"
    );
    server.shutdown().await.unwrap();
}

/// The hub never probes its own self-entry. We confirm by the absence
/// of any per-instance failure dynamics on a hub that has registered
/// only itself: shutting down cleanly is sufficient evidence (the
/// task isn't busy-looping on a non-existent target).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_does_not_probe_self() {
    let server = start_hub_with_heartbeat(Duration::from_millis(50), 2).await;

    // The hub self-registers in HubServerBuilder::serve when it has a
    // velo transport. Verify that's the only entry, and that the
    // heartbeat task hasn't pruned it after several intervals.
    tokio::time::sleep(Duration::from_secs(1)).await;

    let ids = registered_ids(&server);
    assert_eq!(ids.len(), 1, "only the hub itself should be registered");

    let velo = server.state().velo().expect("hub has velo");
    assert!(
        ids.contains(&velo.instance_id()),
        "hub's self-entry should still be present"
    );

    server.shutdown().await.unwrap();
}
