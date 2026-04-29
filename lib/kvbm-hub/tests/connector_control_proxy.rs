// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the hub's HTTP→velo proxy
//! (`ConnectorControlManager`).
//!
//! Stands up a hub plus a fake velo "connector" peer that registers a
//! `kvbm.connector.leader.reset` handler returning canned
//! `ControlReply` values. Drives `PUT /v1/instances/{id}/reset` against
//! the hub and asserts the proxy maps each connector outcome to the
//! correct HTTP status — equivalent to what the connector's local
//! axum shim would have returned for the same logical outcome.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use kvbm_control_protocol::{
    ControlError, ControlReply, RESET_HANDLER, ResetRequest, ResetResponse, Tier, TierError,
};
use kvbm_hub::{ConnectorControlManager, HubServer};
use velo::Handler;
use velo::backend::tcp::TcpTransportBuilder;

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

async fn start_hub_with_proxy() -> HubServer {
    let transport = new_velo_transport();
    let proxy: Arc<ConnectorControlManager> = Arc::new(ConnectorControlManager::new());
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(proxy as Arc<dyn kvbm_hub::FeatureManager>)
        // Disable hub-driven heartbeat by setting an interval longer
        // than each test's runtime — these tests don't exercise the
        // liveness path, and we don't want probes overwriting our
        // canned reset handler responses.
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .serve()
        .await
        .expect("start hub")
}

fn build_client(server: &HubServer) -> Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host(server.discovery_addr().ip().to_string())
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build hub client")
}

/// Register a fake connector peer. Returns `(client, instance_id)`
/// — caller must hold the client to keep the registration alive.
async fn register_connector(
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

/// Install a reset handler on the peer's velo that always returns the
/// given canned reply.
fn install_canned_reset(peer: &velo::Velo, canned: ControlReply<ResetResponse>) {
    peer.register_handler(
        Handler::typed_unary_async::<ResetRequest, ControlReply<ResetResponse>, _, _>(
            RESET_HANDLER,
            move |_ctx| {
                let canned = canned.clone();
                async move { Ok(canned) }
            },
        )
        .build(),
    )
    .unwrap();
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

fn proxy_reset_url(server: &HubServer, id: velo_common::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/reset",
        server.control_addr(),
        id
    )
}

// ---- tests ------------------------------------------------------------------

/// Connector returns `Ok` — proxy unwraps the envelope and returns
/// 200 with the inner `ResetResponse` body.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_ok_returns_200_with_unwrapped_body() {
    let server = start_hub_with_proxy().await;
    let peer = new_velo().await;
    install_canned_reset(
        &peer,
        ControlReply::Ok(ResetResponse {
            reset: vec![Tier::G2],
            failed: vec![],
            skipped_unconfigured: vec![Tier::G3],
        }),
    );
    let (_c, id) = register_connector(&server, &peer).await;

    let resp = http()
        .put(proxy_reset_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["reset"], serde_json::json!(["g2"]));
    assert_eq!(body["skipped_unconfigured"], serde_json::json!(["g3"]));
    // The envelope's "status" field MUST NOT leak through; the proxy
    // unwraps Ok before sending to the HTTP client.
    assert!(body.get("status").is_none(), "envelope leaked: {body}");

    server.shutdown().await.unwrap();
}

/// Connector returns `Err(TierNotConfigured)` — proxy maps to 400.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_tier_not_configured_returns_400() {
    let server = start_hub_with_proxy().await;
    let peer = new_velo().await;
    install_canned_reset(
        &peer,
        ControlReply::Err(ControlError::TierNotConfigured(Tier::G3)),
    );
    let (_c, id) = register_connector(&server, &peer).await;

    let resp = http()
        .put(proxy_reset_url(&server, id))
        .header("content-type", "application/json")
        .body(r#"{"tiers":["g3"]}"#)
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 400);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], "tier_not_configured");
    assert!(body["error"].as_str().unwrap().contains("not configured"));

    server.shutdown().await.unwrap();
}

/// Connector returns `Err(NotInitialized)` — proxy maps to 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_not_initialized_returns_503() {
    let server = start_hub_with_proxy().await;
    let peer = new_velo().await;
    install_canned_reset(&peer, ControlReply::Err(ControlError::NotInitialized));
    let (_c, id) = register_connector(&server, &peer).await;

    let resp = http()
        .put(proxy_reset_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 503);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], "not_initialized");

    server.shutdown().await.unwrap();
}

/// Connector returns `Err(Internal)` — proxy maps to 500.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_internal_error_returns_500() {
    let server = start_hub_with_proxy().await;
    let peer = new_velo().await;
    install_canned_reset(
        &peer,
        ControlReply::Err(ControlError::Internal("boom".into())),
    );
    let (_c, id) = register_connector(&server, &peer).await;

    let resp = http()
        .put(proxy_reset_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 500);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], "internal");

    server.shutdown().await.unwrap();
}

/// Connector returns `Ok` with a non-empty `failed` list — proxy
/// still returns 200; per-tier failures are surfaced inside the body
/// (the request as a whole succeeded at the transport level). This
/// matches the connector's local axum shim, which returns 200 unless
/// `failed` is non-empty AND the connector chose to escalate. Today
/// the connector keeps 200 with details inline.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_partial_failure_in_response_body() {
    let server = start_hub_with_proxy().await;
    let peer = new_velo().await;
    install_canned_reset(
        &peer,
        ControlReply::Ok(ResetResponse {
            reset: vec![Tier::G2],
            failed: vec![TierError {
                tier: Tier::G3,
                message: "g3 manager I/O error".into(),
            }],
            skipped_unconfigured: vec![],
        }),
    );
    let (_c, id) = register_connector(&server, &peer).await;

    let resp = http()
        .put(proxy_reset_url(&server, id))
        .header("content-type", "application/json")
        .body(r#"{"tiers":["g2","g3"]}"#)
        .send()
        .await
        .expect("PUT");
    // Proxy returns 200 because envelope is Ok; partial-failure
    // signaling lives in the body.
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["reset"], serde_json::json!(["g2"]));
    assert_eq!(body["failed"][0]["tier"], "g3");

    server.shutdown().await.unwrap();
}

/// Hub registry doesn't know the instance — proxy returns 404 without
/// touching velo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_unknown_instance_returns_404() {
    let server = start_hub_with_proxy().await;
    let nonexistent = velo_common::InstanceId::new_v4();

    let resp = http()
        .put(proxy_reset_url(&server, nonexistent))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 404);

    server.shutdown().await.unwrap();
}

/// Discovery-only hub (no velo) — proxy routes return 503.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn proxy_discovery_only_returns_503() {
    // Hub WITHOUT a velo transport.
    let proxy: Arc<ConnectorControlManager> = Arc::new(ConnectorControlManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(proxy as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start hub");

    // Even with a registered instance (we'll fake it by skipping
    // registration since the registry check fires first), the route
    // path itself reaches the proxy. Use a bogus id; the manager
    // checks velo presence before the registry, so the response is 503.
    let resp = http()
        .put(proxy_reset_url(&server, velo_common::InstanceId::new_v4()))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("PUT");
    assert_eq!(resp.status().as_u16(), 503);

    server.shutdown().await.unwrap();
}
