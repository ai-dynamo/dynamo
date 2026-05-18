// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase C integration tests: `ControlPlaneManager` describe cache lifecycle.
//!
//! Covers:
//! 1. **Push happy path** — leader fixture pushes via `HubClient::push_describe`;
//!    `GET /describe` serves the cached payload with `source: "push"`.
//! 2. **Pull fallback** — describe handler installed on a leader's velo;
//!    leader never pushes; `GET /describe?force=true` triggers the pull and
//!    populates the cache with `source: "pull_fallback"`.
//! 3. **Pending state** — leader has neither pushed nor installed a velo
//!    handler; `GET /describe` returns `503 describe_pending` with
//!    `registered_secs_ago`.
//! 4. **Stale-insert race** — leader's velo handler sleeps; `?force=true`
//!    fires; instance unregisters mid-pull; cache must NOT hold a stale entry.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use kvbm_hub::{ControlPlaneManager, HubServer};
use kvbm_protocols::control::{
    ControlReply, DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest, HostInfo, InstanceDescription,
};
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;

// ---- fixtures ---------------------------------------------------------------

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
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

async fn start_hub_returning_mgr() -> (HubServer, Arc<ControlPlaneManager>) {
    let transport = new_velo_transport();
    let mgr: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(mgr.clone() as Arc<dyn kvbm_hub::FeatureManager>)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .serve()
        .await
        .expect("start hub");
    (server, mgr)
}

fn build_client(server: &HubServer) -> Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host(server.discovery_addr().ip().to_string())
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build hub client")
}

/// A canned `InstanceDescription` with enough non-default fields that
/// round-trips can detect drift.
fn canned_description(instance_id: &str) -> InstanceDescription {
    InstanceDescription {
        instance_id: instance_id.to_owned(),
        worker_ids: vec![1, 2],
        hub_instance_id: Some("hub-stub".to_owned()),
        block_size: Some(16),
        parallelism: None,
        tier_capacity: Vec::new(),
        workers: Vec::new(),
        modules: Vec::new(),
        role: None,
        config: Some(serde_json::json!({ "model": "test/qwen3" })),
        host: HostInfo {
            hostname: "test-host".into(),
            pid: 1234,
        },
        started_at: SystemTime::UNIX_EPOCH,
        layout_compat: None,
    }
}

/// Install a `describe_instance` velo handler that returns the canned payload.
/// Optionally delays before responding (for the race reproducer).
fn install_canned_describe(
    peer: &velo::Velo,
    payload: InstanceDescription,
    delay: Option<Duration>,
) {
    peer.register_handler(
        Handler::typed_unary_async::<DescribeInstanceRequest, _, _, _>(
            DESCRIBE_INSTANCE_HANDLER,
            move |_ctx| {
                let payload = payload.clone();
                let delay = delay;
                async move {
                    if let Some(d) = delay {
                        tokio::time::sleep(d).await;
                    }
                    Ok(ControlReply::Ok(payload))
                }
            },
        )
        .build(),
    )
    .unwrap();
}

fn describe_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/describe",
        server.control_addr(),
        id
    )
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

// ---- tests ------------------------------------------------------------------

/// Steady state: leader pushes; hub serves the cached payload.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn push_then_get_returns_cached_payload() {
    let (server, _mgr) = start_hub_returning_mgr().await;
    let peer = new_velo().await;

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    // Leader push via the new HubClient method.
    let payload = canned_description(&id.to_string());
    _client.push_describe(id, &payload).await.expect("push");

    let resp = http()
        .get(describe_url(&server, id))
        .send()
        .await
        .expect("GET");
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["cached"], serde_json::json!(true));
    assert_eq!(body["source"], serde_json::json!("push"));
    assert_eq!(
        body["description"]["instance_id"],
        serde_json::json!(id.to_string())
    );
    assert_eq!(
        body["description"]["config"],
        serde_json::json!({ "model": "test/qwen3" })
    );

    server.shutdown().await.unwrap();
}

/// Fallback: leader never pushes; `?force=true` triggers a velo pull.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn force_triggers_pull_fallback() {
    let (server, _mgr) = start_hub_returning_mgr().await;
    let peer = new_velo().await;

    let id = peer.instance_id();
    install_canned_describe(&peer, canned_description(&id.to_string()), None);

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");

    let resp = http()
        .get(format!("{}?force=true", describe_url(&server, id)))
        .send()
        .await
        .expect("GET force");
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["cached"], serde_json::json!(false));
    assert_eq!(body["source"], serde_json::json!("pull_fallback"));

    // Subsequent GET without force serves from cache, source persists.
    let resp2 = http()
        .get(describe_url(&server, id))
        .send()
        .await
        .expect("GET 2");
    let body2: serde_json::Value = resp2.json().await.expect("json2");
    assert_eq!(body2["cached"], serde_json::json!(true));
    assert_eq!(body2["source"], serde_json::json!("pull_fallback"));

    server.shutdown().await.unwrap();
}

/// Pending: leader is registered but has neither pushed nor installed a
/// velo handler. GET (no force) returns 503 `describe_pending`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pending_state_returns_503() {
    let (server, _mgr) = start_hub_returning_mgr().await;
    let peer = new_velo().await;
    // No `install_canned_describe` — handler is absent.

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    let resp = http()
        .get(describe_url(&server, id))
        .send()
        .await
        .expect("GET");
    assert_eq!(resp.status().as_u16(), 503);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], serde_json::json!("describe_pending"));
    assert!(body["registered_secs_ago"].is_number());

    server.shutdown().await.unwrap();
}

/// Race reproducer: leader's velo `describe` handler sleeps. The hub fires
/// `?force=true` which awaits the slow describe; concurrently the test
/// unregisters the instance. The stale `Ok` reply MUST NOT populate the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn describe_drops_stale_insert_after_unregister() {
    let (server, mgr) = start_hub_returning_mgr().await;
    let peer = new_velo().await;
    let id = peer.instance_id();
    install_canned_describe(
        &peer,
        canned_description(&id.to_string()),
        Some(Duration::from_millis(250)),
    );

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");

    // Kick off the forced pull in a background task — it'll await the slow
    // handler. We unregister before it returns.
    let url = format!("{}?force=true", describe_url(&server, id));
    let pull = tokio::spawn(async move { http().get(url).send().await });

    // Give the pull a head start so it's mid-await before we unregister.
    tokio::time::sleep(Duration::from_millis(40)).await;
    client.unregister().await.expect("unregister");

    // Let the pull finish (or fail) before we assert.
    let _ = pull.await;

    // Cache must not contain a stale entry. (`InstanceDescription` doesn't
    // implement `PartialEq` upstream — use `is_none()` instead of `assert_eq`.)
    assert!(
        mgr.describe_for(id).is_none(),
        "stale describe pull re-populated cache after unregister"
    );

    server.shutdown().await.unwrap();
}
