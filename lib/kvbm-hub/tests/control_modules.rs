// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase B integration tests: `ControlPlaneManager` modules-cache lifecycle.
//!
//! Stands up a hub plus a fake leader peer that registers
//! `LIST_MODULES_HANDLER` returning a canned `ListModulesResponse`. Drives
//! `GET /v1/instances/{id}/modules` against the hub and asserts:
//!
//! 1. **Populated on register** — the manager's `on_register_any` fetch fills
//!    the cache without the test poking it; `GET …/modules?cached=true` lands.
//! 2. **Cleared on unregister** — dropping the client (RAII unregister) clears
//!    the entry; the next `GET …/modules` is a cache miss that re-fetches
//!    (and 502s when the peer is gone).
//! 3. **`?force=true` re-fetches** — the canned handler counter increments on
//!    each forced request, but stays flat on cached reads.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use kvbm_hub::{ControlPlaneManager, HubServer};
use kvbm_protocols::control::{
    LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse, ModuleId,
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

async fn start_hub_with_control_plane() -> HubServer {
    start_hub_returning_mgr().await.0
}

/// Stand up the hub and keep an `Arc<ControlPlaneManager>` handle so tests
/// can inspect the cache directly (`modules_for`, `has_module`). The
/// returned `Arc` shares state with the one held inside `HubServer`.
async fn start_hub_returning_mgr() -> (HubServer, Arc<ControlPlaneManager>) {
    let transport = new_velo_transport();
    let mgr: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(mgr.clone() as Arc<dyn kvbm_hub::FeatureManager>)
        // Disable hub-driven heartbeat — the cache tests don't exercise it.
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

/// Install a canned `list_modules` handler that returns `modules` and bumps
/// `counter` on each call. Tests assert against `counter` to distinguish
/// cache hits from re-fetches.
fn install_canned_list_modules(
    peer: &velo::Velo,
    modules: Vec<ModuleId>,
    counter: Arc<AtomicUsize>,
) {
    peer.register_handler(
        Handler::typed_unary_async::<ListModulesRequest, _, _, _>(
            LIST_MODULES_HANDLER,
            move |_ctx| {
                let modules = modules.clone();
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(kvbm_protocols::control::ControlReply::Ok(
                        ListModulesResponse { modules },
                    ))
                }
            },
        )
        .build(),
    )
    .unwrap();
}

/// Install a `list_modules` handler that sleeps for `delay` before
/// responding. Used to force-open the on_register/list_modules ↔ unregister
/// race window.
fn install_slow_list_modules(peer: &velo::Velo, modules: Vec<ModuleId>, delay: Duration) {
    peer.register_handler(
        Handler::typed_unary_async::<ListModulesRequest, _, _, _>(
            LIST_MODULES_HANDLER,
            move |_ctx| {
                let modules = modules.clone();
                async move {
                    tokio::time::sleep(delay).await;
                    Ok(kvbm_protocols::control::ControlReply::Ok(
                        ListModulesResponse { modules },
                    ))
                }
            },
        )
        .build(),
    )
    .unwrap();
}

fn modules_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/modules",
        server.control_addr(),
        id
    )
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

/// Poll `GET /modules` until the response indicates a cached hit, the
/// canned counter shows the on-register fetch landed, or the deadline fires.
async fn wait_for_cache_populated(
    server: &HubServer,
    id: velo_ext::InstanceId,
    deadline: Duration,
) -> serde_json::Value {
    let start = Instant::now();
    let url = modules_url(server, id);
    loop {
        let resp = http().get(&url).send().await.expect("GET modules");
        let status = resp.status().as_u16();
        let body: serde_json::Value = resp.json().await.expect("json");
        if status == 200 && body["cached"] == serde_json::json!(true) {
            return body;
        }
        if start.elapsed() > deadline {
            panic!(
                "cache not populated within {deadline:?}; last body = {}",
                serde_json::to_string_pretty(&body).unwrap_or_default()
            );
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

// ---- tests ------------------------------------------------------------------

/// Cache is populated by `on_register_any` after a successful registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cache_populated_on_register() {
    let server = start_hub_with_control_plane().await;
    let peer = new_velo().await;
    let counter = Arc::new(AtomicUsize::new(0));
    install_canned_list_modules(
        &peer,
        vec![ModuleId::Core, ModuleId::Transfer],
        counter.clone(),
    );

    // Register the peer; the manager fans out `list_modules` in the
    // background. Hold `_client` to keep the registration alive.
    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    let body = wait_for_cache_populated(&server, id, Duration::from_secs(2)).await;
    assert_eq!(body["modules"], serde_json::json!(["core", "transfer"]));
    assert_eq!(body["cached"], serde_json::json!(true));
    // After populating, follow-up reads must be cache hits — counter should
    // not advance. (We can't assert exact count after population because
    // `on_register_any` and the wait loop's initial inline fetch race on a
    // cold cache.)
    let counter_after_warm = counter.load(Ordering::SeqCst);
    assert!(
        counter_after_warm >= 1,
        "expected at least one upstream fetch, got {counter_after_warm}"
    );
    for _ in 0..3 {
        let r = http()
            .get(modules_url(&server, id))
            .send()
            .await
            .expect("GET");
        assert_eq!(r.status().as_u16(), 200);
        let body: serde_json::Value = r.json().await.expect("json");
        assert_eq!(body["cached"], serde_json::json!(true));
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        counter_after_warm,
        "cached reads must not trigger upstream fetches"
    );

    server.shutdown().await.unwrap();
}

/// `?force=true` bypasses the cache and increments the upstream counter.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn force_bypasses_cache() {
    let server = start_hub_with_control_plane().await;
    let peer = new_velo().await;
    let counter = Arc::new(AtomicUsize::new(0));
    install_canned_list_modules(&peer, vec![ModuleId::Core], counter.clone());

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    wait_for_cache_populated(&server, id, Duration::from_secs(2)).await;
    let before = counter.load(Ordering::SeqCst);
    assert!(before >= 1);

    // Two forced reads — counter advances by exactly 2.
    for _ in 0..2 {
        let r = http()
            .get(format!("{}?force=true", modules_url(&server, id)))
            .send()
            .await
            .expect("GET force");
        assert_eq!(r.status().as_u16(), 200);
        let body: serde_json::Value = r.json().await.expect("json");
        assert_eq!(body["cached"], serde_json::json!(false));
    }
    assert_eq!(counter.load(Ordering::SeqCst), before + 2);

    server.shutdown().await.unwrap();
}

/// `on_register_any` spawns a `list_modules` fetch task; if the instance
/// unregisters before that task's `await` returns, the *late* `Ok` reply must
/// not re-populate the cache. Without a post-await registry recheck, the
/// stale insert lands and the cache reports modules for an unregistered id.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cache_drops_stale_insert_after_unregister() {
    let (server, mgr) = start_hub_returning_mgr().await;
    let peer = new_velo().await;
    // 250ms artificial delay → unregister fires well before list_modules returns.
    install_slow_list_modules(&peer, vec![ModuleId::Core], Duration::from_millis(250));

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    // Unregister before the slow list_modules can respond.
    tokio::time::sleep(Duration::from_millis(50)).await;
    client.unregister().await.expect("unregister");

    // Wait until the slow fetch has had time to complete and attempt its insert.
    tokio::time::sleep(Duration::from_millis(400)).await;

    // The cache MUST NOT contain a stale entry for the unregistered id.
    assert_eq!(
        mgr.modules_for(id),
        None,
        "stale list_modules result re-populated cache after unregister"
    );

    server.shutdown().await.unwrap();
}

/// Regression: a control call addressed to the hub's OWN velo instance id must
/// be rejected with 400, not attempt a velo self-call. The hub self-registers
/// in its own registry so its id passes `registry.contains`; without a self
/// short-circuit in `leader_client`, the call routes through velo to a peer
/// that is never in the hub's own messenger peer table — "Peer <id> not
/// registered" → 500 — spammed on every control poll a client makes against
/// the hub's id (which it learns from the registration response).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn control_to_hub_self_id_is_rejected() {
    let server = start_hub_with_control_plane().await;
    let peer = new_velo().await;
    // The hub returns its own velo instance id on register — exactly the
    // `hub_velo_id` a real leader learns and then (in the bug) polls control on.
    let client = build_client(&server);
    let hub_id = client
        .register_instance(peer.peer_info())
        .await
        .expect("register")
        .expect("hub returns its velo instance id on register");

    let r = http()
        .get(modules_url(&server, hub_id))
        .send()
        .await
        .expect("GET modules for hub self id");
    assert_eq!(
        r.status().as_u16(),
        400,
        "control addressed to the hub's own velo id must be rejected with 400, \
         not attempt a velo self-call (got {})",
        r.status().as_u16()
    );

    server.shutdown().await.unwrap();
}

/// Cache is cleared when the instance unregisters via the RAII guard.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cache_cleared_on_unregister() {
    let server = start_hub_with_control_plane().await;
    let peer = new_velo().await;
    let counter = Arc::new(AtomicUsize::new(0));
    install_canned_list_modules(&peer, vec![ModuleId::Core], counter.clone());

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    wait_for_cache_populated(&server, id, Duration::from_secs(2)).await;

    // Explicit unregister — fan-out clears the manager cache. The
    // `HubClient::unregister` method consumes the RAII guard, leaving the
    // client unregistered.
    client.unregister().await.expect("unregister");

    // Give the eviction callback a moment to run.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Next read — instance no longer in registry, so 404.
    let r = http()
        .get(modules_url(&server, id))
        .send()
        .await
        .expect("GET after unregister");
    assert_eq!(r.status().as_u16(), 404);

    server.shutdown().await.unwrap();
}
