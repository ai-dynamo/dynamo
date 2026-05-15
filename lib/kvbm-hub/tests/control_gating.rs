// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase D module-gating reproducer: a leader with `[Core, Transfer]` (no
//! `Dev`) MUST receive a 404 `module_not_enabled` for `POST /control/dev/reset`
//! AND velo's `_kvbm_leader_control_reset` handler MUST NOT be invoked.
//!
//! The flow:
//! 1. Stand up a hub + a fake leader peer.
//! 2. Install a `list_modules` velo handler returning `[Core, Transfer]`.
//! 3. Install a canned `reset` handler that increments a counter. If the
//!    gate works the counter must stay at 0.
//! 4. Register the peer; poll `GET /modules` until the cache populates.
//! 5. `POST /control/dev/reset` → 404 with `kind: "module_not_enabled"`.
//! 6. Assert the reset counter stayed at 0.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use kvbm_hub::{ControlPlaneManager, HubServer};
use kvbm_protocols::control::{
    ControlReply, LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse, ModuleId,
    RESET_HANDLER, ResetRequest, ResetResponse,
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

async fn start_hub() -> HubServer {
    let transport = new_velo_transport();
    let mgr: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(mgr as Arc<dyn kvbm_hub::FeatureManager>)
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

fn install_modules(peer: &velo::Velo, modules: Vec<ModuleId>) {
    peer.register_handler(
        Handler::typed_unary_async::<ListModulesRequest, _, _, _>(
            LIST_MODULES_HANDLER,
            move |_ctx| {
                let modules = modules.clone();
                async move { Ok(ControlReply::Ok(ListModulesResponse { modules })) }
            },
        )
        .build(),
    )
    .unwrap();
}

fn install_counted_reset(peer: &velo::Velo, counter: Arc<AtomicUsize>) {
    peer.register_handler(
        Handler::typed_unary_async::<ResetRequest, ControlReply<ResetResponse>, _, _>(
            RESET_HANDLER,
            move |_ctx| {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(ControlReply::Ok(ResetResponse {
                        reset: vec![],
                        failed: vec![],
                        skipped_unconfigured: vec![],
                    }))
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

fn reset_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/control/dev/reset",
        server.control_addr(),
        id
    )
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

/// Poll `GET /modules` until the response indicates a cached hit (proves
/// `on_register_any` populated the modules cache with the leader's
/// `[Core, Transfer]` list).
async fn wait_for_modules_cached(server: &HubServer, id: velo_ext::InstanceId, deadline: Duration) {
    let start = Instant::now();
    let url = modules_url(server, id);
    loop {
        let resp = http().get(&url).send().await.expect("GET modules");
        let status = resp.status().as_u16();
        let body: serde_json::Value = resp.json().await.expect("json");
        if status == 200 && body["cached"] == serde_json::json!(true) {
            return;
        }
        if start.elapsed() > deadline {
            panic!("modules cache not populated within {deadline:?}");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

// ---- tests ------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dev_reset_404s_when_module_absent_without_calling_velo() {
    let server = start_hub().await;
    let peer = new_velo().await;
    let reset_counter = Arc::new(AtomicUsize::new(0));

    // Leader advertises Core + Transfer; conspicuously absent: Dev.
    install_modules(&peer, vec![ModuleId::Core, ModuleId::Transfer]);
    install_counted_reset(&peer, reset_counter.clone());

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    // Wait until on_register_any has populated the modules cache.
    wait_for_modules_cached(&server, id, Duration::from_secs(2)).await;

    // Now hit the gated route. The hub MUST short-circuit at the gate and
    // never dispatch to velo.
    let resp = http()
        .post(reset_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset");
    assert_eq!(resp.status().as_u16(), 404);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(
        body["kind"],
        serde_json::json!("module_not_enabled"),
        "expected module_not_enabled; got {body}"
    );

    // The leader's velo `reset` handler MUST NOT have been invoked.
    assert_eq!(
        reset_counter.load(Ordering::SeqCst),
        0,
        "module gate failed — velo handler was dispatched"
    );

    server.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dev_reset_passes_through_when_module_present() {
    let server = start_hub().await;
    let peer = new_velo().await;
    let reset_counter = Arc::new(AtomicUsize::new(0));

    install_modules(
        &peer,
        vec![ModuleId::Core, ModuleId::Dev, ModuleId::Transfer],
    );
    install_counted_reset(&peer, reset_counter.clone());

    let _client = build_client(&server);
    _client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();

    wait_for_modules_cached(&server, id, Duration::from_secs(2)).await;

    let resp = http()
        .post(reset_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST reset");
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        reset_counter.load(Ordering::SeqCst),
        1,
        "velo handler should have been dispatched exactly once"
    );

    server.shutdown().await.unwrap();
}
