// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `metrics` control module's HTTP surface on the
//! hub:
//!
//! - `POST /v1/instances/{id}/control/metrics/snapshot` — single leader,
//!   gated on `ModuleId::Metrics`.
//! - `GET /v1/metrics` — fanout across every registered leader that has
//!   the metrics module enabled. Failing/absent leaders surface as per-
//!   instance error entries rather than failing the whole response.
//!
//! The leader side here is a canned velo handler — we don't bring up an
//! `InstanceLeader` because the goal is to exercise the hub's HTTP plumbing
//! (route, gate, fanout, error mapping). Leader-side metric folding has its
//! own unit tests in `kvbm-engine::leader::control::modules::metrics`.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use kvbm_hub::{ControlPlaneManager, HubServer};
use kvbm_protocols::control::{
    ControlError, ControlReply, LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse,
    MetricsSnapshotRequest, MetricsSnapshotResponse, ModuleId, PoolBreakdown, SNAPSHOT_HANDLER,
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
        // Disable hub-driven heartbeat so probes don't race with our canned
        // handlers (mirrors control_routes / control_gating).
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

fn install_canned_snapshot(
    peer: &velo::Velo,
    canned: ControlReply<MetricsSnapshotResponse>,
    counter: Arc<AtomicUsize>,
) {
    peer.register_handler(
        Handler::typed_unary_async::<MetricsSnapshotRequest, _, _, _>(
            SNAPSHOT_HANDLER,
            move |_ctx| {
                let canned = canned.clone();
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(canned)
                }
            },
        )
        .build(),
    )
    .unwrap();
}

/// Install a snapshot handler that sleeps `delay` before responding. Used to
/// drive the per-leader timeout path in the fanout.
fn install_slow_snapshot(peer: &velo::Velo, delay: Duration) {
    peer.register_handler(
        Handler::typed_unary_async::<MetricsSnapshotRequest, _, _, _>(
            SNAPSHOT_HANDLER,
            move |_ctx| async move {
                tokio::time::sleep(delay).await;
                Ok(ControlReply::Ok(MetricsSnapshotResponse {
                    gathered_at_unix_ms: 0,
                    sessions_inflight: 0,
                    pools: vec![],
                    cd: None,
                }))
            },
        )
        .build(),
    )
    .unwrap();
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

fn snapshot_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/control/metrics/snapshot",
        server.control_addr(),
        id
    )
}

fn fanout_url(server: &HubServer) -> String {
    format!("http://{}/v1/metrics", server.control_addr())
}

fn modules_url(server: &HubServer, id: velo_ext::InstanceId) -> String {
    format!(
        "http://{}/v1/instances/{}/modules",
        server.control_addr(),
        id
    )
}

/// Poll `GET /v1/instances/{id}/modules` until the manager reports a cached
/// hit — i.e. `on_register_any`'s background fetch has landed. Without this,
/// the gating tests race the background `list_modules` task and would
/// occasionally flake.
async fn wait_for_modules_cached(server: &HubServer, id: velo_ext::InstanceId, deadline: Duration) {
    let start = Instant::now();
    loop {
        let resp = http()
            .get(modules_url(server, id))
            .send()
            .await
            .expect("GET modules");
        if resp.status().as_u16() == 200 {
            let body: serde_json::Value = resp.json().await.expect("json");
            if body["cached"] == serde_json::json!(true) {
                return;
            }
        }
        if start.elapsed() > deadline {
            panic!("modules cache not populated within {deadline:?}");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn sample_pool(name: &str, mutable: u64, immutable: u64) -> PoolBreakdown {
    PoolBreakdown {
        pool: name.into(),
        mutable,
        immutable,
        reset: 10,
        inactive: 20,
    }
}

fn sample_snapshot() -> MetricsSnapshotResponse {
    MetricsSnapshotResponse {
        gathered_at_unix_ms: 1_700_000_000_000,
        sessions_inflight: 3,
        pools: vec![sample_pool("G2", 4, 5), sample_pool("G3", 1, 2)],
        cd: None,
    }
}

// ---- tests ------------------------------------------------------------------

/// Happy path: leader has the metrics module + handler returns a snapshot;
/// HTTP returns 200 with the unwrapped body.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_ok_returns_200_with_unwrapped_body() {
    let server = start_hub().await;
    let peer = new_velo().await;
    install_modules(&peer, vec![ModuleId::Core, ModuleId::Metrics]);
    let counter = Arc::new(AtomicUsize::new(0));
    install_canned_snapshot(&peer, ControlReply::Ok(sample_snapshot()), counter.clone());

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();
    wait_for_modules_cached(&server, id, Duration::from_secs(2)).await;

    let resp = http()
        .post(snapshot_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST snapshot");
    assert_eq!(resp.status().as_u16(), 200);
    let body: MetricsSnapshotResponse = resp.json().await.expect("snapshot json");
    assert_eq!(body.sessions_inflight, 3);
    assert_eq!(body.pools.len(), 2);
    assert_eq!(body.pools[0].pool, "G2");
    assert_eq!(body.pools[0].mutable, 4);
    assert_eq!(body.pools[1].pool, "G3");
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    server.shutdown().await.unwrap();
}

/// Module gate: a leader without `Metrics` in its module set returns 404
/// `module_not_enabled` *without* hitting the canned handler (counter stays 0).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_returns_404_when_module_absent() {
    let server = start_hub().await;
    let peer = new_velo().await;
    // No `Metrics` in the list.
    install_modules(&peer, vec![ModuleId::Core, ModuleId::Transfer]);
    let counter = Arc::new(AtomicUsize::new(0));
    install_canned_snapshot(&peer, ControlReply::Ok(sample_snapshot()), counter.clone());

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();
    wait_for_modules_cached(&server, id, Duration::from_secs(2)).await;

    let resp = http()
        .post(snapshot_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST snapshot");
    assert_eq!(resp.status().as_u16(), 404);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], serde_json::json!("module_not_enabled"));
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "gate must short-circuit before velo dispatch"
    );

    server.shutdown().await.unwrap();
}

/// Leader-side `ControlError::Internal` propagates as 500 with the error body.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_propagates_internal_error_as_500() {
    let server = start_hub().await;
    let peer = new_velo().await;
    install_modules(&peer, vec![ModuleId::Core, ModuleId::Metrics]);
    install_canned_snapshot(
        &peer,
        ControlReply::Err(ControlError::Internal("boom".into())),
        Arc::new(AtomicUsize::new(0)),
    );

    let client = build_client(&server);
    client
        .register_instance(peer.peer_info())
        .await
        .expect("register");
    let id = peer.instance_id();
    wait_for_modules_cached(&server, id, Duration::from_secs(2)).await;

    let resp = http()
        .post(snapshot_url(&server, id))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .expect("POST snapshot");
    assert_eq!(resp.status().as_u16(), 500);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["kind"], serde_json::json!("internal"));

    server.shutdown().await.unwrap();
}

/// Fanout: two leaders, one with metrics + one without. Response carries an
/// entry for the metrics leader and skips the non-metrics one. Top-level
/// status is always 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_aggregates_and_filters_by_module() {
    let server = start_hub().await;

    let peer_a = new_velo().await;
    install_modules(&peer_a, vec![ModuleId::Core, ModuleId::Metrics]);
    install_canned_snapshot(
        &peer_a,
        ControlReply::Ok(sample_snapshot()),
        Arc::new(AtomicUsize::new(0)),
    );

    let peer_b = new_velo().await;
    // No metrics module — fanout filters this leader out before dispatching.
    install_modules(&peer_b, vec![ModuleId::Core, ModuleId::Transfer]);
    let b_counter = Arc::new(AtomicUsize::new(0));
    install_canned_snapshot(
        &peer_b,
        ControlReply::Ok(sample_snapshot()),
        b_counter.clone(),
    );

    let client_a = build_client(&server);
    client_a
        .register_instance(peer_a.peer_info())
        .await
        .expect("register a");
    let client_b = build_client(&server);
    client_b
        .register_instance(peer_b.peer_info())
        .await
        .expect("register b");

    wait_for_modules_cached(&server, peer_a.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(&server, peer_b.instance_id(), Duration::from_secs(2)).await;

    let resp = http().get(fanout_url(&server)).send().await.expect("GET");
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    let instances = body["instances"]
        .as_object()
        .expect("instances object on fanout response");
    let a_key = peer_a.instance_id().to_string();
    let b_key = peer_b.instance_id().to_string();
    assert!(
        instances.contains_key(&a_key),
        "metrics-enabled leader should appear in fanout"
    );
    assert!(
        !instances.contains_key(&b_key),
        "non-metrics leader should be filtered out, got {:?}",
        instances.keys().collect::<Vec<_>>()
    );
    assert!(
        instances[&a_key]["snapshot"].is_object(),
        "metrics-enabled leader should have a snapshot, body = {body}"
    );
    assert_eq!(
        b_counter.load(Ordering::SeqCst),
        0,
        "fanout should not dispatch to leaders without the module"
    );

    server.shutdown().await.unwrap();
}

/// Fanout: a slow leader must not stall the whole response. It surfaces as
/// a per-instance entry with a `"timeout after ..."` error string while the
/// fast leader's snapshot still lands in the same response. Lower bound on
/// the total elapsed time guards against a regression that drops the per-
/// leader `tokio::time::timeout` wrapper (which would block on the slow leader
/// for ~10s under the velo default unary timeout).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_slow_leader_times_out_does_not_block_others() {
    let server = start_hub().await;

    let peer_fast = new_velo().await;
    install_modules(&peer_fast, vec![ModuleId::Core, ModuleId::Metrics]);
    install_canned_snapshot(
        &peer_fast,
        ControlReply::Ok(sample_snapshot()),
        Arc::new(AtomicUsize::new(0)),
    );

    let peer_slow = new_velo().await;
    install_modules(&peer_slow, vec![ModuleId::Core, ModuleId::Metrics]);
    // Sleep is longer than `METRICS_FANOUT_PER_LEADER` (2s) but short enough
    // not to slow the suite. The handler still returns Ok eventually — the
    // hub-side timeout fires first.
    install_slow_snapshot(&peer_slow, Duration::from_secs(5));

    let client_fast = build_client(&server);
    client_fast
        .register_instance(peer_fast.peer_info())
        .await
        .expect("register fast");
    let client_slow = build_client(&server);
    client_slow
        .register_instance(peer_slow.peer_info())
        .await
        .expect("register slow");

    wait_for_modules_cached(&server, peer_fast.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(&server, peer_slow.instance_id(), Duration::from_secs(2)).await;

    let started = Instant::now();
    let resp = http().get(fanout_url(&server)).send().await.expect("GET");
    let elapsed = started.elapsed();
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    let instances = body["instances"].as_object().expect("instances");

    let fast_key = peer_fast.instance_id().to_string();
    let slow_key = peer_slow.instance_id().to_string();
    assert!(
        instances[&fast_key]["snapshot"].is_object(),
        "fast leader's snapshot should land, got {body}"
    );
    let slow_err = instances[&slow_key]["error"]
        .as_str()
        .expect("slow leader should have an error string");
    assert!(
        slow_err.contains("timeout"),
        "slow leader's error should mention timeout, got: {slow_err}"
    );

    // Per-leader budget is 2s; the entire fanout completes shortly after
    // that. The slow handler sleeps 5s — if we dropped the timeout wrapper
    // we'd be stuck behind the velo default unary timeout (~10s). A 4s
    // ceiling proves the budget is doing its job without being flaky.
    assert!(
        elapsed < Duration::from_secs(4),
        "fanout should not block on slow leader (took {elapsed:?})"
    );

    server.shutdown().await.unwrap();
}

/// Fanout never fails the response on a single bad leader — the failing
/// leader surfaces as `{ "error": ... }` instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fanout_per_instance_error_does_not_fail_response() {
    let server = start_hub().await;

    let peer_ok = new_velo().await;
    install_modules(&peer_ok, vec![ModuleId::Core, ModuleId::Metrics]);
    install_canned_snapshot(
        &peer_ok,
        ControlReply::Ok(sample_snapshot()),
        Arc::new(AtomicUsize::new(0)),
    );

    let peer_err = new_velo().await;
    install_modules(&peer_err, vec![ModuleId::Core, ModuleId::Metrics]);
    install_canned_snapshot(
        &peer_err,
        ControlReply::Err(ControlError::Internal("simulated".into())),
        Arc::new(AtomicUsize::new(0)),
    );

    let client_ok = build_client(&server);
    client_ok
        .register_instance(peer_ok.peer_info())
        .await
        .expect("register ok");
    let client_err = build_client(&server);
    client_err
        .register_instance(peer_err.peer_info())
        .await
        .expect("register err");

    wait_for_modules_cached(&server, peer_ok.instance_id(), Duration::from_secs(2)).await;
    wait_for_modules_cached(&server, peer_err.instance_id(), Duration::from_secs(2)).await;

    let resp = http().get(fanout_url(&server)).send().await.expect("GET");
    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.expect("json");
    let instances = body["instances"].as_object().expect("instances");

    let ok_key = peer_ok.instance_id().to_string();
    let err_key = peer_err.instance_id().to_string();
    assert!(
        instances[&ok_key]["snapshot"].is_object(),
        "ok leader should expose a snapshot"
    );
    assert!(
        instances[&err_key]["error"].is_string(),
        "failing leader should expose an error string, got {body}"
    );

    server.shutdown().await.unwrap();
}
