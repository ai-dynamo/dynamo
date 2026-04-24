// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use std::sync::Arc;

use kvbm_hub::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use kvbm_hub::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggInstancesResponse, ConditionalDisaggRole, ErrorBody,
    ErrorCode, Feature, HeartbeatResponse, ListInstancesResponse, PeerLookupResponse,
    ProbeResponse, RegisterRequest, RegisterResponse, instance_by_id, instance_heartbeat,
    instance_probe, paths, peers_by_instance, peers_by_worker,
};
use kvbm_hub::{ConditionalDisaggClient, ConditionalDisaggManager, HubClientBuilder, HubServer};
use velo::discovery::PeerDiscovery;
use velo_common::{InstanceId, PeerInfo, WorkerAddress};
use velo_transports::Transport;
use velo_transports::tcp::TcpTransportBuilder;

// ---- fixtures ---------------------------------------------------------------

async fn start_server() -> HubServer {
    kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .serve()
        .await
        .expect("start test server")
}

fn make_peer() -> PeerInfo {
    let id = InstanceId::new_v4();
    PeerInfo::new(id, WorkerAddress::from_encoded(b"test-addr".to_vec()))
}

fn build_client(server: &HubServer) -> std::sync::Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host("127.0.0.1")
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build client")
}

fn discovery_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.discovery_addr(), path)
}

fn control_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.control_addr(), path)
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

// ---- handlers module --------------------------------------------------------

#[test]
fn heartbeat_handler_name_is_stable() {
    assert_eq!(HEARTBEAT_HANDLER, "kvbm_hub_heartbeat");
}

#[test]
fn heartbeat_request_default() {
    let req = HeartbeatRequest::default();
    assert_eq!(req.seq, 0);
}

#[test]
fn heartbeat_ack_default() {
    let ack = HeartbeatAck::default();
    assert_eq!(ack.seq, 0);
    assert!(!ack.ok);
}

#[test]
fn heartbeat_request_serde_round_trip() {
    let orig = HeartbeatRequest { seq: 99 };
    let json = serde_json::to_string(&orig).unwrap();
    let back: HeartbeatRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seq, 99);
}

#[test]
fn heartbeat_ack_serde_round_trip() {
    let orig = HeartbeatAck { seq: 7, ok: true };
    let json = serde_json::to_string(&orig).unwrap();
    let back: HeartbeatAck = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seq, 7);
    assert!(back.ok);
}

#[tokio::test]
async fn create_heartbeat_handler_builds() {
    let server = start_server().await;
    let client = build_client(&server);
    let _handler = kvbm_hub::handlers::create_heartbeat_handler(client);
}

// ---- HTTP-direct server tests -----------------------------------------------

#[tokio::test]
async fn health_check_discovery_port() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(&server, paths::HEALTH))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn health_check_control_port() {
    let server = start_server().await;
    let resp = http()
        .get(control_url(&server, paths::HEALTH))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn register_success() {
    let server = start_server().await;
    let peer = make_peer();
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: RegisterResponse = resp.json().await.unwrap();
    assert_eq!(body.instance_id, peer.instance_id());
}

#[tokio::test]
async fn reregister_same_instance_is_idempotent() {
    let server = start_server().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: Vec::new(),
    };
    let post = || {
        http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&req)
            .send()
    };
    assert_eq!(post().await.unwrap().status(), 200);
    assert_eq!(post().await.unwrap().status(), 200);
}

#[tokio::test]
async fn unregister_success() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
}

#[tokio::test]
async fn unregister_not_found() {
    let server = start_server().await;
    let resp = http()
        .delete(control_url(&server, &instance_by_id(InstanceId::new_v4())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn heartbeat_registered_instance() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .post(control_url(
            &server,
            &instance_heartbeat(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: HeartbeatResponse = resp.json().await.unwrap();
    assert!(body.acknowledged);
}

#[tokio::test]
async fn heartbeat_unregistered_instance() {
    let server = start_server().await;
    let resp = http()
        .post(control_url(
            &server,
            &instance_heartbeat(InstanceId::new_v4()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: HeartbeatResponse = resp.json().await.unwrap();
    assert!(!body.acknowledged);
}

#[tokio::test]
async fn get_peer_by_instance_found() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: PeerLookupResponse = resp.json().await.unwrap();
    assert_eq!(body.peer_info.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn get_peer_by_instance_not_found() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(InstanceId::new_v4()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn get_peer_by_worker_found() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(discovery_url(&server, &peers_by_worker(peer.worker_id())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: PeerLookupResponse = resp.json().await.unwrap();
    assert_eq!(body.peer_info.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn get_peer_by_worker_not_found() {
    let server = start_server().await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_worker(InstanceId::new_v4().worker_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::NotFound);
}

#[tokio::test]
async fn control_port_mirrors_discovery_endpoints() {
    let server = start_server().await;
    let peer = make_peer();
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&RegisterRequest {
            peer_info: peer.clone(),
            features: Vec::new(),
        })
        .send()
        .await
        .unwrap();
    let resp = http()
        .get(control_url(&server, &peers_by_instance(peer.instance_id())))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn peers_snapshot_tracks_registrations() {
    let server = start_server().await;
    for _ in 0..3 {
        http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&RegisterRequest {
                peer_info: make_peer(),
                features: Vec::new(),
            })
            .send()
            .await
            .unwrap();
    }
    assert_eq!(server.state().peers().len(), 3);
}

#[tokio::test]
async fn peers_snapshot_tracks_unregistrations() {
    let server = start_server().await;
    let a = make_peer();
    let b = make_peer();
    for peer in [&a, &b] {
        http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&RegisterRequest {
                peer_info: peer.clone(),
                features: Vec::new(),
            })
            .send()
            .await
            .unwrap();
    }
    http()
        .delete(control_url(&server, &instance_by_id(a.instance_id())))
        .send()
        .await
        .unwrap();
    let peers = server.state().peers();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].instance_id(), b.instance_id());
}

// ---- HubClient tests --------------------------------------------------------

#[tokio::test]
async fn client_builder_requires_host() {
    assert!(HubClientBuilder::new().build().is_err());
}

#[tokio::test]
async fn client_starts_unregistered() {
    let server = start_server().await;
    let client = build_client(&server);
    assert!(!client.is_registered());
}

#[tokio::test]
async fn client_register_sets_is_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    assert!(client.is_registered());
}

#[tokio::test]
async fn client_register_twice_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    assert!(client.register_instance(make_peer()).await.is_err());
}

#[tokio::test]
async fn client_heartbeat_while_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.register_instance(make_peer()).await.unwrap();
    client.send_heartbeat().await.unwrap();
}

#[tokio::test]
async fn client_heartbeat_before_register_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    assert!(client.send_heartbeat().await.is_err());
}

#[tokio::test]
async fn client_discover_by_instance_id() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    let found = client
        .discover_by_instance_id(peer.instance_id())
        .await
        .unwrap();
    assert_eq!(found.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn client_discover_by_worker_id() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    let found = client
        .discover_by_worker_id(peer.worker_id())
        .await
        .unwrap();
    assert_eq!(found.instance_id(), peer.instance_id());
}

#[tokio::test]
async fn client_unregister_removes_from_server() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    client.unregister().await.unwrap();
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn client_unregister_noop_when_not_registered() {
    let server = start_server().await;
    let client = build_client(&server);
    client.unregister().await.unwrap();
}

#[tokio::test]
async fn client_discover_after_unregister_errors() {
    let server = start_server().await;
    let client = build_client(&server);
    let peer = make_peer();
    client.register_instance(peer.clone()).await.unwrap();
    client.unregister().await.unwrap();
    assert!(
        client
            .discover_by_instance_id(peer.instance_id())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn registration_guard_drop_fires_delete() {
    let server = start_server().await;
    let peer = make_peer();
    {
        let client = build_client(&server);
        client.register_instance(peer.clone()).await.unwrap();
        // Arc<HubClient> drops here → HubRegistrationGuard::drop spawns DELETE
    }
    // Allow the background DELETE task to complete
    tokio::time::sleep(Duration::from_millis(100)).await;
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn registered_client_visible_in_list_then_removed_on_drop() {
    let server = start_server().await;
    let http = http();

    // List is initially empty
    let resp: ListInstancesResponse = http
        .get(control_url(&server, paths::INSTANCES))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(resp.instances.is_empty());

    // Register a client
    let peer = make_peer();
    {
        let client = build_client(&server);
        client.register_instance(peer.clone()).await.unwrap();

        // Instance is now visible via the view API
        let resp: ListInstancesResponse = http
            .get(control_url(&server, paths::INSTANCES))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(resp.instances.len(), 1);
        assert_eq!(resp.instances[0].instance_id(), peer.instance_id());
        // Arc<HubClient> drops here → HubRegistrationGuard::drop spawns DELETE
    }

    // Allow the background DELETE task to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Instance is gone
    let resp: ListInstancesResponse = http
        .get(control_url(&server, paths::INSTANCES))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(resp.instances.is_empty());
}

// ---- Velo probe tests -------------------------------------------------------

fn new_velo_transport() -> Arc<velo_transports::tcp::TcpTransport> {
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

async fn start_server_with_transport() -> (HubServer, Arc<velo_transports::tcp::TcpTransport>) {
    let transport = new_velo_transport();
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .serve()
        .await
        .expect("start test server");
    (server, transport)
}

async fn wire_mutual_velo(
    server: &HubServer,
    client_velo: &Arc<velo::Velo>,
) -> Arc<kvbm_hub::HubClient> {
    let hub_client = build_client(server);
    hub_client.register_handlers(client_velo).unwrap();
    let hub_id = hub_client
        .register_instance(client_velo.peer_info())
        .await
        .unwrap()
        .expect("hub should return its own instance id when running with a transport");
    let hub_peer = hub_client.discover_by_instance_id(hub_id).await.unwrap();
    client_velo.register_peer(hub_peer).unwrap();
    hub_client
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_probe_happy_path() {
    let (server, _hub_transport) = start_server_with_transport().await;
    let client_velo = new_velo().await;
    let _hub_client = wire_mutual_velo(&server, &client_velo).await;

    tokio::time::sleep(Duration::from_millis(200)).await;

    let instance_id = client_velo.instance_id();
    let resp = http()
        .post(control_url(&server, &instance_probe(instance_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: ProbeResponse = resp.json().await.unwrap();
    assert!(body.ok);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_probe_after_client_velo_shutdown_returns_bad_gateway() {
    let (server, _hub_transport) = start_server_with_transport().await;

    // Build client velo with an explicit transport handle so we can shut it down.
    let client_transport = new_velo_transport();
    let client_velo = velo::Velo::builder()
        .add_transport(Arc::clone(&client_transport) as Arc<dyn Transport>)
        .build()
        .await
        .unwrap();

    let hub_client = wire_mutual_velo(&server, &client_velo).await;
    let instance_id = client_velo.instance_id();

    // Keep hub_client alive so the HTTP registration guard never fires and the
    // instance stays in the registry.
    let _keep_hub_client = Arc::clone(&hub_client);

    // Explicitly shut down the transport — cancels the TCP listener and all
    // connection tasks, making the client unreachable.
    client_transport.shutdown();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let resp = http()
        .post(control_url(&server, &instance_probe(instance_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
}

// ---- Hub self-registration + hub_instance_id tests --------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn register_response_includes_hub_instance_id_when_velo_configured() {
    let (server, _hub_transport) = start_server_with_transport().await;
    let client_velo = new_velo().await;
    let client = build_client(&server);
    let hub_id = client
        .register_instance(client_velo.peer_info())
        .await
        .unwrap();
    assert!(hub_id.is_some(), "hub_instance_id should be Some");
    // Hub is discoverable immediately.
    let _hub_peer = client
        .discover_by_instance_id(hub_id.unwrap())
        .await
        .unwrap();
}

#[tokio::test]
async fn register_response_hub_instance_id_none_without_transport() {
    let server = start_server().await;
    let client = build_client(&server);
    let hub_id = client.register_instance(make_peer()).await.unwrap();
    assert!(hub_id.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_self_registered_in_registry() {
    let (server, hub_transport) = start_server_with_transport().await;
    let hub_velo_id = server.state().velo().expect("hub velo").instance_id();

    // Direct HTTP lookup should resolve the hub's PeerInfo.
    let resp = http()
        .get(discovery_url(&server, &peers_by_instance(hub_velo_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = hub_transport; // keep alive
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hub_self_entry_survives_reaper() {
    // Short TTL; protect() must keep the hub's entry alive.
    let transport = new_velo_transport();
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registration_ttl(Duration::from_millis(50))
        .prune_interval(Duration::from_millis(20))
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .serve()
        .await
        .expect("start test server");

    let hub_velo_id = server.state().velo().expect("hub velo").instance_id();

    // Wait well past TTL + multiple prune cycles.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let resp = http()
        .get(discovery_url(&server, &peers_by_instance(hub_velo_id)))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        200,
        "hub's self-entry should be protected from reaper"
    );
}

// ---- ConditionalDisagg feature tests ---------------------------------------

async fn start_server_with_cd() -> (
    HubServer,
    Arc<velo_transports::tcp::TcpTransport>,
    Arc<ConditionalDisaggManager>,
) {
    let transport = new_velo_transport();
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(Arc::clone(&transport) as Arc<dyn Transport>)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD");
    (server, transport, cd_manager)
}

/// CD-enabled hub without any velo transport. Use this when a test only
/// exercises HTTP registration dispatch and does not need velo peer
/// registration on the hub side — the hub's `velo.register_peer` call
/// would otherwise reject the opaque addresses produced by `make_peer()`.
async fn start_server_with_cd_no_velo() -> (HubServer, Arc<ConditionalDisaggManager>) {
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD");
    (server, cd_manager)
}

#[tokio::test]
async fn feature_register_without_manager_rejects() {
    let server = start_server().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Prefill,
        }))],
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: ErrorBody = resp.json().await.unwrap();
    assert_eq!(body.code, ErrorCode::BadRequest);

    // Base entry should have been rolled back.
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn feature_cd_register_missing_config_rejects() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::ConditionalDisagg(None)],
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);

    // Base entry rolled back.
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn feature_cd_list_empty_on_both_ports() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    for url in [
        discovery_url(&server, paths::CD_INSTANCES),
        control_url(&server, paths::CD_INSTANCES),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(url).send().await.unwrap().json().await.unwrap();
        assert!(body.prefill.is_empty());
        assert!(body.decode.is_empty());
    }
}

#[tokio::test]
async fn register_without_features_field_still_works() {
    // Proves `#[serde(default)]` on RegisterRequest.features is honored for
    // older clients that omit the field entirely.
    let server = start_server().await;
    let peer = make_peer();
    let legacy_json = format!(
        r#"{{"peer_info":{}}}"#,
        serde_json::to_string(&peer).unwrap()
    );
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .header("content-type", "application/json")
        .body(legacy_json)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn feature_cd_role_conflict_on_reregister() {
    let (server, _cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();

    let post = |role: ConditionalDisaggRole| {
        let req = RegisterRequest {
            peer_info: peer.clone(),
            features: vec![Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
                role,
            }))],
        };
        http()
            .post(control_url(&server, paths::INSTANCES))
            .json(&req)
            .send()
    };

    assert_eq!(
        post(ConditionalDisaggRole::Prefill).await.unwrap().status(),
        200
    );
    let resp = post(ConditionalDisaggRole::Decode).await.unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn feature_cd_unregister_removes_from_lists() {
    let (server, cd) = start_server_with_cd_no_velo().await;
    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Prefill,
        }))],
    };
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(cd.snapshot().prefill.len(), 1);

    http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .send()
        .await
        .unwrap();

    assert!(cd.snapshot().prefill.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_cd_reaper_evicts_from_lists() {
    // No transport — the reaper runs off the in-memory registry ticker and
    // the eviction callback fires into the CD manager regardless of velo.
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .registration_ttl(Duration::from_millis(60))
        .prune_interval(Duration::from_millis(20))
        .add_feature_manager(Arc::clone(&cd_manager) as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server");

    let peer = make_peer();
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::ConditionalDisagg(Some(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Prefill,
        }))],
    };
    http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(cd_manager.snapshot().prefill.len(), 1);

    // Wait well past TTL + multiple prune cycles so the reaper evicts the
    // base entry and the eviction callback fans out to the CD manager.
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(
        cd_manager.snapshot().prefill.is_empty(),
        "reaper eviction should fan out to the feature manager"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn feature_cd_prefill_and_decode_register_and_list() {
    let (server, _hub_transport, _cd) = start_server_with_cd().await;

    // Build two velo participants with their own transports + hub clients.
    let p_velo = new_velo().await;
    let d_velo = new_velo().await;

    let p_hub = build_client(&server);
    let d_hub = build_client(&server);
    p_hub.register_handlers(&p_velo).unwrap();
    d_hub.register_handlers(&d_velo).unwrap();

    let p_cd = ConditionalDisaggClient::new(Arc::clone(&p_hub), ConditionalDisaggRole::Prefill);
    let d_cd = ConditionalDisaggClient::new(Arc::clone(&d_hub), ConditionalDisaggRole::Decode);

    let p_hub_id = p_cd
        .register(p_velo.peer_info())
        .await
        .unwrap()
        .expect("hub velo id");
    let d_hub_id = d_cd
        .register(d_velo.peer_info())
        .await
        .unwrap()
        .expect("hub velo id");
    assert_eq!(p_hub_id, d_hub_id, "both clients should see the same hub");

    // Give the server a moment to fully settle (listeners + velo peer table).
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Both endpoints must report the same role split, with the correct ids.
    let p_id = p_velo.instance_id();
    let d_id = d_velo.instance_id();
    for url in [
        discovery_url(&server, paths::CD_INSTANCES),
        control_url(&server, paths::CD_INSTANCES),
    ] {
        let body: ConditionalDisaggInstancesResponse =
            http().get(&url).send().await.unwrap().json().await.unwrap();
        assert_eq!(body.prefill, vec![p_id], "from {url}");
        assert_eq!(body.decode, vec![d_id], "from {url}");
    }

    // Wire the hub's PeerInfo into each participant's velo (needed so velo
    // probes initiated from the hub can route back through the TCP transport).
    let hub_peer = p_hub.discover_by_instance_id(p_hub_id).await.unwrap();
    p_velo.register_peer(hub_peer.clone()).unwrap();
    d_velo.register_peer(hub_peer).unwrap();

    // Prefill side: await the decode peer and register it into prefill's velo.
    let d_peer = p_cd
        .await_peer_of_role(
            ConditionalDisaggRole::Decode,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(d_peer.instance_id(), d_id);
    p_velo.register_peer(d_peer).unwrap();

    // Decode side: symmetric handshake.
    let p_peer = d_cd
        .await_peer_of_role(
            ConditionalDisaggRole::Prefill,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(p_peer.instance_id(), p_id);
    d_velo.register_peer(p_peer).unwrap();

    // Hub can now probe both instances — proves velo handshakes succeeded.
    tokio::time::sleep(Duration::from_millis(200)).await;
    for id in [p_id, d_id] {
        let resp = http()
            .post(control_url(&server, &instance_probe(id)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "probe failed for {id}");
        let body: ProbeResponse = resp.json().await.unwrap();
        assert!(body.ok, "probe returned not-ok for {id}");
    }
}
