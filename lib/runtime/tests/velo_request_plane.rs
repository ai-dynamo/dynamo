// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test: round-trip a request through the velo `RequestPlaneServer`
//! and `RequestPlaneClient` using two velo nodes that share a KV-backed
//! `PeerDiscovery`.
//!
//! Run with: `timeout 60 cargo test -p dynamo-runtime --features velo-transport --test velo_request_plane -- --nocapture`

#![cfg(feature = "velo-transport")]

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use tokio::sync::Notify;

use velo::Velo;
use velo::transports::tcp::TcpTransportBuilder;

use dynamo_runtime::SystemHealth;
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::PushWorkHandler;
use dynamo_runtime::pipeline::network::egress::unified_client::RequestPlaneClient;
use dynamo_runtime::pipeline::network::egress::velo_client::VeloRequestPlaneClient;
use dynamo_runtime::pipeline::network::ingress::unified_server::RequestPlaneServer;
use dynamo_runtime::pipeline::network::ingress::velo_endpoint::VeloRequestPlaneServer;
use dynamo_runtime::pipeline::network::velo::{
    KvPeerDiscovery, encode_velo_address,
};
use dynamo_runtime::storage::kv;

/// Test handler that records every payload and request id it receives.
#[derive(Default)]
struct RecordingHandler {
    received: Mutex<Vec<(Bytes, Option<String>)>>,
}

impl RecordingHandler {
    fn snapshot(&self) -> Vec<(Bytes, Option<String>)> {
        self.received.lock().clone()
    }
}

#[async_trait]
impl PushWorkHandler for RecordingHandler {
    async fn handle_payload(
        &self,
        payload: Bytes,
        request_id: Option<String>,
    ) -> Result<(), PipelineError> {
        self.received.lock().push((payload, request_id));
        Ok(())
    }

    fn add_metrics(
        &self,
        _endpoint: &dynamo_runtime::component::Endpoint,
        _metrics_labels: Option<&[(&str, &str)]>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn set_endpoint_health_check_notifier(&self, _notifier: Arc<Notify>) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Build a velo node and register its `PeerInfo` into the shared discovery.
/// Returns `(velo, guard)` — the guard MUST be kept alive until the test is
/// done dialing this node, otherwise the Drop impl tears down the kv entry.
async fn build_velo_node(
    disco: Arc<KvPeerDiscovery>,
) -> (
    Arc<Velo>,
    dynamo_runtime::pipeline::network::velo::KvPeerRegistrationGuard,
) {
    let bind = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 0);
    let transport = TcpTransportBuilder::new()
        .bind_addr(bind)
        .build()
        .expect("build velo TCP transport");
    let discovery_for_velo: Arc<dyn velo::discovery::PeerDiscovery> = disco.clone();
    let velo = Velo::builder()
        .add_transport(Arc::new(transport))
        .discovery(discovery_for_velo)
        .build()
        .await
        .expect("build velo");

    let guard = disco.register(velo.peer_info()).await.expect("register peer");
    (velo, guard)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_request_plane_round_trip() {
    // Single shared kv::Manager (memory) acts as the discovery surface for both nodes.
    let kv = Arc::new(kv::Manager::memory());
    let disco = Arc::new(KvPeerDiscovery::new(kv));

    // Server side: velo node A + VeloRequestPlaneServer.
    let (velo_server, _server_guard) = build_velo_node(disco.clone()).await;
    let server = VeloRequestPlaneServer::new(velo_server.clone()).expect("server");

    // Register one Dynamo endpoint with a deterministic instance id.
    let dynamo_instance_id: u64 = 0xdead_beef;
    let endpoint = "echo".to_string();
    let handler = Arc::new(RecordingHandler::default());
    let system_health = Arc::new(parking_lot::Mutex::new(SystemHealth::new(
        dynamo_runtime::config::HealthStatus::Ready,
        Vec::<String>::new(),
        false,
        "/health".to_string(),
        "/live".to_string(),
    )));

    server
        .register_endpoint(
            endpoint.clone(),
            handler.clone() as Arc<dyn PushWorkHandler>,
            dynamo_instance_id,
            "ns".to_string(),
            "comp".to_string(),
            system_health.clone(),
        )
        .await
        .expect("register endpoint");

    // Client side: velo node B + VeloRequestPlaneClient. Letting velo discover peer A
    // via the shared KvPeerDiscovery.
    let (velo_client_node, _client_guard) = build_velo_node(disco.clone()).await;
    let client = VeloRequestPlaneClient::new(velo_client_node.clone());

    let address = encode_velo_address(velo_server.instance_id(), dynamo_instance_id, &endpoint);

    let payload = Bytes::from_static(b"hello-velo");
    let mut headers = HashMap::new();
    headers.insert("x-dynamo-request-id".to_string(), "req-42".to_string());

    let ack = client
        .send_request(address, payload.clone(), headers)
        .await
        .expect("send_request");

    // Empty ACK is the contract.
    assert!(ack.is_empty(), "expected empty ACK, got {} bytes", ack.len());

    // Allow velo's spawn-by-default dispatch to land before we read the recording.
    let mut tries = 0;
    while handler.snapshot().is_empty() && tries < 50 {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        tries += 1;
    }
    let recorded = handler.snapshot();
    assert_eq!(recorded.len(), 1, "expected exactly one delivery");
    assert_eq!(recorded[0].0, payload);
    assert_eq!(recorded[0].1.as_deref(), Some("req-42"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_two_nodes_can_discover_each_other_via_kv() {
    let kv = Arc::new(kv::Manager::memory());
    let disco = Arc::new(KvPeerDiscovery::new(kv));

    let (velo_a, _guard_a) = build_velo_node(disco.clone()).await;
    let (velo_b, _guard_b) = build_velo_node(disco.clone()).await;

    // Direct discovery hit.
    let resolved = velo::discovery::PeerDiscovery::discover_by_instance_id(
        &*disco,
        velo_a.instance_id(),
    )
    .await
    .expect("disco direct");
    assert_eq!(resolved.instance_id(), velo_a.instance_id());

    // Velo's own discover_and_register_peer.
    velo_b
        .discover_and_register_peer(velo_a.instance_id())
        .await
        .expect("velo discover");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn velo_request_plane_unknown_endpoint_errors() {
    let kv = Arc::new(kv::Manager::memory());
    let disco = Arc::new(KvPeerDiscovery::new(kv));

    let (velo_server, _server_guard) = build_velo_node(disco.clone()).await;
    let _server = VeloRequestPlaneServer::new(velo_server.clone()).expect("server");

    let (velo_client, _client_guard) = build_velo_node(disco.clone()).await;
    let client = VeloRequestPlaneClient::new(velo_client);

    // No endpoint registered — server should respond with an error.
    let address = encode_velo_address(velo_server.instance_id(), 0, "missing");
    let res = client
        .send_request(address, Bytes::from_static(b"x"), HashMap::new())
        .await;
    assert!(res.is_err(), "expected error for unknown endpoint, got {res:?}");
}
