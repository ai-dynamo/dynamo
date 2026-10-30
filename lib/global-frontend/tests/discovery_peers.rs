// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Global Frontend finds its replica-sync peers through Global Discovery:
//! every replica `/routers` lists but itself, following the list as it
//! changes, and never dropping a peer given statically.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::routing::get;
use dynamo_global_discovery_client::GlobalDiscoveryClient;
use dynamo_global_frontend::poll_replica_peers;
use dynamo_global_router::{CorePolicy, CoreSelector, GrpcReplicaSync, ReplicaTransport};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

const POLL: Duration = Duration::from_millis(50);

/// A fake Global Discovery whose `/routers` the test rewrites at will.
#[derive(Clone)]
struct FakeDiscovery(Arc<Mutex<Vec<(String, String)>>>);

impl FakeDiscovery {
    async fn serve(replicas: &[(&str, &str)]) -> (Self, GlobalDiscoveryClient) {
        let fake = Self(Arc::new(Mutex::new(Vec::new())));
        fake.list(replicas);
        let app = axum::Router::new()
            .route("/routers", get(serve_routers))
            .with_state(fake.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = Url::parse(&format!("http://{address}/")).unwrap();
        let client = GlobalDiscoveryClient::new(base, Duration::from_secs(2)).unwrap();
        (fake, client)
    }

    fn list(&self, replicas: &[(&str, &str)]) {
        *self.0.lock().unwrap() = replicas
            .iter()
            .map(|(id, url)| (id.to_string(), url.to_string()))
            .collect();
    }
}

async fn serve_routers(State(fake): State<FakeDiscovery>) -> Json<Value> {
    let routers: Vec<Value> = fake
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|(id, url)| json!({ "replica": id, "url": url }))
        .collect();
    Json(json!({ "routers": routers }))
}

async fn assert_peers(core: &CoreSelector, expected: &[&str]) {
    let expected: HashSet<Url> = expected.iter().map(|url| url.parse().unwrap()).collect();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let peers: HashSet<Url> = core.replica_peers().unwrap().into_iter().collect();
            if peers == expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "the peers never became {expected:?}; last seen {:?}",
            core.replica_peers()
        )
    });
}

const SELF: &str = "http://10.0.0.1:8000";
const OTHER: &str = "http://10.0.0.2:8000";
const THIRD: &str = "http://10.0.0.3:8000";
const STATIC: &str = "http://10.0.0.9:8000";

async fn core_following(
    client: GlobalDiscoveryClient,
    static_peers: &[&str],
    cancel: &CancellationToken,
) -> Arc<CoreSelector> {
    let core = Arc::new(
        CoreSelector::with_transport(
            CorePolicy::RoundRobin,
            Some(ReplicaTransport::Grpc(GrpcReplicaSync {
                peers: static_peers
                    .iter()
                    .map(|url| url.parse().unwrap())
                    .collect(),
            })),
        )
        .await
        .unwrap(),
    );
    // The stream is served before any peer is known.
    assert!(core.replica_sync_routes().is_some());
    let (listed, peers) = watch::channel(Vec::new());
    tokio::spawn(poll_replica_peers(
        client,
        Some(SELF.parse().unwrap()),
        POLL,
        listed,
        cancel.child_token(),
    ));
    tokio::spawn(core.follow_peers(peers, cancel.child_token()).unwrap());
    core
}

#[tokio::test]
async fn peers_come_from_discovery_and_follow_its_listing() {
    let cancel = CancellationToken::new();
    let (fake, client) = FakeDiscovery::serve(&[("gf-a", SELF), ("gf-b", OTHER)]).await;
    let core = core_following(client, &[], &cancel).await;

    assert_peers(&core, &[OTHER]).await;

    fake.list(&[("gf-a", SELF), ("gf-c", THIRD)]);
    assert_peers(&core, &[THIRD]).await;

    fake.list(&[("gf-a", SELF)]);
    assert_peers(&core, &[]).await;
    cancel.cancel();
}

#[tokio::test]
async fn static_peers_survive_discovery() {
    let cancel = CancellationToken::new();
    let (fake, client) = FakeDiscovery::serve(&[("gf-a", SELF), ("gf-b", OTHER)]).await;
    let core = core_following(client, &[STATIC], &cancel).await;

    assert_peers(&core, &[STATIC, OTHER]).await;
    fake.list(&[("gf-a", SELF)]);
    assert_peers(&core, &[STATIC]).await;
    cancel.cancel();
}
