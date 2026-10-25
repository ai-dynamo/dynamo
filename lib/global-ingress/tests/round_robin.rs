// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Frontend reads pools from the real Global Discovery Service and
//! sends requests to them in turn.

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::HeaderMap;
use common::{MODEL, Server, chat, chat_chunk, post, scripted_pool_frontend};
use dynamo_global_discovery_client::{
    GlobalDiscoveryClient, PoolKey, PoolLocation, PoolRegistration,
};
use dynamo_global_discovery_service::backends::kubernetes::TrustedNetworkAuthenticator;
use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::pool_discovery::LeasePolicy;
use dynamo_global_discovery_service::service::{self, DiscoveryStore};
use dynamo_global_router::{DiscoveryPoolCatalog, PoolCatalog};
use reqwest::StatusCode;
use tokio_util::sync::CancellationToken;

/// A Global Discovery Service with an in-memory store.
async fn discovery_service() -> (Server, GlobalDiscoveryClient) {
    let lease = LeasePolicy {
        heartbeat_interval: Duration::from_secs(5),
        lease_duration: Duration::from_secs(15),
    };
    let pools = Arc::new(MemoryPoolStore::new(lease).unwrap());
    let store = DiscoveryStore {
        pools: pools.clone(),
        routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
        registrar: Some(pools),
        authenticator: Arc::new(TrustedNetworkAuthenticator),
        sharing: None,
    };
    let (listener, url) = common::bind().await;
    let cancel = CancellationToken::new();
    tokio::spawn(service::serve(listener, store, cancel.clone()));
    let client = GlobalDiscoveryClient::new(url.clone(), Duration::from_secs(2)).unwrap();
    (Server { url, cancel }, client)
}

/// A pool frontend registered with the discovery service.
struct RegisteredPool {
    _server: Server,
    requests: Arc<Mutex<Vec<HeaderMap>>>,
}

async fn register_pool(discovery: &GlobalDiscoveryClient, dgd: &str) -> RegisteredPool {
    let sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk(dgd));
    let (server, requests) = scripted_pool_frontend(sse.leak()).await;
    discovery
        .register_pool(&PoolRegistration {
            key: PoolKey {
                site_id: "aws-ohio".into(),
                namespace: "dynamo".into(),
                dgd_name: dgd.into(),
            },
            location: PoolLocation {
                region: "us-east-2".into(),
                availability_zone: None,
                cluster: Some("aws-ohio".into()),
                datacenter: None,
            },
            model: MODEL.into(),
            capabilities: vec!["aggregated".into()],
            frontends: vec![server.url.clone()],
            capacity: Vec::new(),
        })
        .await
        .unwrap();
    RegisteredPool {
        _server: server,
        requests,
    }
}

/// Sends `count` chat requests through the Global Frontend.
async fn send_chats(global: &Server, count: usize) {
    for _ in 0..count {
        let response = post(&global.url, "v1/chat/completions", &chat(MODEL, "hi", true)).await;
        assert_eq!(response.status(), StatusCode::OK);
        response.text().await.unwrap();
    }
}

/// How many requests each pool has received.
fn request_counts(pools: &[(&str, &RegisteredPool)]) -> BTreeMap<String, usize> {
    pools
        .iter()
        .map(|(name, pool)| (name.to_string(), pool.requests.lock().unwrap().len()))
        .collect()
}

#[tokio::test]
async fn rotates_across_the_pools_discovery_lists() {
    let (_discovery_server, discovery) = discovery_service().await;
    let blue = register_pool(&discovery, "blue").await;
    let green = register_pool(&discovery, "green").await;

    let cancel = CancellationToken::new();
    let catalog =
        DiscoveryPoolCatalog::start(discovery.clone(), Duration::from_millis(50), cancel.clone())
            .await;
    let global = common::global_frontend(catalog.subscribe()).await;

    send_chats(&global, 4).await;
    assert_eq!(
        request_counts(&[("blue", &blue), ("green", &green)]),
        BTreeMap::from([("blue".into(), 2), ("green".into(), 2)])
    );

    // A pool registered later joins the rotation without a restart.
    let red = register_pool(&discovery, "red").await;
    let mut pools = catalog.subscribe();
    tokio::time::timeout(
        Duration::from_secs(5),
        pools.wait_for(|snapshot| snapshot.pools().len() == 3),
    )
    .await
    .expect("the catalog did not pick up the new pool")
    .unwrap();

    send_chats(&global, 3).await;
    assert_eq!(
        request_counts(&[("blue", &blue), ("green", &green), ("red", &red)]),
        BTreeMap::from([("blue".into(), 3), ("green".into(), 3), ("red".into(), 1)])
    );
    cancel.cancel();
}
