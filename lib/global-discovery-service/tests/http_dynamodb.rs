// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The HTTP service over the DynamoDB store: a pool registered through one
//! server is listed by a second server, which has its own store over the same
//! table. Skips unless `GDS_TEST_DYNAMODB_ENDPOINT` is set.

mod common;
#[macro_use]
#[path = "common/dynamodb.rs"]
mod dynamodb_support;

use std::sync::Arc;

use dynamo_global_discovery_service::backends::memory::StaticReplicaDirectory;
use dynamo_global_discovery_service::pool_discovery::{
    Lease, PoolDirectory, PoolRegistrar, RouterReplica, RouterReplicaId,
};
use dynamo_global_discovery_service::{DynamoDbPoolStore, PoolDiscoveryBackend, serve};
use dynamodb_support::Fixture;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;

fn router_replica() -> RouterReplica {
    RouterReplica {
        replica: RouterReplicaId("aws-ohio/global-router-0".to_owned()),
        url: Url::parse("http://10.42.0.5:50051").unwrap(),
    }
}

/// What `dynamo-global-discovery-client` sends.
fn registration() -> Value {
    json!({
        "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
        "location": {
            "region": "us-west-2",
            "availability_zone": null,
            "datacenter": null
        },
        "model": "Qwen/Qwen3-0.6B",
        "capabilities": ["aggregated"],
        "frontends": ["http://10.43.1.2:8000/"],
        "capacity": [{"sku": "H100", "workers": 8}]
    })
}

/// Serves one replica of the service over `store`: its pools and its
/// registrations, with static routers.
async fn start(store: Arc<DynamoDbPoolStore>) -> (String, CancellationToken) {
    let pools: Arc<PoolDirectory> = store.clone();
    let registrar: Arc<dyn PoolRegistrar> = store;
    let backend = PoolDiscoveryBackend {
        pools,
        routers: Arc::new(StaticReplicaDirectory::new(vec![router_replica()])),
        registrar: Some(registrar),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let shutdown = CancellationToken::new();
    tokio::spawn(serve(listener, backend, None, shutdown.clone()));
    (base, shutdown)
}

async fn get_json(base: &str, path: &str) -> (reqwest::StatusCode, Value) {
    let response = reqwest::get(format!("{base}{path}")).await.unwrap();
    let status = response.status();
    let body = response.bytes().await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

async fn post_pool(base: &str, body: &Value) -> (reqwest::StatusCode, Value) {
    let response = reqwest::Client::new()
        .post(format!("{base}/pools"))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body = response.bytes().await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

#[tokio::test]
async fn a_pool_registered_on_one_replica_is_listed_by_another() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let (a, shutdown_a) = start(fixture.store()).await;
    let (b, shutdown_b) = start(fixture.store()).await;

    // Both replicas start exact and empty: the table is the shared state.
    let (status, pools) = get_json(&b, "/pools").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(pools, json!({"pools": [], "complete": true}));

    let (status, granted) = post_pool(&a, &registration()).await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(
        granted,
        json!({"heartbeat_interval_s": 5, "lease_duration_s": 15})
    );

    let expected = json!({
        "pools": [{
            "key": {"cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1"},
            "location": {
                "region": "us-west-2",
                "availability_zone": null,
                "datacenter": null
            },
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.43.1.2:8000/"],
            "capacity": [{"sku": "H100", "workers": 8}],
            "relay_identities": []
        }],
        "complete": true
    });
    for base in [&b, &a] {
        let (status, pools) = get_json(base, "/pools").await;
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(pools, expected);
    }
    let (status, _) = get_json(&b, "/healthz").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    let (_, routers) = get_json(&b, "/routers").await;
    assert_eq!(
        routers,
        json!({"routers": [{"replica": "aws-ohio/global-router-0", "url": "http://10.42.0.5:50051/"}]})
    );

    // Past the lease with no heartbeat, both replicas drop it.
    fixture.advance(
        Lease::new(5, 15).unwrap().duration()
            + dynamodb_support::SKEW
            + std::time::Duration::from_secs(1),
    );
    for base in [&a, &b] {
        let (_, pools) = get_json(base, "/pools").await;
        assert_eq!(pools, json!({"pools": [], "complete": true}));
    }
    shutdown_a.cancel();
    shutdown_b.cancel();
}
