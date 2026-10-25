// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use axum::http::header;
use axum::routing::get;
use dynamo_global_discovery_client::{
    Capacity, Completeness, DiscoveryClientError, GlobalDiscoveryClient, PoolKey, PoolListing,
    PoolLocation,
};
use tokio::net::TcpListener;
use url::Url;

/// A client for a service that answers `GET /discovery/pools` with `body`.
async fn client(body: &'static str) -> GlobalDiscoveryClient {
    let app = axum::Router::new().route(
        "/discovery/pools",
        get(move || async move { ([(header::CONTENT_TYPE, "application/json")], body) }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let base_url = Url::parse(&format!("http://{address}/discovery")).unwrap();
    GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap()
}

#[tokio::test]
async fn lists_every_pool() {
    // The second pool is what the Kubernetes stores report: a name and a
    // Relay, but no model or frontends.
    let client = client(
        r#"{ "complete": true, "pools": [
            {
                "key": { "site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1" },
                "location": { "region": "us-west-2", "availability_zone": null, "cluster": "aws-west", "datacenter": null },
                "model": "Qwen/Qwen3-0.6B",
                "capabilities": ["aggregated"],
                "frontends": ["http://10.1.22.20:8000/"],
                "capacity": [{ "sku": "H100", "count": 8 }],
                "relay_identity": null,
                "added_later": "ignored"
            },
            {
                "key": { "site_id": "aws-ohio", "namespace": "dynamo", "dgd_name": "mocker-ohio-1" },
                "location": { "region": "us-east-2", "availability_zone": null, "cluster": null, "datacenter": null },
                "model": null,
                "relay_identity": "10.0.3.4"
            }
        ] }"#,
    )
    .await;

    let snapshot = client.pools().await.unwrap();
    assert_eq!(snapshot.completeness, Completeness::Exact);
    assert_eq!(
        snapshot.items,
        [
            PoolListing {
                key: PoolKey {
                    site_id: "aws-west".into(),
                    namespace: "dynamo".into(),
                    dgd_name: "mocker-west-1".into(),
                },
                location: PoolLocation {
                    region: "us-west-2".into(),
                    availability_zone: None,
                    cluster: Some("aws-west".into()),
                    datacenter: None,
                },
                model: Some("Qwen/Qwen3-0.6B".into()),
                capabilities: vec!["aggregated".into()],
                frontends: vec!["http://10.1.22.20:8000/".parse().unwrap()],
                capacity: vec![Capacity {
                    sku: "H100".into(),
                    count: 8,
                }],
                relay_identity: None,
            },
            PoolListing {
                key: PoolKey {
                    site_id: "aws-ohio".into(),
                    namespace: "dynamo".into(),
                    dgd_name: "mocker-ohio-1".into(),
                },
                location: PoolLocation {
                    region: "us-east-2".into(),
                    availability_zone: None,
                    cluster: None,
                    datacenter: None,
                },
                model: None,
                capabilities: Vec::new(),
                frontends: Vec::new(),
                capacity: Vec::new(),
                relay_identity: Some("10.0.3.4".into()),
            },
        ]
    );
}

#[tokio::test]
async fn a_partial_list_is_estimated() {
    let client = client(r#"{ "complete": false, "pools": [] }"#).await;

    let snapshot = client.pools().await.unwrap();
    assert_eq!(snapshot.completeness, Completeness::Estimated);
    assert!(snapshot.items.is_empty());
}

#[tokio::test]
async fn one_bad_frontend_fails_the_whole_list() {
    let client = client(
        r#"{ "complete": true, "pools": [ {
            "key": { "site_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1" },
            "location": { "region": "us-west-2", "availability_zone": null, "cluster": null, "datacenter": null },
            "model": "Qwen/Qwen3-0.6B",
            "frontends": ["http://10.1.22.20:99999/"]
        } ] }"#,
    )
    .await;

    let error = client.pools().await.unwrap_err();
    assert!(
        matches!(error, DiscoveryClientError::Decode { .. }),
        "{error:?}"
    );
}
