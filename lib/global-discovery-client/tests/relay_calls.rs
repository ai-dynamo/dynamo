// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use dynamo_global_discovery_client::{
    Capacity, DiscoveryClientError, GlobalDiscoveryClient, PoolKey, PoolLocation, PoolRegistration,
    ReplicaEndpoint, ReplicaId,
};
use url::Url;

#[derive(Clone)]
struct FakeService(Arc<Mutex<FakeState>>);

struct FakeState {
    routers_status: StatusCode,
    routers_body: String,
    registrations: Vec<PoolRegistration>,
}

impl FakeService {
    fn new(routers_body: &str) -> Self {
        Self(Arc::new(Mutex::new(FakeState {
            routers_status: StatusCode::OK,
            routers_body: routers_body.to_owned(),
            registrations: Vec::new(),
        })))
    }

    /// Serves under `/discovery`, so every test also checks that the client
    /// keeps the base URL's path.
    async fn client(&self) -> GlobalDiscoveryClient {
        let app = axum::Router::new()
            .route("/discovery/routers", get(serve_routers))
            .route("/discovery/pools", post(serve_register))
            .with_state(self.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base_url = Url::parse(&format!("http://{address}/discovery")).unwrap();
        GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap()
    }
}

async fn serve_routers(State(fake): State<FakeService>) -> Response {
    let state = fake.0.lock().unwrap();
    if state.routers_status != StatusCode::OK {
        return (state.routers_status, "not synced").into_response();
    }
    (
        [(header::CONTENT_TYPE, "application/json")],
        state.routers_body.clone(),
    )
        .into_response()
}

async fn serve_register(
    State(fake): State<FakeService>,
    Json(registration): Json<PoolRegistration>,
) -> StatusCode {
    fake.0.lock().unwrap().registrations.push(registration);
    StatusCode::NO_CONTENT
}

fn replica(id: &str, url: &str) -> ReplicaEndpoint {
    ReplicaEndpoint {
        replica: ReplicaId(id.into()),
        url: url.parse().unwrap(),
    }
}

#[tokio::test]
async fn lists_every_router() {
    let fake = FakeService::new(
        r#"{ "routers": [
            { "replica": "aws-ohio/global-router-0", "url": "http://10.0.12.7:50061" },
            { "replica": "aws-west/global-router-0", "url": "http://[fd00::9]:50061", "zone": "usw2-az1" }
        ] }"#,
    );
    let client = fake.client().await;

    assert_eq!(
        client.routers().await.unwrap(),
        [
            replica("aws-ohio/global-router-0", "http://10.0.12.7:50061"),
            replica("aws-west/global-router-0", "http://[fd00::9]:50061"),
        ]
    );
}

#[tokio::test]
async fn one_bad_url_fails_the_whole_list() {
    let fake = FakeService::new(
        r#"{ "routers": [
            { "replica": "aws-ohio/global-router-0", "url": "http://10.0.12.7:50061" },
            { "replica": "aws-west/global-router-0", "url": "http://10.1.40.3:99999" }
        ] }"#,
    );
    let client = fake.client().await;

    let error = client.routers().await.unwrap_err();
    assert!(
        matches!(error, DiscoveryClientError::Decode { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn reports_a_non_2xx_status() {
    let fake = FakeService::new(r#"{ "routers": [] }"#);
    fake.0.lock().unwrap().routers_status = StatusCode::SERVICE_UNAVAILABLE;
    let client = fake.client().await;

    let error = client.routers().await.unwrap_err();
    assert!(
        matches!(&error, DiscoveryClientError::Status { status, .. } if status.as_u16() == 503),
        "{error:?}"
    );
}

#[tokio::test]
async fn registers_a_pool() {
    let fake = FakeService::new(r#"{ "routers": [] }"#);
    let client = fake.client().await;
    let registration = PoolRegistration {
        key: PoolKey {
            site_id: "aws-west".into(),
            namespace: "dynamo".into(),
            dgd_name: "mocker-west-1".into(),
        },
        location: PoolLocation {
            region: "us-west-2".into(),
            availability_zone: Some("usw2-az1".into()),
            cluster: Some("aws-west".into()),
            datacenter: None,
        },
        model: "Qwen/Qwen3-0.6B".into(),
        capabilities: vec!["aggregated".into()],
        frontends: vec!["http://10.1.22.20:8000".parse().unwrap()],
        capacity: vec![Capacity {
            sku: "H100".into(),
            count: 8,
        }],
    };

    client.register_pool(&registration).await.unwrap();

    assert_eq!(fake.0.lock().unwrap().registrations, [registration]);
}
