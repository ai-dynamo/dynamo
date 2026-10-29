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
    Capacity, DiscoveryClientError, GlobalDiscoveryClient, Lease, PoolKey, PoolLocation,
    PoolRegistration, RouterReplica, RouterReplicaId,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use url::Url;

#[derive(Clone)]
struct FakeService(Arc<Mutex<FakeState>>);

struct FakeState {
    routers_status: StatusCode,
    routers_body: String,
    /// Registrations as the JSON the service received.
    registrations: Vec<Value>,
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
        let address = serve(app).await;
        let base_url = Url::parse(&format!("http://{address}/discovery")).unwrap();
        GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap()
    }
}

async fn serve(app: axum::Router) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    address
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
    Json(registration): Json<Value>,
) -> Json<Value> {
    fake.0.lock().unwrap().registrations.push(registration);
    Json(json!({ "heartbeat_interval_s": 5, "lease_duration_s": 15 }))
}

fn replica(id: &str, url: &str) -> RouterReplica {
    RouterReplica {
        replica: RouterReplicaId(id.into()),
        url: url.parse().unwrap(),
    }
}

fn registration() -> PoolRegistration {
    PoolRegistration {
        key: PoolKey::new("aws-west", "dynamo", "mocker-west-1").unwrap(),
        location: PoolLocation {
            region: "us-west-2".into(),
            availability_zone: Some("usw2-az1".into()),
            datacenter: None,
        },
        model: "Qwen/Qwen3-0.6B".into(),
        capabilities: vec!["aggregated".into()],
        frontends: vec!["http://10.1.22.20:8000".parse().unwrap()],
        capacity: vec![Capacity {
            sku: "H100".into(),
            workers: 8,
        }],
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
        matches!(&error, DiscoveryClientError::Status { status, body, .. } if status.as_u16() == 503 && body == "not synced"),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_cut_off_body_is_a_request_error() {
    // The body ends before its Content-Length. That is a transport failure,
    // not a change in the JSON shape, so it must not be reported as `Decode`.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        let _ = socket.read(&mut request).await.unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 100\r\n\r\n{\"routers\": [")
            .await
            .unwrap();
    });
    let base_url = Url::parse(&format!("http://{address}/")).unwrap();
    let client = GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap();

    let error = client.routers().await.unwrap_err();
    assert!(
        matches!(error, DiscoveryClientError::Request { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn registers_a_pool() {
    let fake = FakeService::new(r#"{ "routers": [] }"#);
    let client = fake.client().await;

    let lease = client.register_pool(&registration()).await.unwrap();

    assert_eq!(lease, Lease::new(5, 15).unwrap());
    // The field names are the contract with the service, so compare the JSON
    // it received rather than decoding it back into the same struct.
    assert_eq!(
        fake.0.lock().unwrap().registrations,
        [json!({
            "key": { "cluster_id": "aws-west", "namespace": "dynamo", "dgd_name": "mocker-west-1" },
            "location": {
                "region": "us-west-2",
                "availability_zone": "usw2-az1",
                "datacenter": null
            },
            "model": "Qwen/Qwen3-0.6B",
            "capabilities": ["aggregated"],
            "frontends": ["http://10.1.22.20:8000/"],
            "capacity": [{ "sku": "H100", "workers": 8 }]
        })]
    );
}

#[tokio::test]
async fn a_redirect_is_an_error() {
    // Following a 301 would resend the registration as a GET without its
    // body, so the client must report the 3xx instead.
    let app = axum::Router::new().fallback(|| async {
        (
            StatusCode::MOVED_PERMANENTLY,
            [(header::LOCATION, "/elsewhere")],
        )
    });
    let address = serve(app).await;
    let base_url = Url::parse(&format!("http://{address}/")).unwrap();
    let client = GlobalDiscoveryClient::new(base_url, Duration::from_secs(2)).unwrap();

    let error = client.register_pool(&registration()).await.unwrap_err();
    assert!(
        matches!(&error, DiscoveryClientError::Status { status, .. } if *status == StatusCode::MOVED_PERMANENTLY),
        "{error:?}"
    );
}
