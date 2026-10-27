// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Servers and requests shared by the Global Frontend tests.

// Each test binary includes this module and uses only some of it.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::http::HeaderMap;
use axum::routing;
use dynamo_global_ingress::{ModelSync, PoolForwardingEngine, RoutingStats};
use dynamo_global_router::{InFlightTracker, Pool, PoolSnapshot, RoundRobinSelector};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_llm::http::service::service_v2::HttpService;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

pub const MODEL: &str = "echo-model";

/// A running HTTP server. Dropping it stops the server.
pub struct Server {
    pub url: Url,
    pub cancel: CancellationToken,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

pub async fn bind() -> (TcpListener, Url) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (
        listener,
        format!("http://127.0.0.1:{port}/").parse().unwrap(),
    )
}

pub fn http_service(url: &Url) -> HttpService {
    HttpService::builder()
        .host("127.0.0.1")
        .port(url.port().unwrap())
        .build()
        .unwrap()
}

/// Serves `service` and waits until it is healthy.
pub async fn serve(service: HttpService, listener: TcpListener, url: Url) -> Server {
    let cancel = CancellationToken::new();
    service.spawn_with_listener(cancel.clone(), listener).await;
    let health = url.join("health").unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !client()
            .get(health.clone())
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("server did not become healthy");
    Server { url, cancel }
}

/// A Global Frontend that routes round robin over the pools `pools` publishes.
pub async fn global_frontend(pools: watch::Receiver<Arc<PoolSnapshot>>) -> Server {
    counting_global_frontend(pools, Arc::new(InFlightTracker::new())).await
}

/// [`global_frontend`], counting its requests in `in_flight`.
pub async fn counting_global_frontend(
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    in_flight: Arc<InFlightTracker>,
) -> Server {
    let (listener, url) = bind().await;
    let service = http_service(&url);
    let engine = Arc::new(PoolForwardingEngine::new(
        pools.clone(),
        Arc::new(RoundRobinSelector::new()),
        in_flight,
        Arc::new(RoutingStats::new(None)),
        client(),
    ));
    let mut models = ModelSync::new(service.clone(), engine);
    models.apply(&pools.borrow());
    let server = serve(service, listener, url).await;
    tokio::spawn(models.follow(pools, server.cancel.child_token()));
    server
}

pub fn pool(dgd: &str, model: &str, frontend: &Url) -> Pool {
    Pool {
        id: V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", dgd).unwrap()),
        name: format!("ohio/dynamo/{dgd}"),
        model: model.to_owned(),
        location: PoolLocation {
            region: "us-east-2".to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends: vec![frontend.clone()],
        relay: None,
    }
}

/// A pool frontend that records the headers of each chat request and answers
/// with a fixed SSE body.
pub async fn scripted_pool_frontend(sse: &'static str) -> (Server, Arc<Mutex<Vec<HeaderMap>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        routing::post(move |headers: HeaderMap| {
            recorder.lock().unwrap().push(headers);
            async move { ([("content-type", "text/event-stream")], sse) }
        }),
    );
    let (listener, url) = bind().await;
    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
    });
    (Server { url, cancel }, seen)
}

/// One chat completion chunk carrying `content`.
pub fn chat_chunk(content: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}],
    })
}

pub fn chat(model: &str, prompt: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    })
}

pub async fn post(base: &Url, route: &str, body: &Value) -> reqwest::Response {
    client()
        .post(base.join(route).unwrap())
        .json(body)
        .send()
        .await
        .unwrap()
}

/// The JSON body of a successful response. On failure, the panic shows the
/// body, which carries the error.
pub async fn ok_json(response: reqwest::Response) -> Value {
    let status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}

/// The `data:` payloads of an SSE body, up to `[DONE]`.
pub fn sse_data(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .take_while(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}
