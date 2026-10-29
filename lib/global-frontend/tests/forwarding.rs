// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client calls the Global Frontend, which forwards to a pool frontend.
//! Most pools here are Dynamo's own HTTP service running the echo engine,
//! which streams the prompt back one character at a time.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::HeaderMap;
use dynamo_global_common::PoolKey;
use dynamo_global_frontend::PoolSnapshot;
use dynamo_global_router::{
    InFlightTracker, Pool, PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest,
};
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::{StreamingEngineAdapter, make_echo_engine};
use dynamo_llm::http::service::service_v2::HttpService;
use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

const MODEL: &str = "echo-model";

/// A running HTTP server. Dropping it stops the server.
struct Server {
    url: Url,
    cancel: CancellationToken,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

async fn bind() -> (TcpListener, Url) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (
        listener,
        format!("http://127.0.0.1:{port}/").parse().unwrap(),
    )
}

/// Waits until `url` answers a GET with success.
async fn wait_until_ok(url: &Url) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !client()
            .get(url.clone())
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("server did not come up");
}

/// Dynamo's HTTP service serving the echo engine for chat, completions and
/// Responses.
async fn echo_pool_frontend() -> Server {
    let (listener, url) = bind().await;
    let service = HttpService::builder()
        .host("127.0.0.1")
        .port(url.port().unwrap())
        .build()
        .unwrap();
    let engine = Arc::new(StreamingEngineAdapter::new(make_echo_engine()));
    let manager = service.model_manager();
    manager
        .add_chat_completions_model(MODEL, "echo", engine.clone())
        .unwrap();
    manager
        .add_completions_model(MODEL, "echo", engine)
        .unwrap();
    for endpoint in [
        EndpointType::Chat,
        EndpointType::Completion,
        EndpointType::Responses,
    ] {
        service.enable_model_endpoint(endpoint, true).unwrap();
    }
    let cancel = CancellationToken::new();
    service.spawn_with_listener(cancel.clone(), listener).await;
    wait_until_ok(&url.join("health").unwrap()).await;
    Server { url, cancel }
}

/// A pool frontend serving `app`.
async fn pool_frontend(app: axum::Router) -> Server {
    let (listener, url) = bind().await;
    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
    });
    Server { url, cancel }
}

/// A pool frontend that records the headers of each chat request and answers
/// with a fixed SSE body.
async fn scripted_pool_frontend(sse: &'static str) -> (Server, Arc<Mutex<Vec<HeaderMap>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |headers: HeaderMap| {
            recorder.lock().unwrap().push(headers);
            async move { ([("content-type", "text/event-stream")], sse) }
        }),
    );
    (pool_frontend(app).await, seen)
}

fn pool(dgd: &str, model: &str, frontend: &Url) -> Pool {
    let key = PoolKey::new("ohio", "dynamo", dgd).unwrap();
    Pool::new(&key, model, "us-east-2", vec![frontend.clone()])
}

/// A running Global Frontend, and the catalog it routes over.
struct GlobalFrontend {
    server: Server,
    pools: watch::Sender<Arc<PoolSnapshot>>,
}

/// A Global Frontend that routes round robin over `pools`.
async fn global_frontend(pools: Vec<Pool>) -> GlobalFrontend {
    let selector = Arc::new(RoundRobinSelector::new());
    global_frontend_with(pools, selector, Arc::new(InFlightTracker::new())).await
}

/// A Global Frontend that routes with `selector` over `pools`, counting
/// requests in flight in `in_flight`.
async fn global_frontend_with(
    pools: Vec<Pool>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
) -> GlobalFrontend {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    let (listener, url) = bind().await;
    let cancel = CancellationToken::new();
    let app = dynamo_global_frontend::app(receiver, selector, in_flight, client());
    tokio::spawn(dynamo_global_frontend::serve(listener, app, cancel.clone()));
    wait_until_ok(&url.join("live").unwrap()).await;
    GlobalFrontend {
        server: Server { url, cancel },
        pools: sender,
    }
}

/// One chat completion chunk carrying `content`.
fn chat_chunk(content: &str) -> Value {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}],
    })
}

fn chat(model: &str, prompt: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    })
}

async fn post(base: &Url, route: &str, body: &Value) -> reqwest::Response {
    client()
        .post(base.join(route).unwrap())
        .json(body)
        .send()
        .await
        .unwrap()
}

/// The JSON body of a successful response. On failure, the panic shows the
/// body, which carries the error.
async fn ok_json(response: reqwest::Response) -> Value {
    let status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}

/// The `data:` payloads of an SSE body, up to `[DONE]`.
fn sse_data(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .take_while(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

fn streamed_text(body: &str) -> String {
    sse_data(body)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect()
}

#[tokio::test]
async fn forwards_each_route_to_the_pool() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    // Each case's route and request, and how to read the echoed text from
    // the pool's response body.
    type EchoedText = fn(&str) -> String;
    let cases: [(&str, Value, EchoedText); 4] = [
        ("v1/chat/completions", chat(MODEL, "hello", true), |body| {
            assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
            streamed_text(body)
        }),
        ("v1/chat/completions", chat(MODEL, "hello", false), |body| {
            let body: Value = serde_json::from_str(body).unwrap();
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_owned()
        }),
        (
            "v1/completions",
            json!({"model": MODEL, "prompt": "hello", "max_tokens": 16}),
            |body| {
                let body: Value = serde_json::from_str(body).unwrap();
                body["choices"][0]["text"].as_str().unwrap().to_owned()
            },
        ),
        (
            "v1/responses",
            json!({"model": MODEL, "input": "hello"}),
            |body| {
                let body: Value = serde_json::from_str(body).unwrap();
                body["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
                    .filter_map(|part| part["text"].as_str().map(str::to_owned))
                    .collect()
            },
        ),
    ];
    for (route, request, text) in cases {
        let response = post(&global.server.url, route, &request).await;
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status, StatusCode::OK, "{route}: {body}");
        assert_eq!(text(&body), "hello", "{route}: {body}");
    }
}

#[tokio::test]
async fn passes_the_pool_stream_through_as_sent() {
    // Keep-alives and a mid-stream error frame are the pool's to send.
    let sse = format!(
        ": keep-alive\n\ndata: {}\n\n: keep-alive\n\ndata: {}\n\ndata: [DONE]\n\n",
        chat_chunk("he"),
        json!({"error": {"message": "worker died", "type": "internal_error", "code": 500}})
    );
    let sse: &'static str = sse.leak();
    let (pool_frontend, _) = scripted_pool_frontend(sse).await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), sse);
}

#[tokio::test]
async fn models_follow_the_catalog() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(Vec::new()).await;
    let request = chat(MODEL, "hi", false);
    assert!(listed_models(&global).await.is_empty());
    let response = post(&global.server.url, "v1/chat/completions", &request).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let blue = pool("blue", MODEL, &pool_frontend.url);
    global
        .pools
        .send_replace(Arc::new(PoolSnapshot::new(vec![blue])));
    assert_eq!(listed_models(&global).await, [MODEL]);
    let response = post(&global.server.url, "v1/chat/completions", &request).await;
    assert_eq!(response.status(), StatusCode::OK);

    global.pools.send_replace(Arc::new(PoolSnapshot::default()));
    assert!(listed_models(&global).await.is_empty());
    let response = post(&global.server.url, "v1/chat/completions", &request).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

async fn listed_models(global: &GlobalFrontend) -> Vec<String> {
    let response = client()
        .get(global.server.url.join("v1/models").unwrap())
        .send()
        .await
        .unwrap();
    let body: Value = response.json().await.unwrap();
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn requests_without_a_model_are_rejected() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    for body in [json!({"messages": []}), json!({"model": 7})] {
        let response = post(&global.server.url, "v1/chat/completions", &body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test]
async fn pool_errors_pass_through() {
    // The catalog says the pool serves a model its frontend does not have.
    let echo = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", "ghost", &echo.url)]).await;
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat("ghost", "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // Whatever the pool sends, status and body, reaches the client.
    let failing = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { (StatusCode::BAD_GATEWAY, "pool says no") }),
    ))
    .await;
    let global = global_frontend(vec![pool("blue", MODEL, &failing.url)]).await;
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(response.text().await.unwrap(), "pool says no");
}

#[tokio::test]
async fn unreachable_pools_are_unavailable() {
    let (listener, closed) = bind().await;
    drop(listener);
    let global = global_frontend(vec![pool("blue", MODEL, &closed)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn forwards_the_request_as_sent() {
    let sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("ok"));
    let (pool_frontend, seen) = scripted_pool_frontend(sse.leak()).await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let request = chat(MODEL, "hi", true);
    let response = client()
        .post(global.server.url.join("v1/chat/completions").unwrap())
        .header("x-dynamo-session-id", "session-123")
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // The client's headers go on, and the body goes as a stream with its
    // length declared rather than chunked.
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["x-dynamo-session-id"], "session-123");
    assert_eq!(seen[0]["content-type"], "application/json");
    let length = serde_json::to_vec(&request).unwrap().len();
    assert_eq!(seen[0]["content-length"], length.to_string().as_str());
    assert!(!seen[0].contains_key("transfer-encoding"));
}

/// Records the session each request carried, and routes round robin.
#[derive(Default)]
struct SessionRecorder {
    sessions: Mutex<Vec<Option<String>>>,
    inner: RoundRobinSelector,
}

impl PoolSelector for SessionRecorder {
    fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        let session = request.session.map(str::to_owned);
        self.sessions.lock().unwrap().push(session);
        self.inner.select(request, candidates)
    }
}

#[tokio::test]
async fn the_selector_sees_the_session() {
    let pool_frontend = echo_pool_frontend().await;
    let recorder = Arc::new(SessionRecorder::default());
    let blue = pool("blue", MODEL, &pool_frontend.url);
    let in_flight = Arc::new(InFlightTracker::new());
    let global = global_frontend_with(vec![blue], recorder.clone(), in_flight).await;

    for session in [Some("session-123"), None] {
        let mut request = client()
            .post(global.server.url.join("v1/chat/completions").unwrap())
            .json(&chat(MODEL, "hi", false));
        if let Some(session) = session {
            request = request.header("x-dynamo-session-id", session);
        }
        ok_json(request.send().await.unwrap()).await;
    }
    let sessions = recorder.sessions.lock().unwrap();
    assert_eq!(*sessions, [Some("session-123".to_owned()), None]);
}

#[tokio::test]
async fn requests_count_in_flight_until_their_response_ends() {
    // A pool that sends one chunk and then nothing until the client leaves.
    let endless = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async {
            let first = futures::stream::once(async { Ok::<_, Infallible>("data: {}\n\n") });
            let body = axum::body::Body::from_stream(first.chain(futures::stream::pending()));
            ([("content-type", "text/event-stream")], body)
        }),
    ))
    .await;
    let blue = pool("blue", MODEL, &endless.url);
    let id = blue.id.clone();
    let in_flight = Arc::new(InFlightTracker::new());
    let selector = Arc::new(RoundRobinSelector::new());
    let global = global_frontend_with(vec![blue], selector, in_flight.clone()).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", true),
    )
    .await;
    let mut chunks = response.bytes_stream();
    assert!(chunks.next().await.is_some());
    assert_eq!(in_flight.get(&id), 1);

    // The client leaves mid-stream.
    drop(chunks);
    tokio::time::timeout(Duration::from_secs(5), async {
        while in_flight.get(&id) != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the request stayed in flight after the client left");
}

#[tokio::test]
async fn rotates_across_pools_and_tracks_catalog_changes() {
    let blue_sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("blue"));
    let green_sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("green"));
    let (blue, blue_requests) = scripted_pool_frontend(blue_sse.leak()).await;
    let (green, green_requests) = scripted_pool_frontend(green_sse.leak()).await;
    let blue_pool = pool("blue", MODEL, &blue.url);
    let green_pool = pool("green", MODEL, &green.url);
    let global = global_frontend(vec![blue_pool.clone()]).await;

    async fn send(global: &GlobalFrontend, count: usize) {
        for _ in 0..count {
            let request = chat(MODEL, "hi", true);
            let response = post(&global.server.url, "v1/chat/completions", &request).await;
            assert_eq!(response.status(), StatusCode::OK);
            response.text().await.unwrap();
        }
    }
    send(&global, 1).await;
    let both = PoolSnapshot::new(vec![blue_pool, green_pool.clone()]);
    global.pools.send_replace(Arc::new(both));
    send(&global, 4).await;
    assert_eq!(blue_requests.lock().unwrap().len(), 3);
    assert_eq!(green_requests.lock().unwrap().len(), 2);

    let green_only = PoolSnapshot::new(vec![green_pool]);
    global.pools.send_replace(Arc::new(green_only));
    send(&global, 2).await;
    assert_eq!(blue_requests.lock().unwrap().len(), 3);
    assert_eq!(green_requests.lock().unwrap().len(), 4);
}
