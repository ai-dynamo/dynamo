// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client calls the Global Frontend, which forwards to a pool frontend.
//! Most pools here are Dynamo's own HTTP service running the echo engine,
//! which streams the prompt back one character at a time.

mod common;

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{
    MODEL, Server, bind, chat, chat_chunk, client, http_service, ok_json, pool, post,
    scripted_pool_frontend, serve, sse_data,
};
use dynamo_global_frontend::{
    Pool, PoolSelector, PoolSnapshot, PoolTarget, RoundRobinSelector, RoutingRequest,
};
use dynamo_global_router::InFlightTracker;
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::{StreamingEngineAdapter, make_echo_engine};
use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

struct GlobalFrontend {
    server: Server,
    pools: watch::Sender<Arc<PoolSnapshot>>,
}

async fn echo_pool_frontend() -> Server {
    let (listener, url) = bind().await;
    let service = http_service(&url);
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
    serve(service, listener, url).await
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

/// A Global Frontend over a catalog the test can change.
async fn global_frontend(pools: Vec<Pool>) -> GlobalFrontend {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    GlobalFrontend {
        server: common::global_frontend(receiver).await,
        pools: sender,
    }
}

fn streamed_text(body: &str) -> String {
    sse_data(body)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect()
}

#[tokio::test]
async fn streams_chat_completions_from_the_pool() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hello", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let body = response.text().await.unwrap();
    assert_eq!(streamed_text(&body), "hello");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
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
async fn passes_non_streaming_responses_through() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hello", false),
    )
    .await;
    let body = ok_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hello", "{body}");
}

#[tokio::test]
async fn forwards_completions() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let request = json!({"model": MODEL, "prompt": "hi there", "max_tokens": 16});
    let response = post(&global.server.url, "v1/completions", &request).await;
    let body = ok_json(response).await;
    assert_eq!(body["choices"][0]["text"], "hi there", "{body}");
}

#[tokio::test]
async fn forwards_the_responses_api_to_the_pool() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let request = json!({"model": MODEL, "input": "hello"});
    let response = post(&global.server.url, "v1/responses", &request).await;
    let body = ok_json(response).await;
    let text = body["output"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|item| item["content"].as_array().cloned().unwrap_or_default())
        .filter_map(|part| part["text"].as_str().map(str::to_owned))
        .collect::<String>();
    assert_eq!(text, "hello", "{body}");
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
    fn select(&self, request: &RoutingRequest<'_>, candidates: &[&Pool]) -> Option<PoolTarget> {
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
    let (_pools, receiver) = watch::channel(Arc::new(PoolSnapshot::new(vec![blue])));
    let in_flight = Arc::new(InFlightTracker::new());
    let global = common::global_frontend_with(receiver, recorder.clone(), in_flight).await;

    for session in [Some("session-123"), None] {
        let mut request = client()
            .post(global.url.join("v1/chat/completions").unwrap())
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
    let (_pools, receiver) = watch::channel(Arc::new(PoolSnapshot::new(vec![blue])));
    let in_flight = Arc::new(InFlightTracker::new());
    let selector = Arc::new(RoundRobinSelector::new());
    let global = common::global_frontend_with(receiver, selector, in_flight.clone()).await;

    let response = post(&global.url, "v1/chat/completions", &chat(MODEL, "hi", true)).await;
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
