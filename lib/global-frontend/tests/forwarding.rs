// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client calls the Global Frontend, which forwards to a pool frontend.
//! Most pools here are Dynamo's own HTTP service running the echo engine,
//! which streams the prompt back one character at a time.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{
    MODEL, Server, bind, chat, chat_chunk, client, http_service, ok_json, pool, post,
    scripted_pool_frontend, serve, sse_data,
};
use dynamo_global_frontend::{
    Pool, PoolSelector, PoolSnapshot, PoolTarget, RoundRobinSelector, RoutingRequest,
};
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::{StreamingEngineAdapter, make_echo_engine};
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::sync::watch;

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
    for endpoint in [EndpointType::Chat, EndpointType::Completion] {
        service.enable_model_endpoint(endpoint, true).unwrap();
    }
    serve(service, listener, url).await
}

/// A Global Frontend over a catalog the test can change.
async fn global_frontend(pools: Vec<Pool>) -> GlobalFrontend {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    GlobalFrontend {
        server: common::global_frontend(receiver).await,
        pools: sender,
    }
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
    let body = response.text().await.unwrap();

    let events = sse_data(&body);
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "{body}"
    );
    let text = events
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(text, "hello");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
}

#[tokio::test]
async fn folds_the_pool_stream_for_non_streaming_clients() {
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

/// The Responses API translates to chat completions at the Global Frontend,
/// so the pool sees a chat request.
#[tokio::test]
async fn serves_the_responses_api_through_chat() {
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
    assert_eq!(
        post(&global.server.url, "v1/chat/completions", &request)
            .await
            .status(),
        StatusCode::NOT_FOUND
    );

    global
        .pools
        .send_replace(Arc::new(PoolSnapshot::new(vec![pool(
            "blue",
            MODEL,
            &pool_frontend.url,
        )])));
    wait_for_models(&global, &[MODEL]).await;
    assert_eq!(
        post(&global.server.url, "v1/chat/completions", &request)
            .await
            .status(),
        StatusCode::OK
    );

    global.pools.send_replace(Arc::new(PoolSnapshot::default()));
    wait_for_models(&global, &[]).await;
    assert_eq!(
        post(&global.server.url, "v1/chat/completions", &request)
            .await
            .status(),
        StatusCode::NOT_FOUND
    );
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

async fn wait_for_models(global: &GlobalFrontend, expected: &[&str]) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while listed_models(global).await != expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the Global Frontend did not pick up the catalog change");
}

#[tokio::test]
async fn pool_errors_keep_their_status() {
    // The catalog says the pool serves a model its frontend does not have.
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", "ghost", &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat("ghost", "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
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
async fn forwards_the_session_and_skips_pool_keep_alives() {
    let sse = format!(
        ": keep-alive\n\ndata: {}\n\n: keep-alive\n\ndata: {}\n\ndata: [DONE]\n\n",
        chat_chunk("he"),
        chat_chunk("llo")
    );
    let (pool_frontend, seen) = scripted_pool_frontend(sse.leak()).await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = client()
        .post(global.server.url.join("v1/chat/completions").unwrap())
        .header("x-dynamo-session-id", "session-123")
        .json(&chat(MODEL, "hi", true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let text = sse_data(&response.text().await.unwrap())
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(text, "hello");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["x-dynamo-session-id"], "session-123");
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
    let global = common::global_frontend_with(receiver, recorder.clone()).await;

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
async fn sends_the_request_body_with_its_length() {
    let sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("ok"));
    let (pool_frontend, seen) = scripted_pool_frontend(sse.leak()).await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    // The body goes as a stream, so the pool sees its declared length
    // rather than chunked encoding.
    let seen = seen.lock().unwrap();
    assert!(
        seen[0]["content-length"]
            .to_str()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            > 0
    );
    assert!(!seen[0].contains_key("transfer-encoding"));
    assert_eq!(seen[0]["content-type"], "application/json");
}

#[tokio::test]
async fn a_pool_error_mid_stream_ends_the_client_stream_with_an_error() {
    let sse = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chat_chunk("he"),
        json!({"error": {"message": "worker died", "type": "internal_error", "code": 500}})
    );
    let (pool_frontend, _) = scripted_pool_frontend(sse.leak()).await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = sse_data(&response.text().await.unwrap());
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "he");
    assert!(
        events.last().unwrap().get("error").is_some(),
        "the client must see an error frame: {events:?}"
    );
}

#[tokio::test]
async fn pool_server_errors_follow_frontend_sanitization() {
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { (StatusCode::BAD_GATEWAY, "x".repeat(64 * 1024)) }),
    );
    let (listener, url) = bind().await;
    let cancel = tokio_util::sync::CancellationToken::new();
    let shutdown = cancel.clone();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
            .unwrap();
    });
    let pool_frontend = Server { url, cancel };
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = response.text().await.unwrap();
    assert!(
        !body.contains(&"x".repeat(100)),
        "upstream diagnostics must not be exposed"
    );
    assert!(body.len() < 1024, "expected a sanitized error response");
}
