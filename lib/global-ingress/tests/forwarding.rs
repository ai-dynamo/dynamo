// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client calls the Global Frontend, which forwards to a pool frontend.
//! Most pools here are Dynamo's own HTTP service running the echo engine,
//! which streams the prompt back one character at a time.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::http::HeaderMap;
use axum::routing;
use dynamo_global_ingress::{ModelSync, PoolForwardingEngine};
use dynamo_global_router::{FirstPoolSelector, Pool, PoolSnapshot};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::{StreamingEngineAdapter, make_echo_engine};
use dynamo_llm::http::service::service_v2::HttpService;
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

struct GlobalFrontend {
    server: Server,
    pools: watch::Sender<Arc<PoolSnapshot>>,
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

async fn bind() -> (TcpListener, Url) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (
        listener,
        format!("http://127.0.0.1:{port}/").parse().unwrap(),
    )
}

fn http_service(url: &Url) -> HttpService {
    HttpService::builder()
        .host("127.0.0.1")
        .port(url.port().unwrap())
        .build()
        .unwrap()
}

async fn serve(service: HttpService, listener: TcpListener, url: Url) -> Server {
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

async fn global_frontend(pools: Vec<Pool>) -> GlobalFrontend {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    let (listener, url) = bind().await;
    let service = http_service(&url);
    let engine = Arc::new(PoolForwardingEngine::new(
        receiver.clone(),
        Arc::new(FirstPoolSelector),
        client(),
    ));
    let mut models = ModelSync::new(service.clone(), engine);
    models.apply(&receiver.borrow());
    let server = serve(service, listener, url).await;
    tokio::spawn(models.follow(receiver, server.cancel.child_token()));
    GlobalFrontend {
        server,
        pools: sender,
    }
}

fn pool(dgd: &str, model: &str, frontend: &Url) -> Pool {
    Pool {
        id: V1PoolIdDeriver.derive(&PoolKey::new("ohio", "dynamo", dgd).unwrap()),
        model: model.to_owned(),
        location: PoolLocation {
            region: "us-east-2".to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends: vec![frontend.clone()],
    }
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

/// A pool that records the headers it receives and answers with a fixed SSE
/// body.
async fn scripted_pool_frontend(sse: &'static str) -> (Server, Arc<Mutex<Vec<HeaderMap>>>) {
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

#[tokio::test]
async fn forwards_the_session_and_skips_pool_keep_alives() {
    let chunk = |content: &str| {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}}],
        })
    };
    let sse = format!(
        ": keep-alive\n\ndata: {}\n\n: keep-alive\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk("he"),
        chunk("llo")
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

#[tokio::test]
async fn a_pool_error_mid_stream_ends_the_client_stream_with_an_error() {
    let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}}],
    });
    let sse = format!(
        "data: {chunk}\n\ndata: {}\n\ndata: [DONE]\n\n",
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
