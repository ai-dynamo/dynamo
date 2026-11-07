// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A client calls the Global Frontend, which forwards to a pool frontend.
//! Most pools here are Dynamo's own HTTP service running the echo engine,
//! which streams the prompt back one character at a time.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::HeaderMap;
use dynamo_global_common::{PoolId, PoolKey};
use dynamo_global_frontend::{Metrics, PoolSnapshot};
use dynamo_global_router::{
    InFlightTracker, Pool, PoolSelector, PoolTarget, RoundRobinSelector, RoutingRequest,
};
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::{StreamingEngineAdapter, make_echo_engine};
use dynamo_llm::http::service::service_v2::HttpService;
use futures::StreamExt;
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpSocket};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

const MODEL: &str = "echo-model";
/// The region of the Global Frontend and of every pool.
const REGION: &str = "us-east-2";

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

/// A URL whose connections are refused. The socket keeps the port bound,
/// but not listening, so no other test's server can take it.
fn closed() -> (TcpSocket, Url) {
    let socket = TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let port = socket.local_addr().unwrap().port();
    (socket, format!("http://127.0.0.1:{port}/").parse().unwrap())
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

/// A pool frontend that answers every chat request with `status` and
/// `body`, and the number of requests it has received.
async fn answering_pool_frontend(
    status: StatusCode,
    body: &'static str,
) -> (Server, Arc<AtomicUsize>) {
    let received = Arc::new(AtomicUsize::new(0));
    let counter = received.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            async move { (status, body) }
        }),
    );
    (pool_frontend(app).await, received)
}

/// A pool frontend that reads each request and closes the connection
/// without answering.
async fn hanging_up_pool_frontend() -> Server {
    let (listener, url) = bind().await;
    let cancel = CancellationToken::new();
    let stop = cancel.clone();
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = tokio::select! {
                _ = stop.cancelled() => return,
                accepted = listener.accept() => accepted.unwrap(),
            };
            let mut request = [0; 4096];
            let _ = stream.read(&mut request).await;
        }
    });
    Server { url, cancel }
}

fn pool(dgd: &str, model: &str, frontend: &Url) -> Pool {
    let key = PoolKey::new("ohio", "dynamo", dgd).unwrap();
    Pool::new(&key, model, REGION, vec![frontend.clone()])
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
    global_frontend_failing_over(pools, selector, in_flight, 1).await
}

/// [`global_frontend_with`], sending a request its pool cannot take to up to
/// `max_failovers` other pools.
async fn global_frontend_failing_over(
    pools: Vec<Pool>,
    selector: Arc<dyn PoolSelector>,
    in_flight: Arc<InFlightTracker>,
    max_failovers: usize,
) -> GlobalFrontend {
    let (sender, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    let (listener, url) = bind().await;
    let cancel = CancellationToken::new();
    let metrics = Metrics::new(Some(REGION.to_owned()), in_flight.clone());
    let app = dynamo_global_frontend::app(
        receiver,
        selector,
        in_flight,
        client(),
        metrics,
        max_failovers,
    );
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

/// The Global Frontend's metrics once `ready` holds for them, scraping
/// until it does: a response's metrics settle just after its client has
/// read it.
async fn metrics_when(global: &Url, ready: impl Fn(&str) -> bool) -> String {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let response = client()
                .get(global.join("metrics").unwrap())
                .send()
                .await
                .unwrap();
            assert_eq!(response.headers()["content-type"], prometheus::TEXT_FORMAT);
            let metrics = response.text().await.unwrap();
            if ready(&metrics) {
                return metrics;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the metrics did not settle")
}

/// The value of `metric`'s series whose labels include `labels`, in
/// Prometheus's text format.
fn sample(metrics: &str, metric: &str, labels: &[(&str, &str)]) -> Option<f64> {
    metrics
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let (series, value) = line.rsplit_once(' ')?;
            let (name, series_labels) = series.split_once('{').unwrap_or((series, ""));
            let matches = name == metric
                && labels
                    .iter()
                    .all(|(key, value)| series_labels.contains(&format!("{key}=\"{value}\"")));
            matches.then(|| value.parse().unwrap())
        })
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
    let (_socket, closed) = closed();
    let global = global_frontend(vec![pool("blue", MODEL, &closed)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let no_response = [("pool", "ohio/dynamo/blue"), ("status", "no_response")];
    let requests = "dynamo_global_frontend_requests_total";
    metrics_when(&global.server.url, |metrics| {
        sample(metrics, requests, &no_response) == Some(1.0)
    })
    .await;
}

#[tokio::test]
async fn a_client_leaving_before_the_pool_answers_counts_as_cancelled() {
    // A pool that queues every request and never answers.
    let queueing = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(std::future::pending::<&'static str>),
    ))
    .await;
    let global = global_frontend(vec![pool("blue", MODEL, &queueing.url)]).await;

    let request = client()
        .post(global.server.url.join("v1/chat/completions").unwrap())
        .json(&chat(MODEL, "hi", true))
        .timeout(Duration::from_millis(200))
        .send()
        .await;
    assert!(request.unwrap_err().is_timeout());

    let cancelled = [("pool", "ohio/dynamo/blue"), ("status", "cancelled")];
    let requests = "dynamo_global_frontend_requests_total";
    metrics_when(&global.server.url, |metrics| {
        sample(metrics, requests, &cancelled) == Some(1.0)
    })
    .await;
}

#[tokio::test]
async fn metrics_count_requests_by_pool() {
    let pool_frontend = echo_pool_frontend().await;
    let global = global_frontend(vec![pool("blue", MODEL, &pool_frontend.url)]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", true),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    response.text().await.unwrap();
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat("unknown", "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let blue = [("pool", "ohio/dynamo/blue"), ("region", REGION)];
    let duration = "dynamo_global_frontend_request_duration_seconds_count";
    let metrics = metrics_when(&global.server.url, |metrics| {
        sample(metrics, duration, &blue) == Some(1.0)
    })
    .await;
    let ok = [blue[0], blue[1], ("local", "true"), ("status", "200")];
    let counts = [
        ("dynamo_global_frontend_requests_total", &ok[..]),
        (
            "dynamo_global_frontend_time_to_first_byte_seconds_count",
            &blue,
        ),
        (
            "dynamo_global_frontend_rejected_requests_total",
            &[("error_type", "not_found")],
        ),
    ];
    for (metric, labels) in counts {
        assert_eq!(sample(&metrics, metric, labels), Some(1.0), "{metric}");
    }
    let in_flight = "dynamo_global_frontend_inflight_requests";
    assert_eq!(sample(&metrics, in_flight, &blue), Some(0.0));
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

/// Records the session and tenant each request carried, and routes round robin.
#[derive(Default)]
struct SessionRecorder {
    sessions: Mutex<Vec<Option<String>>>,
    tenants: Mutex<Vec<String>>,
    inner: RoundRobinSelector,
}

#[async_trait::async_trait]
impl PoolSelector for SessionRecorder {
    async fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        let session = request.session.map(str::to_owned);
        self.sessions.lock().unwrap().push(session);
        self.tenants.lock().unwrap().push(request.tenant.to_owned());
        self.inner.select(request, candidates).await
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
async fn the_selector_sees_a_tenant_but_never_the_credential() {
    use dynamo_global_router::affinity::{ANONYMOUS_TENANT, tenant_id};

    let pool_frontend = echo_pool_frontend().await;
    let recorder = Arc::new(SessionRecorder::default());
    let blue = pool("blue", MODEL, &pool_frontend.url);
    let in_flight = Arc::new(InFlightTracker::new());
    let global = global_frontend_with(vec![blue], recorder.clone(), in_flight).await;

    for credential in [Some("Bearer alice"), Some("Bearer bob"), None] {
        let mut request = client()
            .post(global.server.url.join("v1/chat/completions").unwrap())
            .header("x-dynamo-session-id", "session-123")
            .json(&chat(MODEL, "hi", false));
        if let Some(credential) = credential {
            request = request.header("authorization", credential);
        }
        ok_json(request.send().await.unwrap()).await;
    }
    let tenants = recorder.tenants.lock().unwrap();
    // One session ID, three namespaces: each credential gets its own, and an
    // unauthenticated request is anonymous rather than borrowing one.
    assert_eq!(
        *tenants,
        [
            tenant_id(Some("Bearer alice")),
            tenant_id(Some("Bearer bob")),
            ANONYMOUS_TENANT.to_owned(),
        ]
    );
    // The credential itself never reaches the selector, so it never reaches
    // a key or a stored row.
    assert!(tenants.iter().all(|tenant| !tenant.contains("alice")));
}

#[tokio::test]
async fn independent_frontends_share_session_affinity() {
    use dynamo_global_router::affinity::{
        AffinityConfig, InMemoryAffinityStore, StoreBackedSessionAffinity,
    };
    use dynamo_global_router::{AffinitySelector, PoolSignals};

    let blue_server = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { axum::Json(json!({"pool": "blue"})) }),
    ))
    .await;
    let green_server = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { axum::Json(json!({"pool": "green"})) }),
    ))
    .await;
    let mut blue = pool("blue", MODEL, &blue_server.url);
    blue.location.region = "east".into();
    let mut green = pool("green", MODEL, &green_server.url);
    green.location.region = "west".into();
    let store = Arc::new(InMemoryAffinityStore::default());
    let mut globals = Vec::new();
    for region in ["east", "west"] {
        let in_flight = Arc::new(InFlightTracker::new());
        let selector = Arc::new(AffinitySelector::new(
            Arc::new(PoolSignals::new(in_flight.clone())),
            region.into(),
            2.0,
            Arc::new(
                StoreBackedSessionAffinity::new(store.clone(), Duration::from_secs(1800)).unwrap(),
            ),
            AffinityConfig::default(),
        ));
        globals.push(
            global_frontend_with(vec![blue.clone(), green.clone()], selector, in_flight).await,
        );
    }
    for global in &globals {
        let response = client()
            .post(global.server.url.join("v1/chat/completions").unwrap())
            .header("x-dynamo-session-id", "shared-session")
            .json(&chat(MODEL, "hi", false))
            .send()
            .await
            .unwrap();
        assert_eq!(ok_json(response).await["pool"], "blue");
    }
    let response = client()
        .post(globals[1].server.url.join("v1/chat/completions").unwrap())
        .json(&chat(MODEL, "hi", false))
        .send()
        .await
        .unwrap();
    assert_eq!(ok_json(response).await["pool"], "green");
    assert_eq!(store.len(), 1);
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
    let duration = "dynamo_global_frontend_request_duration_seconds_count";
    metrics_when(&global.server.url, |metrics| {
        sample(metrics, duration, &[("pool", "ohio/dynamo/blue")]) == Some(1.0)
    })
    .await;
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

#[tokio::test]
async fn shutdown_lets_an_open_stream_finish_then_returns() {
    // A pool that sends one chunk, then the second once `release` fires.
    let release = Arc::new(tokio::sync::Notify::new());
    let gate = release.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || {
            let gate = gate.clone();
            async move {
                let first = futures::stream::once(async { Ok::<_, Infallible>("data: one\n\n") });
                let second = futures::stream::once(async move {
                    gate.notified().await;
                    Ok::<_, Infallible>("data: two\n\n")
                });
                let body = axum::body::Body::from_stream(first.chain(second));
                ([("content-type", "text/event-stream")], body)
            }
        }),
    );
    let pool_server = pool_frontend(app).await;
    let pools = vec![pool("gated", MODEL, &pool_server.url)];
    let (_catalog, receiver) = watch::channel(Arc::new(PoolSnapshot::new(pools)));
    let (listener, url) = bind().await;
    let in_flight = Arc::new(InFlightTracker::new());
    let metrics = Metrics::new(Some(REGION.to_owned()), in_flight.clone());
    let selector = Arc::new(RoundRobinSelector::new());
    let app = dynamo_global_frontend::app(receiver, selector, in_flight, client(), metrics, 1);
    let shutdown = CancellationToken::new();
    let serving = tokio::spawn(dynamo_global_frontend::serve(
        listener,
        app,
        shutdown.clone(),
    ));
    wait_until_ok(&url.join("live").unwrap()).await;

    // The client keeps its connection open once idle, so the server must
    // close it when the response ends.
    let client = client();
    let response = client
        .post(url.join("v1/chat/completions").unwrap())
        .json(&chat(MODEL, "hi", true))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    shutdown.cancel();
    release.notify_one();
    assert_eq!(response.text().await.unwrap(), "data: one\n\ndata: two\n\n");
    // The server returns once its last connection closes, long before the
    // grace period ends.
    tokio::time::timeout(Duration::from_secs(5), serving)
        .await
        .unwrap()
        .unwrap();
    drop(client);
}

const FAILOVERS: &str = "dynamo_global_frontend_failovers_total";
const REQUESTS: &str = "dynamo_global_frontend_requests_total";
const FAILOVER_REASONS: [&str; 3] = ["no_response", "unavailable", "overloaded"];

/// The content of a non-streamed chat completion.
async fn chat_content(response: reqwest::Response) -> String {
    let body = ok_json(response).await;
    body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned()
}

/// The Global Frontend's metrics now. Counters are final once the client
/// has the response.
async fn metrics_now(global: &GlobalFrontend) -> String {
    metrics_when(&global.server.url, |_| true).await
}

#[tokio::test]
async fn fails_over_from_an_unavailable_or_overloaded_pool() {
    let green = echo_pool_frontend().await;
    for (status, reason) in [(503, "unavailable"), (529, "overloaded")] {
        let status = StatusCode::from_u16(status).unwrap();
        let (blue, blue_requests) = answering_pool_frontend(status, "busy").await;
        let global = global_frontend(vec![
            pool("blue", MODEL, &blue.url),
            pool("green", MODEL, &green.url),
        ])
        .await;

        let response = post(
            &global.server.url,
            "v1/chat/completions",
            &chat(MODEL, "hi", false),
        )
        .await;
        assert_eq!(chat_content(response).await, "hi", "{status}");
        assert_eq!(blue_requests.load(Ordering::Relaxed), 1, "{status}");

        // Each pool tried counts the request under its own answer.
        let metrics = metrics_now(&global).await;
        let blue = [("pool", "ohio/dynamo/blue"), ("status", status.as_str())];
        let green = [("pool", "ohio/dynamo/green"), ("status", "200")];
        assert_eq!(sample(&metrics, REQUESTS, &blue), Some(1.0), "{status}");
        assert_eq!(sample(&metrics, REQUESTS, &green), Some(1.0), "{status}");
        assert_eq!(
            sample(&metrics, FAILOVERS, &[("reason", reason)]),
            Some(1.0)
        );
    }
}

#[tokio::test]
async fn fails_over_when_the_pool_does_not_answer() {
    let green = echo_pool_frontend().await;
    let (_socket, refused) = closed();
    let hanging_up = hanging_up_pool_frontend().await;
    for blue in [&refused, &hanging_up.url] {
        let global = global_frontend(vec![
            pool("blue", MODEL, blue),
            pool("green", MODEL, &green.url),
        ])
        .await;

        let response = post(
            &global.server.url,
            "v1/chat/completions",
            &chat(MODEL, "hi", false),
        )
        .await;
        assert_eq!(chat_content(response).await, "hi", "{blue}");

        let metrics = metrics_now(&global).await;
        let no_response = [("pool", "ohio/dynamo/blue"), ("status", "no_response")];
        assert_eq!(
            sample(&metrics, REQUESTS, &no_response),
            Some(1.0),
            "{blue}"
        );
        let reason = [("reason", "no_response")];
        assert_eq!(sample(&metrics, FAILOVERS, &reason), Some(1.0), "{blue}");
    }
}

#[tokio::test]
async fn the_last_attempt_passes_through_what_it_got() {
    let unavailable = StatusCode::SERVICE_UNAVAILABLE;
    let overloaded = StatusCode::from_u16(529).unwrap();
    let (blue, _) = answering_pool_frontend(unavailable, "blue is busy").await;
    let (green, _) = answering_pool_frontend(overloaded, "green is overloaded").await;
    let global = global_frontend(vec![
        pool("blue", MODEL, &blue.url),
        pool("green", MODEL, &green.url),
    ])
    .await;
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), overloaded);
    assert_eq!(response.text().await.unwrap(), "green is overloaded");
    let metrics = metrics_now(&global).await;
    let reason = |reason| [("reason", reason)];
    assert_eq!(
        sample(&metrics, FAILOVERS, &reason("unavailable")),
        Some(1.0)
    );
    assert_eq!(
        sample(&metrics, FAILOVERS, &reason("overloaded")),
        Some(0.0)
    );

    // A last pool that cannot be reached gets the Global Frontend's own 503.
    let (_socket, refused) = closed();
    let global = global_frontend(vec![
        pool("blue", MODEL, &blue.url),
        pool("green", MODEL, &refused),
    ])
    .await;
    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), unavailable);
    let body: Value = response.json().await.unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with(&format!(
            "pool frontend {refused}v1/chat/completions failed"
        )),
        "{message}"
    );
}

#[tokio::test]
async fn pools_without_frontends_are_not_tried() {
    let (blue, _) = answering_pool_frontend(StatusCode::SERVICE_UNAVAILABLE, "blue is busy").await;
    let draining = Pool::new(
        &PoolKey::new("ohio", "dynamo", "draining").unwrap(),
        MODEL,
        REGION,
        Vec::new(),
    );
    let global = global_frontend(vec![pool("blue", MODEL, &blue.url), draining]).await;

    let response = post(
        &global.server.url,
        "v1/chat/completions",
        &chat(MODEL, "hi", false),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.text().await.unwrap(), "blue is busy");
    let metrics = metrics_now(&global).await;
    let reason = [("reason", "unavailable")];
    assert_eq!(sample(&metrics, FAILOVERS, &reason), Some(0.0));
    let rejected = "dynamo_global_frontend_rejected_requests_total";
    assert_eq!(sample(&metrics, rejected, &[]), None);
}

#[tokio::test]
async fn failover_tries_each_pool_once_up_to_the_limit() {
    for (max_failovers, tried) in [(0, 1), (1, 2), (5, 3)] {
        let mut servers = Vec::new();
        let mut pools = Vec::new();
        for dgd in ["blue", "green", "red"] {
            let (server, received) =
                answering_pool_frontend(StatusCode::SERVICE_UNAVAILABLE, "busy").await;
            pools.push(pool(dgd, MODEL, &server.url));
            servers.push((server, received));
        }
        let selector = Arc::new(RoundRobinSelector::new());
        let in_flight = Arc::new(InFlightTracker::new());
        let global = global_frontend_failing_over(pools, selector, in_flight, max_failovers).await;

        let response = post(
            &global.server.url,
            "v1/chat/completions",
            &chat(MODEL, "hi", false),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let received = servers
            .iter()
            .map(|(_, received)| received.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        assert!(received.iter().all(|count| *count <= 1), "{received:?}");
        assert_eq!(received.iter().sum::<usize>(), tried, "{received:?}");
        let metrics = metrics_now(&global).await;
        let reason = [("reason", "unavailable")];
        let failovers = (tried - 1) as f64;
        assert_eq!(sample(&metrics, FAILOVERS, &reason), Some(failovers));
    }
}

#[tokio::test]
async fn other_statuses_pass_through_without_failing_over() {
    let (green, green_requests) = answering_pool_frontend(StatusCode::OK, "green").await;
    for status in [429, 500, 502] {
        let status = StatusCode::from_u16(status).unwrap();
        let (blue, _) = answering_pool_frontend(status, "blue says no").await;
        let global = global_frontend(vec![
            pool("blue", MODEL, &blue.url),
            pool("green", MODEL, &green.url),
        ])
        .await;

        let response = post(
            &global.server.url,
            "v1/chat/completions",
            &chat(MODEL, "hi", false),
        )
        .await;
        assert_eq!(response.status(), status);
        assert_eq!(response.text().await.unwrap(), "blue says no");
        // Every reason is exported from the start, at zero.
        let metrics = metrics_now(&global).await;
        for reason in FAILOVER_REASONS {
            let labels = [("reason", reason)];
            assert_eq!(sample(&metrics, FAILOVERS, &labels), Some(0.0), "{status}");
        }
    }
    assert_eq!(green_requests.load(Ordering::Relaxed), 0);
}

/// Records, for each selection, the session it saw and the requests in
/// flight to `watched`, and routes round robin.
struct InFlightRecorder {
    in_flight: Arc<InFlightTracker>,
    watched: PoolId,
    seen: Mutex<Vec<(Option<String>, u64)>>,
    inner: RoundRobinSelector,
}

#[async_trait::async_trait]
impl PoolSelector for InFlightRecorder {
    async fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        let session = request.session.map(str::to_owned);
        let in_flight = self.in_flight.get(&self.watched);
        self.seen.lock().unwrap().push((session, in_flight));
        self.inner.select(request, candidates).await
    }
}

#[tokio::test]
async fn a_failover_releases_the_first_pool_and_resends_the_request() {
    let (blue, _) = answering_pool_frontend(StatusCode::SERVICE_UNAVAILABLE, "busy").await;
    let sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("green"));
    let sse: &'static str = sse.leak();
    let (green, seen) = scripted_pool_frontend(sse).await;
    let blue = pool("blue", MODEL, &blue.url);
    let green = pool("green", MODEL, &green.url);
    let in_flight = Arc::new(InFlightTracker::new());
    let recorder = Arc::new(InFlightRecorder {
        in_flight: in_flight.clone(),
        watched: blue.id.clone(),
        seen: Mutex::default(),
        inner: RoundRobinSelector::new(),
    });
    let ids = [blue.id.clone(), green.id.clone()];
    let global = global_frontend_with(vec![blue, green], recorder.clone(), in_flight.clone()).await;

    let request = chat(MODEL, "hi", true);
    let response = client()
        .post(global.server.url.join("v1/chat/completions").unwrap())
        .header("x-dynamo-session-id", "session-123")
        .header("authorization", "Bearer alice")
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), sse);

    // Blue's request had left its count before the second selection, which
    // saw no session and so could not move the session's pool.
    assert_eq!(
        *recorder.seen.lock().unwrap(),
        [(Some("session-123".to_owned()), 0), (None, 0)]
    );
    // Green got the request as the client sent it.
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0]["x-dynamo-session-id"], "session-123");
        assert_eq!(seen[0]["authorization"], "Bearer alice");
        let length = serde_json::to_vec(&request).unwrap().len();
        assert_eq!(seen[0]["content-length"], length.to_string().as_str());
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        while ids.iter().any(|id| in_flight.get(id) != 0) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("a request stayed in flight");
}

#[tokio::test]
async fn a_failover_does_not_move_the_session_pin() {
    use dynamo_global_router::affinity::{
        AffinityConfig, InMemoryAffinityStore, StoreBackedSessionAffinity,
    };
    use dynamo_global_router::{AffinitySelector, PoolSignals};

    // Blue, in the Global Frontend's region, is busy for its first request.
    let busy = Arc::new(AtomicBool::new(true));
    let blue_server = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move || {
            let busy = busy.swap(false, Ordering::Relaxed);
            async move {
                if busy {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        axum::Json(json!({"pool": "busy"})),
                    )
                } else {
                    (StatusCode::OK, axum::Json(json!({"pool": "blue"})))
                }
            }
        }),
    ))
    .await;
    let green_server = pool_frontend(axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(|| async { axum::Json(json!({"pool": "green"})) }),
    ))
    .await;
    let mut blue = pool("blue", MODEL, &blue_server.url);
    blue.location.region = "east".into();
    let mut green = pool("green", MODEL, &green_server.url);
    green.location.region = "west".into();
    let in_flight = Arc::new(InFlightTracker::new());
    let store = Arc::new(InMemoryAffinityStore::default());
    let selector = Arc::new(AffinitySelector::new(
        Arc::new(PoolSignals::new(in_flight.clone())),
        "east".into(),
        2.0,
        Arc::new(StoreBackedSessionAffinity::new(store, Duration::from_secs(1800)).unwrap()),
        AffinityConfig::default(),
    ));
    let stats = selector.stats();
    let global = global_frontend_with(vec![blue, green], selector, in_flight).await;

    let mut served = Vec::new();
    for _ in 0..2 {
        let response = client()
            .post(global.server.url.join("v1/chat/completions").unwrap())
            .header("x-dynamo-session-id", "session-123")
            .json(&chat(MODEL, "hi", false))
            .send()
            .await
            .unwrap();
        served.push(ok_json(response).await["pool"].clone());
    }
    // Green serves the request blue could not take, but the session stays on
    // blue, and the retry does not touch the store.
    assert_eq!(served, ["green", "blue"]);
    assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
    assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
}

/// Routes round robin for its first selection and finds no pool after that,
/// as a selector may once the candidates are narrowed.
#[derive(Default)]
struct ChoosesOnce {
    selections: AtomicUsize,
    inner: RoundRobinSelector,
}

#[async_trait::async_trait]
impl PoolSelector for ChoosesOnce {
    async fn select<'a>(
        &self,
        request: &RoutingRequest<'_>,
        candidates: &[&'a Pool],
    ) -> Option<PoolTarget<'a>> {
        if self.selections.fetch_add(1, Ordering::Relaxed) > 0 {
            return None;
        }
        self.inner.select(request, candidates).await
    }
}

#[tokio::test]
async fn a_failure_passes_through_when_no_other_pool_is_chosen() {
    let green = echo_pool_frontend().await;
    let (busy, _) = answering_pool_frontend(StatusCode::SERVICE_UNAVAILABLE, "blue is busy").await;
    let (_socket, refused) = closed();
    for blue in [&busy.url, &refused] {
        let blue_pool = pool("blue", MODEL, blue);
        let id = blue_pool.id.clone();
        let in_flight = Arc::new(InFlightTracker::new());
        let selector = Arc::new(ChoosesOnce::default());
        let global = global_frontend_with(
            vec![blue_pool, pool("green", MODEL, &green.url)],
            selector.clone(),
            in_flight.clone(),
        )
        .await;

        let response = post(
            &global.server.url,
            "v1/chat/completions",
            &chat(MODEL, "hi", false),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{blue}");
        let body = response.text().await.unwrap();
        if blue == &busy.url {
            assert_eq!(body, "blue is busy");
        } else {
            let body: Value = serde_json::from_str(&body).unwrap();
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.starts_with("pool frontend "), "{message}");
        }
        assert_eq!(selector.selections.load(Ordering::Relaxed), 2, "{blue}");
        assert_eq!(in_flight.get(&id), 0, "{blue}");
        let metrics = metrics_now(&global).await;
        for reason in FAILOVER_REASONS {
            let labels = [("reason", reason)];
            assert_eq!(sample(&metrics, FAILOVERS, &labels), Some(0.0), "{blue}");
        }
        let rejected = "dynamo_global_frontend_rejected_requests_total";
        assert_eq!(sample(&metrics, rejected, &[]), None, "{blue}");
    }
}
