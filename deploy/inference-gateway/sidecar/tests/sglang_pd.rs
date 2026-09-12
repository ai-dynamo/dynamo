// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Request, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use dynamo_epp_sidecar::{
    PREFILLER_HOST_PORT, PdAdapter, PrefillEndpoint, SglangPdAdapter, SidecarState, router,
};
use futures::{StreamExt, stream};
use reqwest::Url;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{Barrier, Semaphore, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

const DEADLINE: Duration = Duration::from_secs(10);
const PREFILL_SSE: &str = "data: {\"id\":\"prefill\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n";
const DECODE_SSE: &str = ": keepalive\n\ndata: {\"id\":\"decode\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\r\n\r\ndata: [DONE]\n\n";
const DECODE_FIRST_SSE: &[u8] = b"data: {\"id\":\"decode\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null}]}\n\n";

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(DEADLINE, future)
        .await
        .expect("mock HTTP operation did not complete")
}

#[derive(Clone, Debug)]
struct Observed {
    path: String,
    headers: HeaderMap,
    body: Bytes,
}

type MockReceiver<T> = Arc<Mutex<Option<mpsc::UnboundedReceiver<T>>>>;

#[derive(Clone)]
enum Reply {
    Full(Bytes),
    Streaming(MockReceiver<Bytes>),
    Failing(MockReceiver<Result<Bytes, std::io::Error>>),
}

#[derive(Clone)]
struct MockSpec {
    info: Value,
    status: StatusCode,
    content_type: &'static str,
    retry_after: Option<&'static str>,
    reply: Reply,
    barrier: Option<Arc<Barrier>>,
    before_headers: Option<Arc<Semaphore>>,
    abort_gate: Option<Arc<Semaphore>>,
}

impl MockSpec {
    fn new(mode: &str) -> Self {
        Self {
            info: json!({
                "version": "0.5.19",
                "disaggregation_mode": mode,
                "disaggregation_bootstrap_port": 8998,
            }),
            status: StatusCode::OK,
            content_type: "application/json",
            retry_after: None,
            reply: Reply::Full(Bytes::from(
                json!({
                    "id": mode,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "a"}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                })
                .to_string(),
            )),
            barrier: None,
            before_headers: None,
            abort_gate: None,
        }
    }

    fn sse(mut self, body: &'static str) -> Self {
        self.content_type = "text/event-stream";
        self.reply = Reply::Full(Bytes::from_static(body.as_bytes()));
        self
    }

    fn held(mut self) -> Self {
        self.before_headers = Some(Arc::new(Semaphore::new(0)));
        self
    }

    fn streaming(mut self) -> (Self, mpsc::UnboundedSender<Bytes>) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.content_type = "text/event-stream";
        self.reply = Reply::Streaming(Arc::new(Mutex::new(Some(rx))));
        (self, tx)
    }

    fn failing_stream(mut self) -> (Self, mpsc::UnboundedSender<Result<Bytes, std::io::Error>>) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.content_type = "text/event-stream";
        self.reply = Reply::Failing(Arc::new(Mutex::new(Some(rx))));
        (self, tx)
    }
}

#[derive(Clone)]
struct MockState {
    spec: MockSpec,
    observed: Arc<Mutex<Vec<Observed>>>,
    generated: Arc<Semaphore>,
    aborted: Arc<Semaphore>,
    emitted: Arc<Semaphore>,
    stop: CancellationToken,
}

async fn mock_request(State(state): State<MockState>, request: Request<Body>) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap();
    let path = parts.uri.path().to_owned();
    state.observed.lock().unwrap().push(Observed {
        path: path.clone(),
        headers: parts.headers,
        body,
    });
    match path.as_str() {
        "/server_info" => axum::Json(state.spec.info).into_response(),
        "/abort_request" => {
            state.aborted.add_permits(1);
            if let Some(gate) = &state.spec.abort_gate {
                tokio::select! {
                    permit = gate.acquire() => permit.unwrap().forget(),
                    () = state.stop.cancelled() => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
                }
            }
            StatusCode::OK.into_response()
        }
        "/v1/chat/completions" => {
            state.generated.add_permits(1);
            if let Some(barrier) = &state.spec.barrier {
                tokio::select! {
                    _ = barrier.wait() => {},
                    () = state.stop.cancelled() => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
                }
            }
            if let Some(gate) = &state.spec.before_headers {
                tokio::select! {
                    permit = gate.acquire() => permit.unwrap().forget(),
                    () = state.stop.cancelled() => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
                }
            }
            let body = match state.spec.reply {
                Reply::Full(bytes) => Body::from(bytes),
                Reply::Streaming(receiver) => {
                    let receiver = receiver.lock().unwrap().take().unwrap();
                    Body::from_stream(stream::unfold(
                        (receiver, state.emitted.clone()),
                        |(mut receiver, emitted)| async move {
                            receiver.recv().await.map(|bytes| {
                                emitted.add_permits(1);
                                (Ok::<_, Infallible>(bytes), (receiver, emitted))
                            })
                        },
                    ))
                }
                Reply::Failing(receiver) => {
                    let receiver = receiver.lock().unwrap().take().unwrap();
                    Body::from_stream(stream::unfold(
                        (receiver, state.emitted.clone()),
                        |(mut receiver, emitted)| async move {
                            receiver.recv().await.map(|chunk| {
                                if chunk.is_ok() {
                                    emitted.add_permits(1);
                                }
                                (chunk, (receiver, emitted))
                            })
                        },
                    ))
                }
            };
            let mut response = Response::builder()
                .status(state.spec.status)
                .header("content-type", state.spec.content_type)
                .header("x-engine-response", "untouched");
            if let Some(retry_after) = state.spec.retry_after {
                response = response.header("retry-after", retry_after);
            }
            response.body(body).unwrap()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

struct MockServer {
    url: Url,
    state: MockState,
    task: JoinHandle<()>,
}

impl MockServer {
    async fn start(spec: MockSpec) -> Self {
        let state = MockState {
            spec,
            observed: Arc::default(),
            generated: Arc::new(Semaphore::new(0)),
            aborted: Arc::new(Semaphore::new(0)),
            emitted: Arc::new(Semaphore::new(0)),
            stop: CancellationToken::new(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
        let app = Router::new()
            .route("/server_info", get(mock_request))
            .route("/abort_request", post(mock_request))
            .route("/v1/chat/completions", post(mock_request))
            .with_state(state.clone());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, state, task }
    }

    async fn generated(&self) {
        bounded(self.state.generated.acquire())
            .await
            .unwrap()
            .forget();
    }

    async fn aborted(&self) {
        bounded(self.state.aborted.acquire())
            .await
            .unwrap()
            .forget();
    }

    async fn emitted(&self) {
        bounded(self.state.emitted.acquire())
            .await
            .unwrap()
            .forget();
    }

    fn requests(&self, path: &str) -> Vec<Observed> {
        self.state
            .observed
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path == path)
            .cloned()
            .collect()
    }

    fn generations(&self) -> Vec<Value> {
        self.requests("/v1/chat/completions")
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    fn aborts(&self) -> Vec<Value> {
        self.requests("/abort_request")
            .iter()
            .map(|r| serde_json::from_slice(&r.body).unwrap())
            .collect()
    }

    fn endpoint(&self) -> PrefillEndpoint {
        let mut headers = HeaderMap::new();
        headers.insert(PREFILLER_HOST_PORT, self.authority().parse().unwrap());
        PrefillEndpoint::parse_headers(&headers).unwrap().unwrap()
    }

    fn authority(&self) -> String {
        format!(
            "{}:{}",
            self.url.host_str().unwrap(),
            self.url.port().unwrap()
        )
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.state.stop.cancel();
        self.task.abort();
    }
}

struct Pair {
    prefill: MockServer,
    decode: MockServer,
    adapter: Arc<SglangPdAdapter>,
}

impl Pair {
    async fn new(prefill: MockSpec, decode: MockSpec) -> Self {
        Self::with_timeout(prefill, decode, Duration::from_secs(5)).await
    }

    async fn with_timeout(prefill: MockSpec, decode: MockSpec, read_timeout: Duration) -> Self {
        let prefill = MockServer::start(prefill).await;
        let decode = MockServer::start(decode).await;
        let adapter = Arc::new(
            SglangPdAdapter::new(decode.url.clone(), Duration::from_secs(2), read_timeout).unwrap(),
        );
        Self {
            prefill,
            decode,
            adapter,
        }
    }

    fn app(&self, force_shutdown: CancellationToken) -> Router {
        router(
            SidecarState::new(
                self.decode.url.clone(),
                Duration::from_secs(2),
                Duration::from_secs(5),
                self.adapter.clone(),
                CancellationToken::new(),
                force_shutdown,
            )
            .unwrap(),
        )
    }

    fn request(&self, body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", "Bearer inference-key")
            .header(PREFILLER_HOST_PORT, self.prefill.authority())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn response(&self, body: &Value) -> Response<Body> {
        bounded(
            self.app(CancellationToken::new())
                .oneshot(self.request(body)),
        )
        .await
        .unwrap()
    }

    async fn both_generated(&self) {
        tokio::join!(self.prefill.generated(), self.decode.generated());
    }

    async fn shutdown(&self) {
        bounded(self.adapter.shutdown()).await;
    }

    fn assert_abort_matches(&self, server: &MockServer) {
        let generations = server.generations();
        let aborts = server.aborts();
        assert_eq!(aborts.len(), 1, "expected exactly one request-scoped abort");
        assert_eq!(
            aborts[0],
            json!({"rid": generations[0]["rid"], "abort_all": false})
        );
    }
}

fn chat() -> Value {
    json!({"model":"test", "messages":[{"role":"user","content":"hello"}], "max_tokens":32, "stream":false})
}

fn simultaneous(mut prefill: MockSpec, mut decode: MockSpec) -> (MockSpec, MockSpec) {
    let barrier = Arc::new(Barrier::new(2));
    prefill.barrier = Some(barrier.clone());
    decode.barrier = Some(barrier);
    (prefill, decode)
}

#[tokio::test]
async fn dispatches_both_legs_concurrently_and_preserves_chat_fields() {
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").sse(PREFILL_SSE),
        MockSpec::new("decode").sse(DECODE_SSE),
    );
    let pair = Pair::new(prefill, decode).await;
    let mut rooms = Vec::new();
    let mut rids = Vec::new();
    for n in [None, Some(json!(1))] {
        let mut body = json!({
            "model":"test", "messages":[{"role":"user","content":"hello"}],
            "stream":true, "stream_options":{"include_usage":true},
            "max_tokens":256, "max_completion_tokens":128, "min_tokens":3,
            "temperature":0.25, "seed":17, "stop":["end"],
            "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],
            "tool_choice":"auto", "response_format":{"type":"json_object"},
            "unknown_extension":{"preserve":[1,2,3]},
            "rid":"client-forged", "bootstrap_host":"forged.invalid", "bootstrap_port":1, "bootstrap_room":0,
        });
        if let Some(n) = n {
            body["n"] = n;
        }
        let response = pair.response(&body).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-engine-response"], "untouched");
        assert_eq!(
            bounded(to_bytes(response.into_body(), usize::MAX))
                .await
                .unwrap(),
            DECODE_SSE
        );
        let prefill = pair.prefill.generations().pop().unwrap();
        let decode = pair.decode.generations().pop().unwrap();
        assert_eq!(prefill, decode);
        assert_eq!(prefill["bootstrap_host"], "127.0.0.1");
        assert_eq!(prefill["bootstrap_port"], 8998);
        assert_ne!(prefill["rid"], body["rid"]);
        let room = prefill["bootstrap_room"].as_u64().unwrap();
        assert!(room <= i64::MAX as u64);
        assert!(!rooms.contains(&room));
        rooms.push(room);
        let rid = prefill["rid"].as_str().unwrap().to_owned();
        assert!(!rids.contains(&rid));
        rids.push(rid);
        let mut actual = prefill;
        for field in ["bootstrap_host", "bootstrap_port", "bootstrap_room", "rid"] {
            actual.as_object_mut().unwrap().remove(field);
            body.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(actual, body);
    }
    pair.shutdown().await;
    for server in [&pair.prefill, &pair.decode] {
        assert_eq!(server.requests("/server_info").len(), 2);
        assert!(
            server.aborts().is_empty(),
            "normal EOS must not abort completed requests"
        );
        for request in server.requests("/v1/chat/completions") {
            assert!(!request.headers.contains_key(PREFILLER_HOST_PORT));
            assert_eq!(request.headers["authorization"], "Bearer inference-key");
        }
    }
}

#[tokio::test]
async fn prefill_http_failure_aborts_decode_waiting_for_headers() {
    let mut prefill = MockSpec::new("prefill");
    prefill.status = StatusCode::SERVICE_UNAVAILABLE;
    prefill.reply = Reply::Full(Bytes::from_static(
        b"{\"object\":\"error\",\"message\":\"prefill unavailable\",\"code\":503}",
    ));
    let (prefill, decode) = simultaneous(prefill, MockSpec::new("decode").held());
    let pair = Pair::new(prefill, decode).await;
    assert_eq!(
        pair.response(&chat()).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn prefill_sse_error_after_success_chunk_prevents_decode_response() {
    let error_sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\ndata: {\"error\":{\"object\":\"error\",\"message\":\"KV transfer failed\",\"type\":\"INTERNAL_SERVER_ERROR\",\"param\":null,\"code\":500}}\n\ndata: [DONE]\n\n";
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").sse(error_sse),
        MockSpec::new("decode").sse(DECODE_SSE),
    );
    let pair = Pair::new(prefill, decode).await;
    let mut body = chat();
    body["stream"] = json!(true);
    assert_eq!(pair.response(&body).await.status(), StatusCode::BAD_GATEWAY);
    pair.shutdown().await;
}

#[tokio::test]
async fn prefill_sse_without_done_aborts_pending_decode() {
    let truncated = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":\"length\"}]}\n\n";
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").sse(truncated),
        MockSpec::new("decode").held(),
    );
    let pair = Pair::new(prefill, decode).await;
    let mut body = chat();
    body["stream"] = json!(true);
    assert_eq!(pair.response(&body).await.status(), StatusCode::BAD_GATEWAY);
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn graceful_prefill_abort_is_failure_in_both_json_and_sse() {
    for streaming in [false, true] {
        let mut prefill = MockSpec::new("prefill");
        if streaming {
            prefill = prefill.sse("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"abort\"}]}\n\ndata: [DONE]\n\n");
        } else {
            prefill.reply = Reply::Full(Bytes::from(json!({
                "id":"prefill", "object":"chat.completion", "created":1, "model":"test",
                "choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"abort"}],
            }).to_string()));
        }
        let (prefill, decode) = simultaneous(prefill, MockSpec::new("decode").held());
        let pair = Pair::new(prefill, decode).await;
        let mut body = chat();
        body["stream"] = json!(streaming);
        assert_eq!(pair.response(&body).await.status(), StatusCode::BAD_GATEWAY);
        pair.shutdown().await;
        pair.assert_abort_matches(&pair.decode);
    }
}

#[tokio::test]
async fn validates_server_version_mode_and_bootstrap_before_dispatch() {
    for (target, key, value) in [
        ("prefill", "version", json!("0.5.18")),
        ("decode", "disaggregation_mode", json!("prefill")),
        ("prefill", "disaggregation_bootstrap_port", json!(0)),
    ] {
        let mut prefill = MockSpec::new("prefill");
        let mut decode = MockSpec::new("decode");
        if target == "prefill" {
            prefill.info[key] = value;
        } else {
            decode.info[key] = value;
        }
        let pair = Pair::new(prefill, decode).await;
        assert_eq!(
            pair.response(&chat()).await.status(),
            StatusCode::BAD_GATEWAY
        );
        pair.shutdown().await;
        assert!(pair.prefill.generations().is_empty());
        assert!(pair.decode.generations().is_empty());
        assert!(pair.prefill.aborts().is_empty());
        assert!(pair.decode.aborts().is_empty());
    }
}

#[tokio::test]
async fn rejects_parallel_sampling_before_engine_generation() {
    let pair = Pair::new(MockSpec::new("prefill"), MockSpec::new("decode")).await;
    for n in [
        Value::Null,
        json!(0),
        json!(2),
        json!(-1),
        json!("1"),
        json!(true),
    ] {
        let mut body = chat();
        body["n"] = n;
        assert_eq!(pair.response(&body).await.status(), StatusCode::BAD_REQUEST);
    }
    pair.shutdown().await;
    assert!(pair.prefill.generations().is_empty());
    assert!(pair.decode.generations().is_empty());
    assert!(pair.prefill.aborts().is_empty());
    assert!(pair.decode.aborts().is_empty());
}

#[tokio::test]
async fn decode_error_cancels_prefill_without_waiting_for_prefill_completion() {
    let mut decode = MockSpec::new("decode");
    decode.status = StatusCode::BAD_REQUEST;
    decode.reply = Reply::Full(Bytes::from_static(
        b"{\"object\":\"error\",\"message\":\"decode rejected\",\"code\":400}",
    ));
    let (prefill, decode) = simultaneous(MockSpec::new("prefill").held(), decode);
    let pair = Pair::new(prefill, decode).await;
    assert_eq!(
        pair.response(&chat()).await.status(),
        StatusCode::BAD_REQUEST
    );
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
}

#[tokio::test]
async fn decode_sse_error_cancels_prefill_before_prefill_completes() {
    let decode_error = "data: {\"error\":{\"object\":\"error\",\"message\":\"decode failed\",\"type\":\"INTERNAL_SERVER_ERROR\",\"param\":null,\"code\":500}}\n\ndata: [DONE]\n\n";
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").held(),
        MockSpec::new("decode").sse(decode_error),
    );
    let pair = Pair::with_timeout(prefill, decode, Duration::from_secs(30)).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let response = tokio::time::timeout(Duration::from_secs(1), pair.response(&body))
        .await
        .expect("decode SSE error must cancel prefill before its read timeout");
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
}

#[tokio::test]
async fn inference_rate_limit_preserves_status_and_retry_after_on_either_leg() {
    for failing_prefill in [true, false] {
        let mut failed = MockSpec::new(if failing_prefill { "prefill" } else { "decode" });
        failed.status = StatusCode::TOO_MANY_REQUESTS;
        failed.retry_after = Some("7");
        failed.reply = Reply::Full(Bytes::from_static(b"{\"object\":\"error\",\"message\":\"engine overloaded\",\"type\":\"TooManyRequests\",\"code\":429}"));
        let (prefill, decode) = if failing_prefill {
            simultaneous(failed, MockSpec::new("decode").held())
        } else {
            simultaneous(MockSpec::new("prefill").held(), failed)
        };
        let pair = Pair::new(prefill, decode).await;
        let response = pair.response(&chat()).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "7");
        let body: Value = serde_json::from_slice(
            &bounded(to_bytes(response.into_body(), usize::MAX))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["error"]["message"].as_str().is_some());
        pair.shutdown().await;
        pair.assert_abort_matches(if failing_prefill {
            &pair.decode
        } else {
            &pair.prefill
        });
    }
}

#[tokio::test]
async fn read_timeout_maps_to_504_and_cleans_up_both_legs() {
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").held(),
        MockSpec::new("decode").held(),
    );
    let pair = Pair::with_timeout(prefill, decode, Duration::from_millis(100)).await;
    assert_eq!(
        pair.response(&chat()).await.status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn dropping_execute_before_headers_aborts_both_dispatched_requests() {
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").held(),
        MockSpec::new("decode").held(),
    );
    let pair = Pair::new(prefill, decode).await;
    let adapter = pair.adapter.clone();
    let request = pair.request(&chat());
    let endpoint = pair.prefill.endpoint();
    let mut task = AbortOnDrop(tokio::spawn(async move {
        adapter
            .execute(request, endpoint, CancellationToken::new())
            .await
    }));
    pair.both_generated().await;
    task.0.abort();
    assert!((&mut task.0).await.unwrap_err().is_cancelled());
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn dropping_decode_stream_aborts_decode_and_closes_upstream_stream() {
    let (decode, sender) = MockSpec::new("decode").streaming();
    sender.send(Bytes::from_static(DECODE_FIRST_SSE)).unwrap();
    let pair = Pair::new(MockSpec::new("prefill").sse(PREFILL_SSE), decode).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let response = pair.response(&body).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    assert_eq!(
        bounded(stream.next()).await.unwrap().unwrap(),
        DECODE_FIRST_SSE
    );
    drop(stream);
    bounded(sender.closed()).await;
    pair.decode.aborted().await;
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.decode);
    assert!(pair.prefill.aborts().is_empty());
}

#[tokio::test]
async fn force_cancellation_cleans_up_both_inflight_legs() {
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").held(),
        MockSpec::new("decode").held(),
    );
    let pair = Pair::new(prefill, decode).await;
    let force_shutdown = CancellationToken::new();
    let app = pair.app(force_shutdown.clone());
    let mut task = AbortOnDrop(tokio::spawn(app.oneshot(pair.request(&chat()))));
    pair.both_generated().await;
    force_shutdown.cancel();
    assert_eq!(
        bounded(&mut task.0).await.unwrap().unwrap().status(),
        StatusCode::BAD_GATEWAY
    );
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn requests_without_prefill_metadata_keep_decode_only_wire_body() {
    let pair = Pair::new(
        MockSpec::new("prefill"),
        MockSpec::new("decode").sse(DECODE_SSE),
    )
    .await;
    let original = b"{  \"stream\": true, \"n\": 3, \"unknown\": [1, 2], \"messages\": [] }";
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(original.as_slice()))
        .unwrap();
    let response = bounded(pair.app(CancellationToken::new()).oneshot(request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        bounded(to_bytes(response.into_body(), usize::MAX))
            .await
            .unwrap(),
        DECODE_SSE
    );
    pair.shutdown().await;
    assert!(pair.prefill.state.observed.lock().unwrap().is_empty());
    assert_eq!(
        pair.decode.requests("/v1/chat/completions")[0].body,
        original.as_slice()
    );
    assert!(pair.decode.requests("/server_info").is_empty());
    assert!(pair.decode.aborts().is_empty());
}

struct AbortOnDrop<T>(JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[tokio::test]
async fn prefill_completion_preserves_partial_sse_and_utf8_decode_prefixes_byte_for_byte() {
    const WIRE: &[u8] = b"data: {\"id\":\"decode\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\xe4\xbd\xa0\xe5\xa5\xbd\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let utf8_split = WIRE.iter().position(|byte| *byte == 0xe4).unwrap() + 1;
    for split in [2, utf8_split] {
        let prefill = MockSpec::new("prefill").sse(PREFILL_SSE).held();
        let prefill_gate = prefill.before_headers.as_ref().unwrap().clone();
        let (decode, sender) = MockSpec::new("decode").streaming();
        let (prefill, decode) = simultaneous(prefill, decode);
        let pair = Pair::new(prefill, decode).await;
        let mut body = chat();
        body["stream"] = json!(true);
        let mut task = AbortOnDrop(tokio::spawn(
            pair.app(CancellationToken::new())
                .oneshot(pair.request(&body)),
        ));
        pair.both_generated().await;
        sender.send(Bytes::copy_from_slice(&WIRE[..split])).unwrap();
        pair.decode.emitted().await;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut task.0)
                .await
                .is_err()
        );
        prefill_gate.add_permits(1);
        let response = tokio::time::timeout(Duration::from_secs(1), &mut task.0)
            .await
            .expect("complete prefill must release even an incomplete decode SSE event")
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.into_body().into_data_stream();
        let prefix = bounded(stream.next()).await.unwrap().unwrap();
        assert_eq!(prefix, &WIRE[..split]);
        sender.send(Bytes::copy_from_slice(&WIRE[split..])).unwrap();
        drop(sender);
        let mut received = prefix.to_vec();
        while let Some(chunk) = bounded(stream.next()).await {
            received.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(received, WIRE);
        pair.shutdown().await;
        assert!(pair.prefill.aborts().is_empty());
        assert!(pair.decode.aborts().is_empty());
    }
}

#[tokio::test]
async fn decode_body_connection_failure_after_200_cancels_pending_prefill() {
    let (decode, sender) = MockSpec::new("decode").failing_stream();
    let (prefill, decode) = simultaneous(MockSpec::new("prefill").held(), decode);
    let pair = Pair::with_timeout(prefill, decode, Duration::from_secs(30)).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let mut task = AbortOnDrop(tokio::spawn(
        pair.app(CancellationToken::new())
            .oneshot(pair.request(&body)),
    ));
    pair.both_generated().await;
    sender
        .send(Ok(Bytes::from_static(DECODE_FIRST_SSE)))
        .unwrap();
    pair.decode.emitted().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut task.0)
            .await
            .is_err()
    );
    sender
        .send(Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "mock decode connection reset",
        )))
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(1), &mut task.0)
        .await
        .expect("decode body failure must interrupt pending prefill")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
}

#[tokio::test]
async fn decode_prefix_above_32_mib_is_bounded_while_prefill_is_pending() {
    let (decode, sender) = MockSpec::new("decode").streaming();
    let (prefill, decode) = simultaneous(MockSpec::new("prefill").held(), decode);
    let pair = Pair::with_timeout(prefill, decode, Duration::from_secs(30)).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let mut task = AbortOnDrop(tokio::spawn(
        pair.app(CancellationToken::new())
            .oneshot(pair.request(&body)),
    ));
    pair.both_generated().await;
    let mut event = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"".to_vec();
    event.extend(std::iter::repeat_n(b'x', 64 * 1024));
    event.extend_from_slice(b"\"},\"finish_reason\":null}]}\n\n");
    let mut oversized = Vec::with_capacity(32 * 1024 * 1024 + event.len());
    while oversized.len() <= 32 * 1024 * 1024 {
        oversized.extend_from_slice(&event);
    }
    sender.send(Bytes::from(oversized)).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(5), &mut task.0)
        .await
        .expect("oversized decode prefix must fail before prefill's 30-second timeout")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    bounded(sender.closed()).await;
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn client_tcp_disconnect_before_headers_aborts_both_legs() {
    let (prefill, decode) = simultaneous(
        MockSpec::new("prefill").held(),
        MockSpec::new("decode").held(),
    );
    let pair = Pair::new(prefill, decode).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = pair.app(CancellationToken::new());
    let _server = AbortOnDrop(tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap()
    }));
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let request = client
        .post(format!("http://{address}/v1/chat/completions"))
        .header(PREFILLER_HOST_PORT, pair.prefill.authority())
        .json(&chat());
    let mut task = AbortOnDrop(tokio::spawn(async move { request.send().await }));
    pair.both_generated().await;
    task.0.abort();
    assert!((&mut task.0).await.unwrap_err().is_cancelled());
    drop(client);
    tokio::join!(pair.prefill.aborted(), pair.decode.aborted());
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.prefill);
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn decode_response_waits_for_complete_prefill_stream_even_after_decode_bytes_arrive() {
    let (prefill, prefill_sender) = MockSpec::new("prefill").streaming();
    let (decode, decode_sender) = MockSpec::new("decode").streaming();
    let (prefill, decode) = simultaneous(prefill, decode);
    let pair = Pair::new(prefill, decode).await;
    let mut body = chat();
    body["stream"] = json!(true);
    let mut task = AbortOnDrop(tokio::spawn(
        pair.app(CancellationToken::new())
            .oneshot(pair.request(&body)),
    ));
    pair.both_generated().await;
    prefill_sender.send(Bytes::from_static(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n")).unwrap();
    decode_sender
        .send(Bytes::from_static(DECODE_SSE.as_bytes()))
        .unwrap();
    tokio::join!(pair.prefill.emitted(), pair.decode.emitted());
    // Both engines have emitted bytes. P remains open until its explicit error below.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut task.0)
            .await
            .is_err(),
        "decode response escaped before prefill completed"
    );
    prefill_sender.send(Bytes::from_static(b"data: {\"error\":{\"object\":\"error\",\"message\":\"transfer failed\",\"type\":\"INTERNAL_SERVER_ERROR\",\"param\":null,\"code\":500}}\n\ndata: [DONE]\n\n")).unwrap();
    drop(prefill_sender);
    assert_eq!(
        bounded(&mut task.0).await.unwrap().unwrap().status(),
        StatusCode::BAD_GATEWAY
    );
    bounded(decode_sender.closed()).await;
    pair.shutdown().await;
    pair.assert_abort_matches(&pair.decode);
}

#[tokio::test]
async fn shutdown_waits_for_abort_response_and_bounds_a_stalled_abort() {
    for release_abort in [true, false] {
        let abort_gate = Arc::new(Semaphore::new(0));
        let mut decode = MockSpec::new("decode");
        decode.abort_gate = Some(abort_gate.clone());
        let (decode, sender) = decode.streaming();
        sender.send(Bytes::from_static(DECODE_FIRST_SSE)).unwrap();
        // Keep the transport's read deadline above the dedicated five-second abort deadline.
        let pair = Pair::with_timeout(
            MockSpec::new("prefill").sse(PREFILL_SSE),
            decode,
            Duration::from_secs(30),
        )
        .await;
        let mut body = chat();
        body["stream"] = json!(true);
        let response = pair.response(&body).await;
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        pair.decode.aborted().await;
        let mut shutdown = Box::pin(pair.adapter.shutdown());
        assert!(
            futures::poll!(&mut shutdown).is_pending(),
            "shutdown skipped its active abort cleanup"
        );
        if release_abort {
            abort_gate.add_permits(1);
        }
        tokio::time::timeout(Duration::from_secs(6), &mut shutdown)
            .await
            .expect("abort cleanup exceeded its five-second deadline plus scheduling allowance");
        bounded(sender.closed()).await;
        pair.assert_abort_matches(&pair.decode);
        assert!(pair.prefill.aborts().is_empty());
    }
}
