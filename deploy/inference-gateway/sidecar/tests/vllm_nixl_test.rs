// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol and adapter tests for the raw-vLLM NIXL P/D path.
//!
//! These run against ephemeral local fake workers, so they need no GPU, no
//! vLLM, and no NIXL. They prove what the sidecar does with the HTTP exchange:
//! which request each leg sends, which legs must *not* run, where the error
//! boundary is drawn, and that cancellation reaches the backend future.
//!
//! What they do not prove: that a real NIXL connector accepts the handoff, or
//! that any KV byte moved. That needs the two-worker smoke test plus the
//! backend connector's own logs.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, Request, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::any;
use dynamo_epp_sidecar::metadata::PREFILLER_HOST_PORT;
use dynamo_epp_sidecar::server::{SidecarState, router};
use dynamo_epp_sidecar::vllm_nixl::{self, VllmNixlAdapter};
use dynamo_epp_sidecar::{PdAdapter, UnavailablePdAdapter};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Fake worker
// ---------------------------------------------------------------------------

/// One captured upstream request.
#[derive(Debug, Clone)]
struct Captured {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
}

impl Captured {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("captured body is JSON")
    }
}

/// How a fake worker answers.
#[derive(Clone)]
enum Behaviour {
    /// 200 with this JSON body.
    Json(Value),
    /// 200 with an SSE stream sent as these chunks.
    Sse(Vec<String>),
    /// An HTTP error status with a JSON body.
    Status(StatusCode, Value),
    /// 200 with a non-JSON body.
    Raw(&'static str),
    /// 200 with a body that is not valid JSON.
    TruncatedJson,
    /// 200 with a body larger than any legitimate handoff.
    Oversized(usize),
    /// 200 whose streamed body is a valid prefill response followed by nothing,
    /// with a `Content-Length` that promises more. Parks the caller inside the
    /// bounded body read, deterministically.
    PartialBodyThenHang(Value),
    /// 200 whose streamed body never yields a byte and never ends. Parked
    /// inside the body read with a satisfied `Content-Length`, so only
    /// end-of-stream can finish it.
    HeadersThenStall,
    /// 200 whose streamed body emits one small chunk every `interval` forever.
    /// Each gap stays under the read timeout, so only a total deadline can end
    /// the leg. `Content-Length` promises more than is ever delivered.
    ChunkEveryMillis(u64),
    /// Never answer at all.
    Hang,
}

struct Fake {
    captured: Mutex<Vec<Captured>>,
    behaviour: Behaviour,
    address: SocketAddr,
}

impl Fake {
    /// Bind a fake worker and start serving it.
    async fn start(behaviour: Behaviour) -> Arc<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fake = Arc::new(Self {
            captured: Mutex::new(Vec::new()),
            behaviour,
            address,
        });
        let app = Router::new()
            .fallback(any(fake_handler))
            .with_state(fake.clone());
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        fake
    }

    fn address(&self) -> SocketAddr {
        self.address
    }

    fn captured(&self) -> Vec<Captured> {
        self.captured.lock().unwrap().clone()
    }

    fn call_count(&self) -> usize {
        self.captured.lock().unwrap().len()
    }

    fn last(&self) -> Captured {
        self.captured()
            .last()
            .cloned()
            .expect("the fake worker was called")
    }
}

async fn fake_handler(State(fake): State<Arc<Fake>>, request: Request<Body>) -> Response<Body> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap_or_default();
    fake.captured.lock().unwrap().push(Captured {
        method: parts.method,
        uri: parts.uri,
        headers: parts.headers,
        body,
    });

    match fake.behaviour.clone() {
        Behaviour::Json(value) => json_response(StatusCode::OK, &value),
        Behaviour::Sse(chunks) => {
            let frames: Vec<Result<Bytes, Infallible>> = chunks
                .into_iter()
                .map(|chunk| Ok(Bytes::from(chunk.into_bytes())))
                .collect();
            (
                StatusCode::OK,
                [("content-type", "text/event-stream")],
                Body::from_stream(futures::stream::iter(frames)),
            )
                .into_response()
        }
        Behaviour::Status(status, value) => json_response(status, &value),
        Behaviour::Raw(text) => (
            StatusCode::OK,
            [("content-type", "text/plain")],
            Body::from(text),
        )
            .into_response(),
        Behaviour::TruncatedJson => (
            StatusCode::OK,
            [("content-type", "application/json")],
            Body::from(r#"{"choices":[],"kv_transfer_p"#),
        )
            .into_response(),
        Behaviour::Oversized(size) => (
            StatusCode::OK,
            [("content-type", "application/json")],
            Body::from(vec![b'x'; size]),
        )
            .into_response(),
        Behaviour::PartialBodyThenHang(handoff) => {
            let first = serde_json::to_vec(&json!({
                "choices": [],
                "kv_transfer_params": handoff
            }))
            .expect("fixture serializes");
            // Promise the whole body, deliver only the first chunk: the peer then
            // stalls without reaching end-of-stream.
            let promised = first.len() + 4096;
            let head =
                futures::stream::once(async move { Ok::<_, Infallible>(Bytes::from(first)) });
            (
                StatusCode::OK,
                [
                    ("content-type", "application/json"),
                    ("content-length", &promised.to_string()),
                ],
                Body::from_stream(head.chain(futures::stream::pending())),
            )
                .into_response()
        }
        Behaviour::HeadersThenStall => (
            StatusCode::OK,
            [
                ("content-type", "application/json"),
                ("content-length", "32"),
            ],
            Body::from_stream(futures::stream::pending::<Result<Bytes, Infallible>>()),
        )
            .into_response(),
        Behaviour::ChunkEveryMillis(interval_ms) => {
            let ticks = futures::stream::unfold(0usize, move |n| async move {
                tokio::time::sleep(Duration::from_millis(interval_ms)).await;
                Some((Ok::<_, Infallible>(Bytes::from(vec![b'.'; 256])), n + 1))
            });
            (
                StatusCode::OK,
                [
                    ("content-type", "application/json"),
                    ("content-length", "1000000"),
                ],
                Body::from_stream(ticks),
            )
                .into_response()
        }
        Behaviour::Hang => {
            futures::future::pending::<()>().await;
            unreachable!("a hanging fake never returns")
        }
    }
}

fn json_response(status: StatusCode, value: &Value) -> Response<Body> {
    (
        status,
        [("content-type", "application/json")],
        Body::from(value.to_string()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A handoff as vLLM v0.29.0 `NixlPullConnectorScheduler::request_finished`
/// emits it, with two KV cache groups.
fn handoff() -> Value {
    json!({
        "do_remote_prefill": true,
        "do_remote_decode": false,
        "remote_block_ids": [[1, 2, 3], [7]],
        "remote_engine_id": "engine-abc",
        "remote_request_id": "req-123",
        "remote_host": "10.0.0.5",
        "remote_port": 5600,
        "tp_size": 1,
        "dcp_size": 1,
        "pp_size": 1,
        "remote_num_tokens": 128,
        "remote_blocks_expiry_time": null,
        "transfer_mode": "pull"
    })
}

fn prefill_success() -> Value {
    json!({
        "id": "chatcmpl-prefill",
        "object": "chat.completion",
        "model": "Qwen/Qwen3-0.6B",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "PRE-FILL-TOKEN"},
            "finish_reason": "length"
        }],
        "usage": {"prompt_tokens": 128, "completion_tokens": 1, "total_tokens": 129},
        "kv_transfer_params": handoff()
    })
}

fn decode_success() -> Value {
    json!({
        "id": "chatcmpl-decode",
        "object": "chat.completion",
        "model": "Qwen/Qwen3-0.6B",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "DECODED"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 128, "completion_tokens": 2, "total_tokens": 130}
    })
}

fn original_request() -> Value {
    json!({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 64,
        "min_tokens": 4,
        "temperature": 0.7,
        "tools": [{"type": "function", "function": {"name": "get_time"}}]
    })
}

// ---------------------------------------------------------------------------
// Sidecar under test
// ---------------------------------------------------------------------------

/// Everything a test needs to drive the sidecar and inspect both legs.
struct Harness {
    app: Router,
    prefill: Arc<Fake>,
    decode: Arc<Fake>,
    _adapter: Arc<dyn PdAdapter>,
}

impl Harness {
    /// Build the two fakes and a sidecar whose adapter points at them.
    async fn build(prefill_behaviour: Behaviour, decode_behaviour: Behaviour) -> Self {
        Self::build_with_limits(
            prefill_behaviour,
            decode_behaviour,
            1024 * 1024,
            1024 * 1024,
        )
        .await
    }

    async fn build_with_limits(
        prefill_behaviour: Behaviour,
        decode_behaviour: Behaviour,
        max_request_bytes: usize,
        max_prefill_response_bytes: usize,
    ) -> Self {
        let prefill = Fake::start(prefill_behaviour).await;
        let decode = Fake::start(decode_behaviour).await;
        let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
            decode_url(&decode),
            Duration::from_secs(5),
            Duration::from_secs(5),
            vllm_nixl::Config {
                model: "Qwen/Qwen3-0.6B".to_string(),
                max_request_bytes,
                // Long enough that no test trips it by accident.
                client_body_timeout: Duration::from_secs(30),
                max_prefill_response_bytes,
                prefill_deadline: Duration::from_secs(60),
            },
        )
        .unwrap();
        Self::from_adapter(adapter, prefill, decode)
    }

    /// A P/D sidecar with explicit client timeouts, for the timeout cases.
    async fn with_timeouts(
        prefill_behaviour: Behaviour,
        connect_timeout: Duration,
        read_timeout: Duration,
    ) -> (Router, Arc<Fake>, Arc<Fake>) {
        let prefill = Fake::start(prefill_behaviour).await;
        let decode = Fake::start(Behaviour::Json(decode_success())).await;
        let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
            decode_url(&decode),
            connect_timeout,
            read_timeout,
            vllm_nixl::Config::default(),
        )
        .unwrap();
        let state = SidecarState::new(
            decode_url(&decode),
            connect_timeout,
            read_timeout,
            adapter,
            CancellationToken::new(),
            CancellationToken::new(),
        )
        .unwrap();
        (router(state), prefill, decode)
    }

    /// Build a sidecar around an explicitly supplied adapter, for the tests
    /// that need different client timeouts.
    fn from_adapter(adapter: Arc<dyn PdAdapter>, prefill: Arc<Fake>, decode: Arc<Fake>) -> Self {
        let state = SidecarState::new(
            decode_url(&decode),
            Duration::from_secs(5),
            Duration::from_secs(5),
            adapter.clone(),
            CancellationToken::new(),
            CancellationToken::new(),
        )
        .unwrap();
        Self {
            app: router(state),
            prefill,
            decode,
            _adapter: adapter,
        }
    }
}

fn decode_url(decode: &Arc<Fake>) -> reqwest::Url {
    format!("http://{}", decode.address()).parse().unwrap()
}

fn pd_request(prefill: SocketAddr, body: &Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(PREFILLER_HOST_PORT, prefill.to_string())
        .header("content-type", "application/json")
        .header("x-request-id", "client-request-id")
        .header("authorization", "Bearer secret")
        .header("x-gateway-destination-endpoint", "decode:8000")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body_bytes(response: Response<Body>) -> Bytes {
    to_bytes(response.into_body(), usize::MAX).await.unwrap()
}

async fn body_json(response: Response<Body>) -> Value {
    serde_json::from_slice(&body_bytes(response).await).expect("response body is JSON")
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().expect("error code present")
}

// ---------------------------------------------------------------------------
// C01-C03: the request path
// ---------------------------------------------------------------------------

/// C01: with no prefill header the request is a plain decode passthrough.
#[tokio::test]
async fn c01_no_prefill_header_passes_through_to_decode() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions?stream=true")
        .header("x-custom", "preserved")
        .body(Body::from("raw-body"))
        .unwrap();

    let response = harness.app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    assert_eq!(
        harness.prefill.call_count(),
        0,
        "the prefill leg must not run"
    );
    let captured = harness.decode.last();
    assert_eq!(captured.uri.to_string(), "/v1/chat/completions?stream=true");
    assert_eq!(captured.headers["x-custom"], "preserved");
    assert_eq!(captured.body, "raw-body");
}

/// C02: a non-streaming P/D request returns the decode response.
#[tokio::test]
async fn c02_pd_non_streaming_success_returns_the_decode_response() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut body = original_request();
    body["stream"] = json!(false);
    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &body))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_json(response).await, decode_success());
    assert_eq!(harness.prefill.call_count(), 1);
    assert_eq!(harness.decode.call_count(), 1);
}

/// C02: the decode leg receives the caller's generation parameters and the
/// validated handoff.
#[tokio::test]
async fn c02_decode_receives_the_original_request_plus_the_handoff() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;
    let original = original_request();

    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original))
        .await
        .unwrap();

    let decode_body = harness.decode.last().json();
    assert_eq!(decode_body["stream"], json!(true));
    assert_eq!(decode_body["max_tokens"], json!(64));
    assert_eq!(decode_body["min_tokens"], json!(4));
    assert_eq!(
        decode_body["stream_options"],
        json!({"include_usage": true})
    );
    assert_eq!(decode_body["tools"], original["tools"]);
    assert_eq!(decode_body["messages"], original["messages"]);
    assert_eq!(decode_body["kv_transfer_params"], handoff());
}

/// C03: a streaming P/D request keeps the decode leg streaming and never leaks
/// the prefill leg's generated token to the caller.
#[tokio::test]
async fn c03_pd_streaming_success_keeps_decode_streaming() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Sse(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n".to_string(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ]),
    )
    .await;

    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"],
        "text/event-stream",
        "the decode content type must survive"
    );
    let text = String::from_utf8(body_bytes(response).await.to_vec()).unwrap();
    assert!(text.contains("\"he\""), "got:\n{text}");
    assert!(text.contains("\"llo\""), "got:\n{text}");
    assert!(text.ends_with("data: [DONE]\n\n"), "got:\n{text}");
    assert!(
        !text.contains("PRE-FILL-TOKEN"),
        "the prefill leg's token must never reach the caller"
    );

    let prefill_body = harness.prefill.last().json();
    assert_eq!(
        prefill_body["stream"],
        json!(false),
        "the prefill leg must be non-streaming"
    );
    assert_eq!(harness.decode.last().json()["stream"], json!(true));
}

// ---------------------------------------------------------------------------
// C04-C08: request derivation and pre-flight rejection
// ---------------------------------------------------------------------------

/// C04: `max_tokens` and `max_completion_tokens` combinations.
#[tokio::test]
async fn c04_max_tokens_combinations() {
    for (original, expect_completion) in [
        (json!({"model": "m", "max_tokens": 64}), None),
        (json!({"model": "m"}), None),
        (json!({"model": "m", "max_completion_tokens": 99}), Some(1)),
        (
            json!({"model": "m", "max_tokens": 8, "max_completion_tokens": 99}),
            Some(1),
        ),
    ] {
        let harness = Harness::build(
            Behaviour::Json(prefill_success()),
            Behaviour::Json(decode_success()),
        )
        .await;

        harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original))
            .await
            .unwrap();

        let prefill_body = harness.prefill.last().json();
        assert_eq!(prefill_body["max_tokens"], json!(1), "for {original}");
        assert_eq!(
            prefill_body.get("max_completion_tokens").cloned(),
            expect_completion.map(Value::from),
            "for {original}"
        );

        let decode_body = harness.decode.last().json();
        assert_eq!(
            decode_body.get("max_tokens").cloned(),
            original.get("max_tokens").cloned(),
            "decode keeps the caller's max_tokens for {original}"
        );
        assert_eq!(
            decode_body.get("max_completion_tokens").cloned(),
            original.get("max_completion_tokens").cloned(),
            "for {original}"
        );
    }
}

/// C05: `min_tokens`, `min_completion_tokens`, and `stream_options` are removed
/// from the prefill leg only.
#[tokio::test]
async fn c05_min_tokens_and_stream_options_are_prefill_only() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut original = original_request();
    original["min_completion_tokens"] = json!(3);
    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original))
        .await
        .unwrap();

    let prefill_body = harness.prefill.last().json();
    assert!(prefill_body.get("min_tokens").is_none());
    assert!(prefill_body.get("min_completion_tokens").is_none());
    assert!(prefill_body.get("stream_options").is_none());

    let decode_body = harness.decode.last().json();
    assert_eq!(decode_body["min_tokens"], json!(4));
    assert_eq!(decode_body["min_completion_tokens"], json!(3));
    assert_eq!(
        decode_body["stream_options"],
        json!({"include_usage": true})
    );
}

/// C06: unknown fields and tool definitions are not lossily converted.
#[tokio::test]
async fn c06_unknown_fields_and_tools_survive_both_legs() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut original = original_request();
    original["response_format"] = json!({"type": "json_object"});
    original["nvext"] = json!({"agent_hints": {"osl": 128}});
    original["future_vendor_field"] = json!({"deeply": {"nested": [1, 2, 3]}});

    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original))
        .await
        .unwrap();

    for body in [harness.prefill.last().json(), harness.decode.last().json()] {
        assert_eq!(body["response_format"], original["response_format"]);
        assert_eq!(body["nvext"], original["nvext"]);
        assert_eq!(body["future_vendor_field"], original["future_vendor_field"]);
        assert_eq!(body["tools"], original["tools"]);
    }
}

/// C07: `n != 1` is rejected before either leg, not silently rewritten.
#[tokio::test]
async fn c07_n_greater_than_one_is_rejected_before_any_leg() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut original = original_request();
    original["n"] = json!(3);
    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error_code(&body_json(response).await),
        "unsupported_pd_variant"
    );
    assert_eq!(harness.prefill.call_count(), 0);
    assert_eq!(harness.decode.call_count(), 0);
}

/// C08: a client-injected handoff is rejected; null and empty are tolerated.
#[tokio::test]
async fn c08_client_injected_handoff_is_rejected_but_empty_is_tolerated() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut injected = original_request();
    injected["kv_transfer_params"] = json!({
        "do_remote_prefill": true,
        "remote_block_ids": [[999]],
        "remote_engine_id": "attacker",
        "remote_request_id": "x",
        "remote_host": "evil",
        "remote_port": 1
    });
    let response = harness
        .app
        .clone()
        .oneshot(pd_request(harness.prefill.address(), &injected))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body_json(response).await), "invalid_pd_request");
    assert_eq!(
        harness.prefill.call_count(),
        0,
        "no prefill for an injected handoff"
    );
    assert_eq!(harness.decode.call_count(), 0);

    for tolerated in [Value::Null, json!({})] {
        let mut original = original_request();
        original["kv_transfer_params"] = tolerated;
        let response = harness
            .app
            .clone()
            .oneshot(pd_request(harness.prefill.address(), &original))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}

// ---------------------------------------------------------------------------
// C09-C15: handoff validation
// ---------------------------------------------------------------------------

/// C09: a missing, null, or non-object handoff is a protocol error and decode
/// is never called.
#[tokio::test]
async fn c09_missing_null_or_non_object_handoff_never_reaches_decode() {
    for response in [
        json!({"choices": [], "usage": {}}),
        json!({"choices": [], "kv_transfer_params": null}),
        json!({"choices": [], "kv_transfer_params": "pull"}),
        json!([1, 2, 3]),
    ] {
        let harness = Harness::build(
            Behaviour::Json(response.clone()),
            Behaviour::Json(decode_success()),
        )
        .await;

        let result = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();

        assert_eq!(result.status(), StatusCode::BAD_GATEWAY, "for {response}");
        assert_eq!(
            error_code(&body_json(result).await),
            "invalid_prefill_handoff",
            "for {response}"
        );
        assert_eq!(
            harness.decode.call_count(),
            0,
            "decode must not run for {response}"
        );
    }
}

/// C10: field-level handoff defects are locatable protocol errors.
#[tokio::test]
async fn c10_field_level_handoff_errors_never_reach_decode() {
    let mut cases: Vec<(String, Value)> = Vec::new();
    for field in [
        "remote_engine_id",
        "remote_request_id",
        "remote_host",
        "remote_port",
        "remote_block_ids",
    ] {
        let mut broken = handoff();
        broken.as_object_mut().unwrap().remove(field);
        cases.push((format!("missing {field}"), broken));
    }
    let mut wrong_direction = handoff();
    wrong_direction["do_remote_prefill"] = json!(false);
    cases.push(("do_remote_prefill=false".to_string(), wrong_direction));
    let mut producer_flag = handoff();
    producer_flag["do_remote_decode"] = json!(true);
    cases.push(("do_remote_decode=true".to_string(), producer_flag));
    let mut missing_decode = handoff();
    missing_decode
        .as_object_mut()
        .unwrap()
        .remove("do_remote_decode");
    cases.push(("missing do_remote_decode".to_string(), missing_decode));
    let mut bad_port = handoff();
    bad_port["remote_port"] = json!(0);
    cases.push(("remote_port=0".to_string(), bad_port));
    let mut huge_port = handoff();
    huge_port["remote_port"] = json!(70000);
    cases.push(("remote_port=70000".to_string(), huge_port));
    let mut bad_blocks = handoff();
    bad_blocks["remote_block_ids"] = json!([[1, "two"]]);
    cases.push(("non-integer block id".to_string(), bad_blocks));
    let mut flat_blocks = handoff();
    flat_blocks["remote_block_ids"] = json!([1, 2, 3]);
    cases.push(("flat block ids".to_string(), flat_blocks));
    let mut empty_engine = handoff();
    empty_engine["remote_engine_id"] = json!("");
    cases.push(("empty remote_engine_id".to_string(), empty_engine));

    for (name, broken) in cases {
        let harness = Harness::build(
            Behaviour::Json(json!({"choices": [], "kv_transfer_params": broken})),
            Behaviour::Json(decode_success()),
        )
        .await;

        let response = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{name}");
        assert_eq!(
            error_code(&body_json(response).await),
            "invalid_prefill_handoff",
            "{name}"
        );
        assert_eq!(harness.decode.call_count(), 0, "{name}");
    }
}

/// C11: grouped block ids keep their nesting.
#[tokio::test]
async fn c11_nested_block_groups_are_preserved() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();

    let sent = harness.decode.last().json();
    assert_eq!(
        sent["kv_transfer_params"]["remote_block_ids"],
        json!([[1, 2, 3], [7]])
    );
    assert!(
        sent["kv_transfer_params"]["remote_block_ids"][0].is_array(),
        "the group nesting must not be flattened"
    );
}

/// C12: legally empty block groups are a valid handoff, not a missing one.
#[tokio::test]
async fn c12_legally_empty_block_groups_are_not_a_missing_handoff() {
    for blocks in [json!([]), json!([[]]), json!([[], []])] {
        let mut valid = handoff();
        valid["remote_block_ids"] = blocks.clone();
        let harness = Harness::build(
            Behaviour::Json(json!({"choices": [], "kv_transfer_params": valid})),
            Behaviour::Json(decode_success()),
        )
        .await;

        let response = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK, "blocks={blocks}");
        assert_eq!(harness.decode.call_count(), 1, "blocks={blocks}");
        assert_eq!(
            harness.decode.last().json()["kv_transfer_params"]["remote_block_ids"],
            blocks,
            "the empty structure must survive verbatim"
        );
    }
}

/// C13: extra connector fields reach the decode worker intact, and nothing is
/// rebuilt or dropped.
#[tokio::test]
async fn c13_extra_handoff_fields_reach_decode_intact() {
    let mut extended = handoff();
    extended["future_connector_field"] = json!({"nested": [1, 2]});
    extended["tp_size"] = json!(4);
    extended["pp_size"] = json!(2);
    extended["dcp_size"] = json!(8);
    extended["remote_num_tokens"] = json!(777);
    extended["remote_blocks_expiry_time"] = json!(1234.5);
    extended["transfer_mode"] = json!("push");

    let harness = Harness::build(
        Behaviour::Json(json!({"choices": [], "kv_transfer_params": extended.clone()})),
        Behaviour::Json(decode_success()),
    )
    .await;

    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(
        harness.decode.last().json()["kv_transfer_params"],
        extended,
        "the backend metadata must arrive verbatim"
    );
}

/// C14: the connector's own request id is forwarded unchanged and is not
/// conflated with the HTTP correlation header.
#[tokio::test]
async fn c14_backend_request_id_is_forwarded_unchanged() {
    let mut backend_handoff = handoff();
    backend_handoff["remote_request_id"] = json!("backend-issued-id");

    let harness = Harness::build(
        Behaviour::Json(json!({"choices": [], "kv_transfer_params": backend_handoff})),
        Behaviour::Json(decode_success()),
    )
    .await;

    harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();

    let sent = harness.decode.last();
    assert_eq!(
        sent.json()["kv_transfer_params"]["remote_request_id"],
        json!("backend-issued-id"),
        "the connector's request id must not be rewritten"
    );
    assert_eq!(
        sent.headers["x-request-id"], "client-request-id",
        "the HTTP correlation header is a separate concern"
    );
}

/// C15: `remote_host`/`remote_port` are side-channel metadata and never change
/// the decode HTTP destination.
#[tokio::test]
async fn c15_remote_host_and_port_do_not_change_the_decode_destination() {
    let mut remote = handoff();
    remote["remote_host"] = json!("192.0.2.99");
    remote["remote_port"] = json!(9);

    let harness = Harness::build(
        Behaviour::Json(json!({"choices": [], "kv_transfer_params": remote})),
        Behaviour::Json(decode_success()),
    )
    .await;

    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        harness.decode.call_count(),
        1,
        "the request must land on the configured decode worker"
    );
}

// ---------------------------------------------------------------------------
// C16-C21: error boundaries
// ---------------------------------------------------------------------------

/// C16: a prefill HTTP error stops the request and its body is not relayed.
#[tokio::test]
async fn c16_prefill_http_error_stops_before_decode() {
    for status in [
        StatusCode::BAD_REQUEST,
        StatusCode::UNAUTHORIZED,
        StatusCode::INTERNAL_SERVER_ERROR,
    ] {
        let harness = Harness::build(
            Behaviour::Status(
                status,
                json!({"error": {"message": "prefill refused", "code": "boom"}}),
            ),
            Behaviour::Json(decode_success()),
        )
        .await;

        let response = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            status,
            "the upstream status is preserved"
        );
        let body = body_json(response).await;
        assert_eq!(error_code(&body), "prefill_upstream_error");
        assert!(
            !body.to_string().contains("prefill refused"),
            "the upstream body may quote the request and must not be relayed"
        );
        assert_eq!(
            harness.decode.call_count(),
            0,
            "decode must not run after a {status}"
        );
    }
}

/// C17: a non-JSON, truncated, or SSE prefill response is a protocol error and
/// decode is never called.
#[tokio::test]
async fn c17_non_json_prefill_responses_stop_before_decode() {
    for (name, behaviour) in [
        ("non-json", Behaviour::Raw("not json at all")),
        (
            "sse",
            Behaviour::Sse(vec![
                "data: {\"choices\":[]}\n\n".to_string(),
                "data: [DONE]\n\n".to_string(),
            ]),
        ),
    ] {
        let harness = Harness::build(behaviour, Behaviour::Json(decode_success())).await;

        let response = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{name}");
        assert_eq!(
            error_code(&body_json(response).await),
            "invalid_prefill_handoff",
            "{name}"
        );
        assert_eq!(harness.decode.call_count(), 0, "{name}");
    }
}

/// A peer that stalls mid-body is an upstream timeout (504), not a malformed
/// handoff (502). The fixture promises a larger Content-Length than it delivers,
/// so the read is parked inside the body with bytes still outstanding.
#[tokio::test]
async fn c17b_prefill_mid_body_stall_is_an_upstream_timeout() {
    let (app, prefill, decode) = Harness::with_timeouts(
        Behaviour::PartialBodyThenHang(handoff()),
        // A local connect is immediate, so only the read timeout can fire here.
        std::time::Duration::from_secs(10),
        std::time::Duration::from_millis(500),
    )
    .await;

    let response = app
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_upstream_timeout"
    );
    assert_eq!(
        decode.call_count(),
        0,
        "a stalled prefill must not be followed by a decode dispatch"
    );
}

/// A peer that accepts the request, sends response headers, and then produces no
/// body at all is the same 504 case.
#[tokio::test]
async fn c17c_prefill_header_then_body_stall_is_an_upstream_timeout() {
    let (app, prefill, decode) = Harness::with_timeouts(
        Behaviour::HeadersThenStall,
        std::time::Duration::from_secs(10),
        std::time::Duration::from_millis(500),
    )
    .await;

    let response = app
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_upstream_timeout"
    );
    assert_eq!(decode.call_count(), 0);
}

/// The counterpart: a malformed body that ends immediately is a protocol error
/// (502), not a timeout. Timeout classification must not swallow this case.
#[tokio::test]
async fn c17d_malformed_prefill_body_that_ends_immediately_is_a_protocol_error() {
    let (app, prefill, decode) = Harness::with_timeouts(
        Behaviour::TruncatedJson,
        std::time::Duration::from_secs(10),
        // Much longer than an immediate end-of-body needs, so a misclassified
        // read failure cannot pass this case by timing out instead.
        std::time::Duration::from_secs(5),
    )
    .await;

    let response = app
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        error_code(&body_json(response).await),
        "invalid_prefill_handoff"
    );
    assert_eq!(decode.call_count(), 0);
}

/// A peer that keeps producing small chunks must not extend the prefill leg
/// indefinitely. The read timeout bounds one gap between reads, so each gap here
/// stays under it and only the total deadline can end the leg.
#[tokio::test]
async fn c17e_prefill_leg_deadline_bounds_a_chatty_peer() {
    let prefill = Fake::start(Behaviour::ChunkEveryMillis(20)).await;
    let decode = Fake::start(Behaviour::Json(decode_success())).await;
    let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
        decode_url(&decode),
        Duration::from_secs(10),
        // Longer than any gap this peer leaves between chunks.
        Duration::from_millis(500),
        vllm_nixl::Config {
            prefill_deadline: Duration::from_millis(400),
            ..vllm_nixl::Config::default()
        },
    )
    .unwrap();
    let state = SidecarState::new(
        decode_url(&decode),
        Duration::from_secs(10),
        Duration::from_secs(10),
        adapter,
        CancellationToken::new(),
        CancellationToken::new(),
    )
    .unwrap();

    let started = std::time::Instant::now();
    let response = router(state)
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();
    let elapsed = started.elapsed();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_deadline_exceeded",
        "a chatty peer must hit the leg deadline, not the per-read timeout"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "the leg must end near its deadline, took {elapsed:?}"
    );
    assert_eq!(
        decode.call_count(),
        0,
        "a prefill leg that never finishes must not be followed by a decode dispatch"
    );
}

/// A client that sends an under-cap body slowly must not hold the handler open.
/// The configured upstream read timeout does not cover the client body, so this
/// deadline is the only bound on a partial-body client.
#[tokio::test]
async fn c18b_client_body_deadline_rejects_a_dribbling_client() {
    let decode = Fake::start(Behaviour::Json(decode_success())).await;
    let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
        decode_url(&decode),
        Duration::from_secs(10),
        Duration::from_secs(10),
        vllm_nixl::Config {
            client_body_timeout: Duration::from_millis(300),
            ..vllm_nixl::Config::default()
        },
    )
    .unwrap();
    let app = router(
        SidecarState::new(
            decode_url(&decode),
            Duration::from_secs(10),
            Duration::from_secs(10),
            adapter,
            CancellationToken::new(),
            CancellationToken::new(),
        )
        .unwrap(),
    );

    // A real socket, because a slow client is a transport behaviour: send the
    // head and a partial body, then stop without finishing the request.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let sidecar = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let mut socket = tokio::net::TcpStream::connect(sidecar).await.unwrap();
    use tokio::io::AsyncWriteExt;
    socket
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\n\
              Host: sidecar\r\n\
              x-prefiller-host-port: 127.0.0.1:1\r\n\
              Content-Type: application/json\r\n\
              Content-Length: 100000\r\n\r\n\
              {\"model\":",
        )
        .await
        .unwrap();

    // Read the response without ever completing the body.
    use tokio::io::AsyncReadExt;
    let mut buf = vec![0u8; 4096];
    let read = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buf))
        .await
        .expect("the deadline must produce a response, not a hang")
        .unwrap();
    let response = String::from_utf8_lossy(&buf[..read]);

    assert!(
        response.starts_with("HTTP/1.1 408"),
        "a stalled request body must time out, got: {response}"
    );
    assert!(
        response.contains("pd_request_timeout"),
        "expected the request-timeout code, got: {response}"
    );
    assert_eq!(
        decode.call_count(),
        0,
        "a body that never arrived must not reach a backend leg"
    );
}

/// C18: request and prefill-response buffering are both bounded.
#[tokio::test]
async fn c18_oversized_request_and_prefill_response_are_rejected_with_bounds() {
    let harness = Harness::build_with_limits(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
        256,
        512,
    )
    .await;

    let mut huge = original_request();
    huge["messages"] = json!([{"role": "user", "content": "x".repeat(4096)}]);
    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &huge))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        error_code(&body_json(response).await),
        "pd_request_too_large"
    );
    assert_eq!(
        harness.prefill.call_count(),
        0,
        "an oversized request must not reach the prefill worker"
    );

    let harness = Harness::build_with_limits(
        Behaviour::Oversized(64 * 1024),
        Behaviour::Json(decode_success()),
        1024 * 1024,
        512,
    )
    .await;
    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_response_too_large"
    );
    assert_eq!(harness.decode.call_count(), 0);
}

/// C19: prefill connect and read failures are prefill-stage errors, never
/// reported as decode failures.
#[tokio::test]
async fn c19_prefill_connect_and_read_failures_are_prefill_stage_errors() {
    // Nothing listening on the selected prefill endpoint.
    let dead = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_address = dead.local_addr().unwrap();
    drop(dead);

    let harness = Harness::build(Behaviour::Hang, Behaviour::Json(decode_success())).await;
    let decode = harness.decode.clone();
    let response = harness
        .app
        .oneshot(pd_request(dead_address, &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = body_json(response).await;
    assert_eq!(error_code(&body), "prefill_upstream_unavailable");
    assert!(
        !body["error"]["message"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("decode"),
        "a prefill failure must not be described as a decode failure"
    );
    assert_eq!(decode.call_count(), 0);

    // A prefill worker that accepts the request and never answers.
    let (app, prefill, decode) = Harness::with_timeouts(
        Behaviour::Hang,
        std::time::Duration::from_millis(200),
        std::time::Duration::from_millis(300),
    )
    .await;

    let response = app
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_upstream_timeout"
    );
    assert_eq!(decode.call_count(), 0);
}

/// C20: decode connect failure and decode HTTP status are surfaced, not
/// swallowed.
#[tokio::test]
async fn c20_decode_connect_error_and_status_are_not_swallowed() {
    let dead = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_address = dead.local_addr().unwrap();
    drop(dead);

    let prefill = Fake::start(Behaviour::Json(prefill_success())).await;
    let decode = Fake::start(Behaviour::Json(decode_success())).await;
    let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
        format!("http://{dead_address}").parse().unwrap(),
        Duration::from_secs(2),
        Duration::from_secs(2),
        vllm_nixl::Config::default(),
    )
    .unwrap();
    let state = SidecarState::new(
        format!("http://{dead_address}").parse().unwrap(),
        Duration::from_secs(2),
        Duration::from_secs(2),
        adapter,
        CancellationToken::new(),
        CancellationToken::new(),
    )
    .unwrap();
    drop(decode);

    let response = router(state)
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        error_code(&body_json(response).await),
        "decode_upstream_error"
    );

    // A decode HTTP error status and its OpenAI error body are forwarded.
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Status(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": {"message": "decode busy", "code": "rate_limited"}}),
        ),
    )
    .await;
    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body_json(response).await["error"]["code"], "rate_limited");
}

/// C21: the sidecar relays exactly the decode bytes and never fabricates a
/// success terminator.
#[tokio::test]
async fn c21_decode_body_is_relayed_verbatim_without_a_fake_terminator() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_bytes(response).await;
    assert_eq!(body, decode_success().to_string());
    assert!(
        !String::from_utf8_lossy(&body).contains("[DONE]"),
        "the sidecar must not synthesize a stream terminator"
    );
}

// ---------------------------------------------------------------------------
// C22-C27: cancellation, streaming, concurrency
// ---------------------------------------------------------------------------

/// C22: a client that goes away while the prefill leg is pending must not
/// trigger a decode dispatch.
#[tokio::test]
async fn c22_cancellation_during_pending_prefill_prevents_decode() {
    // The prefill handler records the request, then never answers.
    let harness = Harness::build(Behaviour::Hang, Behaviour::Json(decode_success())).await;
    let prefill = harness.prefill.clone();
    let decode = harness.decode.clone();

    let app = harness.app.clone();
    let address = prefill.address();
    let request = pd_request(address, &original_request());
    let task = tokio::spawn(async move { app.oneshot(request).await });

    // Barrier: wait until the prefill leg has actually been entered.
    let mut spins = 0;
    while prefill.call_count() == 0 {
        assert!(spins < 2_000, "the prefill leg never started");
        spins += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    // Give any orphaned continuation a chance to misbehave before asserting.
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(
        decode.call_count(),
        0,
        "a cancelled prefill must not be followed by a decode dispatch"
    );
}

/// C23: a client that goes away after the prefill response has been accepted
/// but before the decode leg is dispatched must not trigger a decode.
///
/// The prefill worker is parked mid-body, so the adapter is provably inside
/// the handoff read when the task is aborted.
#[tokio::test]
async fn c23_cancellation_after_partial_handoff_prevents_decode() {
    let harness = Harness::build(
        Behaviour::PartialBodyThenHang(handoff()),
        Behaviour::Json(decode_success()),
    )
    .await;
    let prefill = harness.prefill.clone();
    let decode = harness.decode.clone();

    let app = harness.app.clone();
    let request = pd_request(prefill.address(), &original_request());
    let task = tokio::spawn(async move { app.oneshot(request).await });

    let mut spins = 0;
    while prefill.call_count() == 0 {
        assert!(spins < 2_000, "the prefill leg never started");
        spins += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(
        decode.call_count(),
        0,
        "aborting inside the handoff read must not produce a decode dispatch"
    );
}

/// C24: an abort while the decode leg is pending ends the request.
#[tokio::test]
async fn c24_cancellation_during_pending_decode_ends_the_request() {
    let harness = Harness::build(Behaviour::Json(prefill_success()), Behaviour::Hang).await;
    let prefill = harness.prefill.clone();
    let decode = harness.decode.clone();

    let app = harness.app.clone();
    let request = pd_request(prefill.address(), &original_request());
    let task = tokio::spawn(async move { app.oneshot(request).await });

    let mut spins = 0;
    while decode.call_count() == 0 {
        assert!(spins < 2_000, "the decode leg never started");
        spins += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(prefill.call_count(), 1);
    assert_eq!(decode.call_count(), 1);
}

/// C24: force shutdown interrupts a pending decode dispatch.
#[tokio::test]
async fn c24_force_shutdown_interrupts_a_pending_decode_dispatch() {
    let prefill = Fake::start(Behaviour::Json(prefill_success())).await;
    let decode = Fake::start(Behaviour::Hang).await;
    let force_shutdown = CancellationToken::new();

    let adapter: Arc<dyn PdAdapter> = VllmNixlAdapter::new(
        decode_url(&decode),
        Duration::from_secs(30),
        Duration::from_secs(30),
        vllm_nixl::Config::default(),
    )
    .unwrap();
    let state = SidecarState::new(
        decode_url(&decode),
        Duration::from_secs(30),
        Duration::from_secs(30),
        adapter,
        CancellationToken::new(),
        force_shutdown.clone(),
    )
    .unwrap();
    let app = router(state);

    let task = tokio::spawn(async move {
        app.oneshot(pd_request(prefill.address(), &original_request()))
            .await
    });

    let mut spins = 0;
    while decode.call_count() == 0 {
        assert!(spins < 2_000, "the decode leg never started");
        spins += 1;
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    force_shutdown.cancel();

    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("force shutdown must interrupt a pending decode dispatch")
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(error_code(&body_json(response).await), "request_cancelled");
}

/// C25: dropping a P/D response body ends the caller's stream cleanly.
#[tokio::test]
async fn c25_downstream_drop_ends_a_pd_stream() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Sse(vec!["data: one\n\n".to_string()]),
    )
    .await;

    let response = harness
        .app
        .oneshot(pd_request(harness.prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let mut stream = response.into_body().into_data_stream();
    assert_eq!(stream.next().await.unwrap().unwrap(), "data: one\n\n");
    drop(stream);
    assert_eq!(harness.decode.call_count(), 1);
}

/// C26: SSE byte content is independent of how the upstream splits chunks, and
/// each chunk is delivered as it arrives.
#[tokio::test]
async fn c26_sse_chunk_boundaries_do_not_change_the_byte_stream() {
    let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"abc\"}}]}\n\ndata: [DONE]\n\n";
    let split_first = "data: {\"choices\":[{\"delta\":{\"content\":\"abc\"}}]}\n\n";
    let split_second = "data: [DONE]\n\n";

    for chunks in [
        vec![payload.to_string()],
        vec![split_first.to_string(), split_second.to_string()],
    ] {
        let harness = Harness::build(
            Behaviour::Json(prefill_success()),
            Behaviour::Sse(chunks.clone()),
        )
        .await;

        let response = harness
            .app
            .oneshot(pd_request(harness.prefill.address(), &original_request()))
            .await
            .unwrap();
        assert_eq!(
            String::from_utf8(body_bytes(response).await.to_vec()).unwrap(),
            payload,
            "chunking {chunks:?} must not change the bytes"
        );
    }
}

/// C27: concurrent requests do not cross handoffs or request identities, even
/// when they share a client-supplied request id.
#[tokio::test]
async fn c27_concurrent_requests_do_not_cross_handoffs() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;
    let app = harness.app.clone();
    let address = harness.prefill.address();

    let mut tasks = Vec::new();
    for index in 0..8 {
        let app = app.clone();
        let mut body = original_request();
        // Same client request id for every stream, as a misbehaving client would
        // send; the handoff must still not cross.
        body["user_marker"] = json!(index);
        tasks.push(tokio::spawn(async move {
            app.oneshot(pd_request(address, &body)).await
        }));
    }

    for task in tasks {
        let response = task.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    assert_eq!(harness.prefill.call_count(), 8);
    assert_eq!(harness.decode.call_count(), 8);

    let mut markers = Vec::new();
    for captured in harness.decode.captured() {
        let body = captured.json();
        assert_eq!(
            body["kv_transfer_params"],
            handoff(),
            "every decode request must carry the validated handoff"
        );
        assert_eq!(captured.headers["x-request-id"], "client-request-id");
        markers.push(body["user_marker"].as_i64().unwrap());
    }
    markers.sort_unstable();
    assert_eq!(markers, (0..8).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------
// C28-C30: endpoints, headers, configuration
// ---------------------------------------------------------------------------

/// C28: the endpoint may be an IPv6 literal, and the request path and query are
/// preserved on both legs.
#[tokio::test]
async fn c28_ipv6_endpoint_and_query_and_base_path_are_preserved() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut request = pd_request(harness.prefill.address(), &original_request());
    *request.uri_mut() = "/v1/chat/completions?trace=1&stream=true".parse().unwrap();
    harness.app.clone().oneshot(request).await.unwrap();

    assert_eq!(
        harness.prefill.last().uri.to_string(),
        "/v1/chat/completions?trace=1&stream=true"
    );
    assert_eq!(
        harness.decode.last().uri.to_string(),
        "/v1/chat/completions?trace=1&stream=true"
    );

    // A closed loopback port, so the failure is a connection refusal on any
    // runner. A documentation-prefix address is not portable: where IPv6 is
    // routable it black-holes until the timeout and reports 504 instead.
    let Ok(listener) = TcpListener::bind("[::1]:0").await else {
        // No IPv6 loopback here; the IPv6-specific subcase does not apply.
        return;
    };
    // Let the os pick the port and release it, so this is a port that is closed
    // rather than a documentation prefix that may or may not be routable here.
    let closed = listener.local_addr().unwrap();
    drop(listener);

    let mut request = pd_request(harness.prefill.address(), &original_request());
    request
        .headers_mut()
        .insert(PREFILLER_HOST_PORT, closed.to_string().parse().unwrap());
    let response = harness.app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_GATEWAY,
        "a refused IPv6 connection is an unavailable upstream, not a timeout"
    );
    assert_eq!(
        error_code(&body_json(response).await),
        "prefill_upstream_unavailable"
    );
}

/// C29: hop-by-hop headers are stripped precisely; authorization and routing
/// headers survive on both legs.
#[tokio::test]
async fn c29_hop_by_hop_headers_are_stripped_and_semantic_headers_kept() {
    let harness = Harness::build(
        Behaviour::Json(prefill_success()),
        Behaviour::Json(decode_success()),
    )
    .await;

    let mut request = pd_request(harness.prefill.address(), &original_request());
    request
        .headers_mut()
        .insert("connection", "keep-alive, x-nominated".parse().unwrap());
    request
        .headers_mut()
        .insert("x-nominated", "drop-me".parse().unwrap());
    request
        .headers_mut()
        .insert("keep-alive", "timeout=5".parse().unwrap());
    request
        .headers_mut()
        .insert("te", "trailers".parse().unwrap());
    request
        .headers_mut()
        .insert("transfer-encoding", "chunked".parse().unwrap());
    request
        .headers_mut()
        .insert("x-custom", "preserved".parse().unwrap());

    harness.app.oneshot(request).await.unwrap();

    for (leg, captured) in [
        ("prefill", harness.prefill.last()),
        ("decode", harness.decode.last()),
    ] {
        assert_eq!(
            captured.method,
            Method::POST,
            "{leg} must reach the worker as a POST"
        );
        assert!(!captured.headers.contains_key("connection"), "{leg}");
        assert!(!captured.headers.contains_key("x-nominated"), "{leg}");
        assert!(!captured.headers.contains_key("keep-alive"), "{leg}");
        assert!(!captured.headers.contains_key("te"), "{leg}");
        assert!(!captured.headers.contains_key("transfer-encoding"), "{leg}");
        assert!(
            !captured.headers.contains_key(PREFILLER_HOST_PORT),
            "{leg} must not forward EPP metadata"
        );
        assert_eq!(captured.headers["x-custom"], "preserved", "{leg}");
        assert_eq!(captured.headers["authorization"], "Bearer secret", "{leg}");
        assert_eq!(
            captured.headers["x-gateway-destination-endpoint"], "decode:8000",
            "{leg}"
        );
        assert_eq!(
            captured.headers["content-type"], "application/json",
            "{leg}"
        );
    }
}

/// C30: the adapter is off by default, an invalid configuration is rejected,
/// and the protocol revision is asserted.
#[tokio::test]
async fn c30_disabled_adapter_and_protocol_assertion() {
    let prefill = Fake::start(Behaviour::Json(prefill_success())).await;
    let decode = Fake::start(Behaviour::Json(decode_success())).await;

    // With no adapter configured, a P/D request keeps failing explicitly.
    let state = SidecarState::new(
        decode_url(&decode),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Arc::new(UnavailablePdAdapter),
        CancellationToken::new(),
        CancellationToken::new(),
    )
    .unwrap();
    let app = router(state);

    let response = app
        .clone()
        .oneshot(pd_request(prefill.address(), &original_request()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    assert_eq!(
        error_code(&body_json(response).await),
        "pd_adapter_unavailable"
    );
    assert_eq!(prefill.call_count(), 0);
    assert_eq!(decode.call_count(), 0);

    // The no-header path still passes through with the adapter disabled.
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .body(Body::from("passthrough"))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(decode.last().body, "passthrough");

    // The protocol revision is pinned; anything else is refused at startup.
    assert!(VllmNixlAdapter::check_protocol_version(vllm_nixl::SUPPORTED_PROTOCOL_VERSION).is_ok());
    assert!(VllmNixlAdapter::check_protocol_version("vllm-v0.28.0-nixl-pull").is_err());
    assert_eq!(vllm_nixl::SUPPORTED_VLLM_VERSION, "v0.29.0");
}
