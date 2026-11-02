// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    body::Body,
    http::{HeaderMap, Request, StatusCode},
    response::Response,
};
use dynamo_llm::generate_format::{ClientRequest, SharedFormat};
use dynamo_llm::http::service::{format_generate, service_v2};
use dynamo_llm::protocols::{
    Annotated,
    common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};
use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Error, ManyOut, SingleIn};
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;
type Input = SingleIn<PreprocessedRequest>;
type Output = ManyOut<Annotated<LLMEngineOutput>>;
const CANCELLED: StatusCode = match StatusCode::from_u16(499) {
    Ok(status) => status,
    Err(_) => unreachable!(),
};

struct FailingEngine(dynamo_runtime::error::ErrorType);

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for FailingEngine {
    async fn generate(&self, _request: Input) -> Result<Output, Error> {
        let mut error = DynamoError::builder()
            .error_type(self.0)
            .message("private worker details");
        if self.0 == dynamo_runtime::error::ErrorType::DeadlineExceeded {
            error = error.reason(
                dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded").unwrap(),
            );
        }
        Err(error.build().into())
    }
}

use dynamo_llm::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
use dynamo_runtime::protocols::maybe_error::MaybeError;

fn sglang() -> SharedFormat {
    Arc::new(dynamo_sglang_generate::SglangFormat::native())
}

async fn respond(
    state: Arc<service_v2::State>,
    format: SharedFormat,
    engine: dynamo_llm::types::openai::generate::GenerateStreamingEngine,
    body: Value,
    mut headers: HeaderMap,
) -> Response {
    let mut card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
    card.runtime_config
        .runtime_data
        .insert(format.capability().into(), Value::Bool(true));
    let ws = dynamo_llm::discovery::WorkerSet::new("test".into(), "test".into(), card)
        .with_generate_engine(engine);
    assert!(state.manager().add_worker_set("test-model", "test", ws));
    let (_, router) = format_generate::router(state, None, format);
    let mut request = Request::post("/generate")
        .body(Body::from(body.to_string()))
        .unwrap();
    headers.insert("content-type", "application/json".parse().unwrap());
    *request.headers_mut() = headers;
    router.oneshot(request).await.unwrap()
}

fn state() -> Arc<service_v2::State> {
    service_v2::HttpService::builder()
        .build()
        .unwrap()
        .state_clone()
}

async fn json(response: Response) -> Value {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

struct ReplayEngine(Vec<Annotated<LLMEngineOutput>>, bool);

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for ReplayEngine {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        let stream = Box::pin(
            futures::stream::iter(self.0.clone())
                .chain(futures::stream::pending().take(usize::from(self.1))),
        );
        Ok(ResponseStream::new(stream, request.context()))
    }
}

/// A vLLM worker that generates `OUTPUT` one token per chunk and whose first
/// `failures` attempts drop after two chunks.
struct FlakyVllmWorker {
    calls: std::sync::atomic::AtomicUsize,
    failures: usize,
    prompt: usize,
}

const VLLM_OUTPUT: [u32; 5] = [10, 11, 12, 13, 14];

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for FlakyVllmWorker {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (request, context) = request.transfer(());
        // A retry's prompt is the client's prompt plus the delivered tokens.
        let delivered = request.token_ids.len() - self.prompt;
        assert_eq!(request.token_ids[self.prompt..], VLLM_OUTPUT[..delivered]);
        let budget = request.stop_conditions.max_tokens.unwrap() as usize;
        let sampling = &request.extra_args.as_ref().unwrap()["vllm_tito"]["sampling_params"];
        assert_eq!(sampling["max_tokens"], budget);
        let tokens = &VLLM_OUTPUT[delivered..(delivered + budget).min(VLLM_OUTPUT.len())];
        // Like the vLLM worker, prompt logprobs ride on the final chunk.
        let prompt_logprobs = request.output_options.prompt_logprobs.map(|_| {
            let mut entries = vec![Value::Null];
            entries.extend(request.token_ids[1..].iter().enumerate().map(|(i, token)| {
                serde_json::json!({ token.to_string(): {"logprob": -(i as f32) - 1.0} })
            }));
            serde_json::json!({ "prompt_logprobs": entries })
        });
        let mut items: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(i, &token)| {
                let last = i + 1 == tokens.len();
                Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![token],
                    finish_reason: last
                        .then_some(dynamo_llm::protocols::common::FinishReason::Length),
                    engine_data: prompt_logprobs.clone().filter(|_| last),
                    ..Default::default()
                })
            })
            .collect();
        if call < self.failures {
            items.truncate(2);
            items.push(Annotated::from_err(
                DynamoError::builder()
                    .error_type(dynamo_runtime::error::ErrorType::Backend(
                        dynamo_runtime::error::BackendError::StreamIncomplete,
                    ))
                    .message("engine died")
                    .build(),
            ));
        }
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            context.context(),
        ))
    }
}

struct MigratingVllm(Arc<dynamo_llm::migration::Migration>, Arc<FlakyVllmWorker>);

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for MigratingVllm {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        use dynamo_runtime::pipeline::Operator;
        self.0.generate(request, self.1.clone()).await
    }
}

#[tokio::test]
async fn a_vllm_request_migrated_mid_generation_returns_what_an_unbroken_one_does() {
    let respond = |failures: usize, sampling: Value| async move {
        let body = serde_json::json!({
            "token_ids": [1, 2, 3], "request_id": "request-id", "sampling_params": sampling,
        });
        let worker = Arc::new(FlakyVllmWorker {
            calls: Default::default(),
            failures,
            prompt: 3,
        });
        let card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
        let metrics = Arc::new(dynamo_llm::http::service::metrics::Metrics::new());
        let migration = dynamo_llm::migration::Migration::for_generate(&card, 3, None, metrics);
        let response = respond(
            state(),
            Arc::new(dynamo_vllm_generate::VllmFormat),
            Arc::new(MigratingVllm(migration, worker.clone())),
            body,
            HeaderMap::new(),
        )
        .await;
        let status = response.status();
        let calls = worker.calls.load(std::sync::atomic::Ordering::Relaxed);
        (status, json(response).await, calls)
    };
    let budget = serde_json::json!({"max_tokens": 5});
    let (status, unbroken, calls) = respond(0, budget.clone()).await;
    assert_eq!((status, calls), (StatusCode::OK, 1));
    assert_eq!(
        unbroken["choices"][0]["token_ids"],
        serde_json::json!(VLLM_OUTPUT)
    );
    let (status, migrated, calls) = respond(2, budget).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(calls, 3, "two failed attempts, then one that finishes");
    assert_eq!(migrated, unbroken);
    // Prompt logprobs arrive only with the final chunk, so the retry recomputes
    // them over a longer prompt and the client sees the original prompt's.
    let prompt = serde_json::json!({"max_tokens": 5, "prompt_logprobs": 1});
    let (status, unbroken, _) = respond(0, prompt.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unbroken["prompt_logprobs"].as_array().unwrap().len(), 3);
    let (status, migrated, calls) = respond(2, prompt).await;
    assert_eq!((status, calls), (StatusCode::OK, 3));
    assert_eq!(migrated, unbroken);
    // A seeded request cannot be continued with the same RNG state.
    let (status, _, calls) = respond(1, serde_json::json!({"max_tokens": 5, "seed": 7})).await;
    assert_eq!((status, calls), (StatusCode::INTERNAL_SERVER_ERROR, 1));
}

#[tokio::test]
async fn a_vllm_stream_migrated_mid_generation_streams_what_an_unbroken_one_does() {
    let respond = |failures: usize| async move {
        let body = serde_json::json!({
            "token_ids": [1, 2, 3], "request_id": "request-id", "stream": true,
            "stream_options": {"include_usage": true},
            "sampling_params": {"max_tokens": 5},
        });
        let worker = Arc::new(FlakyVllmWorker {
            calls: Default::default(),
            failures,
            prompt: 3,
        });
        let card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
        let metrics = Arc::new(dynamo_llm::http::service::metrics::Metrics::new());
        let migration = dynamo_llm::migration::Migration::for_generate(&card, 3, None, metrics);
        let response = respond(
            state(),
            Arc::new(dynamo_vllm_generate::VllmFormat),
            Arc::new(MigratingVllm(migration, worker.clone())),
            body,
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let calls = worker.calls.load(std::sync::atomic::Ordering::Relaxed);
        (String::from_utf8(body.to_vec()).unwrap(), calls)
    };
    let (unbroken, calls) = respond(0).await;
    assert_eq!(calls, 1);
    let events: Vec<Value> = unbroken
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let tokens: Vec<u64> = events
        .iter()
        .flat_map(|event| event["choices"].as_array().unwrap())
        .flat_map(|choice| choice["token_ids"].as_array().unwrap())
        .map(|token| token.as_u64().unwrap())
        .collect();
    assert_eq!(tokens, VLLM_OUTPUT.map(u64::from));
    assert_eq!(
        events.last().unwrap()["usage"],
        serde_json::json!({"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
    );
    assert!(unbroken.ends_with("data: [DONE]\n\n"));
    let (migrated, calls) = respond(2).await;
    assert_eq!(calls, 3, "two failed attempts, then one that finishes");
    assert_eq!(migrated, unbroken);
}

#[tokio::test]
async fn native_responses_record_cached_token_metrics() {
    struct MetricsEngine(Arc<prometheus::Registry>, LLMEngineOutput);

    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for MetricsEngine {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            let families = self.0.gather();
            for (name, expected) in [
                ("dynamo_frontend_awaiting_first_token_requests", 1.0),
                ("dynamo_frontend_inflight_input_tokens", 3.0),
            ] {
                let family = families
                    .iter()
                    .find(|family| family.name() == name)
                    .unwrap();
                let value: f64 = family
                    .get_metric()
                    .iter()
                    .map(|metric| metric.get_gauge().value())
                    .sum();
                assert_eq!(value, expected, "{name} before engine output");
            }
            ReplayEngine(vec![Annotated::from_data(self.1.clone())], false)
                .generate(request)
                .await
        }
    }
    for stream in [false, true] {
        let state = state();
        let registry = Arc::new(prometheus::Registry::new());
        state.metrics_clone().register(&registry).unwrap();
        let output = dynamo_sglang_generate::inspect(
            serde_json::json!({"output_ids": [7], "meta_info": {
                "finish_reason": {"type": "length"}, "prompt_tokens": 3,
                "completion_tokens": 1, "cached_tokens": 2,
            }}),
            false,
        )
        .unwrap()
        .output;
        let response = respond(
            state,
            sglang(),
            Arc::new(MetricsEngine(registry.clone(), output)),
            serde_json::json!({"input_ids": [1, 2, 3], "stream": stream}),
            HeaderMap::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let families = registry.gather();
        let cached = families
            .iter()
            .find(|family| family.name() == "dynamo_frontend_cached_tokens")
            .unwrap();
        let histogram = cached.get_metric()[0].get_histogram();
        assert_eq!(histogram.get_sample_count(), 1);
        assert_eq!(histogram.get_sample_sum(), 2.0);
        for (name, expected) in [
            ("dynamo_frontend_input_tokens_total", 3.0),
            ("dynamo_frontend_output_tokens_total", 1.0),
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .unwrap();
            assert_eq!(
                family.get_metric()[0].get_counter().value(),
                expected,
                "{name}"
            );
        }
        for name in [
            "dynamo_frontend_awaiting_first_token_requests",
            "dynamo_frontend_inflight_input_tokens",
        ] {
            let family = families
                .iter()
                .find(|family| family.name() == name)
                .unwrap();
            assert!(
                family
                    .get_metric()
                    .iter()
                    .all(|metric| metric.get_gauge().value() == 0.0),
                "{name} after completion"
            );
        }
    }
}

#[tokio::test]
async fn vllm_format_preserves_binding_terminal_status_and_wire_shape() {
    use dynamo_llm::protocols::common::FinishReason;
    for (finish, missing_logprobs, status) in [
        (FinishReason::Length, false, StatusCode::OK),
        (FinishReason::Cancelled, false, CANCELLED),
        (
            FinishReason::Length,
            true,
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            FinishReason::Error("private engine detail".into()),
            false,
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ] {
        let format: SharedFormat = Arc::new(dynamo_vllm_generate::VllmFormat);
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("x-request-id", "header-id".parse().unwrap());
        headers.insert("x-data-parallel-rank", "3".parse().unwrap());
        let body = serde_json::json!({
            "token_ids": [1], "sampling_params": if missing_logprobs {
                serde_json::json!({"logprobs": 1})
            } else { serde_json::json!({}) },
            "request_id": "body-id", "priority": -7,
        })
        .to_string();
        let opened = format
            .open(ClientRequest {
                body: body.as_bytes(),
                headers: &headers,
                default_request_id: "default-id",
            })
            .unwrap();
        assert_eq!(opened.request_id, "header-id");
        let bound = (opened.bind)(&dynamo_sglang_generate::testkit::worker()).unwrap();
        assert!(bound.migration.is_ok());
        let hints = bound.request.routing.as_ref().unwrap();
        assert_eq!(hints.dp_rank, Some(3));
        assert_eq!(hints.priority, Some(7));
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let output = LLMEngineOutput {
            token_ids: vec![2, 3],
            finish_reason: (!missing_logprobs).then_some(finish),
            ..Default::default()
        };
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            respond(
                state,
                format,
                Arc::new(ReplayEngine(
                    vec![Annotated::from_data(output)],
                    status != StatusCode::OK,
                )),
                serde_json::from_str(&body).unwrap(),
                headers,
            ),
        )
        .await
        .expect("invalid output must fail before the worker stream ends");
        assert_eq!(response.status(), status);
        let body = json(response).await;
        if status == StatusCode::OK {
            assert_eq!(body["request_id"], "header-id");
            assert_eq!(body["choices"][0]["token_ids"], serde_json::json!([2, 3]));
            assert_eq!(body["choices"][0]["finish_reason"], "length");
        } else {
            assert_eq!(body["error"]["code"], status.as_u16());
            assert!(!body.to_string().contains("private engine detail"));
        }
    }
}

#[tokio::test]
async fn unary_dispatch_joins_chunks_and_maps_stream_errors() {
    let chunk = |ids: Value| {
        Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": ids}})),
            ..Default::default()
        })
    };
    let invalid = DynamoError::builder()
        .error_type(dynamo_runtime::error::ErrorType::InvalidArgument)
        .message("bad sampling params")
        .build();
    for (outputs, status, body) in [
        (
            vec![chunk(serde_json::json!([5])), chunk(serde_json::json!([6]))],
            StatusCode::OK,
            serde_json::json!({"output_ids": [5, 6]}),
        ),
        (
            vec![chunk(serde_json::json!([5])), Annotated::from_err(invalid)],
            StatusCode::BAD_REQUEST,
            serde_json::json!({"error": {"message": "bad sampling params"}}),
        ),
        (
            vec![Annotated::from_data(LLMEngineOutput::default())],
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({"error": {"message": "Internal server error"}}),
        ),
    ] {
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            respond(
                state,
                sglang(),
                Arc::new(ReplayEngine(outputs, status != StatusCode::OK)),
                serde_json::json!({"input_ids":[1]}),
                HeaderMap::new(),
            ),
        )
        .await
        .expect("stream errors must fail before the worker stream ends");
        assert_eq!(response.status(), status);
        assert_eq!(json(response).await, body);
    }
}

#[tokio::test]
async fn custom_codec_renders_raw_and_decoded_request_errors() {
    use dynamo_sglang_generate::{ClientCodec, SglangFormat};

    struct Codec;
    impl ClientCodec for Codec {
        fn error_body(&self, message: &str) -> Value {
            serde_json::json!({"custom_failure": message})
        }
    }

    let format: SharedFormat = Arc::new(SglangFormat::with_codec("custom-errors", Codec));
    for body in [b"{".as_slice(), br#"{"input_ids":"invalid"}"#] {
        for (format, custom) in [(sglang(), false), (format.clone(), true)] {
            let (_, router) = format_generate::router(state(), None, format);
            let response = router
                .oneshot(
                    Request::post("/generate")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_vec()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = json(response).await;
            if !custom {
                assert_eq!(body["object"], "error");
                assert_eq!(body["code"], 400);
                continue;
            }
            assert!(
                body["custom_failure"]
                    .as_str()
                    .unwrap()
                    .starts_with("Failed to ")
            );
            assert_eq!(body.as_object().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn overlapping_public_ids_have_independent_cancellation_contexts() {
    use dynamo_runtime::pipeline::AsyncEngineContext;
    type Captured = Arc<dyn AsyncEngineContext>;
    struct Capture(tokio::sync::mpsc::UnboundedSender<Captured>);
    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for Capture {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            assert_eq!(
                request.content().extra_args.as_ref().unwrap()["sglang_tito"]["rid"],
                "same-public-id"
            );
            self.0.send(request.context()).unwrap();
            Ok(ResponseStream::new(
                Box::pin(futures::stream::pending()),
                request.context(),
            ))
        }
    }
    let state = state();
    let format = sglang();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
    card.runtime_config
        .runtime_data
        .insert(format.capability().into(), Value::Bool(true));
    let ws = dynamo_llm::discovery::WorkerSet::new("test".into(), "test".into(), card)
        .with_generate_engine(Arc::new(Capture(tx)));
    assert!(state.manager().add_worker_set("test-model", "test", ws));
    let (_, router) = format_generate::router(state, None, format);
    let request = || {
        Request::post("/generate")
            .header("content-type", "application/json")
            .header("x-request-id", "same-public-id")
            .body(Body::from(r#"{"rid":"same-public-id","input_ids":[1]}"#))
            .unwrap()
    };
    let a = tokio::spawn(router.clone().oneshot(request()));
    let b = tokio::spawn(router.oneshot(request()));
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    let second = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_ne!(first.id(), "same-public-id");
    assert_ne!(first.id(), second.id());
    assert!(!Arc::ptr_eq(&first, &second));
    first.stop_generating();
    assert!(!second.is_stopped());
    second.stop_generating();
    a.abort();
    b.abort();
}

#[tokio::test]
async fn format_request_id_is_correlation_only() {
    struct IdentityEngine;
    #[async_trait::async_trait]
    impl AsyncEngine<Input, Output, Error> for IdentityEngine {
        async fn generate(&self, request: Input) -> Result<Output, Error> {
            // The server mints the context id; the client's id only reaches the envelope.
            assert!(uuid::Uuid::parse_str(request.context().id()).is_ok());
            let extra = request.content().extra_args.as_ref().unwrap();
            let envelope_id = extra["sglang_tito"]["rid"]
                .as_str()
                .or_else(|| extra["vllm_tito"]["request_id"].as_str());
            assert_eq!(envelope_id, Some("client-id"));
            let output = LLMEngineOutput {
                token_ids: vec![7],
                finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
                engine_data: Some(serde_json::json!({"sglang_response": {
                    "output_ids": [7], "meta_info": {"id": "engine-private"}
                }})),
                ..Default::default()
            };
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter([Annotated::from_data(output)])),
                request.context(),
            ))
        }
    }
    for (format, body, header_id) in [
        (
            sglang(),
            serde_json::json!({"input_ids": [1], "rid": "client-id"}),
            "header-id",
        ),
        (
            Arc::new(dynamo_vllm_generate::VllmFormat) as SharedFormat,
            serde_json::json!({"token_ids": [1], "sampling_params": {}, "request_id": "body-id"}),
            "client-id",
        ),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", header_id.parse().unwrap());
        let response = respond(state(), format, Arc::new(IdentityEngine), body, headers).await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
    }
}

/// Echoes each request's tokens after both are in flight, and records the
/// context id and envelope id the worker saw.
struct ConcurrentEchoEngine {
    barrier: tokio::sync::Barrier,
    seen: std::sync::Mutex<Vec<(String, Option<String>)>>,
}

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for ConcurrentEchoEngine {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        let extra = request.content().extra_args.as_ref().unwrap();
        let envelope_id = extra["sglang_tito"]["rid"]
            .as_str()
            .or_else(|| extra["vllm_tito"]["request_id"].as_str())
            .map(str::to_string);
        self.seen
            .lock()
            .unwrap()
            .push((request.context().id().to_string(), envelope_id));
        self.barrier.wait().await;
        let token_ids = request.content().token_ids.to_vec();
        let output = LLMEngineOutput {
            token_ids: token_ids.clone(),
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
            engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": token_ids}})),
            ..Default::default()
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            request.context(),
        ))
    }
}

/// Concurrent requests sharing a client id each get their own output through
/// the frontend pipeline, which matches responses to callers by context id.
#[tokio::test]
async fn concurrent_requests_sharing_a_client_id_get_their_own_output() {
    use dynamo_runtime::pipeline::{ServiceBackend, ServiceFrontend, Source};
    for (format, body, header_id) in [
        (
            sglang(),
            serde_json::json!({"rid": "shared-id", "stream": false}),
            None,
        ),
        (
            Arc::new(dynamo_vllm_generate::VllmFormat) as SharedFormat,
            serde_json::json!({"sampling_params": {}}),
            Some("shared-id"),
        ),
    ] {
        let engine = Arc::new(ConcurrentEchoEngine {
            barrier: tokio::sync::Barrier::new(2),
            seen: std::sync::Mutex::new(Vec::new()),
        });
        let frontend = ServiceFrontend::<Input, Output>::new();
        let pipeline = frontend
            .link(ServiceBackend::from_engine(engine.clone()))
            .unwrap()
            .link_terminal(frontend)
            .unwrap();
        let state = state();
        let mut card = dynamo_llm::model_card::ModelDeploymentCard::with_name_only("test-model");
        card.runtime_config
            .runtime_data
            .insert(format.capability().into(), Value::Bool(true));
        let ws = dynamo_llm::discovery::WorkerSet::new("test".into(), "test".into(), card)
            .with_generate_engine(pipeline);
        assert!(state.manager().add_worker_set("test-model", "test", ws));
        let (_, router) = format_generate::router(state, None, format.clone());
        let tokens_key = if format.name() == "sglang" {
            "input_ids"
        } else {
            "token_ids"
        };
        let send = |tokens: Vec<u32>| {
            let mut body = body.clone();
            body[tokens_key] = serde_json::json!(tokens);
            let mut request = Request::post("/generate").header("content-type", "application/json");
            if let Some(id) = header_id {
                request = request.header("x-request-id", id);
            }
            router
                .clone()
                .oneshot(request.body(Body::from(body.to_string())).unwrap())
        };
        let (first, second) = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            futures::future::join(send(vec![1, 2, 3]), send(vec![7, 8])),
        )
        .await
        .expect("concurrent generate requests must not hang");
        for (response, expected) in [(first, [1, 2, 3].as_slice()), (second, &[7, 8])] {
            let response = response.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = json(response).await;
            let token_ids = if format.name() == "sglang" {
                &body["output_ids"]
            } else {
                assert_eq!(body["request_id"], "shared-id");
                &body["choices"][0]["token_ids"]
            };
            assert_eq!(token_ids, &serde_json::json!(expected));
        }
        let seen = engine.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_ne!(
            seen[0].0, seen[1].0,
            "each request needs its own context id"
        );
        for (context_id, envelope_id) in seen.iter() {
            assert_ne!(context_id, "shared-id");
            assert_eq!(envelope_id.as_deref(), Some("shared-id"));
        }
    }
}

#[path = "http/compatibility.rs"]
mod compatibility;

#[tokio::test]
async fn streaming_preserves_json_bytes_and_event_order() {
    let values = [
        serde_json::json!({"output_ids": [1], "text": "\"雪\"\n\r\\"}),
        serde_json::json!({"output_ids": [2], "custom": [null, true, {"a": 1.25}]}),
    ];
    let outputs = values
        .iter()
        .map(|value| {
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![1],
                engine_data: Some(serde_json::json!({"sglang_response": value})),
                ..Default::default()
            })
        })
        .collect();
    let response = respond(
        state(),
        sglang(),
        Arc::new(ReplayEngine(outputs, false)),
        serde_json::json!({"input_ids": [1], "stream": true}),
        HeaderMap::new(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let actual = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let mut expected = values
        .iter()
        .map(|value| format!("data: {value}\n\n"))
        .collect::<String>();
    expected.push_str("data: [DONE]\n\n");
    assert_eq!(actual.as_ref(), expected.as_bytes());
}

#[path = "http/error_policy.rs"]
mod error_policy;
