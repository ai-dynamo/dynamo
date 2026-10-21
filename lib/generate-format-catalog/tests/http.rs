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
    headers: HeaderMap,
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

#[tokio::test]
async fn dispatch_errors_preserve_status_shape_and_metrics() {
    use dynamo_runtime::error::ErrorType as Backend;
    for (kind, status, metric, message) in [
        (
            Backend::WorkerUnavailable,
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            Backend::DeadlineExceeded,
            StatusCode::TOO_MANY_REQUESTS,
            ErrorType::Cancelled,
            "request deadline exceeded",
        ),
    ] {
        for streaming in [false, true] {
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let response = respond(
                state.clone(),
                sglang(),
                Arc::new(FailingEngine(kind)),
                serde_json::json!({"input_ids": [1], "stream": streaming}),
                HeaderMap::new(),
            )
            .await;
            assert_eq!(response.status(), status);
            assert_eq!(json(response).await["error"]["message"], message);
            let request_type = if streaming {
                RequestType::Stream
            } else {
                RequestType::Unary
            };
            assert_eq!(
                state.metrics_clone().get_request_counter(
                    state.manager().metric_model_for("test-model"),
                    &Endpoint::Generate,
                    &request_type,
                    &Status::Error,
                    &metric
                ),
                1
            );
        }
    }
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

#[tokio::test]
async fn native_responses_record_cached_token_metrics() {
    for stream in [false, true] {
        let state = state();
        let registry = prometheus::Registry::new();
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
            Arc::new(ReplayEngine(vec![Annotated::from_data(output)], false)),
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
            serde_json::json!({"error": {"message": "Invalid request"}}),
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
                    .starts_with("Failed to deserialize")
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
    assert_ne!(first.id(), second.id());
    first.stop_generating();
    assert!(!second.is_stopped());
    second.stop_generating();
    a.abort();
    b.abort();
}
