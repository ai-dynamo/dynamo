// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use dynamo_runtime::engine::{AsyncEngine, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
type Input = SingleIn<PreprocessedRequest>;
type Output = ManyOut<Annotated<LLMEngineOutput>>;

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

fn dispatch_test_context() -> Context<PreprocessedRequest> {
    Context::new(
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .expect("build dispatch test request"),
    )
}
use crate::http::service::metrics::{Endpoint, RequestType, Status};
use dynamo_runtime::protocols::maybe_error::MaybeError;

fn sglang() -> SharedFormat {
    Arc::new(crate::protocols::sglang::SglangFormat::native())
}

fn dispatch(
    state: Arc<service_v2::State>,
    streaming: bool,
    stream_handle: ConnectionHandle,
) -> Dispatch {
    let body = serde_json::json!({"input_ids": [1], "stream": streaming}).to_string();
    let Opened { bind, .. } = sglang()
        .open(ClientRequest {
            body: body.as_bytes(),
            headers: &HeaderMap::new(),
            default_request_id: "request-id",
        })
        .unwrap();
    let exchange = bind(&crate::generate_format::testkit::worker(&Value::Null))
        .unwrap()
        .exchange;
    Dispatch {
        tracker: Arc::new(RequestTracker::new()),
        request_id: "req".to_string(),
        model: "test-model".to_string(),
        state,
        stream_handle,
        streaming,
        exchange: Arc::new(Mutex::new(exchange)),
        format: sglang(),
    }
}

async fn json(response: Response) -> Value {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[test]
fn model_aliases_are_deduplicated_before_implicit_selection() {
    let manager = crate::discovery::ModelManager::new();
    assert!(manager.register_alias("alias", "primary"));
    let models = ["primary", "alias", "other"].map(String::from).to_vec();
    assert_eq!(
        canonical_generate_models(&manager, models),
        ["other", "primary"]
    );
}

#[test]
fn stream_errors_keep_their_classification_and_shape() {
    use dynamo_runtime::error::{ErrorReason, ErrorType as DynamoErrorType};

    let typed = |error_type, message: &str| {
        DynamoError::builder()
            .error_type(error_type)
            .message(message)
            .build()
    };
    let deadline = DynamoError::builder()
        .error_type(DynamoErrorType::WorkerOverloaded)
        .message("worker overloaded")
        .cause(
            DynamoError::builder()
                .error_type(DynamoErrorType::DeadlineExceeded)
                .reason(ErrorReason::new("router.queue_deadline_exceeded").unwrap())
                .message("internal deadline detail")
                .build(),
        )
        .build();
    for (error, expected_type, expected_message) in [
        (
            typed(DynamoErrorType::InvalidArgument, "bad params"),
            ErrorType::Validation,
            "Invalid request",
        ),
        (
            typed(DynamoErrorType::Unavailable, "x"),
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            typed(DynamoErrorType::WorkerUnavailable, "x"),
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            typed(DynamoErrorType::Cancelled, "x"),
            ErrorType::Cancelled,
            "Request cancelled",
        ),
        (
            DynamoError::msg("backend failure"),
            ErrorType::Internal,
            "Internal server error",
        ),
        // Deadline wins over overload.
        (deadline, ErrorType::Cancelled, "request deadline exceeded"),
    ] {
        let (error_type, body) = stream_error(&sglang(), &axum::Error::new(error));
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(error_type, expected_type);
        assert_eq!(
            body,
            serde_json::json!({"error": {"message": expected_message}})
        );
    }
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
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let response = dispatch(
                state.clone(),
                streaming,
                ConnectionHandle::create_disabled(tx),
            )
            .run(Arc::new(FailingEngine(kind)), dispatch_test_context())
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

/// A simulated SGLang worker whose first `failures` attempts drop after
/// two chunks, as when an engine dies mid-stream.
struct FlakyWorker {
    calls: std::sync::atomic::AtomicUsize,
    failures: usize,
    truth: crate::generate_format::testkit::sim::Truth,
}

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for FlakyWorker {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        use crate::generate_format::testkit::sim;
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (request, context) = request.transfer(());
        let body = request.extra_args.as_ref().unwrap()["sglang_tito"].clone();
        let chunks = sim::stream(
            &self.truth,
            body.as_object().unwrap(),
            &request.token_ids,
            sim::Role::Aggregated,
            &Value::Null,
        );
        let mut items: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                Annotated::from_data(
                    crate::protocols::sglang::inspect(chunk, false)
                        .unwrap()
                        .output,
                )
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
        let stream = Box::pin(futures::stream::iter(items));
        Ok(ResponseStream::new(stream, context.context()))
    }
}

/// Generate's migration operator in front of a worker.
struct Migrating(Arc<crate::migration::Migration>, Arc<FlakyWorker>);

#[async_trait::async_trait]
impl AsyncEngine<Input, Output, Error> for Migrating {
    async fn generate(&self, request: Input) -> Result<Output, Error> {
        use dynamo_runtime::pipeline::Operator;
        self.0.generate(request, self.1.clone()).await
    }
}

#[tokio::test]
async fn a_request_migrated_mid_stream_returns_what_an_unbroken_one_does() {
    let truth = crate::generate_format::testkit::sim::Truth {
        prompt: vec![1, 2, 3],
        prompt_logprobs: vec![0.0, -0.5, -0.25],
        output: (10..15).map(|id| (id, -1.0)).collect(),
        stop: false,
        row_bytes: 1,
        experts: vec![0; 16],
        cuts: (1..=5).collect(),
        cached_tokens: 2,
    };
    let respond = |stream: bool, failures: usize, max_seq_len| {
        let truth = truth.clone();
        async move {
            let body = serde_json::json!({
                "input_ids": [1, 2, 3], "stream": stream, "return_logprob": true,
                "logprob_start_len": 0, "sampling_params": {"max_new_tokens": 5},
            })
            .to_string();
            let Opened { bind, .. } = sglang()
                .open(ClientRequest {
                    body: body.as_bytes(),
                    headers: &HeaderMap::new(),
                    default_request_id: "request-id",
                })
                .unwrap();
            let worker = crate::generate_format::testkit::worker(&Value::Null);
            let Bound {
                mut request,
                migration,
                exchange,
            } = bind(&worker).unwrap();
            assert!(migration.is_ok());
            let exchange: SharedExchange = Arc::new(Mutex::new(exchange));
            request.continuation = Some(exchange.clone());
            let worker = Arc::new(FlakyWorker {
                calls: Default::default(),
                failures,
                truth,
            });
            let card = crate::model_card::ModelDeploymentCard::with_name_only("test-model");
            let metrics = Arc::new(super::super::metrics::Metrics::new());
            let migration =
                crate::migration::Migration::for_generate(&card, 3, max_seq_len, metrics);
            let state = service_v2::HttpService::builder()
                .build()
                .unwrap()
                .state_clone();
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let mut dispatch = dispatch(state, stream, ConnectionHandle::create_disabled(tx));
            dispatch.exchange = exchange;
            let response = dispatch
                .run(
                    Arc::new(Migrating(migration, worker.clone())),
                    Context::new(request),
                )
                .await;
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let calls = worker.calls.load(std::sync::atomic::Ordering::Relaxed);
            (status, String::from_utf8(body.to_vec()).unwrap(), calls)
        }
    };
    for stream in [false, true] {
        let (status, unbroken, calls) = respond(stream, 0, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(calls, 1);
        let (status, migrated, calls) = respond(stream, 2, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(calls, 3, "two failed attempts, then one that finishes");
        assert_eq!(migrated, unbroken, "stream={stream}");
        let (status, body, calls) = respond(stream, 2, Some(4)).await;
        assert_eq!(
            calls, 1,
            "a retry must not cross the configured sequence limit"
        );
        assert_eq!(
            status,
            if stream {
                StatusCode::OK
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        );
        assert!(body.contains("Internal server error"), "{body}");
    }
}

#[tokio::test]
async fn vllm_format_preserves_binding_terminal_status_and_wire_shape() {
    use crate::protocols::common::FinishReason;
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
        let format = crate::generate_format::named("vllm").unwrap();
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
        let bound = (opened.bind)(&crate::generate_format::testkit::worker(&Value::Null)).unwrap();
        assert!(bound.migration.is_err());
        let hints = bound.request.routing.as_ref().unwrap();
        assert_eq!(hints.dp_rank, Some(3));
        assert_eq!(hints.priority, Some(7));
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut dispatch = dispatch(state, false, ConnectionHandle::create_disabled(tx));
        dispatch.exchange = Arc::new(Mutex::new(bound.exchange));
        dispatch.format = format;
        let output = LLMEngineOutput {
            token_ids: vec![2, 3],
            finish_reason: (!missing_logprobs).then_some(finish),
            ..Default::default()
        };
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            dispatch.run(
                Arc::new(ReplayEngine(
                    vec![Annotated::from_data(output)],
                    status != StatusCode::OK,
                )),
                Context::new(bound.request),
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
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            dispatch(state, false, ConnectionHandle::create_disabled(tx)).run(
                Arc::new(ReplayEngine(outputs, status != StatusCode::OK)),
                dispatch_test_context(),
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
    use crate::protocols::sglang::{ClientCodec, SglangFormat};

    struct Codec;
    impl ClientCodec for Codec {
        fn error_body(&self, message: &str) -> Value {
            serde_json::json!({"custom_failure": message})
        }
    }

    let format: SharedFormat = Arc::new(SglangFormat::with_codec("custom-errors", Codec));
    for body in [b"{".as_slice(), br#"{"input_ids":"invalid"}"#] {
        let native = sglang()
            .open(ClientRequest {
                body,
                headers: &HeaderMap::new(),
                default_request_id: "request-id",
            })
            .err()
            .unwrap();
        let native_body = json(client_error(&sglang(), native)).await;
        assert_eq!(native_body["object"], "error");
        assert_eq!(native_body["code"], 400);
        let error = format
            .open(ClientRequest {
                body,
                headers: &HeaderMap::new(),
                default_request_id: "request-id",
            })
            .err()
            .expect("malformed requests must fail");
        let response = client_error(&format, error);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json(response).await;
        assert!(
            body["custom_failure"]
                .as_str()
                .unwrap()
                .starts_with("Failed to deserialize")
        );
        assert_eq!(body.as_object().unwrap().len(), 1);
    }
}

#[test]
fn overlapping_public_ids_have_independent_cancellation_contexts() {
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", "same-public-id".parse().unwrap());
    let open = || {
        let opened = sglang()
            .open(ClientRequest {
                body: br#"{"rid":"same-public-id","input_ids":[1]}"#,
                headers: &headers,
                default_request_id: "default-id",
            })
            .unwrap();
        assert_eq!(opened.request_id, "same-public-id");
        let bound = (opened.bind)(&crate::generate_format::testkit::worker(&Value::Null)).unwrap();
        preprocessed_context_from_headers(bound.request, &headers).unwrap()
    };
    let requests = [open(), open()];
    assert_ne!(requests[0].context().id(), requests[1].context().id());
    for request in &requests {
        assert_eq!(
            request.content().extra_args.as_ref().unwrap()["sglang_tito"]["rid"],
            "same-public-id"
        );
    }
    requests[0].context().stop_generating();
    assert!(!requests[1].context().is_stopped());
}
