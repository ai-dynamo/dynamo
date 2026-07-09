// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::{UNKNOWN_METRIC_MODEL, WorkerSet};
use crate::http::service::metrics::{RequestType, Status};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::inference::generate::GenerateBackendMetadata;
use crate::types::Annotated;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use futures::stream;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context as TaskContext, Poll};
use tokio::sync::Notify;

#[derive(Debug)]
struct CapturedGenerate {
    request: GenerateRequest,
    dp_rank: Option<u32>,
}

struct CaptureGenerateEngine {
    captured: Arc<Mutex<Option<CapturedGenerate>>>,
}

struct ErrorGenerateEngine;

struct TypedErrorGenerateEngine(DynamoErrorType);

struct TypedStreamErrorGenerateEngine(DynamoErrorType);

struct PendingGenerateEngine {
    context: Arc<Mutex<Option<Arc<dyn AsyncEngineContext>>>>,
}

struct DistinctContextPendingEngine {
    context: Arc<Mutex<Option<Arc<dyn AsyncEngineContext>>>>,
    started: Arc<Notify>,
    dropped: Arc<AtomicBool>,
}

struct TerminalGenerateEngine(FinishReason);

#[derive(Clone, Copy)]
enum PendingPhase {
    Generate,
    Stream,
}

struct InterruptibleGenerateEngine {
    phase: PendingPhase,
    started: Arc<Notify>,
    dropped: Arc<AtomicBool>,
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct PendingResponse {
    started: Arc<Notify>,
    _drop_flag: DropFlag,
}

impl futures::Stream for PendingResponse {
    type Item = Annotated<LLMEngineOutput>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.started.notify_one();
        Poll::Pending
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for CaptureGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let dp_rank = request
            .get::<u32>(GENERATE_DP_RANK_CONTEXT_KEY)
            .ok()
            .map(|rank| *rank);
        self.captured.lock().unwrap().replace(CapturedGenerate {
            request: request.content().clone(),
            dp_rank,
        });
        let context = request.context();
        let response = Annotated::from_data(LLMEngineOutput {
            token_ids: vec![99],
            finish_reason: Some(FinishReason::Stop),
            index: Some(0),
            completion_usage: Some(CompletionUsage {
                prompt_tokens: 3,
                completion_tokens: 1,
                total_tokens: 4,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::iter([response])),
            context,
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for ErrorGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        Ok(ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_error(
                "private backend detail",
            )])),
            context,
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for TypedErrorGenerateEngine
{
    async fn generate(
        &self,
        _request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        Err(Error::new(
            DynamoError::builder()
                .error_type(self.0)
                .message("private typed engine failure")
                .build(),
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for TypedStreamErrorGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        let response = Annotated {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(
                DynamoError::builder()
                    .error_type(self.0)
                    .message("private typed stream failure")
                    .build(),
            ),
        };
        Ok(ResponseStream::new(
            Box::pin(stream::iter([response])),
            context,
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for PendingGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        self.context.lock().unwrap().replace(context.clone());
        Ok(ResponseStream::new(Box::pin(stream::pending()), context))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for DistinctContextPendingEngine
{
    async fn generate(
        &self,
        _request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let engine_context = dynamo_runtime::pipeline::Context::new(()).context();
        self.context.lock().unwrap().replace(engine_context.clone());
        Ok(ResponseStream::new(
            Box::pin(PendingResponse {
                started: self.started.clone(),
                _drop_flag: DropFlag(self.dropped.clone()),
            }),
            engine_context,
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for TerminalGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let context = request.context();
        let response = Annotated::from_data(LLMEngineOutput {
            finish_reason: Some(self.0.clone()),
            index: Some(0),
            ..Default::default()
        });
        Ok(ResponseStream::new(
            Box::pin(stream::iter([response])),
            context,
        ))
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<GenerateRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for InterruptibleGenerateEngine
{
    async fn generate(
        &self,
        request: SingleIn<GenerateRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        match self.phase {
            PendingPhase::Generate => {
                let _drop_flag = DropFlag(self.dropped.clone());
                self.started.notify_one();
                futures::future::pending().await
            }
            PendingPhase::Stream => Ok(ResponseStream::new(
                Box::pin(PendingResponse {
                    started: self.started.clone(),
                    _drop_flag: DropFlag(self.dropped.clone()),
                }),
                request.context(),
            )),
        }
    }
}

fn output(index: u32, token_id: u32, finished: bool) -> LLMEngineOutput {
    let top = TopLogprob {
        rank: 1,
        token_id,
        token: Some(format!("token_id:{token_id}")),
        logprob: -0.25,
        bytes: None,
    };
    LLMEngineOutput {
        token_ids: vec![token_id],
        log_probs: Some(vec![-0.25]),
        top_logprobs: Some(vec![vec![top]]),
        finish_reason: finished.then_some(FinishReason::Stop),
        index: Some(index),
        generate_metadata: finished.then_some(GenerateBackendMetadata::default()),
        ..Default::default()
    }
}

fn request_for(model: &str) -> GenerateRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "token_ids": [11],
        "sampling_params": {}
    }))
    .unwrap()
}

fn service_with_generate_engine(
    engine: crate::types::inference::generate::GenerateStreamingEngine,
) -> service_v2::HttpService {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(engine);
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);
    service
}

async fn direct_unary(
    engine: crate::types::inference::generate::GenerateStreamingEngine,
) -> (
    tokio::task::JoinHandle<Result<GenerateResponse, ErrorResponse>>,
    Arc<dyn AsyncEngineContext>,
    super::super::disconnect::ConnectionHandle,
    Arc<service_v2::State>,
) {
    let service = service_with_generate_engine(engine);
    let state = service.state_clone();
    let request = dynamo_runtime::pipeline::Context::new(request_for("test-model"));
    let context = request.context();
    let cancellation_labels = CancellationLabels {
        model: "test-model".to_string(),
        endpoint: Endpoint::InferenceGenerate.to_string(),
        request_type: "unary".to_string(),
    };
    let (connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), None, cancellation_labels).await;
    let task = tokio::spawn(generate_unary(
        state.clone(),
        request,
        "test-request".to_string(),
        "test-model".to_string(),
        1,
        false,
        stream_handle,
    ));
    (task, context, connection_handle, state)
}

fn assert_cancelled_metrics(state: &service_v2::State) {
    let metrics = state.metrics_clone();
    assert_eq!(metrics.get_inflight_count("test-model"), 0);
    assert_eq!(
        metrics.get_request_counter(
            "test-model",
            &Endpoint::InferenceGenerate,
            &RequestType::Unary,
            &Status::Error,
            &ErrorType::Cancelled,
        ),
        1
    );
}

fn assert_engine_error_metrics(state: &service_v2::State, error_type: ErrorType) {
    let metrics = state.metrics_clone();
    assert_eq!(metrics.get_inflight_count("test-model"), 0);
    assert_eq!(
        metrics.get_request_counter(
            "test-model",
            &Endpoint::InferenceGenerate,
            &RequestType::Unary,
            &Status::Error,
            &error_type,
        ),
        1
    );
}

#[test]
fn unary_accumulator_preserves_choice_order_and_alignment() {
    let mut accumulator = GenerateAccumulator::new("request-1".into(), 2, true);
    accumulator.push(output(1, 21, false)).unwrap();
    accumulator.push(output(0, 11, true)).unwrap();
    accumulator.push(output(1, 22, true)).unwrap();
    let response = accumulator.finish().unwrap();
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[1].token_ids, vec![21, 22]);
    assert_eq!(
        response.choices[1]
            .logprobs
            .as_ref()
            .unwrap()
            .content
            .as_ref()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn unary_accumulator_rejects_missing_requested_logprobs() {
    let mut accumulator = GenerateAccumulator::new("request-1".into(), 1, true);
    let mut broken = output(0, 11, true);
    broken.log_probs = None;
    assert!(accumulator.push(broken).is_err());
}

#[test]
fn token_logprob_preserves_backend_bytes_and_selected_candidate() {
    let selected_bytes = vec![0xf0, 0x9f, 0x98, 0x80];
    let token = build_token_logprob(
        42,
        -0.125,
        &[
            TopLogprob {
                rank: 1,
                token_id: 7,
                token: Some("other".to_string()),
                logprob: -0.25,
                bytes: Some(b"other".to_vec()),
            },
            TopLogprob {
                rank: 2,
                token_id: 42,
                token: Some("😀".to_string()),
                logprob: -0.125,
                bytes: Some(selected_bytes.clone()),
            },
        ],
    );

    assert_eq!(token.token, "😀");
    assert_eq!(token.bytes, Some(selected_bytes.clone()));
    assert_eq!(
        token
            .top_logprobs
            .iter()
            .filter(|candidate| candidate.token == "😀")
            .count(),
        1
    );
    assert_eq!(token.top_logprobs[1].bytes, Some(selected_bytes));
}

#[test]
fn token_logprob_adds_missing_selected_token_and_clamps_non_finite_values() {
    let token = build_token_logprob(
        42,
        f64::NEG_INFINITY,
        &[TopLogprob {
            rank: 1,
            token_id: 7,
            token: Some("other".to_string()),
            logprob: f64::NAN,
            bytes: Some(b"other".to_vec()),
        }],
    );

    assert_eq!(token.token, "token_id:42");
    assert_eq!(token.bytes, Some(b"token_id:42".to_vec()));
    assert_eq!(token.logprob, -9999.0);
    assert!(
        token
            .top_logprobs
            .iter()
            .all(|candidate| candidate.logprob.is_finite())
    );
    let selected = token
        .top_logprobs
        .iter()
        .find(|candidate| candidate.token == "token_id:42")
        .expect("selected token must be represented in top_logprobs");
    assert_eq!(selected.logprob, -9999.0);
    assert_eq!(selected.bytes, Some(b"token_id:42".to_vec()));
}

#[test]
fn prompt_logprobs_are_clamped_before_response_serialization() {
    let mut prompt_logprobs = vec![Some(std::collections::HashMap::from([(
        42,
        GenerateLogprob {
            logprob: f32::NEG_INFINITY,
            rank: Some(1),
            decoded_token: Some("answer".to_string()),
        },
    )]))];

    normalize_prompt_logprobs(&mut prompt_logprobs);

    assert_eq!(prompt_logprobs[0].as_ref().unwrap()[&42].logprob, -9999.0);
}

#[test]
fn cancelled_finish_reason_is_not_a_successful_stop_reason() {
    assert!(public_finish_reason(FinishReason::Cancelled).is_err());
}

async fn assert_nested_boundary_error(
    response: Response,
    expected_status: u16,
    expected_type: &str,
) -> serde_json::Value {
    assert_eq!(response.status().as_u16(), expected_status);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(body.get("message").is_none());
    assert_eq!(body["error"]["code"], expected_status);
    assert_eq!(body["error"]["type"], expected_type);
    assert!(body["error"]["message"].is_string());
    body
}

#[tokio::test]
async fn vllm_error_envelope_is_nested_for_boundary_statuses() {
    let cases = [
        (
            protocol_error(GenerateProtocolError::InvalidRequest(
                "bad sampling value".to_string(),
            )),
            400,
            "invalid_request_error",
        ),
        (ErrorMessage::model_not_found(), 404, "not_found"),
        (
            ErrorMessage::from_http_error(HttpError {
                code: 499,
                message: "private cancellation context".to_string(),
            }),
            499,
            "request_cancelled",
        ),
        (
            ErrorMessage::internal_server_error_with_details(
                "Internal server error",
                "private backend detail",
            ),
            500,
            "internal_error",
        ),
    ];

    for (error, expected_status, expected_type) in cases {
        let response = vllm_error_response(error);
        assert_eq!(response.status().as_u16(), expected_status);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body.get("message").is_none());
        assert_eq!(body["error"]["code"], expected_status);
        assert_eq!(body["error"]["type"], expected_type);
        assert!(body["error"]["message"].is_string());
        assert!(!body.to_string().contains("private backend detail"));
        assert!(!body.to_string().contains("private cancellation context"));
    }
}

#[tokio::test]
async fn handler_returns_nested_vllm_errors_for_400_404_499_and_500() {
    let empty_service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let invalid_request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [11],
        "sampling_params": {"n": 0}
    }))
    .unwrap();
    let response = handler(
        State(empty_service.state_clone()),
        HeaderMap::new(),
        Json(invalid_request),
    )
    .await;
    assert_nested_boundary_error(response, 400, "invalid_request_error").await;

    let invalid_thinking_budget: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [11],
        "sampling_params": {"thinking_token_budget": -2}
    }))
    .unwrap();
    let response = handler(
        State(empty_service.state_clone()),
        HeaderMap::new(),
        Json(invalid_thinking_budget),
    )
    .await;
    assert_nested_boundary_error(response, 400, "invalid_request_error").await;

    let response = handler(
        State(empty_service.state_clone()),
        HeaderMap::new(),
        Json(request_for("missing-model")),
    )
    .await;
    assert_nested_boundary_error(response, 404, "not_found").await;

    let error_service = service_with_generate_engine(Arc::new(ErrorGenerateEngine));
    let response = handler(
        State(error_service.state_clone()),
        HeaderMap::new(),
        Json(request_for("test-model")),
    )
    .await;
    let body = assert_nested_boundary_error(response, 500, "internal_error").await;
    assert!(!body.to_string().contains("private backend detail"));

    let cancelled_service =
        service_with_generate_engine(Arc::new(TerminalGenerateEngine(FinishReason::Cancelled)));
    let response = handler(
        State(cancelled_service.state_clone()),
        HeaderMap::new(),
        Json(request_for("test-model")),
    )
    .await;
    assert_nested_boundary_error(response, 499, "request_cancelled").await;
}

#[tokio::test]
async fn request_kill_interrupts_pending_generate_and_stream_phases() {
    for phase in [PendingPhase::Generate, PendingPhase::Stream] {
        let started = Arc::new(Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let engine: crate::types::inference::generate::GenerateStreamingEngine =
            Arc::new(InterruptibleGenerateEngine {
                phase,
                started: started.clone(),
                dropped: dropped.clone(),
            });
        let (task, context, mut connection_handle, state) = direct_unary(engine).await;
        started.notified().await;
        context.kill();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("unary generate did not stop after request cancellation")
            .expect("unary generate task panicked")
            .expect_err("cancelled unary generate must return an error response");
        connection_handle.disarm();
        assert_eq!(result.0.as_u16(), 499);
        assert!(dropped.load(Ordering::SeqCst));
        assert_cancelled_metrics(state.as_ref());
    }
}

#[tokio::test]
async fn engine_context_kill_interrupts_pending_unary_stream() {
    let engine_context = Arc::new(Mutex::new(None));
    let started = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let engine: crate::types::inference::generate::GenerateStreamingEngine =
        Arc::new(DistinctContextPendingEngine {
            context: engine_context.clone(),
            started: started.clone(),
            dropped: dropped.clone(),
        });
    let (task, request_context, mut connection_handle, state) = direct_unary(engine).await;
    started.notified().await;
    let engine_context = engine_context.lock().unwrap().as_ref().unwrap().clone();
    assert!(!Arc::ptr_eq(&request_context, &engine_context));
    engine_context.kill();

    let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .expect("unary generate did not stop after engine cancellation")
        .expect("unary generate task panicked")
        .expect_err("cancelled unary generate must return an error response");
    connection_handle.disarm();
    assert_eq!(result.0.as_u16(), 499);
    assert!(!request_context.is_killed());
    assert!(dropped.load(Ordering::SeqCst));
    assert_cancelled_metrics(state.as_ref());
}

#[tokio::test]
async fn typed_engine_errors_use_canonical_http_metrics_and_rejection_count() {
    let overload_status = crate::http::service::error::overload_status_code();
    let overload_metric = crate::http::service::openai::classify_error_for_metrics(
        overload_status,
        "Service temporarily overloaded",
    );
    let cases = [
        (
            DynamoErrorType::ResourceExhausted,
            overload_status.as_u16(),
            overload_metric,
            1,
        ),
        (
            DynamoErrorType::Unavailable,
            StatusCode::SERVICE_UNAVAILABLE.as_u16(),
            ErrorType::Unavailable,
            0,
        ),
        (DynamoErrorType::Cancelled, 499, ErrorType::Cancelled, 0),
        (
            DynamoErrorType::Unknown,
            StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            ErrorType::Internal,
            0,
        ),
    ];

    for (engine_error, expected_status, expected_metric, expected_rejections) in cases {
        let engine: crate::types::inference::generate::GenerateStreamingEngine =
            Arc::new(TypedErrorGenerateEngine(engine_error));
        let (task, _context, mut connection_handle, state) = direct_unary(engine).await;
        let response = task
            .await
            .expect("unary generate task panicked")
            .expect_err("typed engine error must fail the request");
        connection_handle.disarm();

        assert_eq!(response.0.as_u16(), expected_status);
        assert_engine_error_metrics(state.as_ref(), expected_metric);
        assert_eq!(
            state
                .metrics_clone()
                .get_rejection_count("test-model", Endpoint::InferenceGenerate),
            expected_rejections
        );
    }
}

#[tokio::test]
async fn cancelled_finish_reason_returns_499_and_cancelled_metrics() {
    let engine: crate::types::inference::generate::GenerateStreamingEngine =
        Arc::new(TerminalGenerateEngine(FinishReason::Cancelled));
    let (task, _context, mut connection_handle, state) = direct_unary(engine).await;

    let result = task
        .await
        .expect("unary generate task panicked")
        .expect_err("cancelled finish reason must not be reported as success");
    connection_handle.disarm();
    assert_eq!(result.0.as_u16(), 499);
    assert_cancelled_metrics(state.as_ref());
}

#[test]
fn stream_accumulator_emits_delta_finish_only_and_usage_chunks() {
    let mut accumulator = GenerateStreamAccumulator::new("request-1".into(), 1, true);
    let mut delta = output(0, 11, false);
    delta.completion_usage = Some(CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 1,
        total_tokens: 4,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });
    let delta = accumulator.push(delta, true).unwrap().unwrap();
    assert_eq!(delta.choices[0].token_ids, vec![11]);
    assert_eq!(delta.usage.as_ref().unwrap().total_tokens, 4);

    let terminal = LLMEngineOutput {
        token_ids: Vec::new(),
        finish_reason: Some(FinishReason::Stop),
        index: Some(0),
        completion_usage: Some(CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
        ..Default::default()
    };
    let terminal = accumulator.push(terminal, true).unwrap().unwrap();
    assert!(terminal.choices[0].token_ids.is_empty());
    assert_eq!(terminal.choices[0].finish_reason.as_deref(), Some("stop"));

    let final_usage = accumulator.finish(true).unwrap().unwrap();
    assert!(final_usage.choices.is_empty());
    assert_eq!(final_usage.usage.unwrap().completion_tokens, 1);
}

#[test]
fn stream_accumulator_rejects_missing_terminal_choice() {
    let mut accumulator = GenerateStreamAccumulator::new("request-1".into(), 1, false);
    accumulator.push(output(0, 11, false), false).unwrap();
    assert!(accumulator.finish(false).is_err());
}

#[test]
fn request_context_matches_vllm_header_precedence() {
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", "header-id".parse().unwrap());
    headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "7".parse().unwrap());

    assert_eq!(resolve_request_id(&headers, Some("body-id")), "header-id");
    assert_eq!(data_parallel_rank_from_headers(&headers), Some(7));
}

#[test]
fn malformed_data_parallel_rank_is_ignored_like_vllm() {
    let mut headers = HeaderMap::new();
    headers.insert(
        HEADER_DATA_PARALLEL_RANK_ALIAS,
        "not-a-rank".parse().unwrap(),
    );
    assert_eq!(data_parallel_rank_from_headers(&headers), None);
}

#[tokio::test]
async fn unary_http_boundary_preserves_token_native_request_and_private_context() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let captured = Arc::new(Mutex::new(None));
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(Arc::new(CaptureGenerateEngine {
        captured: captured.clone(),
    }));
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);

    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", "header-request".parse().unwrap());
    headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "3".parse().unwrap());
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "request_id": "body-request",
        "token_ids": [11, 22, 33],
        "sampling_params": {"max_tokens": 4},
        "priority": -2
    }))
    .unwrap();

    let response = handler(State(service.state_clone()), headers, Json(request)).await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response: GenerateResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(response.request_id, "header-request");
    assert_eq!(response.choices[0].token_ids, vec![99]);

    let captured = captured.lock().unwrap();
    let captured = captured.as_ref().unwrap();
    assert_eq!(captured.request.model.as_deref(), Some("test-model"));
    assert_eq!(
        captured.request.request_id.as_deref(),
        Some("header-request")
    );
    assert_eq!(captured.request.token_ids, vec![11, 22, 33]);
    assert_eq!(captured.request.priority, -2);
    assert_eq!(captured.dp_rank, Some(3));
}

#[tokio::test]
async fn streaming_http_boundary_emits_usage_and_done() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let captured = Arc::new(Mutex::new(None));
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(Arc::new(CaptureGenerateEngine {
        captured: captured.clone(),
    }));
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);

    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "request_id": "stream-request",
        "token_ids": [11, 22, 33],
        "sampling_params": {"max_tokens": 4},
        "stream": true,
        "stream_options": {
            "include_usage": true,
            "continuous_usage_stats": true
        }
    }))
    .unwrap();
    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("\"token_ids\":[99]"));
    assert!(body.contains("\"choices\":[]"));
    assert!(body.contains("\"prompt_tokens\":3"));
    assert!(body.ends_with("data: [DONE]\n\n"));

    let captured = captured.lock().unwrap();
    assert!(captured.as_ref().unwrap().request.stream);
}

#[tokio::test]
async fn streaming_engine_cancellation_emits_499_error_done_and_cancelled_metrics() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let engine_context = Arc::new(Mutex::new(None));
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(Arc::new(PendingGenerateEngine {
        context: engine_context.clone(),
    }));
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);
    let mut request = request_for("test-model");
    request.stream = true;

    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let context = engine_context.lock().unwrap().as_ref().unwrap().clone();
    context.kill();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    assert!(body.contains(r#""type":"request_cancelled""#));
    assert!(body.contains(r#""code":499"#));
    assert!(body.contains("Request cancelled"));
    assert!(body.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        service.state_clone().metrics_clone().get_request_counter(
            "test-model",
            &Endpoint::InferenceGenerate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::Cancelled,
        ),
        1
    );
}

#[tokio::test]
async fn streaming_cancelled_finish_reason_emits_499_error_and_done() {
    let service =
        service_with_generate_engine(Arc::new(TerminalGenerateEngine(FinishReason::Cancelled)));
    let mut request = request_for("test-model");
    request.stream = true;

    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    assert!(body.contains(r#""type":"request_cancelled""#));
    assert!(body.contains(r#""code":499"#));
    assert!(body.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        service.state_clone().metrics_clone().get_request_counter(
            "test-model",
            &Endpoint::InferenceGenerate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::Cancelled,
        ),
        1
    );
}

#[tokio::test]
async fn streaming_typed_overload_uses_canonical_sse_metrics_and_rejection_count() {
    let service = service_with_generate_engine(Arc::new(TypedStreamErrorGenerateEngine(
        DynamoErrorType::ResourceExhausted,
    )));
    let mut request = request_for("test-model");
    request.stream = true;

    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    let overload_status = crate::http::service::error::overload_status_code();
    let overload_metric = crate::http::service::openai::classify_error_for_metrics(
        overload_status,
        "Service temporarily overloaded",
    );

    assert!(body.contains(r#""type":"service_unavailable""#));
    assert!(body.contains(&format!(r#""code":{}"#, overload_status.as_u16())));
    assert!(!body.contains("private typed stream failure"));
    assert!(body.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        service.state_clone().metrics_clone().get_request_counter(
            "test-model",
            &Endpoint::InferenceGenerate,
            &RequestType::Stream,
            &Status::Error,
            &overload_metric,
        ),
        1
    );
    assert_eq!(
        service
            .state_clone()
            .metrics_clone()
            .get_rejection_count("test-model", Endpoint::InferenceGenerate),
        1
    );
}

#[tokio::test]
async fn streaming_missing_model_uses_bounded_metric_label() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let state = service.state_clone();
    let request = dynamo_runtime::pipeline::Context::new(request_for("attacker-model-label"));
    let context = request.context();
    let cancellation_labels = CancellationLabels {
        model: UNKNOWN_METRIC_MODEL.to_string(),
        endpoint: Endpoint::InferenceGenerate.to_string(),
        request_type: "stream".to_string(),
    };
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context, None, cancellation_labels).await;

    let response = generate_streaming(
        state.clone(),
        request,
        "test-request".to_string(),
        "attacker-model-label".to_string(),
        1,
        false,
        false,
        false,
        stream_handle,
    )
    .await
    .expect_err("missing model must fail before producing an SSE response");
    connection_handle.disarm();
    assert_eq!(response.0, StatusCode::NOT_FOUND);
    assert_eq!(
        state.metrics_clone().get_request_counter(
            UNKNOWN_METRIC_MODEL,
            &Endpoint::InferenceGenerate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::NotFound,
        ),
        1
    );
    assert_eq!(
        state.metrics_clone().get_request_counter(
            "attacker-model-label",
            &Endpoint::InferenceGenerate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::NotFound,
        ),
        0
    );
}

#[tokio::test]
async fn streaming_backend_error_is_sanitized_and_terminated() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(Arc::new(ErrorGenerateEngine));
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "token_ids": [11],
        "sampling_params": {},
        "stream": true
    }))
    .unwrap();

    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("Internal server error"));
    assert!(!body.contains("private backend detail"));
    assert!(body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn dropping_streaming_response_cancels_engine_context() {
    let service = service_v2::HttpService::builder()
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let engine_context = Arc::new(Mutex::new(None));
    let mut worker_set = WorkerSet::new(
        "test".to_string(),
        "test-mdc".to_string(),
        ModelDeploymentCard::default(),
    );
    worker_set.generate_engine = Some(Arc::new(PendingGenerateEngine {
        context: engine_context.clone(),
    }));
    service
        .model_manager()
        .add_worker_set("test-model", "test", worker_set);
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "token_ids": [11],
        "sampling_params": {},
        "stream": true
    }))
    .unwrap();

    let response = handler(
        State(service.state_clone()),
        HeaderMap::new(),
        Json(request),
    )
    .await;
    let context = engine_context.lock().unwrap().as_ref().unwrap().clone();
    assert!(!context.is_killed());
    drop(response);

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !context.is_killed() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the SSE body should cancel generation");
}
