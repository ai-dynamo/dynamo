// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::{UNKNOWN_METRIC_MODEL, WorkerSet};
use crate::http::service::metrics::{RequestType, Status};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::extensions::{
    HEADER_DATA_PARALLEL_RANK_ALIAS, HEADER_PREFILL_DP_RANK, HEADER_PREFILL_INSTANCE_ID,
    HEADER_WORKER_INSTANCE_ID, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
};
use crate::protocols::common::preprocessor::RoutingHints;
use crate::protocols::inference::generate::{
    GENERATE_ROUTING_HINTS_CONTEXT_KEY, GenerateBackendMetadata,
};
use crate::types::Annotated;
use base64::Engine as _;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use futures::stream;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context as TaskContext, Poll};
use tokio::sync::Notify;

#[test]
fn token_id_logprob_selection_enables_response_logprobs() {
    let sampling = GenerateSamplingParams {
        logprob_token_ids: Some(vec![11, 22, 33]),
        ..Default::default()
    };
    assert_eq!(requested_completion_logprobs(&sampling), Some(4));

    let sampling = GenerateSamplingParams {
        logprobs: Some(1),
        logprob_token_ids: Some(vec![11]),
        ..Default::default()
    };
    assert_eq!(requested_completion_logprobs(&sampling), Some(2));
}

#[test]
fn token_id_and_all_candidate_modes_do_not_truncate_logprobs() {
    let candidates = [
        TopLogprob {
            rank: 1,
            token_id: 42,
            token: None,
            logprob: -0.1,
            bytes: None,
        },
        TopLogprob {
            rank: 2,
            token_id: 11,
            token: None,
            logprob: -0.2,
            bytes: None,
        },
        TopLogprob {
            rank: 3,
            token_id: 22,
            token: None,
            logprob: -0.3,
            bytes: None,
        },
    ];

    let token_ids = build_token_logprob(42, &candidates, 3);
    assert_eq!(token_ids.top_logprobs.len(), 3);
    assert_eq!(token_ids.top_logprobs[2].token, "token_id:22");

    let all = build_token_logprob(42, &candidates, -1);
    assert_eq!(all.top_logprobs.len(), 3);
}

#[derive(Debug)]
struct CapturedGenerate {
    request: GenerateRequest,
    routing: Option<RoutingHints>,
    session_affinity: Option<String>,
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
        let routing = request
            .get::<RoutingHints>(GENERATE_ROUTING_HINTS_CONTEXT_KEY)
            .ok()
            .map(|routing| routing.as_ref().clone());
        let session_affinity = request
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .ok()
            .map(|session| session.as_str().to_string());
        self.captured.lock().unwrap().replace(CapturedGenerate {
            request: request.content().clone(),
            routing,
            session_affinity,
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

fn routed_expert_payload(values: Vec<i16>, rows: usize) -> String {
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[rows, 1, 1]), values).unwrap();
    let mut bytes = Vec::new();
    array.write_npy(&mut bytes).unwrap();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn large_routed_expert_payload(data_bytes: usize) -> String {
    assert_eq!(data_bytes % std::mem::size_of::<u32>(), 0);
    let array = ndarray::Array1::<u32>::zeros(data_bytes / std::mem::size_of::<u32>());
    let mut bytes = Vec::new();
    array.write_npy(&mut bytes).unwrap();
    base64::engine::general_purpose::STANDARD.encode(bytes)
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
        None,
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
    let mut accumulator = GenerateAccumulator::new("request-1".into(), 2, Some(1));
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
fn routed_expert_numpy_rows_merge_across_bootstrap_prefill_and_decode() {
    let merged = merge_routed_expert_payloads(
        Some(routed_expert_payload(vec![10, 11], 2)),
        Some(routed_expert_payload(vec![12], 1)),
    )
    .unwrap()
    .unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(merged)
        .unwrap();
    let array = ndarray::ArrayD::<i16>::read_npy(std::io::Cursor::new(bytes)).unwrap();

    assert_eq!(array.shape(), &[3, 1, 1]);
    assert_eq!(array.iter().copied().collect::<Vec<_>>(), vec![10, 11, 12]);
}

#[test]
fn routed_expert_budget_is_cumulative_across_unary_and_streaming_choices() {
    // Each payload is below the 32 MiB decoded per-payload limit, while two
    // base64 payloads together exceed the 64 MiB response budget.
    let payload = large_routed_expert_payload(24 * 1024 * 1024);

    {
        let mut accumulator = GenerateAccumulator::new("request-unary".into(), 2, None);
        let mut first = output(0, 11, true);
        first.generate_metadata.as_mut().unwrap().routed_experts = Some(payload.clone());
        accumulator.push(first).unwrap();
        let mut second = output(1, 12, true);
        second.generate_metadata.as_mut().unwrap().routed_experts = Some(payload.clone());
        let error = accumulator
            .push(second)
            .expect_err("unary routed-expert budget must span all choices");
        assert!(error.to_string().contains("cumulative"));
    }

    {
        let mut accumulator = GenerateStreamAccumulator::new("request-stream".into(), 2, None);
        let mut first = output(0, 11, true);
        first.generate_metadata.as_mut().unwrap().routed_experts = Some(payload.clone());
        accumulator.push(first, false).unwrap();
        let mut second = output(1, 12, true);
        second.generate_metadata.as_mut().unwrap().routed_experts = Some(payload);
        let error = accumulator
            .push(second, false)
            .expect_err("stream routed-expert budget must span all choices");
        assert!(error.to_string().contains("cumulative"));
    }
}

#[test]
fn unary_accumulator_preserves_backend_completion_usage() {
    let mut accumulator = GenerateAccumulator::new("request-1".into(), 1, None);
    let mut terminal = output(0, 11, true);
    terminal.completion_usage = Some(CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 1,
        total_tokens: 4,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });
    accumulator.push(terminal).unwrap();

    let response = accumulator.finish().unwrap();

    assert_eq!(response.usage.unwrap().total_tokens, 4);
}

#[test]
fn unary_accumulator_rejects_missing_requested_logprobs() {
    let mut accumulator = GenerateAccumulator::new("request-1".into(), 1, Some(1));
    let mut broken = output(0, 11, true);
    broken.log_probs = None;
    assert!(accumulator.push(broken).is_err());
}

#[test]
fn token_logprob_matches_target_vllm_token_id_shape_and_requested_cap() {
    let selected_bytes = vec![0xf0, 0x9f, 0x98, 0x80];
    let token = build_token_logprob(
        42,
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
        1,
    );

    assert_eq!(token.token, "token_id:42");
    assert_eq!(token.logprob, -0.125);
    assert_eq!(token.bytes, None);
    assert_eq!(token.top_logprobs.len(), 1);
    assert_eq!(token.top_logprobs[0].token, "token_id:7");
    assert_eq!(token.top_logprobs[0].bytes, None);
}

#[test]
fn token_logprob_does_not_leak_decoded_or_uncapped_candidates() {
    let token = build_token_logprob(
        42,
        &[
            TopLogprob {
                rank: 1,
                token_id: 42,
                token: Some("decoded-selected".to_string()),
                logprob: f64::NEG_INFINITY,
                bytes: Some(b"decoded-selected".to_vec()),
            },
            TopLogprob {
                rank: 2,
                token_id: 7,
                token: Some("decoded-seven".to_string()),
                logprob: f64::NAN,
                bytes: Some(b"decoded-seven".to_vec()),
            },
            TopLogprob {
                rank: 3,
                token_id: 8,
                token: Some("decoded-eight".to_string()),
                logprob: -0.5,
                bytes: Some(b"decoded-eight".to_vec()),
            },
        ],
        2,
    );

    assert_eq!(token.token, "token_id:42");
    assert_eq!(token.bytes, None);
    assert_eq!(token.logprob, -9999.0);
    assert_eq!(token.top_logprobs.len(), 2);
    assert_eq!(token.top_logprobs[0].token, "token_id:42");
    assert_eq!(token.top_logprobs[1].token, "token_id:7");
    assert!(token.top_logprobs.iter().all(|candidate| {
        candidate.bytes.is_none()
            && candidate.logprob.is_finite()
            && !candidate.token.contains("decoded")
    }));
}

#[test]
fn token_logprob_omits_unrelated_candidates_when_sampled_token_is_missing() {
    let token = build_token_logprob(
        42,
        &[TopLogprob {
            rank: 1,
            token_id: 7,
            token: Some("decoded-seven".to_string()),
            logprob: -0.25,
            bytes: Some(b"decoded-seven".to_vec()),
        }],
        1,
    );

    assert_eq!(token.token, "token_id:42");
    assert_eq!(token.logprob, -9999.0);
    assert_eq!(token.bytes, None);
    assert!(token.top_logprobs.is_empty());
}

#[test]
fn unary_and_stream_accumulators_share_target_vllm_logprob_shape() {
    let mut backend_output = output(0, 42, true);
    backend_output.log_probs = Some(vec![-0.125]);
    backend_output.top_logprobs = Some(vec![vec![
        TopLogprob {
            rank: 1,
            token_id: 42,
            token: Some("decoded-selected".to_string()),
            logprob: -0.125,
            bytes: Some(b"decoded-selected".to_vec()),
        },
        TopLogprob {
            rank: 2,
            token_id: 7,
            token: Some("decoded-seven".to_string()),
            logprob: -0.25,
            bytes: Some(b"decoded-seven".to_vec()),
        },
        TopLogprob {
            rank: 3,
            token_id: 8,
            token: Some("decoded-eight".to_string()),
            logprob: -0.5,
            bytes: Some(b"decoded-eight".to_vec()),
        },
    ]]);

    let mut unary = GenerateAccumulator::new("request-1".into(), 1, Some(2));
    unary.push(backend_output.clone()).unwrap();
    let unary = unary.finish().unwrap();
    let unary_logprob = &unary.choices[0]
        .logprobs
        .as_ref()
        .unwrap()
        .content
        .as_ref()
        .unwrap()[0];

    let mut stream = GenerateStreamAccumulator::new("request-1".into(), 1, Some(2));
    let stream = stream.push(backend_output, false).unwrap().unwrap();
    let stream_logprob = &stream.choices[0]
        .logprobs
        .as_ref()
        .unwrap()
        .content
        .as_ref()
        .unwrap()[0];

    for logprob in [unary_logprob, stream_logprob] {
        assert_eq!(logprob.token, "token_id:42");
        assert_eq!(logprob.bytes, None);
        assert_eq!(logprob.top_logprobs.len(), 2);
        assert_eq!(logprob.top_logprobs[1].token, "token_id:7");
        assert!(
            logprob
                .top_logprobs
                .iter()
                .all(|candidate| candidate.bytes.is_none())
        );
    }
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

    let invalid_explicit_null: GenerateRequest = serde_json::from_value(serde_json::json!({
        "token_ids": [11],
        "sampling_params": {"temperature": null}
    }))
    .unwrap();
    let response = handler(
        State(empty_service.state_clone()),
        HeaderMap::new(),
        Json(invalid_explicit_null),
    )
    .await;
    let body = assert_nested_boundary_error(response, 400, "invalid_request_error").await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("sampling_params.temperature")
    );

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
async fn worker_sampling_validation_errors_return_http_400() {
    for engine_error in [
        DynamoErrorType::InvalidArgument,
        DynamoErrorType::Backend(dynamo_runtime::error::BackendError::InvalidArgument),
    ] {
        let engine: crate::types::inference::generate::GenerateStreamingEngine =
            Arc::new(TypedErrorGenerateEngine(engine_error));
        let (task, _context, mut connection_handle, _state) = direct_unary(engine).await;
        let response = task
            .await
            .expect("unary generate task panicked")
            .expect_err("sampling validation must fail the request");
        connection_handle.disarm();

        assert_eq!(response.0, StatusCode::BAD_REQUEST);
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
    let mut accumulator = GenerateStreamAccumulator::new("request-1".into(), 1, Some(1));
    let mut delta = output(0, 11, false);
    let routed_experts = routed_expert_payload(vec![7], 1);
    delta.completion_usage = Some(CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 1,
        total_tokens: 4,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });
    delta.generate_metadata = Some(GenerateBackendMetadata {
        routed_experts: Some(routed_experts.clone()),
        prefill_routed_experts: None,
        ..Default::default()
    });
    let delta = accumulator.push(delta, true).unwrap().unwrap();
    assert_eq!(delta.choices[0].token_ids, vec![11]);
    assert_eq!(
        delta.choices[0].routed_experts.as_deref(),
        Some(routed_experts.as_str())
    );
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
fn continuous_stream_usage_is_per_choice_and_final_usage_is_aggregate() {
    let usage = || CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 0,
        total_tokens: 3,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    };
    let mut accumulator = GenerateStreamAccumulator::new("request-1".into(), 2, None);

    let mut first = output(0, 11, false);
    first.completion_usage = Some(usage());
    assert_eq!(
        accumulator
            .push(first, true)
            .unwrap()
            .unwrap()
            .usage
            .unwrap()
            .completion_tokens,
        1
    );

    let mut first_terminal = output(0, 12, true);
    first_terminal.completion_usage = Some(usage());
    assert_eq!(
        accumulator
            .push(first_terminal, true)
            .unwrap()
            .unwrap()
            .usage
            .unwrap()
            .completion_tokens,
        2
    );

    let mut second_terminal = output(1, 21, true);
    second_terminal.completion_usage = Some(usage());
    assert_eq!(
        accumulator
            .push(second_terminal, true)
            .unwrap()
            .unwrap()
            .usage
            .unwrap()
            .completion_tokens,
        1
    );

    let final_chunk = accumulator.finish(true).unwrap().unwrap();
    assert!(final_chunk.choices.is_empty());
    assert_eq!(final_chunk.usage.unwrap().completion_tokens, 3);
}

#[test]
fn stream_accumulator_rejects_missing_terminal_choice() {
    let mut accumulator = GenerateStreamAccumulator::new("request-1".into(), 1, None);
    accumulator.push(output(0, 11, false), false).unwrap();
    assert!(accumulator.finish(false).is_err());
}

#[test]
fn request_context_matches_vllm_header_precedence() {
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", "header-id".parse().unwrap());
    headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "7".parse().unwrap());

    assert_eq!(resolve_request_id(&headers, Some("body-id")), "header-id");
    assert_eq!(
        routing_hints_from_headers(&headers)
            .unwrap()
            .unwrap()
            .dp_rank,
        Some(7)
    );
}

#[test]
fn malformed_data_parallel_rank_is_rejected_instead_of_unpinned() {
    let mut headers = HeaderMap::new();
    headers.insert(
        HEADER_DATA_PARALLEL_RANK_ALIAS,
        "not-a-rank".parse().unwrap(),
    );
    assert!(routing_hints_from_headers(&headers).is_err());
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
    headers.insert(HEADER_WORKER_INSTANCE_ID, "17".parse().unwrap());
    headers.insert(HEADER_PREFILL_INSTANCE_ID, "19".parse().unwrap());
    headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "3".parse().unwrap());
    headers.insert(HEADER_PREFILL_DP_RANK, "5".parse().unwrap());
    headers.insert("x-dynamo-session-id", "rollout-1".parse().unwrap());
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "request_id": "body-request",
        "token_ids": [11, 22, 33],
        "sampling_params": {"max_tokens": 4},
        "priority": -2
    }))
    .unwrap();

    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let response = handler(State(service.state_clone()), headers, Json(request)).await;
    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response: GenerateResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(response.request_id, "header-request");
    assert_eq!(response.model.as_deref(), Some("test-model"));
    assert!(
        response
            .created
            .is_some_and(|created| (before..=after).contains(&created))
    );
    assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 3);
    assert_eq!(response.usage.as_ref().unwrap().completion_tokens, 1);
    assert_eq!(response.usage.as_ref().unwrap().total_tokens, 4);
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
    let routing = captured.routing.as_ref().unwrap();
    assert_eq!(routing.backend_instance_id, Some(17));
    assert_eq!(routing.decode_worker_id, Some(17));
    assert_eq!(routing.prefill_worker_id, Some(19));
    assert_eq!(routing.dp_rank, Some(3));
    assert_eq!(routing.prefill_dp_rank, Some(5));
    assert_eq!(captured.session_affinity.as_deref(), Some("rollout-1"));
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
        None,
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
