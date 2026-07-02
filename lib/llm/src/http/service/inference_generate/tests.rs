// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::discovery::WorkerSet;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::inference::generate::GenerateBackendMetadata;
use crate::types::Annotated;
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContext, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use futures::stream;
use std::sync::Mutex;

#[derive(Debug)]
struct CapturedGenerate {
    request: GenerateRequest,
    dp_rank: Option<u32>,
}

struct CaptureGenerateEngine {
    captured: Arc<Mutex<Option<CapturedGenerate>>>,
}

struct ErrorGenerateEngine;

struct PendingGenerateEngine {
    context: Arc<Mutex<Option<Arc<dyn AsyncEngineContext>>>>,
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

    let response = handler(State(service.state_clone()), headers, Json(request))
        .await
        .unwrap();
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
    .await
    .unwrap();
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
    .await
    .unwrap();
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
    .await
    .unwrap();
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
