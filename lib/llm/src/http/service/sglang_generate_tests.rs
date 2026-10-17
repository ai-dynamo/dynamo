// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::http::service::generate::tests::{WorkerUnavailableEngine, dispatch_test_context};
use crate::http::service::metrics::{Endpoint, RequestType, Status};
use crate::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_runtime::protocols::maybe_error::MaybeError;

/// `dispatch` takes the metric label the handler already resolved.
const METRIC_MODEL: &str = "test-model";

#[test]
fn cache_salt_matches_routing_and_native_payload() {
    for salt in [None, Some(""), Some("tenant-a")] {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1, 2],
            "cache_salt": salt,
        }))
        .unwrap();
        let request = preprocessed_request(request, "test-model", None, "request-id").unwrap();
        assert_eq!(request.tracker.unwrap().isl_tokens(), Some(2));
        let expected = salt.filter(|salt| !salt.is_empty());
        assert_eq!(
            request.routing.unwrap().cache_namespace.as_deref(),
            expected
        );
        assert_eq!(
            request.extra_args.unwrap()["sglang_tito"]["cache_salt"].as_str(),
            expected,
        );
    }
}

#[tokio::test]
async fn worker_unavailable_dispatch_returns_503() {
    let engine: crate::types::openai::generate::GenerateStreamingEngine =
        std::sync::Arc::new(WorkerUnavailableEngine);
    let state = crate::http::service::service_v2::HttpService::builder()
        .build()
        .unwrap()
        .state_clone();
    let (tx, _rx) = tokio::sync::oneshot::channel();

    let response = dispatch(
        engine,
        dispatch_test_context(),
        METRIC_MODEL.to_string(),
        state.clone(),
        ConnectionHandle::create_disabled(tx),
        true,
        None,
    )
    .await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read error response");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("parse error response");
    assert_eq!(body["error"]["message"], "Service temporarily unavailable");
    assert!(!body.to_string().contains("unknown endpoint"));

    assert_eq!(
        state.metrics_clone().get_request_counter(
            METRIC_MODEL,
            &Endpoint::Generate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::Unavailable,
        ),
        1
    );
}

#[test]
fn model_aliases_are_deduplicated_before_implicit_selection() {
    let manager = crate::discovery::ModelManager::new();
    assert!(manager.register_alias("alias", "primary"));

    let models = canonical_generate_models(
        &manager,
        vec![
            "primary".to_string(),
            "alias".to_string(),
            "other".to_string(),
        ],
    );

    assert_eq!(models, vec!["other".to_string(), "primary".to_string()]);
}

#[test]
fn streaming_errors_use_sglang_shape() {
    let error = anyhow::anyhow!("backend failure");
    let (error_type, body) = stream_error(error.as_ref());
    let error: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(error_type, ErrorType::Internal);
    assert_eq!(
        error,
        serde_json::json!({"error": {"message": "Internal server error"}})
    );
}

#[test]
fn streaming_errors_preserve_typed_classification() {
    use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

    for (dynamo_type, expected_type, expected_message) in [
        (
            DynamoErrorType::InvalidArgument,
            ErrorType::Validation,
            "invalid sampling parameters",
        ),
        (
            DynamoErrorType::Unavailable,
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            DynamoErrorType::WorkerUnavailable,
            ErrorType::Unavailable,
            "Service temporarily unavailable",
        ),
        (
            DynamoErrorType::Cancelled,
            ErrorType::Cancelled,
            "Request cancelled",
        ),
    ] {
        let error = DynamoError::builder()
            .error_type(dynamo_type)
            .message("invalid sampling parameters")
            .build();
        let error = axum::Error::new(error);
        let (error_type, body) = stream_error(&error);
        let body: serde_json::Value = serde_json::from_str(&body).unwrap();

        assert_eq!(error_type, expected_type);
        assert_eq!(body["error"]["message"], expected_message);
    }
}

#[test]
fn streaming_deadline_wins_over_overload_classification() {
    use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType};

    let error = DynamoError::builder()
        .error_type(DynamoErrorType::WorkerOverloaded)
        .message("worker overloaded")
        .cause(
            DynamoError::builder()
                .error_type(DynamoErrorType::DeadlineExceeded)
                .reason(
                    dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded")
                        .unwrap(),
                )
                .message("internal deadline detail")
                .build(),
        )
        .build();
    let (error_type, body) = stream_error(&error);
    let body: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(error_type, ErrorType::Cancelled);
    assert_eq!(body["error"]["message"], "request deadline exceeded");
}

struct DeadlineEngine;

#[async_trait::async_trait]
impl
    dynamo_runtime::engine::AsyncEngine<
        dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        dynamo_runtime::pipeline::ManyOut<
            crate::protocols::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
        >,
        dynamo_runtime::pipeline::Error,
    > for DeadlineEngine
{
    async fn generate(
        &self,
        _request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
    ) -> Result<
        dynamo_runtime::pipeline::ManyOut<
            crate::protocols::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>,
        >,
        dynamo_runtime::pipeline::Error,
    > {
        Err(dynamo_runtime::error::DynamoError::builder()
            .error_type(dynamo_runtime::error::ErrorType::DeadlineExceeded)
            .reason(
                dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded").unwrap(),
            )
            .message("router deadline exceeded before stream start")
            .build()
            .into())
    }
}

#[tokio::test]
async fn pre_stream_deadline_maps_to_http_429_with_cancelled_metric() {
    let engine: crate::types::openai::generate::GenerateStreamingEngine = Arc::new(DeadlineEngine);
    let service = service_v2::HttpService::builder().build().unwrap();
    let state = service.state_clone();
    let context = Context::new(
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .expect("build dispatch test request"),
    );
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.context(),
        None,
        CancellationLabels {
            model: "test-model".to_string(),
            endpoint: Endpoint::Generate.to_string(),
            request_type: "streaming".to_string(),
        },
    )
    .await;

    let response = dispatch(
        engine,
        context,
        METRIC_MODEL.to_string(),
        state.clone(),
        stream_handle,
        true,
        None,
    )
    .await;
    connection_handle.disarm();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"]["message"], "request deadline exceeded");
    assert_eq!(
        state.metrics_clone().get_request_counter(
            METRIC_MODEL,
            &Endpoint::Generate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::Cancelled,
        ),
        1
    );
}

struct ReplayEngine(Vec<crate::protocols::Annotated<LLMEngineOutput>>);

#[async_trait::async_trait]
impl
    dynamo_runtime::engine::AsyncEngine<
        dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
        dynamo_runtime::pipeline::Error,
    > for ReplayEngine
{
    async fn generate(
        &self,
        request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
    ) -> Result<
        dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
        dynamo_runtime::pipeline::Error,
    > {
        let stream = Box::pin(futures::stream::iter(self.0.clone()));
        Ok(dynamo_runtime::engine::ResponseStream::new(
            stream,
            request.context(),
        ))
    }
}

struct FirstChunkThenPendingEngine;

#[async_trait::async_trait]
impl
    dynamo_runtime::engine::AsyncEngine<
        dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
        dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
        dynamo_runtime::pipeline::Error,
    > for FirstChunkThenPendingEngine
{
    async fn generate(
        &self,
        request: dynamo_runtime::pipeline::SingleIn<PreprocessedRequest>,
    ) -> Result<
        dynamo_runtime::pipeline::ManyOut<crate::protocols::Annotated<LLMEngineOutput>>,
        dynamo_runtime::pipeline::Error,
    > {
        let stream =
            futures::stream::iter([sglang_chunk(vec![5])]).chain(futures::stream::pending());
        Ok(dynamo_runtime::engine::ResponseStream::new(
            Box::pin(stream),
            request.context(),
        ))
    }
}

fn sglang_chunk(token_ids: Vec<u32>) -> crate::protocols::Annotated<LLMEngineOutput> {
    crate::protocols::Annotated::from_data(LLMEngineOutput {
        engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": token_ids}})),
        token_ids,
        index: Some(0),
        ..Default::default()
    })
}

/// A P/D stream: prefill's output item, then two decode chunks with
/// terminal finish reason and usage.
fn observed_outputs() -> Vec<crate::protocols::Annotated<LLMEngineOutput>> {
    let prefill = crate::protocols::Annotated {
        event: Some(PREFILL_OUTPUT_ANNOTATION.to_string()),
        ..crate::protocols::Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({"sglang_response": {"meta_info": {}}})),
            ..Default::default()
        })
    };
    let mut last = sglang_chunk(vec![6]);
    let output = last.data.as_mut().unwrap();
    output.finish_reason = Some(crate::protocols::common::FinishReason::Stop);
    output.completion_usage = Some(dynamo_protocols::types::CompletionUsage {
        prompt_tokens: 3,
        completion_tokens: 2,
        total_tokens: 5,
        prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
            cached_tokens: Some(2),
            ..Default::default()
        }),
        completion_tokens_details: None,
    });
    vec![prefill, sglang_chunk(vec![5]), last]
}

fn observed_context(request_id: &str, streaming: bool) -> Context<PreprocessedRequest> {
    let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
        "input_ids": [1, 2, 3],
        "stream": streaming,
    }))
    .unwrap();
    let request = preprocessed_request(request, "test-model", None, request_id).unwrap();
    Context::with_id_and_metadata(request, request_id.to_string(), Default::default())
}

/// Attach agent context and x-request-id as the handler's header parsing
/// does, then build trace state through the policy seam.
fn traced_context(
    mut context: Context<PreprocessedRequest>,
) -> (Context<PreprocessedRequest>, Option<RequestEndTraceState>) {
    context.agent_context = Some(crate::protocols::common::extensions::AgentContext {
        session_id: "sglang-session".to_string(),
        parent_session_id: None,
        session_final: None,
        agent_headers: Default::default(),
        input_trigger: None,
    });
    context.insert(
        crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
        "client-x-request-id".to_string(),
    );
    let trace_state = crate::request_trace::build_request_end_trace_state_for_policy(
        &context,
        &context.tracker,
        &context,
        2,
        true,
    );
    (context, trace_state)
}

async fn request_end_records(
    receiver: &mut tokio::sync::broadcast::Receiver<crate::request_trace::RequestTraceRecord>,
    request_id: &str,
) -> Vec<crate::request_trace::RequestTraceRecord> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut records = Vec::new();
    while let Ok(result) = tokio::time::timeout_at(deadline, receiver.recv()).await {
        match result {
            Ok(record)
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == request_id) =>
            {
                records.push(record);
            }
            Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
    records
}

#[tokio::test]
async fn dispatch_records_frontend_metrics() {
    use crate::http::service::engine_output_metrics::test_support::metric_value;

    for (streaming, request_type) in [(false, RequestType::Unary), (true, RequestType::Stream)] {
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let registry = prometheus::Registry::new();
        state.metrics_clone().register(&registry).unwrap();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = dispatch(
            Arc::new(ReplayEngine(observed_outputs())),
            observed_context("req-sglang-metrics", streaming),
            METRIC_MODEL.to_string(),
            state.clone(),
            ConnectionHandle::create_disabled(tx),
            streaming,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        assert_eq!(state.metrics_clone().get_inflight_count(METRIC_MODEL), 0);
        assert_eq!(
            state.metrics_clone().get_request_counter(
                METRIC_MODEL,
                &Endpoint::Generate,
                &request_type,
                &Status::Success,
                &ErrorType::None,
            ),
            1
        );
        // The P/D prefill-output item carries no tokens and must not count as
        // the first token or add to the output length.
        let families = registry.gather();
        let histogram =
            |name| metric_value(&families, name, &[("model", METRIC_MODEL)]).get_histogram();
        assert_eq!(
            histogram("dynamo_frontend_input_sequence_tokens").get_sample_sum(),
            3.0
        );
        assert_eq!(
            histogram("dynamo_frontend_output_sequence_tokens").get_sample_sum(),
            2.0
        );
        assert_eq!(
            histogram("dynamo_frontend_time_to_first_token_seconds").get_sample_count(),
            1
        );
    }
}

#[tokio::test]
async fn dispatch_emits_request_end_trace_once() {
    crate::request_trace::init_bus_for_test(64);
    let mut receiver = crate::request_trace::subscribe();
    for streaming in [false, true] {
        let request_id = format!("req-sglang-trace-{streaming}");
        let (context, trace_state) = traced_context(observed_context(&request_id, streaming));
        assert!(trace_state.is_some());
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = dispatch(
            Arc::new(ReplayEngine(observed_outputs())),
            context,
            METRIC_MODEL.to_string(),
            state,
            ConnectionHandle::create_disabled(tx),
            streaming,
            trace_state,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let records = request_end_records(&mut receiver, &request_id).await;
        assert_eq!(records.len(), 1, "streaming={streaming}");
        let record = &records[0];
        assert_eq!(
            record.agent_context.as_ref().unwrap().session_id,
            "sglang-session"
        );
        let request = record.request.as_ref().unwrap();
        assert_eq!(request.x_request_id.as_deref(), Some("client-x-request-id"));
        assert_eq!(request.model.as_deref(), Some("test-model"));
        assert_eq!(request.input_tokens, Some(3));
        assert_eq!(request.output_tokens, Some(2));
        assert_eq!(request.cached_tokens, Some(2));
        assert_eq!(request.replay.as_ref().unwrap().input_length, 3);
        let finish = request.finish_reason_metadata.as_ref().unwrap();
        assert_eq!(finish.backend_finish_reason.as_deref(), Some("stop"));
    }
}

#[tokio::test]
async fn streaming_cancellation_emits_request_end_trace_once() {
    crate::request_trace::init_bus_for_test(64);
    let mut receiver = crate::request_trace::subscribe();
    let request_id = "req-sglang-trace-cancel";
    let (context, trace_state) = traced_context(observed_context(request_id, true));
    let request_context = context.context();
    let state = service_v2::HttpService::builder()
        .build()
        .unwrap()
        .state_clone();
    let (tx, _rx) = tokio::sync::oneshot::channel();
    let response = dispatch(
        Arc::new(FirstChunkThenPendingEngine),
        context,
        METRIC_MODEL.to_string(),
        state.clone(),
        ConnectionHandle::create_disabled(tx),
        true,
        trace_state,
    )
    .await;
    let mut body = response.into_body().into_data_stream();
    assert!(body.next().await.unwrap().is_ok());
    request_context.kill();
    while body.next().await.is_some() {}

    let records = request_end_records(&mut receiver, request_id).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].request.as_ref().unwrap().output_tokens, Some(1));
    assert_eq!(
        state.metrics_clone().get_request_counter(
            METRIC_MODEL,
            &Endpoint::Generate,
            &RequestType::Stream,
            &Status::Error,
            &ErrorType::Cancelled,
        ),
        1
    );
}

#[tokio::test]
async fn unary_dispatch_joins_chunks_and_maps_stream_errors() {
    let chunk = |ids: serde_json::Value| {
        crate::protocols::Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({"sglang_response": {"output_ids": ids}})),
            ..Default::default()
        })
    };
    let invalid = dynamo_runtime::error::DynamoError::builder()
        .error_type(dynamo_runtime::error::ErrorType::InvalidArgument)
        .message("bad sampling params")
        .build();
    let failed = crate::protocols::Annotated::from_err(invalid);
    for (outputs, status, body) in [
        (
            vec![chunk(serde_json::json!([5])), chunk(serde_json::json!([6]))],
            StatusCode::OK,
            serde_json::json!({"output_ids": [5, 6]}),
        ),
        (
            vec![chunk(serde_json::json!([5])), failed],
            StatusCode::BAD_REQUEST,
            serde_json::json!({"error": {"message": "bad sampling params"}}),
        ),
    ] {
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let response = dispatch(
            Arc::new(ReplayEngine(outputs)),
            dispatch_test_context(),
            METRIC_MODEL.to_string(),
            state,
            ConnectionHandle::create_disabled(tx),
            false,
            None,
        )
        .await;
        assert_eq!(response.status(), status);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            body
        );
    }
}
