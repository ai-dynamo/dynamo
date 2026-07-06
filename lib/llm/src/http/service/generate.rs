// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental engine-native endpoint, **disabled by default**;
//! opt in via the `enable_engine_apis` builder flag or the
//! `DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE` env var. When enabled it registers
//! a frontend-native handler that preserves the complete request in an opaque
//! backend envelope. Streaming (`stream=true`) remains unimplemented.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::create_connection_monitor;
use super::metrics::{CancellationLabels, ErrorType};
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{
    GenerateRequest, GenerateResponse, GenerateResponseOptions,
};

const X_REQUEST_ID_HEADER: &str = "x-request-id";
const X_DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

#[derive(Debug, PartialEq, Eq)]
struct GenerateRequestContext {
    request_id: String,
    data_parallel_rank: Option<u32>,
}

/// vLLM-style nested error body: `{"error": {"message", "type", "code"}}`.
#[derive(Serialize, Debug)]
struct GenerateError {
    error: GenerateErrorBody,
}

#[derive(Serialize, Debug)]
struct GenerateErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    code: u16,
}

/// Create an Axum [`Router`] for the token-in/token-out `Generate` endpoint.
/// If no path is provided, the default path is `/inference/v1/generate`.
pub fn generate_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/inference/v1/generate".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_generate))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// Build a vLLM-style nested-`error` response.
fn generate_error_response(code: StatusCode, error_type: &str, message: String) -> Response {
    (
        code,
        Json(GenerateError {
            error: GenerateErrorBody {
                message,
                error_type: error_type.to_string(),
                code: code.as_u16(),
            },
        }),
    )
        .into_response()
}

/// Resolve the request metadata that vLLM keeps outside the public JSON body.
fn resolve_generate_request_context(
    headers: &HeaderMap,
    body_request_id: Option<&str>,
) -> GenerateRequestContext {
    let request_id = headers
        .get(X_REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
        .or_else(|| body_request_id.map(ToOwned::to_owned))
        .unwrap_or_else(|| get_or_create_request_id(headers));
    let data_parallel_rank = headers
        .get(X_DATA_PARALLEL_RANK_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());

    GenerateRequestContext {
        request_id,
        data_parallel_rank,
    }
}

/// Convert vLLM's lower-is-higher priority to Dynamo's higher-is-higher scale.
fn dynamo_routing_priority(vllm_priority: i32) -> i32 {
    vllm_priority.saturating_neg()
}

fn generate_dispatch_span(request_id: &str) -> tracing::Span {
    tracing::info_span!(target: "request_span", "generate", request_id = %request_id)
}

/// Project routing controls while retaining the complete client request in
/// `extra_args.vllm_tito`. The backend remains the authority for interpreting
/// every vLLM-specific field.
fn preprocessed_from_generate(
    request: &GenerateRequest,
    model: &str,
    data_parallel_rank: Option<u32>,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &request.sampling_params;
    let max_tokens = sampling.max_tokens();
    let routing_priority = dynamo_routing_priority(request.priority);

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(request.token_ids.clone())
        .stop_conditions(StopConditions {
            max_tokens,
            min_tokens: sampling.min_tokens(),
            ignore_eos: Some(sampling.ignore_eos()),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            n: Some(1),
            ..Default::default()
        })
        .output_options(Default::default())
        .routing(Some(crate::protocols::common::preprocessor::RoutingHints {
            dp_rank: data_parallel_rank,
            expected_output_tokens: max_tokens,
            // `priority_jump` is a boost-only scheduler input. Preserve penalties
            // in signed `priority`, matching the standard preprocessor projection.
            priority_jump: Some(routing_priority.max(0) as f64),
            priority: Some(routing_priority),
            ..Default::default()
        }))
        .extra_args(Some(serde_json::json!({
            "vllm_tito": serde_json::to_value(request)?,
        })))
        .build()
        .map_err(|error| anyhow::anyhow!("failed to build PreprocessedRequest: {error}"))
}

/// Resolve, route, and dispatch a frontend-native token-in/token-out request.
async fn handler_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    if let Err(response) = check_ready(&state) {
        return response.into_response();
    }

    if let Err(message) = request.validate() {
        return generate_error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }

    if request.stream {
        return generate_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for /inference/v1/generate yet".to_string(),
        );
    }
    let response_options = request.response_options();

    let model = match &request.model {
        Some(model) => model.clone(),
        None => {
            let models = state.manager().list_generate_models();
            match models.len() {
                1 => models.into_iter().next().unwrap(),
                0 => {
                    return generate_error_response(
                        StatusCode::NOT_FOUND,
                        "not_found",
                        "no generate-capable model is registered".to_string(),
                    );
                }
                _ => {
                    return generate_error_response(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "multiple models are registered; specify `model` in the request"
                            .to_string(),
                    );
                }
            }
        }
    };

    if let Err(response) = check_model_serving_ready(&state, &model) {
        return response.into_response();
    }

    let (engine, _parsing) = match state.manager().get_generate_engine_with_parsing(&model) {
        Ok(engine) => engine,
        Err(error) => {
            let (status, error_type) = match error {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return generate_error_response(status, error_type, error.to_string());
        }
    };

    let request_context = resolve_generate_request_context(&headers, request.request_id.as_deref());
    let preprocessed =
        match preprocessed_from_generate(&request, &model, request_context.data_parallel_rank) {
            Ok(preprocessed) => preprocessed,
            Err(error) => {
                return generate_error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    error.to_string(),
                );
            }
        };

    let request_id = request_context.request_id;
    let context: Context<PreprocessedRequest> =
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(response) => return response.into_response(),
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        engine_context,
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    let dispatch_span = generate_dispatch_span(&request_id);
    // Unary work must outlive the Axum handler so dropping the handler can signal
    // the armed connection monitor, which kills the shared engine context.
    let response = match tokio::spawn(
        generate_dispatch(
            engine,
            context,
            request_id,
            model,
            state.clone(),
            response_options,
        )
        .instrument(dispatch_span),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => generate_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("generate task panicked: {error}"),
        ),
    };

    connection_handle.disarm();
    response
}

async fn generate_dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
    response_options: GenerateResponseOptions,
) -> Response {
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        state.manager().metric_model_for(&model),
        super::metrics::Endpoint::Generate,
        false,
        &request_id,
    );
    let request_context = context.context();
    let stream = match engine.generate(context).await {
        Ok(stream) => stream,
        Err(error) => {
            let was_rejected = super::metrics::request_was_rejected(error.as_ref());
            inflight_guard.mark_error(if request_context.is_killed() {
                ErrorType::Cancelled
            } else if was_rejected {
                ErrorType::Unavailable
            } else {
                ErrorType::Internal
            });
            if was_rejected {
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
                return generate_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    format!("engine rejected the request: {error:#}"),
                );
            }
            return generate_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("failed to generate: {error:#}"),
            );
        }
    };

    let engine_context = stream.context();
    match GenerateResponse::from_annotated_stream_with_options(
        stream,
        request_id.clone(),
        response_options,
    )
    .await
    {
        Ok(response) => {
            if engine_context.is_killed() {
                inflight_guard.mark_error(ErrorType::Cancelled);
                return generate_error_response(
                    StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                    "request_cancelled",
                    "request was cancelled".to_string(),
                );
            }
            if !response.is_complete_unary() {
                inflight_guard.mark_error(ErrorType::Internal);
                tracing::error!(%request_id, "generate stream ended without a complete choice");
                return generate_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("generation produced no complete choice for {request_id}"),
                );
            }
            inflight_guard.mark_ok();
            Json(response).into_response()
        }
        Err(error) => {
            inflight_guard.mark_error(if engine_context.is_killed() {
                ErrorType::Cancelled
            } else {
                ErrorType::Internal
            });
            tracing::error!(%request_id, %error, "failed to fold generate stream");
            generate_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("failed to fold generate stream for {request_id}"),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::service_v2::{HttpService, VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV};
    use super::*;
    use tokio_util::sync::CancellationToken;
    use tracing::field::{Field, Visit};
    use tracing::{Subscriber, span};
    use tracing_subscriber::Layer;
    use tracing_subscriber::prelude::*;

    #[derive(Clone)]
    struct RequestIdCaptureLayer(Arc<Mutex<Option<String>>>);

    impl<S: Subscriber> Layer<S> for RequestIdCaptureLayer {
        fn on_new_span(
            &self,
            attrs: &span::Attributes<'_>,
            _id: &span::Id,
            _context: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = RequestIdVisitor::default();
            attrs.record(&mut visitor);
            if visitor.request_id.is_some() {
                *self.0.lock().unwrap() = visitor.request_id;
            }
        }
    }

    #[derive(Default)]
    struct RequestIdVisitor {
        request_id: Option<String>,
    }

    impl Visit for RequestIdVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "request_id" {
                self.request_id = Some(value.to_string());
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "request_id" {
                self.request_id = Some(format!("{value:?}"));
            }
        }
    }

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: Option<bool>) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let builder = HttpService::builder().port(port);
        let builder = match enable_generate {
            Some(enabled) => builder.enable_engine_apis(enabled),
            None => builder,
        };
        let service = builder.build().unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_structured_404() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_found");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{},"stream":true}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_IMPLEMENTED.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "not_implemented");
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_rejects_empty_token_ids() {
        let (port, handle) = serve(Some(true)).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");

        assert_eq!(resp.status().as_u16(), StatusCode::BAD_REQUEST.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"].as_str().is_some_and(
                |message| message.contains("token_ids must contain at least one token")
            )
        );
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_enforces_vllm_rust_request_rules() {
        let (port, handle) = serve(Some(true)).await;
        let client = reqwest::Client::new();
        let invalid = [
            r#"{"token_ids":[1],"sampling_params":{},"stream_options":{"include_usage":true}}"#,
            r#"{"token_ids":[1],"sampling_params":{"max_tokens":0}}"#,
            r#"{"token_ids":[1],"sampling_params":{"prompt_logprobs":-2}}"#,
            r#"{"token_ids":[1],"sampling_params":{"min_tokens":3,"max_tokens":2}}"#,
        ];

        for body in invalid {
            let resp = client
                .post(format!("http://localhost:{port}/inference/v1/generate"))
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .expect("generate request failed");
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            let body: serde_json::Value = resp.json().await.expect("json body");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }

        handle.abort();
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_404_by_default() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, None::<&str>)],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                handle.abort();
            },
        )
        .await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn generate_route_is_registered_when_enabled_by_env() {
        temp_env::async_with_vars(
            [(VLLM_ENABLE_INFERENCE_V1_GENERATE_ENV, Some("1"))],
            async {
                let (port, handle) = serve(None).await;
                let resp = reqwest::Client::new()
                    .post(format!("http://localhost:{}/inference/v1/generate", port))
                    .header("content-type", "application/json")
                    .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
                    .send()
                    .await
                    .expect("generate request failed");
                assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
                let body: serde_json::Value = resp.json().await.expect("json body");
                assert_eq!(body["error"]["type"], "not_found");
                handle.abort();
            },
        )
        .await;
    }

    #[test]
    fn complete_request_envelope_reaches_preprocessed_request_unchanged() {
        let raw = serde_json::json!({
            "request_id": "req-forward",
            "token_ids": [1, 2],
            "sampling_params": {
                "max_tokens": 8,
                "future_sampling_field": {"nested": true}
            },
            "model": "test-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "features": {"future_feature": [1, 2, 3]},
            "priority": 7,
            "kv_transfer_params": {"remote": "worker-a"},
            "future_top_level_field": {"anything": "works"}
        });
        let request: GenerateRequest =
            serde_json::from_value(raw.clone()).expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None).expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, Some(8));
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            Some(8)
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority),
            Some(-7),
            "vLLM lower-is-higher priority must be inverted for Dynamo routing"
        );
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.priority_jump),
            Some(0.0)
        );
        let envelope = preprocessed
            .extra_args
            .as_ref()
            .and_then(|extra| extra.get("vllm_tito"))
            .expect("vllm_tito envelope");

        assert_eq!(envelope, &raw);
    }

    #[test]
    fn omitted_max_tokens_stays_omitted_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None).expect("build request");
        assert_eq!(preprocessed.stop_conditions.max_tokens, None);
        assert_eq!(preprocessed.stop_conditions.min_tokens, None);
        assert_eq!(
            preprocessed
                .routing
                .as_ref()
                .and_then(|routing| routing.expected_output_tokens),
            None
        );
    }

    #[test]
    fn explicit_zero_min_tokens_stays_explicit_in_control_shadow() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {"min_tokens": 0},
            "model": "test-model"
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", None).expect("build request");
        assert_eq!(preprocessed.stop_conditions.min_tokens, Some(0));
    }

    #[test]
    fn generate_request_context_matches_vllm_header_precedence() {
        let mut headers = HeaderMap::new();
        headers.insert(X_REQUEST_ID_HEADER, "header-request".parse().unwrap());
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "3".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "header-request");
        assert_eq!(context.data_parallel_rank, Some(3));
    }

    #[test]
    fn generate_request_context_falls_back_and_ignores_invalid_dp_rank() {
        let mut headers = HeaderMap::new();
        headers.insert(X_DATA_PARALLEL_RANK_HEADER, "invalid".parse().unwrap());

        let context = resolve_generate_request_context(&headers, Some("body-request"));

        assert_eq!(context.request_id, "body-request");
        assert_eq!(context.data_parallel_rank, None);
    }

    #[test]
    fn generate_dispatch_span_uses_resolved_request_id() {
        let captured_request_id = Arc::new(Mutex::new(None));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(RequestIdCaptureLayer(captured_request_id.clone())),
        );

        let _dispatch_span = generate_dispatch_span("header-request");

        assert_eq!(
            captured_request_id.lock().unwrap().as_deref(),
            Some("header-request")
        );
    }

    #[test]
    fn generate_control_shadow_carries_dp_rank_and_inverted_priority() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2],
            "sampling_params": {},
            "priority": -7
        }))
        .expect("deserialize request");

        let preprocessed =
            preprocessed_from_generate(&request, "test-model", Some(3)).expect("build request");
        let routing = preprocessed.routing.as_ref().expect("routing hints");

        assert_eq!(routing.dp_rank, Some(3));
        assert_eq!(routing.priority, Some(7));
        assert_eq!(routing.priority_jump, Some(7.0));
    }

    #[test]
    fn priority_inversion_saturates_at_i32_min() {
        assert_eq!(dynamo_routing_priority(i32::MIN), i32::MAX);
    }
}
