// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the token-in/token-out `Generate` API
//! (`POST /inference/v1/generate`).
//!
//! This is an experimental endpoint, enabled by default (matching vLLM,
//! which mounts `/inference/v1/generate` for any generate-capable model).
//! The handler dispatches the raw `token_ids` to the model's Backend-free
//! `PreprocessedRequest -> LLMEngineOutput` pipeline and folds the streamed
//! deltas into a single `GenerateResponse`. Disable via
//! `enable_generate_endpoints`. Streaming (`stream=true`) is not yet
//! implemented and returns HTTP 501.

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

use super::disconnect::create_connection_monitor;
use super::metrics::CancellationLabels;
use super::openai::{
    check_model_serving_ready, check_ready, context_from_headers, get_body_limit,
    get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::{SamplingOptions, StopConditions};
use crate::protocols::openai::generate::{GenerateRequest, GenerateResponse};
use tracing::Instrument;

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

/// Build a vLLM-style nested-`error` [`Response`] with the given status and type.
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

/// vLLM's default `max_tokens` when the request omits it.
const DEFAULT_MAX_TOKENS: u64 = 16;

/// Build a [`PreprocessedRequest`] from a [`GenerateRequest`].
///
/// The whole raw request rides opaquely in `extra_args.vllm_tito` so the
/// backend worker can re-parse it against vLLM's typed `GenerateRequest`;
/// the scalar sampling controls that Dynamo core needs for
/// stop/routing decisions are shadowed into the core fields.
fn preprocessed_from_generate(
    req: &GenerateRequest,
    model: &str,
) -> anyhow::Result<PreprocessedRequest> {
    let sampling = &req.sampling_params;
    let max_tokens = sampling
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let ignore_eos = sampling
        .get("ignore_eos")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let min_tokens = sampling
        .get("min_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let stop_conditions = StopConditions {
        max_tokens: Some(max_tokens as u32),
        min_tokens: Some(min_tokens as u32),
        ignore_eos: Some(ignore_eos),
        ..Default::default()
    };

    // PR2 is n=1; sampling knobs beyond n stay in the opaque envelope.
    let sampling_options = SamplingOptions {
        n: Some(1),
        ..Default::default()
    };

    let routing = crate::protocols::common::preprocessor::RoutingHints {
        expected_output_tokens: Some(max_tokens as u32),
        ..Default::default()
    };

    PreprocessedRequest::builder()
        .model(model.to_string())
        .token_ids(req.token_ids.clone())
        .stop_conditions(stop_conditions)
        .sampling_options(sampling_options)
        .output_options(Default::default())
        .routing(Some(routing))
        .extra_args(Some(serde_json::json!({
            "vllm_tito": serde_json::to_value(req)?,
        })))
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build PreprocessedRequest: {e}"))
}

/// Handler for the token-in/token-out `Generate` endpoint.
///
/// Resolves the target model, dispatches the raw `token_ids` to the model's
/// Backend-free `PreprocessedRequest -> LLMEngineOutput` pipeline, and folds
/// the streamed deltas into a single [`GenerateResponse`]. A client disconnect
/// kills the engine context (mirroring the OpenAI completions path).
async fn handler_generate(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    // return a 503 if the service is not ready
    if let Err(err_response) = check_ready(&state) {
        return err_response.into_response();
    }

    // Streaming is a later PR.
    if request.stream {
        return generate_error_response(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "streaming (stream=true) is not implemented for /inference/v1/generate yet".to_string(),
        );
    }

    // Resolve the target model: use `request.model` if given, else the sole
    // registered generate model (error if none or ambiguous).
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

    // Gate on per-model serving readiness: an aggregated request routed to a
    // decode-only / not-yet-ready worker set would otherwise hang on the worker.
    if let Err(err_response) = check_model_serving_ready(&state, &model) {
        return err_response.into_response();
    }

    let (engine, _parsing) = match state.manager().get_generate_engine_with_parsing(&model) {
        Ok(pair) => pair,
        Err(e) => {
            // Select status and error type together so a 503 (model-removal
            // race) never carries a "not_found" type, and vice versa.
            let (code, error_type) = match e {
                crate::discovery::ModelManagerError::ModelUnavailable(_) => {
                    (StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
                }
                _ => (StatusCode::NOT_FOUND, "not_found"),
            };
            return generate_error_response(code, error_type, e.to_string());
        }
    };

    let preprocessed = match preprocessed_from_generate(&request, &model) {
        Ok(preprocessed) => preprocessed,
        Err(e) => {
            return generate_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
            );
        }
    };

    // Build the request context (carrying the request id) and wire a
    // connection monitor so a client disconnect kills the engine context.
    let request_id = get_or_create_request_id(&headers);
    let context: Context<PreprocessedRequest> =
        match context_from_headers(preprocessed, request_id.clone(), &headers) {
            Ok(context) => context,
            Err(err_response) => {
                return err_response.into_response();
            }
        };
    let engine_context = context.context();
    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: super::metrics::Endpoint::Generate.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        engine_context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

    // Spawn the dispatch so the generation task runs (and cleans up) out of
    // band from the axum handler future, mirroring the OpenAI handlers.
    let response = match tokio::spawn(
        generate_dispatch(engine, context, request_id, model, state.clone()).in_current_span(),
    )
    .await
    {
        Ok(response) => response,
        Err(join_err) => generate_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("generate task panicked: {join_err}"),
        ),
    };

    // Long-running work completed without needing cancellation.
    connection_handle.disarm();
    response
}

/// Dispatch a prepared [`Context<PreprocessedRequest>`] to the generate engine
/// and fold the streamed deltas into a single [`GenerateResponse`].
async fn generate_dispatch(
    engine: crate::types::openai::generate::GenerateStreamingEngine,
    context: Context<PreprocessedRequest>,
    request_id: String,
    model: String,
    state: Arc<service_v2::State>,
) -> Response {
    let stream = match engine.generate(context).await {
        Ok(stream) => stream,
        Err(e) => {
            if super::metrics::request_was_rejected(e.as_ref()) {
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Generate);
                return generate_error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service_unavailable",
                    format!("engine rejected the request: {e:#}"),
                );
            }
            return generate_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                format!("failed to generate: {e:#}"),
            );
        }
    };

    let engine_context = stream.context();

    match GenerateResponse::from_annotated_stream(stream, request_id.clone()).await {
        Ok(response) => {
            // If the engine context was killed (client disconnect), the response
            // was assembled but never delivered.
            if engine_context.is_killed() {
                return generate_error_response(
                    StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                    "request_cancelled",
                    "request was cancelled".to_string(),
                );
            }
            // Guard the unary n=1 profile: a completed generation must yield at
            // least one choice, each carrying a terminal finish_reason. An empty
            // stream or a premature (finish_reason-less) EOF would otherwise
            // surface as a misleading HTTP 200 with empty/partial choices.
            if !response.is_complete_unary() {
                tracing::error!("generate stream for {request_id} ended without a complete choice");
                return generate_error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    format!("generation produced no complete choice for {request_id}"),
                );
            }
            Json(response).into_response()
        }
        Err(e) => {
            tracing::error!("Failed to fold generate stream for {request_id}: {e:?}");
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
    use super::StatusCode;
    use super::service_v2::HttpService;
    use tokio_util::sync::CancellationToken;

    /// Spin up an `HttpService` bound to an ephemeral port and return the port
    /// plus the run handle. Mirrors the reqwest-based router tests in
    /// `service_v2`.
    async fn serve(enable_generate: bool) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind ephemeral port");
        let port = listener.local_addr().unwrap().port();
        let service = HttpService::builder()
            .port(port)
            .enable_generate_endpoints(enable_generate)
            .build()
            .unwrap();
        let cancel_token = CancellationToken::new();
        let handle = tokio::spawn(async move {
            service.run_with_listener(cancel_token, listener).await.ok();
        });
        // Give the server a moment to start listening.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        (port, handle)
    }

    #[tokio::test]
    async fn generate_route_no_model_returns_404() {
        // Endpoint enabled + service ready (default stage) but no model
        // registered: the handler runs, model resolution finds zero generate
        // models, and returns a structured 404 nested-`error` body. This proves
        // the route is mounted AND the real handler executed (a route-not-found
        // 404 has an empty body).
        let (port, handle) = serve(true).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert!(
            body.get("error").is_some(),
            "expected nested-`error` body, got {body}"
        );
        handle.abort();
    }

    #[tokio::test]
    async fn generate_route_streaming_returns_501() {
        // stream=true is not implemented in this PR: the handler returns a 501
        // nested-`error` body after the readiness gate passes (default stage).
        let (port, handle) = serve(true).await;
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
    async fn generate_route_404_when_disabled() {
        let (port, handle) = serve(false).await;
        let resp = reqwest::Client::new()
            .post(format!("http://localhost:{}/inference/v1/generate", port))
            .header("content-type", "application/json")
            .body(r#"{"token_ids":[1,2,3],"sampling_params":{}}"#)
            .send()
            .await
            .expect("generate request failed");
        assert_eq!(resp.status().as_u16(), StatusCode::NOT_FOUND.as_u16());
        handle.abort();
    }

    /// Fidelity of the ★ envelope: the full practical sampling param set
    /// (top_p / top_k / seed / stop_token_ids / penalties) plus an unknown
    /// nested field ride the opaque `vllm_tito` envelope UNCHANGED, while only
    /// the core scalars Dynamo needs for stop/routing are shadowed. This proves
    /// PR3's param set flows without any core-type coercion.
    #[test]
    fn preprocessed_from_generate_carries_full_param_set() {
        use crate::protocols::openai::generate::GenerateRequest;

        let req: GenerateRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3],
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "seed": 12345,
                "stop_token_ids": [151643],
                "stop": ["<|endoftext|>"],
                "presence_penalty": 0.5,
                "min_tokens": 2,
                "max_tokens": 8,
                "ignore_eos": true,
                "future_nested_field": "ignored"
            },
            "priority": 3
        }))
        .expect("deserialize GenerateRequest");

        let pre = super::preprocessed_from_generate(&req, "test-model").expect("build");

        // Core token ids are the request token ids.
        assert_eq!(pre.token_ids, vec![1, 2, 3]);

        // Control shadow: only the scalar knobs core needs are projected.
        assert_eq!(pre.stop_conditions.max_tokens, Some(8));
        assert_eq!(pre.stop_conditions.min_tokens, Some(2));
        assert_eq!(pre.stop_conditions.ignore_eos, Some(true));
        assert_eq!(pre.sampling_options.n, Some(1));
        assert_eq!(
            pre.routing.as_ref().and_then(|r| r.expected_output_tokens),
            Some(8)
        );

        // Drift-containment (negative): sampling knobs must NOT be lowered into
        // core — they live only in the envelope. If a future change lowered any
        // of these, this fails.
        assert_eq!(pre.sampling_options.temperature, None);
        assert_eq!(pre.sampling_options.top_p, None);
        assert_eq!(pre.sampling_options.top_k, None);
        assert_eq!(pre.sampling_options.presence_penalty, None);
        assert_eq!(pre.sampling_options.seed, None);
        assert_eq!(pre.stop_conditions.stop, None);
        assert_eq!(pre.stop_conditions.stop_token_ids, None);
        // Engine priority rides the envelope; it must not lower into Dynamo's
        // routing priority tiers.
        assert_eq!(pre.routing.as_ref().and_then(|r| r.priority_jump), None);
        assert_eq!(pre.routing.as_ref().and_then(|r| r.strict_priority), None);

        // Envelope fidelity: every sampling knob — including ones NOT shadowed
        // into core (top_p/top_k/seed/stop_token_ids/penalties) — and the
        // unknown nested field survive verbatim for the worker to reconstruct.
        let envelope = pre
            .extra_args
            .as_ref()
            .and_then(|e| e.get("vllm_tito"))
            .expect("vllm_tito envelope present");
        let sp = &envelope["sampling_params"];
        assert_eq!(sp["temperature"], serde_json::json!(0.8));
        assert_eq!(sp["top_p"], serde_json::json!(0.9));
        assert_eq!(sp["top_k"], serde_json::json!(50));
        assert_eq!(sp["seed"], serde_json::json!(12345));
        assert_eq!(sp["stop_token_ids"], serde_json::json!([151643]));
        assert_eq!(sp["stop"], serde_json::json!(["<|endoftext|>"]));
        assert_eq!(sp["presence_penalty"], serde_json::json!(0.5));
        assert_eq!(sp["future_nested_field"], serde_json::json!("ignored"));
        // Engine priority rides the envelope, separate from Dynamo routing priority.
        assert_eq!(envelope["priority"], serde_json::json!(3));
        assert_eq!(envelope["token_ids"], serde_json::json!([1, 2, 3]));

        // Strongest fidelity check: the whole envelope equals the raw request
        // serialized verbatim — covers every field (incl. min_tokens/max_tokens/
        // ignore_eos), not just the sampled keys above.
        let expected_envelope = serde_json::to_value(&req).expect("serialize req");
        assert_eq!(envelope, &expected_envelope);
    }
}
