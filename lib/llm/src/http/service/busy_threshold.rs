// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for dynamically getting/setting the busy thresholds per model.
//!
//! The busy thresholds control when workers are marked as "busy" based on their
//! KV cache block utilization and prefill token utilization. When all workers
//! for a model exceed their thresholds, new requests are rejected with a 529
//! response.
//!
//! ## Endpoints
//!
//! ### POST /busy_threshold
//!
//! Get or set a model's busy thresholds.
//!
//! **Set thresholds:**
//! ```json
//! // Request
//! {"model": "llama-3-70b", "active_decode_blocks_threshold": 0.85, "active_prefill_tokens_threshold": 1000, "active_prefill_tokens_threshold_frac": 0.8}
//! // Response
//! {"model": "llama-3-70b", "active_decode_blocks_threshold": 0.85, "active_prefill_tokens_threshold": 1000, "active_prefill_tokens_threshold_frac": 0.8}
//! ```
//!
//! **Get thresholds (omit thresholds):**
//! ```json
//! // Request
//! {"model": "llama-3-70b"}
//! // Response (if configured)
//! {"model": "llama-3-70b", "active_decode_blocks_threshold": 0.85, "active_prefill_tokens_threshold": 1000, "active_prefill_tokens_threshold_frac": 0.8}
//! // Response (if not configured)
//! {"model": "llama-3-70b", "active_decode_blocks_threshold": null, "active_prefill_tokens_threshold": null, "active_prefill_tokens_threshold_frac": null}
//! ```
//!
//! ### GET /busy_threshold
//!
//! List all configured busy thresholds.
//!
//! ```json
//! // Response
//! {"thresholds": [{"model": "llama-3-70b", "active_decode_blocks_threshold": 0.85, "active_prefill_tokens_threshold": 1000, "active_prefill_tokens_threshold_frac": 0.8}]}
//! ```

use super::openai::{get_body_limit, is_json_content_type};
use super::{RouteDoc, service_v2};
use crate::discovery::LoadThresholdConfig;
use axum::{
    Json, Router,
    body::Body,
    extract::Request,
    http::{HeaderMap, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request body for getting or setting busy thresholds.
///
/// - If thresholds are provided: sets/creates the thresholds and returns the new values
/// - If thresholds are null/omitted: returns the existing thresholds if any
#[derive(Debug, Deserialize)]
pub struct BusyThresholdRequest {
    /// The model name
    pub model: String,
    /// The active decode blocks threshold value (0.0 to 1.0), or null to just get the current value
    pub active_decode_blocks_threshold: Option<f64>,
    /// The active prefill tokens threshold value (literal token count), or null to just get the current value
    pub active_prefill_tokens_threshold: Option<u64>,
    /// The active prefill tokens threshold as fraction of max_num_batched_tokens, or null to just get the current value
    pub active_prefill_tokens_threshold_frac: Option<f64>,
}

/// Response for a threshold operation
#[derive(Debug, Serialize)]
pub struct BusyThresholdResponse {
    /// The model name
    pub model: String,
    /// The active decode blocks threshold value (null if no threshold is configured)
    pub active_decode_blocks_threshold: Option<f64>,
    /// The active prefill tokens threshold value (null if no threshold is configured)
    pub active_prefill_tokens_threshold: Option<u64>,
    /// The active prefill tokens threshold as fraction of max_num_batched_tokens
    pub active_prefill_tokens_threshold_frac: Option<f64>,
}

/// Response for listing all thresholds
#[derive(Debug, Serialize)]
pub struct ListBusyThresholdsResponse {
    /// List of model thresholds
    pub thresholds: Vec<BusyThresholdResponse>,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Middleware to convert 422 Unprocessable Entity responses (from JSON deserialization errors)
/// to JSON format instead of text/plain.
async fn json_error_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;

    if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
        let (_parts, body) = response.into_parts();
        let body_bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_default();
        let error_message = String::from_utf8_lossy(&body_bytes).to_string();
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!(ErrorResponse {
                error: error_message,
            })),
        )
            .into_response()
    } else {
        response
    }
}

pub fn busy_threshold_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let base_path = path.unwrap_or_else(|| "/busy_threshold".to_string());

    let docs: Vec<RouteDoc> = vec![
        RouteDoc::new(Method::POST, &base_path),
        RouteDoc::new(Method::GET, &base_path),
    ];

    let router = Router::new()
        .route(&base_path, post(busy_threshold_handler))
        .route(&base_path, get(list_busy_thresholds_handler))
        .layer(axum::middleware::from_fn(json_error_middleware))
        .with_state(state);

    (docs, router)
}

/// Read the request body, reporting a bad `Content-Type`, an oversized body or
/// malformed JSON in this route's `{"error": "..."}` envelope.
///
/// Axum's `Json` extractor rejects all three as `text/plain`, and
/// `json_error_middleware` only rewrites 422, so a caller parsing the envelope
/// this endpoint documents got an unparseable body for the most common
/// mistakes.
async fn read_busy_threshold_request(
    headers: &HeaderMap,
    body: Body,
) -> Result<BusyThresholdRequest, (StatusCode, Json<serde_json::Value>)> {
    let error = |code: StatusCode, message: String| {
        (
            code,
            Json(serde_json::json!(ErrorResponse { error: message })),
        )
    };

    let is_json = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_json_content_type);
    if !is_json {
        return Err(error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Expected request with Content-Type application/json".to_string(),
        ));
    }

    let bytes = axum::body::to_bytes(body, get_body_limit())
        .await
        .map_err(|err| {
            // `to_bytes` wraps an oversized-body failure in its error source
            // rather than returning `LengthLimitError` directly.
            if std::error::Error::source(&err)
                .is_some_and(|source| source.is::<http_body_util::LengthLimitError>())
            {
                error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!(
                        "Request body exceeds the limit of {} bytes",
                        get_body_limit()
                    ),
                )
            } else {
                error(
                    StatusCode::BAD_REQUEST,
                    "Failed to read the request body".to_string(),
                )
            }
        })?;

    serde_json::from_slice(&bytes).map_err(|err| {
        error(
            StatusCode::BAD_REQUEST,
            format!("Failed to parse the request body as JSON: {err}"),
        )
    })
}

async fn busy_threshold_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    headers: HeaderMap,
    body: Body,
) -> impl IntoResponse {
    let request = match read_busy_threshold_request(&headers, body).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    let requested_config = LoadThresholdConfig {
        active_decode_blocks_threshold: request.active_decode_blocks_threshold,
        active_prefill_tokens_threshold: request.active_prefill_tokens_threshold,
        active_prefill_tokens_threshold_frac: request.active_prefill_tokens_threshold_frac,
    };
    if let Err(error) = requested_config.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!(ErrorResponse { error })),
        );
    }

    let manager = state.manager();

    // Build LoadThresholdConfig from request if any threshold is being set
    let is_setting = requested_config.is_configured();
    let update_config = is_setting.then_some(requested_config);

    // Get or set the thresholds via the model's worker monitor
    let config = manager.load_threshold_config(&request.model, update_config.as_ref());

    // If trying to SET but model has no monitor, return 404
    if is_setting && config.is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!(ErrorResponse {
                error: format!(
                    "Model '{}' not found. Thresholds can only be set for discovered models.",
                    request.model
                ),
            })),
        );
    }

    if is_setting {
        tracing::info!(
            model = %request.model,
            config = ?config,
            "Updated busy thresholds"
        );
    }

    let (
        active_decode_blocks_threshold,
        active_prefill_tokens_threshold,
        active_prefill_tokens_threshold_frac,
    ) = config.map_or((None, None, None), |c| {
        (
            c.active_decode_blocks_threshold,
            c.active_prefill_tokens_threshold,
            c.active_prefill_tokens_threshold_frac,
        )
    });

    (
        StatusCode::OK,
        Json(serde_json::json!(BusyThresholdResponse {
            model: request.model,
            active_decode_blocks_threshold,
            active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac,
        })),
    )
}

async fn list_busy_thresholds_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let manager = state.manager();
    let thresholds = manager.list_busy_thresholds();

    let response = ListBusyThresholdsResponse {
        thresholds: thresholds
            .into_iter()
            .map(|(model, config)| BusyThresholdResponse {
                model,
                active_decode_blocks_threshold: config.active_decode_blocks_threshold,
                active_prefill_tokens_threshold: config.active_prefill_tokens_threshold,
                active_prefill_tokens_threshold_frac: config.active_prefill_tokens_threshold_frac,
            })
            .collect(),
    };

    Json(serde_json::json!(response))
}
