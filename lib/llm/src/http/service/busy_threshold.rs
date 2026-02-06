// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for dynamically getting/setting the busy thresholds per model.
//!
//! The busy thresholds control when workers are marked as "busy" based on their
//! KV cache block utilization and prefill token utilization. When all workers
//! for a model exceed their thresholds, new requests are rejected with a 503
//! Service Unavailable response.
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

use super::{RouteDoc, service_v2};
use crate::discovery::LoadThresholdConfig;
use axum::{
    Json, Router,
    extract::Request,
    http::{Method, StatusCode},
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

async fn busy_threshold_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    Json(request): Json<BusyThresholdRequest>,
) -> impl IntoResponse {
    // Validate active decode blocks threshold range if provided (must be 0.0-1.0)
    if let Some(threshold) = request.active_decode_blocks_threshold
        && !(0.0..=1.0).contains(&threshold)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!(ErrorResponse {
                error: format!(
                    "active_decode_blocks_threshold must be between 0.0 and 1.0, got {}",
                    threshold
                ),
            })),
        );
    }

    let manager = state.manager();

    // Build LoadThresholdConfig from request if any threshold is being set
    let is_setting = request.active_decode_blocks_threshold.is_some()
        || request.active_prefill_tokens_threshold.is_some()
        || request.active_prefill_tokens_threshold_frac.is_some();

    let update_config = if is_setting {
        Some(LoadThresholdConfig {
            active_decode_blocks_threshold: request.active_decode_blocks_threshold,
            active_prefill_tokens_threshold: request.active_prefill_tokens_threshold,
            active_prefill_tokens_threshold_frac: request.active_prefill_tokens_threshold_frac,
        })
    } else {
        None
    };

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
