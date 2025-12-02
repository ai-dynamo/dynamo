// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for dynamically getting/setting the busy threshold per model.
//!
//! The busy threshold controls when workers are marked as "busy" based on their
//! KV cache utilization. When all workers for a model exceed their threshold,
//! new requests are rejected with a 503 Service Unavailable response.
//!
//! ## Endpoints
//!
//! ### POST /busy_threshold
//!
//! Get or set a model's busy threshold.
//!
//! **Set threshold:**
//! ```json
//! // Request
//! {"model": "llama-3-70b", "threshold": 0.85}
//! // Response
//! {"model": "llama-3-70b", "threshold": 0.85}
//! ```
//!
//! **Get threshold (omit or null threshold):**
//! ```json
//! // Request
//! {"model": "llama-3-70b"}
//! // Response (if configured)
//! {"model": "llama-3-70b", "threshold": 0.85}
//! // Response (if not configured)
//! {"model": "llama-3-70b", "threshold": null}
//! ```
//!
//! ### GET /busy_threshold
//!
//! List all configured busy thresholds.
//!
//! ```json
//! // Response
//! {"thresholds": [{"model": "llama-3-70b", "threshold": 0.85}]}
//! ```

use super::{RouteDoc, service_v2};
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request body for getting or setting a busy threshold.
///
/// - If `threshold` is provided: sets/creates the threshold and returns the new value
/// - If `threshold` is null/omitted: returns the existing threshold if any
#[derive(Debug, Deserialize)]
pub struct BusyThresholdRequest {
    /// The model name
    pub model: String,
    /// The threshold value (0.0 to 1.0), or null to just get the current value
    pub threshold: Option<f64>,
}

/// Response for a threshold operation
#[derive(Debug, Serialize)]
pub struct BusyThresholdResponse {
    /// The model name
    pub model: String,
    /// The threshold value (null if no threshold is configured)
    pub threshold: Option<f64>,
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
        .with_state(state);

    (docs, router)
}

async fn busy_threshold_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    Json(request): Json<BusyThresholdRequest>,
) -> impl IntoResponse {
    // Validate threshold range if provided
    if let Some(threshold) = request.threshold
        && !(0.0..=1.0).contains(&threshold)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!(ErrorResponse {
                error: format!("Threshold must be between 0.0 and 1.0, got {}", threshold),
            })),
        );
    }

    let manager = state.manager();

    // Get or set the threshold via the model's worker monitor
    let threshold = manager.busy_threshold(&request.model, request.threshold);

    // If trying to SET but model has no monitor, return 404
    if request.threshold.is_some() && threshold.is_none() {
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

    if request.threshold.is_some() {
        tracing::info!(
            model = %request.model,
            threshold = ?threshold,
            "Updated busy threshold"
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!(BusyThresholdResponse {
            model: request.model,
            threshold,
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
            .map(|(model, threshold)| BusyThresholdResponse {
                model,
                threshold: Some(threshold),
            })
            .collect(),
    };

    Json(serde_json::json!(response))
}
