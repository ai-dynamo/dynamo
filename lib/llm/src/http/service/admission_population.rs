// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal producer control for finite prefill-admission populations.
//!
//! Requests opt into a population with
//! `x-dynamo-meta-admission-population-id` and
//! `x-dynamo-meta-admission-population-index`. After it has stopped producing,
//! the same producer closes the population here with its exact final count.
//! The scheduler can then distinguish a true one-to-three-request terminal tail
//! from requests that have merely not arrived yet, without an idle timeout.

use std::sync::Arc;

use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_kv_router::scheduling::AdmissionPopulationClose;
use serde::{Deserialize, Serialize};

use super::{RouteDoc, service_v2};
use crate::discovery::ModelManagerError;

const DEFAULT_PATH: &str = "/admission_population/close";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseAdmissionPopulationRequest {
    pub model: String,
    pub namespace: Option<String>,
    pub policy_class: String,
    pub population_id: String,
    pub final_count: u64,
}

#[derive(Debug, Serialize)]
struct CloseAdmissionPopulationResponse {
    status: &'static str,
    model: String,
    namespace: String,
    policy_class: String,
    population_id: String,
    final_count: u64,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

fn nonempty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

pub fn admission_population_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    let docs = vec![RouteDoc::new(Method::POST, &path)];
    let router = Router::new()
        .route(&path, post(close_admission_population_handler))
        .with_state(state);
    (docs, router)
}

async fn close_admission_population_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    Json(request): Json<CloseAdmissionPopulationRequest>,
) -> impl IntoResponse {
    for (field, value) in [
        ("model", request.model.as_str()),
        ("policy_class", request.policy_class.as_str()),
        ("population_id", request.population_id.as_str()),
    ] {
        if let Err(error) = nonempty(field, value) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!(ErrorResponse { error })),
            );
        }
    }
    if let Some(namespace) = request.namespace.as_deref()
        && let Err(error) = nonempty("namespace", namespace)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!(ErrorResponse { error })),
        );
    }

    let close =
        match AdmissionPopulationClose::new(request.population_id.clone(), request.final_count) {
            Ok(close) => close,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!(ErrorResponse { error })),
                );
            }
        };
    let (namespace, router) = match state
        .manager()
        .resolve_prefill_router(&request.model, request.namespace.as_deref())
    {
        Ok(router) => router,
        Err(ModelManagerError::ModelNotFound(_)) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!(ErrorResponse {
                    error: format!("model {:?} was not found", request.model),
                })),
            );
        }
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!(ErrorResponse {
                    error: error.to_string(),
                })),
            );
        }
    };

    if let Err(error) = router
        .close_admission_population(request.policy_class.clone(), close)
        .await
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!(ErrorResponse {
                error: error.to_string(),
            })),
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!(CloseAdmissionPopulationResponse {
            status: "closed",
            model: request.model,
            namespace,
            policy_class: request.policy_class,
            population_id: request.population_id,
            final_count: request.final_count,
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_request_rejects_unknown_fields() {
        let error = serde_json::from_value::<CloseAdmissionPopulationRequest>(serde_json::json!({
            "model": "m",
            "namespace": "n",
            "policy_class": "p",
            "population_id": "id",
            "final_count": 3,
            "idle_timeout_ms": 10
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn required_control_fields_must_be_nonempty() {
        for field in ["model", "namespace", "policy_class", "population_id"] {
            assert!(nonempty(field, "").is_err());
            assert!(nonempty(field, "  ").is_err());
            assert!(nonempty(field, "value").is_ok());
        }
    }
}
