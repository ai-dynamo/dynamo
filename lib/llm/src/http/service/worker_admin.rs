// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

use super::{RouteDoc, service_v2};
use crate::discovery::{ModelManagerError, WorkerDrainSelector, WorkerStatus};

#[derive(Debug, Deserialize)]
struct WorkerQuery {
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WorkerActionRequest {
    model: String,
    namespace: Option<String>,
    worker_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct WorkersResponse {
    workers: Vec<WorkerStatus>,
}

#[derive(Debug, Serialize)]
struct WorkerActionResponse {
    status: &'static str,
    worker: WorkerStatus,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn worker_admin_router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let docs = vec![
        RouteDoc::new(Method::GET, "/workers"),
        RouteDoc::new(Method::POST, "/workers/{worker_id}/drain"),
        RouteDoc::new(Method::POST, "/workers/{worker_id}/resume"),
    ];
    let router = Router::new()
        .route("/workers", get(list_workers))
        .route("/workers/{worker_id}/drain", post(drain_worker))
        .route("/workers/{worker_id}/resume", post(resume_worker))
        .with_state(state);
    (docs, router)
}

async fn list_workers(
    State(state): State<Arc<service_v2::State>>,
    Query(query): Query<WorkerQuery>,
) -> Response {
    Json(WorkersResponse {
        workers: state.manager().worker_statuses(query.model.as_deref()),
    })
    .into_response()
}

async fn drain_worker(
    State(state): State<Arc<service_v2::State>>,
    Path(worker_id): Path<u64>,
    Json(request): Json<WorkerActionRequest>,
) -> Response {
    set_worker_drained(state, worker_id, request, true).await
}

async fn resume_worker(
    State(state): State<Arc<service_v2::State>>,
    Path(worker_id): Path<u64>,
    Json(request): Json<WorkerActionRequest>,
) -> Response {
    set_worker_drained(state, worker_id, request, false).await
}

async fn set_worker_drained(
    state: Arc<service_v2::State>,
    worker_id: u64,
    request: WorkerActionRequest,
    drained: bool,
) -> Response {
    let model = request.model.trim();
    if model.is_empty() {
        return json_error(StatusCode::BAD_REQUEST, "model must not be empty");
    }
    let selector = WorkerDrainSelector {
        namespace: normalize_selector(request.namespace),
        worker_type: normalize_selector(request.worker_type),
    };

    if let Err(error) = state
        .manager()
        .set_worker_drained(model, worker_id, &selector, drained)
    {
        return model_error_response(error);
    }
    let Some(worker) = state
        .manager()
        .worker_statuses(Some(model))
        .into_iter()
        .find(|worker| worker_matches(worker, worker_id, &selector))
    else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("worker {worker_id} not found for model {model}"),
        );
    };
    Json(WorkerActionResponse {
        status: "ok",
        worker,
    })
    .into_response()
}

fn normalize_selector(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn worker_matches(worker: &WorkerStatus, worker_id: u64, selector: &WorkerDrainSelector) -> bool {
    if worker.worker_id != worker_id {
        return false;
    }
    if let Some(namespace) = selector.namespace.as_deref()
        && worker.namespace != namespace
    {
        return false;
    }
    if let Some(worker_type) = selector.worker_type.as_deref()
        && worker.worker_type.as_deref() != Some(worker_type)
    {
        return false;
    }
    true
}

fn model_error_response(error: ModelManagerError) -> Response {
    match error {
        ModelManagerError::ModelNotFound(_) | ModelManagerError::ModelUnavailable(_) => {
            json_error(StatusCode::NOT_FOUND, error)
        }
        ModelManagerError::WorkerSelectionConflict(_)
        | ModelManagerError::ModelAlreadyExists(_) => json_error(StatusCode::CONFLICT, error),
    }
}

fn json_error(status: StatusCode, error: impl std::fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!(ErrorResponse {
            error: error.to_string(),
        })),
    )
        .into_response()
}
