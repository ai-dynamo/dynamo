// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::net::TcpListener;

use crate::protocols::WorkerId;
use crate::services::common::replica_sync::ReplicaPeerError;

use super::core::SelectionServiceConfig;
use super::service::SelectionService;
use super::types::{
    OutputBlockRequest, OverlapScoresRequest, PotentialLoadsRequest, REQUEST_BODY_LIMIT_BYTES,
    ReservationRequest, SelectAndReserveRequest, SelectRequest, WorkerPatchRequest, WorkerRequest,
};

#[derive(Debug, Deserialize)]
struct FilterQuery {
    model_name: Option<String>,
    routing_group: Option<String>,
}

pub struct AppState {
    pub service: Arc<SelectionService>,
}

#[derive(Debug, Deserialize)]
struct PeerRequest {
    endpoint: String,
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<WorkerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.upsert_worker(req).await {
        Ok(worker) => (StatusCode::CREATED, Json(worker)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn patch_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<WorkerId>,
    payload: Result<Json<WorkerPatchRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.patch_worker(worker_id, req).await {
        Ok(worker) => Json(worker).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<WorkerId>,
) -> Response {
    match state.service.delete_worker(worker_id).await {
        Ok(worker) => Json(worker).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn list_workers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilterQuery>,
) -> Response {
    Json(state.service.list_workers(
        params.model_name.as_deref(),
        params.routing_group.as_deref(),
    ))
    .into_response()
}

async fn select(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Result<Json<SelectRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state
        .service
        .select_with_policy_class(req, policy_class_from_headers(&headers))
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn select_and_reserve(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    payload: Result<Json<SelectAndReserveRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state
        .service
        .select_and_reserve_with_policy_class(req, policy_class_from_headers(&headers))
        .await
    {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn create_reservation(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ReservationRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.create_reservation(req).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn prefill_complete(
    State(state): State<Arc<AppState>>,
    Path(reservation_id): Path<String>,
) -> Response {
    match state.service.prefill_complete(&reservation_id).await {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => error.into_response(),
    }
}

async fn delete_reservation(
    State(state): State<Arc<AppState>>,
    Path(reservation_id): Path<String>,
) -> Response {
    match state.service.free_reservation(&reservation_id).await {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => error.into_response(),
    }
}

async fn add_output_block(
    State(state): State<Arc<AppState>>,
    Path(reservation_id): Path<String>,
    payload: Result<Json<OutputBlockRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state
        .service
        .add_output_block(&reservation_id, req.decay_fraction)
    {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => error.into_response(),
    }
}

async fn health() -> Response {
    json_ok(StatusCode::OK)
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let response = state.service.ready();
    if response.ready {
        Json(response).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
    }
}

async fn loads(State(state): State<Arc<AppState>>, Query(params): Query<FilterQuery>) -> Response {
    Json(state.service.loads(
        params.model_name.as_deref(),
        params.routing_group.as_deref(),
    ))
    .into_response()
}

async fn potential_loads(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<PotentialLoadsRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.potential_loads(req).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn overlap_scores(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<OverlapScoresRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.overlap_scores(req).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn dump_events(State(state): State<Arc<AppState>>) -> Response {
    Json(state.service.indexer_snapshot().await).into_response()
}

async fn register_peer(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<PeerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.register_replica_peer(req.endpoint).await {
        Ok(true) => json_ok(StatusCode::CREATED),
        Ok(false) => json_ok(StatusCode::OK),
        Err(error) => replica_peer_error(error),
    }
}

async fn deregister_peer(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<PeerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.service.deregister_replica_peer(req.endpoint).await {
        Ok(true) => json_ok(StatusCode::OK),
        Ok(false) => json_error(StatusCode::NOT_FOUND, "peer not found"),
        Err(error) => replica_peer_error(error),
    }
}

async fn list_peers(State(state): State<Arc<AppState>>) -> Response {
    Json(state.service.list_replica_peers()).into_response()
}

async fn not_found() -> Response {
    json_error(StatusCode::NOT_FOUND, "route not found")
}

async fn method_not_allowed() -> Response {
    json_error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
}

fn json_ok(status: StatusCode) -> Response {
    (status, Json(serde_json::json!({"status": "ok"}))).into_response()
}

fn json_error(status: StatusCode, error: impl fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!({"error": error.to_string()})),
    )
        .into_response()
}

fn json_rejection(error: JsonRejection) -> Response {
    json_error(error.status(), error.body_text())
}

fn replica_peer_error(error: ReplicaPeerError) -> Response {
    let status = match &error {
        ReplicaPeerError::InvalidEndpoint(_) => StatusCode::BAD_REQUEST,
        ReplicaPeerError::Disabled => StatusCode::CONFLICT,
        ReplicaPeerError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    };
    json_error(status, error)
}

fn policy_class_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-dynamo-meta-policy-class")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/select", post(select))
        .route("/select_and_reserve", post(select_and_reserve))
        .route("/reservations", post(create_reservation))
        .route(
            "/reservations/{reservation_id}/prefill_complete",
            post(prefill_complete),
        )
        .route(
            "/reservations/{reservation_id}/output_block",
            post(add_output_block),
        )
        .route("/reservations/{reservation_id}", delete(delete_reservation))
        .route("/workers", post(create_worker).get(list_workers))
        .route(
            "/workers/{worker_id}",
            patch(patch_worker).delete(delete_worker),
        )
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/loads", get(loads))
        .route("/potential_loads", post(potential_loads))
        .route("/overlap_scores", post(overlap_scores))
        .route("/dump", get(dump_events))
        .route("/replica_sync/register_peer", post(register_peer))
        .route("/replica_sync/deregister_peer", post(deregister_peer))
        .route("/replica_sync/peers", get(list_peers))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(axum::extract::DefaultBodyLimit::max(
            REQUEST_BODY_LIMIT_BYTES,
        ))
        .with_state(state)
}

pub async fn run_server(config: SelectionServiceConfig) -> anyhow::Result<()> {
    tracing::info!(
        port = config.port,
        threads = config.threads,
        indexer_peers = config.indexer_peers.len(),
        replica_sync = config.replica_sync_port.is_some(),
        "Starting Dynamo selection service"
    );

    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    let service = Arc::new(config.service_builder().build().await?);
    let app = create_router(Arc::new(AppState {
        service: Arc::clone(&service),
    }));
    let shutdown_service = Arc::clone(&service);
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("received shutdown signal");
            shutdown_service.shutdown().await;
        })
        .await;
    service.shutdown().await;
    result?;
    Ok(())
}
