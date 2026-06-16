// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::protocols::WorkerId;
use crate::services::common::replica_sync::{PeerManager, setup_replica_sync};
use crate::services::common::replica_sync_http;

use super::core::{SelectionCore, SelectionServiceConfig};
use super::types::{
    OutputBlockRequest, OverlapScoresRequest, PotentialLoadsRequest, REQUEST_BODY_LIMIT_BYTES,
    ReservationRequest, SelectAndReserveRequest, SelectRequest, WorkerPatchRequest, WorkerRequest,
};

#[derive(Debug, Deserialize)]
struct FilterQuery {
    model_name: Option<String>,
    tenant_id: Option<String>,
}

pub struct AppState {
    pub core: Arc<SelectionCore>,
}

async fn create_worker(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<WorkerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.core.upsert_worker(req).await {
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
    match state.core.patch_worker(worker_id, req).await {
        Ok(worker) => Json(worker).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn delete_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<WorkerId>,
) -> Response {
    match state.core.delete_worker(worker_id).await {
        Ok(worker) => Json(worker).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn list_workers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilterQuery>,
) -> Response {
    Json(
        state
            .core
            .list_workers(params.model_name.as_deref(), params.tenant_id.as_deref()),
    )
    .into_response()
}

async fn select(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<SelectRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.core.select(req).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn select_and_reserve(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<SelectAndReserveRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    match state.core.select_and_reserve(req).await {
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
    match state.core.create_reservation(req).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn prefill_complete(
    State(state): State<Arc<AppState>>,
    Path(reservation_id): Path<String>,
) -> Response {
    match state.core.prefill_complete(&reservation_id).await {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => error.into_response(),
    }
}

async fn delete_reservation(
    State(state): State<Arc<AppState>>,
    Path(reservation_id): Path<String>,
) -> Response {
    match state.core.free_reservation(&reservation_id).await {
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
        .core
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
    let response = state.core.ready();
    if response.ready {
        Json(response).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
    }
}

async fn loads(State(state): State<Arc<AppState>>, Query(params): Query<FilterQuery>) -> Response {
    Json(
        state
            .core
            .loads(params.model_name.as_deref(), params.tenant_id.as_deref()),
    )
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
    match state.core.potential_loads(req).await {
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
    match state.core.overlap_scores(req).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}

async fn dump_events(State(state): State<Arc<AppState>>) -> Response {
    Json(state.core.dump_indexer_events().await).into_response()
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

pub(crate) fn create_router(state: Arc<AppState>, peer_manager: Option<PeerManager>) -> Router {
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
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(axum::extract::DefaultBodyLimit::max(
            REQUEST_BODY_LIMIT_BYTES,
        ))
        .with_state(state)
        .merge(replica_sync_http::router(peer_manager))
}

pub async fn run_server(config: SelectionServiceConfig) -> anyhow::Result<()> {
    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("received shutdown signal");
        shutdown_token.cancel();
    });

    let replica_runtime = setup_replica_sync(
        config.replica_sync_port,
        &config.replica_sync_peers,
        cancel_token.child_token(),
    )?;

    tracing::info!(
        port = config.port,
        threads = config.threads,
        indexer_peers = config.indexer_peers.len(),
        replica_sync = replica_runtime.is_some(),
        "Starting Dynamo selection service"
    );

    let core = Arc::new(SelectionCore::new_for_server(
        config.kv_router_config,
        config.threads,
        cancel_token.clone(),
        replica_runtime,
    ));
    if !config.indexer_peers.is_empty() {
        match core.recover_indexer_from_peers(&config.indexer_peers).await {
            Ok(true) => tracing::info!("Selection indexer recovery completed"),
            Ok(false) => {
                tracing::warn!("No reachable selection indexer peers; starting with empty state")
            }
            Err(error) => {
                tracing::warn!(%error, "Selection indexer recovery failed; starting with empty state")
            }
        }
    }
    core.signal_indexer_ready();

    let peer_manager = if config.replica_sync_port.is_some() {
        let dispatch_core = Arc::clone(&core);
        Some(PeerManager::start(
            config.replica_sync_peers,
            cancel_token.child_token(),
            move |event| dispatch_core.dispatch_replica_event(event),
        )?)
    } else {
        None
    };

    let app = create_router(Arc::new(AppState { core }), peer_manager);
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
        })
        .await?;
    Ok(())
}
