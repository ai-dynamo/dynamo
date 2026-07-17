// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dynamo_tokens::SequenceHash;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::ActiveSequenceStride;
use crate::protocols::WorkerWithDpRank;
use crate::sequences::SequenceError;
use crate::services::common::replica_sync::PeerManager;
use crate::services::common::replica_sync_http;

use super::registry::{RegistryError, ServiceError, SlotTrackerRegistry, TrackerKey};

pub struct AppState {
    pub registry: Arc<SlotTrackerRegistry>,
}

fn default_routing_group() -> String {
    "default".to_string()
}

#[derive(Deserialize)]
struct RegisterRequest {
    worker_id: u64,
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    block_size: u32,
    dp_start: u32,
    dp_size: u32,
}

#[derive(Deserialize)]
struct UnregisterRequest {
    worker_id: u64,
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
}

#[derive(Deserialize)]
struct AddRequest {
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    request_id: String,
    worker_id: u64,
    dp_rank: u32,
    #[serde(deserialize_with = "deserialize_sequence_hashes")]
    sequence_hashes: Vec<SequenceHash>,
    #[serde(default)]
    new_isl_tokens: usize,
}

#[derive(Deserialize)]
struct LifecycleRequest {
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    request_id: String,
}

#[derive(Deserialize)]
struct PotentialLoadsRequest {
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    #[serde(deserialize_with = "deserialize_sequence_hashes")]
    sequence_hashes: Vec<SequenceHash>,
    #[serde(default)]
    new_isl_tokens: usize,
}

#[derive(Deserialize)]
struct FilterQuery {
    model_name: Option<String>,
    routing_group: Option<String>,
}

fn deserialize_sequence_hashes<'de, D>(deserializer: D) -> Result<Vec<SequenceHash>, D::Error>
where
    D: Deserializer<'de>,
{
    struct SequenceHashesVisitor;

    impl<'de> Visitor<'de> for SequenceHashesVisitor {
        type Value = Vec<SequenceHash>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an array of signed 64-bit sequence hashes")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut hashes = Vec::with_capacity(seq.size_hint().unwrap_or(0));
            while let Some(hash) = seq.next_element::<i64>()? {
                hashes.push(hash as u64);
            }
            Ok(hashes)
        }
    }

    deserializer.deserialize_seq(SequenceHashesVisitor)
}

async fn register(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<RegisterRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));
    match state.registry.register(
        key,
        req.worker_id,
        req.block_size,
        req.dp_start,
        req.dp_size,
    ) {
        Ok(()) => json_ok(StatusCode::CREATED),
        Err(error) => registry_error(error),
    }
}

async fn unregister(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<UnregisterRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));
    match state.registry.unregister(&key, req.worker_id) {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => registry_error(error),
    }
}

async fn list_workers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilterQuery>,
) -> Response {
    Json(state.registry.list_workers(
        params.model_name.as_deref(),
        params.routing_group.as_deref(),
    ))
    .into_response()
}

async fn add(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<AddRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));

    // Lifecycle delivery is intentionally arrival-ordered. Consumers should
    // normally await /add before sending /prefill_complete or /free.
    match state.registry.add_request(
        &key,
        req.request_id,
        WorkerWithDpRank::new(req.worker_id, req.dp_rank),
        ActiveSequenceStride::ONE.sample_dense(req.sequence_hashes),
        req.new_isl_tokens,
    ) {
        Ok(()) => json_ok(StatusCode::CREATED),
        Err(error) => service_error(error),
    }
}

async fn prefill_complete(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<LifecycleRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));
    match state.registry.mark_prefill_completed(&key, &req.request_id) {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => service_error(error),
    }
}

async fn free(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<LifecycleRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_rejection(error),
    };
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));
    match state.registry.free(&key, &req.request_id) {
        Ok(()) => json_ok(StatusCode::OK),
        Err(error) => service_error(error),
    }
}

async fn list_loads(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilterQuery>,
) -> Response {
    Json(state.registry.list_loads(
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
    let key = TrackerKey::new(req.model_name, Some(req.routing_group));
    let sequence_hashes = ActiveSequenceStride::ONE.sample_dense(req.sequence_hashes);
    match state
        .registry
        .potential_loads(&key, &sequence_hashes, req.new_isl_tokens)
    {
        Ok(loads) => Json(loads).into_response(),
        Err(error) => registry_error(error),
    }
}

async fn health() -> StatusCode {
    StatusCode::OK
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

fn registry_error(error: RegistryError) -> Response {
    let status = match &error {
        RegistryError::InvalidBlockSize
        | RegistryError::InvalidDpSize
        | RegistryError::InvalidDpRange { .. } => StatusCode::BAD_REQUEST,
        RegistryError::BlockSizeMismatch { .. } | RegistryError::DuplicateWorker { .. } => {
            StatusCode::CONFLICT
        }
        RegistryError::WorkerNotFound { .. } | RegistryError::TrackerNotFound { .. } => {
            StatusCode::NOT_FOUND
        }
    };
    json_error(status, error)
}

fn service_error(error: ServiceError) -> Response {
    match error {
        ServiceError::Registry(error) => registry_error(error),
        ServiceError::Sequence(SequenceError::WorkerNotFound { .. })
        | ServiceError::Sequence(SequenceError::RequestNotFound { .. }) => {
            json_error(StatusCode::NOT_FOUND, error)
        }
        ServiceError::Sequence(SequenceError::DuplicateRequest { .. }) => {
            json_error(StatusCode::CONFLICT, error)
        }
        ServiceError::Sequence(SequenceError::ReplicaSyncPublishFailed(_)) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, error)
        }
    }
}

pub(crate) fn create_router(
    state: Arc<AppState>,
    peer_manager: Option<Arc<PeerManager>>,
) -> Router {
    Router::new()
        .route("/register", post(register))
        .route("/unregister", post(unregister))
        .route("/workers", get(list_workers))
        .route("/add", post(add))
        .route("/prefill_complete", post(prefill_complete))
        .route("/free", post(free))
        .route("/loads", get(list_loads))
        .route("/potential_loads", post(potential_loads))
        .route("/health", get(health))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(state)
        .merge(replica_sync_http::router(peer_manager))
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        create_router(
            Arc::new(AppState {
                registry: Arc::new(SlotTrackerRegistry::new(CancellationToken::new())),
            }),
            None,
        )
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        serde_json::from_slice(&body).expect("response JSON")
    }

    #[tokio::test]
    async fn malformed_json_returns_json_error() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/register")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response_json(response).await["error"].is_string());
    }

    #[tokio::test]
    async fn replica_sync_routes_are_mounted() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/replica_sync/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_route_and_method_return_json_errors() {
        let route_response = app()
            .oneshot(
                Request::builder()
                    .uri("/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(route_response.status(), StatusCode::NOT_FOUND);
        assert!(response_json(route_response).await["error"].is_string());

        let method_response = app()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/register")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(method_response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(response_json(method_response).await["error"].is_string());
    }

    #[tokio::test]
    async fn signed_hashes_are_reinterpreted_bit_for_bit() {
        let app = app();
        let register_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/register")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"worker_id":1,"model_name":"model","block_size":16,"dp_start":0,"dp_size":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(register_response.status(), StatusCode::CREATED);

        let add_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/add")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"model_name":"model","request_id":"req","worker_id":1,"dp_rank":0,"sequence_hashes":[-1],"new_isl_tokens":4}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(add_response.status(), StatusCode::CREATED);

        let loads_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/potential_loads")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"model_name":"model","sequence_hashes":[-1],"new_isl_tokens":0}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(loads_response.status(), StatusCode::OK);
        let loads = response_json(loads_response).await;
        assert_eq!(loads[0]["potential_decode_blocks"], 1);
    }
}
