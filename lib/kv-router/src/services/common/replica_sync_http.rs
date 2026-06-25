// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;

use super::replica_sync::{PeerError, PeerManager};

#[derive(Debug, Deserialize)]
struct PeerRequest {
    endpoint: String,
}

async fn register_peer(
    State(peer_manager): State<Option<PeerManager>>,
    payload: Result<Json<PeerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_error(error.status(), error.body_text()),
    };
    let Some(peer_manager) = peer_manager else {
        return json_error(StatusCode::CONFLICT, "replica sync is disabled");
    };
    match peer_manager.register_peer(req.endpoint).await {
        Ok(true) => json_ok(StatusCode::CREATED),
        Ok(false) => json_ok(StatusCode::OK),
        Err(error) => peer_error(error),
    }
}

async fn deregister_peer(
    State(peer_manager): State<Option<PeerManager>>,
    payload: Result<Json<PeerRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(error) => return json_error(error.status(), error.body_text()),
    };
    let Some(peer_manager) = peer_manager else {
        return json_error(StatusCode::CONFLICT, "replica sync is disabled");
    };
    match peer_manager.deregister_peer(req.endpoint).await {
        Ok(true) => json_ok(StatusCode::OK),
        Ok(false) => json_error(StatusCode::NOT_FOUND, "peer not found"),
        Err(error) => peer_error(error),
    }
}

async fn list_peers(State(peer_manager): State<Option<PeerManager>>) -> Response {
    Json(
        peer_manager
            .as_ref()
            .map(PeerManager::list_peers)
            .unwrap_or_default(),
    )
    .into_response()
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

fn peer_error(error: PeerError) -> Response {
    let status = match &error {
        PeerError::InvalidEndpoint(_) => StatusCode::BAD_REQUEST,
        PeerError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    };
    json_error(status, error)
}

pub(crate) fn router(peer_manager: Option<PeerManager>) -> Router {
    Router::new()
        .route("/replica_sync/register_peer", post(register_peer))
        .route("/replica_sync/deregister_peer", post(deregister_peer))
        .route("/replica_sync/peers", get(list_peers))
        .with_state(peer_manager)
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    use super::*;

    async fn post(app: Router, uri: &str, body: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn disabled_replica_sync_lists_no_peers_and_rejects_mutation() {
        let app = router(None);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/replica_sync/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response_json(response).await, serde_json::json!([]));
        assert_eq!(
            post(
                app,
                "/replica_sync/register_peer",
                r#"{"endpoint":"tcp://127.0.0.1:19092"}"#,
            )
            .await
            .status(),
            StatusCode::CONFLICT
        );
    }

    #[tokio::test]
    async fn manages_replica_sync_peers() {
        let cancel_token = CancellationToken::new();
        let peer_manager = PeerManager::start(Vec::new(), cancel_token.clone(), |_| {}).unwrap();
        let app = router(Some(peer_manager));
        let endpoint = "tcp://127.0.0.1:19092";
        let body = format!(r#"{{"endpoint":"{endpoint}"}}"#);

        assert_eq!(
            post(app.clone(), "/replica_sync/register_peer", &body)
                .await
                .status(),
            StatusCode::CREATED
        );
        assert_eq!(
            post(app.clone(), "/replica_sync/register_peer", &body)
                .await
                .status(),
            StatusCode::OK
        );

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/replica_sync/peers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response_json(response).await, serde_json::json!([endpoint]));

        assert_eq!(
            post(
                app.clone(),
                "/replica_sync/register_peer",
                r#"{"endpoint":"invalid"}"#,
            )
            .await
            .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            post(app.clone(), "/replica_sync/deregister_peer", &body)
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            post(app, "/replica_sync/deregister_peer", &body)
                .await
                .status(),
            StatusCode::NOT_FOUND
        );

        cancel_token.cancel();
    }
}
