// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in component control handlers on the system listener.

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize)]
pub struct ControlRequest {
    pub method: String,
    /// Path relative to the registered component, without a leading slash.
    pub path: String,
    pub body: Value,
}

#[derive(Deserialize)]
pub struct ControlResponse {
    pub status: u16,
    pub body: Value,
}

pub type ControlHandler = Arc<
    dyn Fn(ControlRequest) -> Pin<Box<dyn Future<Output = anyhow::Result<ControlResponse>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone, Default)]
pub struct ControlRegistry(Arc<RwLock<HashMap<String, ControlHandler>>>);

impl ControlRegistry {
    /// Register once under `/control/{name}/...`; no handlers are installed by default.
    pub fn register(&self, name: String, handler: ControlHandler) -> anyhow::Result<()> {
        anyhow::ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
            "control handler name must be a nonempty URL segment"
        );
        let mut handlers = self.0.write();
        anyhow::ensure!(
            !handlers.contains_key(&name),
            "control handler already registered"
        );
        handlers.insert(name, handler);
        Ok(())
    }

    pub fn unregister(&self, name: &str) {
        self.0.write().remove(name);
    }
}

pub(crate) fn router(registry: ControlRegistry) -> Router {
    Router::new()
        .route("/control/{name}/{*path}", any(dispatch))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .with_state(registry)
}

async fn dispatch(
    State(registry): State<ControlRegistry>,
    Path((name, path)): Path<(String, String)>,
    method: Method,
    body: Bytes,
) -> Response {
    let Some(handler) = registry.0.read().get(&name).cloned() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let body = if body.is_empty() {
        Value::Null
    } else {
        match serde_json::from_slice(&body) {
            Ok(body) => body,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        }
    };
    let result = handler(ControlRequest {
        method: method.to_string(),
        path,
        body,
    })
    .await;
    match result {
        Ok(response) if (200..600).contains(&response.status) => {
            match StatusCode::from_u16(response.status) {
                Ok(status) => (status, Json(response.body)).into_response(),
                Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            }
        }
        Ok(_) => {
            tracing::error!(name, "Control handler returned an invalid status");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(error) => {
            tracing::error!(name, %error, "Control handler failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    #[tokio::test]
    async fn control_routes_preserve_methods_status_and_registration_lifetime() {
        let registry = ControlRegistry::default();
        registry
            .register(
                "planner".into(),
                Arc::new(|request| {
                    Box::pin(async move {
                        assert_eq!(request.path, "config");
                        let status = match request.method.as_str() {
                            "GET" => 200,
                            "PATCH" if request.body["expected_revision"] == 3 => 202,
                            "PATCH" => 409,
                            _ => 405,
                        };
                        Ok(ControlResponse {
                            status,
                            body: request.body,
                        })
                    })
                }),
            )
            .unwrap();
        let app = router(registry.clone());
        for (method, path, body, expected) in [
            ("GET", "/control/planner/config", "", 200),
            (
                "PATCH",
                "/control/planner/config",
                r#"{"expected_revision":3}"#,
                202,
            ),
            ("PATCH", "/control/planner/config", "{}", 409),
            ("DELETE", "/control/planner/config", "", 405),
            ("GET", "/control/missing/config", "", 404),
            ("PATCH", "/control/planner/config", "{", 400),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/control/planner/config")
                    .body(Body::from("x".repeat(65537)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        registry.unregister("planner");
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/control/planner/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
