// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use axum::{
    Json,
    body::Bytes,
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Legacy body-only callbacks preserve their JSON response and 200/500 status contract.
pub type EngineRouteCallback =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = anyhow::Result<Value>> + Send>> + Send + Sync>;

#[derive(Serialize)]
pub struct EngineHttpRequest {
    pub method: String,
    /// Path relative to the registered handler name.
    pub path: String,
    pub body: Value,
}

#[derive(Deserialize)]
pub struct EngineHttpResponse {
    pub status: u16,
    pub body: Value,
}

pub type EngineHttpCallback = Arc<
    dyn Fn(
            EngineHttpRequest,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<EngineHttpResponse>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
enum Handler {
    Legacy(EngineRouteCallback),
    Http(EngineHttpCallback),
}

/// Registry for both legacy callbacks and method-aware handlers on `/engine/*`.
#[derive(Clone, Default)]
pub struct EngineRouteRegistry {
    routes: Arc<RwLock<HashMap<String, Handler>>>,
}

impl EngineRouteRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, route: &str, callback: EngineRouteCallback) {
        if self
            .routes
            .write()
            .insert(route.to_owned(), Handler::Legacy(callback))
            .is_some()
        {
            tracing::warn!(route, "Overwriting already-registered engine route");
        }
    }

    pub fn get(&self, route: &str) -> Option<EngineRouteCallback> {
        match self.routes.read().get(route) {
            Some(Handler::Legacy(callback)) => Some(Arc::clone(callback)),
            _ => None,
        }
    }

    pub fn routes(&self) -> Vec<String> {
        self.routes.read().keys().cloned().collect()
    }

    /// Register a handler for `/engine/{name}/...` without changing legacy callbacks.
    pub fn register_http(&self, name: String, callback: EngineHttpCallback) -> anyhow::Result<()> {
        anyhow::ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
            "handler name must be a nonempty URL segment"
        );
        let mut routes = self.routes.write();
        let prefix = format!("{name}/");
        anyhow::ensure!(
            !routes
                .keys()
                .any(|key| key == &name || key.starts_with(&prefix)),
            "engine route namespace already registered"
        );
        routes.insert(name, Handler::Http(callback));
        Ok(())
    }

    pub fn unregister_http(&self, name: &str) {
        let mut routes = self.routes.write();
        if matches!(routes.get(name), Some(Handler::Http(_))) {
            routes.remove(name);
        }
    }

    fn get_http(&self, path: &str) -> Option<(EngineHttpCallback, String)> {
        let (name, suffix) = path.split_once('/').unwrap_or((path, ""));
        match self.routes.read().get(name) {
            Some(Handler::Http(callback)) => Some((Arc::clone(callback), suffix.to_owned())),
            _ => None,
        }
    }
}

fn parse_json(body: &[u8], empty: Value) -> Result<Value, serde_json::Error> {
    if body.is_empty() {
        return Ok(empty);
    }
    serde_json::from_slice(body)
}

fn invalid_json(error: serde_json::Error) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error":"Invalid JSON", "message":error.to_string()})),
    )
        .into_response()
}

pub(crate) async fn dispatch(
    registry: &EngineRouteRegistry,
    path: String,
    method: Method,
    body: Bytes,
) -> Response {
    // Exact legacy routes keep their precedence and body-only callback contract.
    if let Some(callback) = registry.get(&path) {
        let payload = match parse_json(&body, json!({})) {
            Ok(value) => value,
            Err(error) => return invalid_json(error),
        };
        return match callback(payload).await {
            Ok(response) => (StatusCode::OK, response.to_string()).into_response(),
            Err(error) => {
                tracing::error!(path, %error, "Engine route handler failed");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error":"Handler error", "message":error.to_string()}).to_string(),
                )
                    .into_response()
            }
        };
    }
    let Some((callback, relative_path)) = registry.get_http(&path) else {
        if let Err(error) = parse_json(&body, Value::Null) {
            return invalid_json(error);
        }
        return (
            StatusCode::NOT_FOUND,
            json!({"error":"Route not found", "message":format!("Route /engine/{path} not found")})
                .to_string(),
        )
            .into_response();
    };
    if body.len() > 64 * 1024 {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let payload = match parse_json(&body, Value::Null) {
        Ok(value) => value,
        Err(error) => return invalid_json(error),
    };
    match callback(EngineHttpRequest {
        method: method.to_string(),
        path: relative_path,
        body: payload,
    })
    .await
    {
        Ok(response) => match StatusCode::from_u16(response.status) {
            Ok(status) if (200..600).contains(&response.status) => {
                (status, Json(response.body)).into_response()
            }
            _ => {
                tracing::error!(path, "Engine HTTP handler returned an invalid status");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        },
        Err(error) => {
            tracing::error!(path, %error, "Engine HTTP handler failed");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::{Path, State},
        http::Request,
        routing::any,
    };
    use tower::ServiceExt;

    fn router(registry: EngineRouteRegistry) -> Router {
        Router::new()
            .route(
                "/engine/{*path}",
                any(
                    |State(registry): State<EngineRouteRegistry>,
                     Path(path): Path<String>,
                     method: Method,
                     body: Bytes| async move {
                        dispatch(&registry, path, method, body).await
                    },
                ),
            )
            .with_state(registry)
    }

    #[tokio::test]
    async fn test_registry_basic() {
        let registry = EngineRouteRegistry::new();

        // Register a simple callback
        let callback: EngineRouteCallback =
            Arc::new(|body| Box::pin(async move { Ok(serde_json::json!({"echo": body})) }));

        registry.register("test", callback);

        // Verify it's registered
        assert!(registry.get("test").is_some());
        assert!(registry.get("nonexistent").is_none());

        // Verify routes list
        let routes = registry.routes();
        assert_eq!(routes.len(), 1);
        assert!(routes.contains(&"test".to_string()));
    }

    #[tokio::test]
    async fn test_callback_execution() {
        let registry = EngineRouteRegistry::new();

        let callback: EngineRouteCallback = Arc::new(|body| {
            Box::pin(async move {
                let input = body.get("input").and_then(|v| v.as_str()).unwrap_or("");
                Ok(serde_json::json!({
                    "output": format!("processed: {}", input)
                }))
            })
        });

        registry.register("process", callback);

        // Get and execute callback
        let cb = registry.get("process").unwrap();
        let result = cb(serde_json::json!({"input": "test"})).await.unwrap();

        assert_eq!(result["output"], "processed: test");
    }

    #[tokio::test]
    async fn test_clone_shares_routes() {
        let registry = EngineRouteRegistry::new();

        let callback: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": true})) }));
        registry.register("test", callback);

        // Clone the registry
        let cloned = registry.clone();

        // Both should see the same route
        assert!(registry.get("test").is_some());
        assert!(cloned.get("test").is_some());

        // Register on clone
        let callback2: EngineRouteCallback =
            Arc::new(|_| Box::pin(async { Ok(serde_json::json!({"ok": false})) }));
        cloned.register("test2", callback2);

        // Original should also see it (they share the Arc)
        assert!(registry.get("test2").is_some());
    }
    #[tokio::test]
    async fn http_handlers_preserve_methods_status_and_registration_lifetime() {
        let registry = EngineRouteRegistry::default();
        registry
            .register_http(
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
                        Ok(EngineHttpResponse {
                            status,
                            body: request.body,
                        })
                    })
                }),
            )
            .unwrap();
        let app = router(registry.clone());
        for (method, path, body, expected) in [
            ("GET", "/engine/planner/config", "", 200),
            (
                "PATCH",
                "/engine/planner/config",
                r#"{"expected_revision":3}"#,
                202,
            ),
            ("PATCH", "/engine/planner/config", "{}", 409),
            ("DELETE", "/engine/planner/config", "", 405),
            ("GET", "/engine/missing/config", "", 404),
            ("PATCH", "/engine/planner/config", "{", 400),
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
            if expected == 400 {
                assert_eq!(response.headers()["content-type"], "application/json");
                let body = to_bytes(response.into_body(), 1024).await.unwrap();
                let error: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(error["error"], "Invalid JSON");
                assert!(error["message"].as_str().is_some());
            }
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri("/engine/planner/config")
                    .body(Body::from("x".repeat(65537)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        registry.unregister_http("planner");
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/engine/planner/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
    #[tokio::test]
    async fn legacy_payloads_are_not_interpreted_as_http_envelopes() {
        let registry = EngineRouteRegistry::new();
        registry.register("legacy", Arc::new(|body| Box::pin(async move { Ok(body) })));
        let app = router(registry);
        let payload = json!({"status":418, "body":{"value":7}});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/engine/legacy")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&body).unwrap(), payload);
    }
}
