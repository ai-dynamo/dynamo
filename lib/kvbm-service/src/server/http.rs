// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! axum HTTP sidecar (liveness, readiness, metrics, discovery).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use axum::{
    Router,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::get,
};
use dynamo_memory::resources::Resources;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::metrics::ServiceMetrics;
use crate::pool::HostMemoryPool;
use crate::registry::Registry;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared state for axum handlers. Cheap to clone.
#[derive(Clone)]
pub struct HttpState {
    pub registry: Registry,
    pub resources: Arc<Resources>,
    pub metrics: ServiceMetrics,
    /// UDS path the gRPC listener is bound to. Set after binding; until then
    /// `/ready` returns 503.
    pub uds_path: Arc<parking_lot::RwLock<Option<PathBuf>>>,
    /// Always `true` in the MVP (no allocator), but wired for the follow-on PR.
    pub allocator_ready: Arc<AtomicBool>,
    /// Host-memory pool, when constructed via
    /// [`crate::KvbmService::start_with_pool`]. `None` for shell-only
    /// deployments; `/v1/pool` returns 503 in that case.
    pub pool: Option<Arc<HostMemoryPool>>,
}

impl HttpState {
    pub fn new(registry: Registry, resources: Arc<Resources>, metrics: ServiceMetrics) -> Self {
        Self {
            registry,
            resources,
            metrics,
            uds_path: Arc::new(parking_lot::RwLock::new(None)),
            allocator_ready: Arc::new(AtomicBool::new(true)),
            pool: None,
        }
    }

    pub fn with_pool(mut self, pool: Arc<HostMemoryPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn set_uds_path(&self, path: PathBuf) {
        *self.uds_path.write() = Some(path);
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all routes wired to `state`.
pub fn build_router(state: HttpState) -> Router {
    Router::new()
        .route("/live", get(live))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/v1/discovery/socket", get(discovery_socket))
        .route("/v1/discovery/resources", get(discovery_resources))
        .route("/v1/registrations", get(registrations))
        .route("/v1/pool", get(pool_snapshot))
        .with_state(state)
}

/// Bind `addr` and run the router until `cancel` fires.
///
/// Returns the actual bound `SocketAddr` (relevant when the caller passed port
/// 0) and a [`tokio::task::JoinHandle`] for the server task.
pub async fn serve(
    addr: SocketAddr,
    state: HttpState,
    cancel: CancellationToken,
) -> std::io::Result<(SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let router = build_router(state);
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router)
            .with_graceful_shutdown(cancel.cancelled_owned())
            .await
        {
            tracing::error!("HTTP sidecar error: {e}");
        }
    });
    Ok((bound, handle))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn live() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

#[derive(Serialize)]
struct ReadyBody {
    ready: bool,
    reason: Option<String>,
}

async fn ready(State(state): State<HttpState>) -> Response {
    let allocator_ok = state.allocator_ready.load(Ordering::Relaxed);
    let uds_ok = state.uds_path.read().is_some();

    if allocator_ok && uds_ok {
        (
            StatusCode::OK,
            Json(ReadyBody {
                ready: true,
                reason: None,
            }),
        )
            .into_response()
    } else {
        let reason = if !uds_ok {
            "gRPC socket not yet bound"
        } else {
            "allocator not ready"
        };
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadyBody {
                ready: false,
                reason: Some(reason.to_owned()),
            }),
        )
            .into_response()
    }
}

async fn metrics(State(state): State<HttpState>) -> Response {
    match state.metrics.encode_text() {
        Ok(body) => {
            let mut resp = (StatusCode::OK, body).into_response();
            resp.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
            );
            resp
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metrics encoding error: {e}"),
        )
            .into_response(),
    }
}

#[derive(Serialize)]
struct SocketBody {
    uds_path: String,
}

async fn discovery_socket(State(state): State<HttpState>) -> Response {
    match state.uds_path.read().as_deref() {
        Some(p) => Json(SocketBody {
            uds_path: p.display().to_string(),
        })
        .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": "socket not yet bound" })),
        )
            .into_response(),
    }
}

async fn discovery_resources(State(state): State<HttpState>) -> impl IntoResponse {
    Json((*state.resources).clone())
}

async fn registrations(State(state): State<HttpState>) -> impl IntoResponse {
    Json(state.registry.snapshot())
}

async fn pool_snapshot(State(state): State<HttpState>) -> Response {
    match state.pool.as_deref() {
        Some(pool) => Json(pool.snapshot()).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "host-memory pool not configured; \
                          start the service via KvbmService::start_with_pool"
            })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::metrics::ServiceMetrics;
    use crate::registry::Registry;

    fn test_state() -> HttpState {
        let metrics = ServiceMetrics::new();
        let registry = Registry::new(8, metrics.clone());
        let resources = Arc::new(Resources::discover());
        HttpState::new(registry, resources, metrics)
    }

    async fn bound_addr(state: HttpState) -> (SocketAddr, CancellationToken) {
        let cancel = CancellationToken::new();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let (bound, _handle) = serve(addr, state, cancel.clone()).await.unwrap();
        (bound, cancel)
    }

    #[tokio::test]
    async fn live_always_200() {
        let state = test_state();
        let (addr, cancel) = bound_addr(state).await;
        let url = format!("http://{addr}/live");
        let resp = reqwest::get(&url).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.text().await.unwrap(), "ok");
        cancel.cancel();
    }

    #[tokio::test]
    async fn ready_503_before_uds_set() {
        let state = test_state();
        let (addr, cancel) = bound_addr(state).await;
        let resp = reqwest::get(format!("http://{addr}/ready")).await.unwrap();
        assert_eq!(resp.status(), 503);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ready"], false);
        assert!(body["reason"].as_str().is_some());
        cancel.cancel();
    }

    #[tokio::test]
    async fn ready_200_after_uds_set() {
        let state = test_state();
        state.set_uds_path(PathBuf::from("/tmp/test.sock"));
        let (addr, cancel) = bound_addr(state).await;
        let resp = reqwest::get(format!("http://{addr}/ready")).await.unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["ready"], true);
        cancel.cancel();
    }

    #[tokio::test]
    async fn registrations_empty() {
        let state = test_state();
        let (addr, cancel) = bound_addr(state).await;
        let resp = reqwest::get(format!("http://{addr}/v1/registrations"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["state"], "Empty");
        assert_eq!(body["used_slots"], 0);
        cancel.cancel();
    }

    #[tokio::test]
    async fn metrics_content_type() {
        let state = test_state();
        let (addr, cancel) = bound_addr(state).await;
        let resp = reqwest::get(format!("http://{addr}/metrics"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let ct = resp.headers()[reqwest::header::CONTENT_TYPE]
            .to_str()
            .unwrap();
        assert!(
            ct.contains("version=0.0.4"),
            "unexpected content-type: {ct}"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn discovery_socket_503_then_200() {
        let state = test_state();
        let (addr, cancel) = bound_addr(state.clone()).await;

        // Before UDS is set.
        let resp = reqwest::get(format!("http://{addr}/v1/discovery/socket"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 503);

        // After UDS is set.
        state.set_uds_path(PathBuf::from("/var/run/kvbm.sock"));
        let resp = reqwest::get(format!("http://{addr}/v1/discovery/socket"))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["uds_path"], "/var/run/kvbm.sock");
        cancel.cancel();
    }
}
