// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::instances::list_all_instances;
use serde_json::json;
use std::sync::Arc;

/// Selects the semantics of the `/ready` endpoint. `/health` and `/live` stay
/// process-lifecycle signals; `/ready` means "able to serve traffic".
pub const READINESS_MODE_ENV: &str = "DYN_FRONTEND_READINESS_MODE";

/// Policy for the `/ready` readiness probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadinessMode {
    /// Ready once the process is accepting requests. Keeps a standalone frontend
    /// deployed ahead of its workers Ready (the historical `/health` behavior).
    Process,
    /// Ready only when the process is accepting requests AND a model has a live,
    /// complete serving topology. Used by GAIE frontend sidecars so a worker pod
    /// is Ready only when its colocated model is routable (with container-scoped
    /// discovery a sidecar only sees its own worker, so "any ready model" ==
    /// "my worker's model"; no cross-worker identity check).
    LocalWorker,
}

/// A set-but-unrecognized [`READINESS_MODE_ENV`] value. Kept distinct from
/// "unset" so callers can fail closed instead of silently downgrading to
/// `Process` (a typo must not let a sidecar report Ready without a routable
/// model).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidReadinessMode(pub String);

impl ReadinessMode {
    /// Resolve the mode from the environment. An unset (or empty) variable
    /// defaults to `Process`; a set-but-unrecognized value is an error so the
    /// caller can fail closed.
    pub fn from_env() -> Result<Self, InvalidReadinessMode> {
        match std::env::var(READINESS_MODE_ENV) {
            Ok(value) => Self::parse(&value),
            Err(_) => Ok(Self::Process),
        }
    }

    fn parse(value: &str) -> Result<Self, InvalidReadinessMode> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "process" => Ok(Self::Process),
            "local-worker" | "local_worker" | "localworker" | "model" => Ok(Self::LocalWorker),
            _ => Err(InvalidReadinessMode(value.trim().to_string())),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::LocalWorker => "local-worker",
        }
    }
}

/// Kubernetes traffic-readiness endpoint (default `/ready`); policy set by
/// [`ReadinessMode`] via [`READINESS_MODE_ENV`].
pub fn ready_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let ready_path = path.unwrap_or_else(|| "/ready".to_string());

    let mode = match ReadinessMode::from_env() {
        Ok(mode) => mode.as_str(),
        Err(InvalidReadinessMode(value)) => {
            tracing::error!(
                value = %value,
                env = READINESS_MODE_ENV,
                "invalid frontend readiness mode; /ready will fail closed (503) until corrected"
            );
            "invalid"
        }
    };
    tracing::info!(
        mode,
        path = %ready_path,
        "registering frontend readiness endpoint"
    );

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &ready_path)];

    let router = Router::new()
        .route(&ready_path, get(ready_handler))
        .with_state(state);

    (docs, router)
}

async fn ready_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    // Read per-request: probes are infrequent and this keeps a plain handler.
    let mode = match ReadinessMode::from_env() {
        Ok(mode) => mode,
        // Fail closed on a misconfigured mode instead of silently serving as
        // `Process`, which could let a sidecar report Ready without a model.
        Err(InvalidReadinessMode(value)) => {
            tracing::warn!(
                value = %value,
                env = READINESS_MODE_ENV,
                "invalid frontend readiness mode; failing readiness closed"
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "not_ready",
                    "message": format!(
                        "invalid {READINESS_MODE_ENV} value '{value}'; expected 'process' or 'local-worker'"
                    )
                })),
            );
        }
    };

    // Draining/stopping is never ready, regardless of mode.
    if !state.is_ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "mode": mode.as_str(),
                "stage": state.service_stage().to_string(),
                "message": "frontend is not accepting requests"
            })),
        );
    }

    match mode {
        ReadinessMode::Process => (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "mode": mode.as_str()
            })),
        ),
        ReadinessMode::LocalWorker => {
            if state.manager().has_any_ready_model() {
                let mut ready_models: Vec<String> = state
                    .manager()
                    .serving_ready_display_names()
                    .into_iter()
                    .collect();
                ready_models.sort();
                (
                    StatusCode::OK,
                    Json(json!({
                        "status": "ready",
                        "mode": mode.as_str(),
                        "ready_models": ready_models
                    })),
                )
            } else {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "status": "not_ready",
                        "mode": mode.as_str(),
                        "message": "no model has a live, complete serving topology yet"
                    })),
                )
            }
        }
    }
}

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let health_path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &health_path)];

    let router = Router::new()
        .route(&health_path, get(health_handler))
        .with_state(state);

    (docs, router)
}

pub fn live_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let live_path = path.unwrap_or_else(|| "/live".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &live_path)];

    let router = Router::new()
        .route(&live_path, get(live_handler))
        .with_state(state);

    (docs, router)
}

async fn live_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    // Check if the http service is being cancelled/shutdown
    if state.is_cancelled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "shutting_down",
                "message": "Service is shutting down"
            })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "live",
            "message": "Service is live"
        })),
    )
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    if !state.is_ready() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "stage": state.service_stage().to_string(),
                "message": "Service is not ready"
            })),
        );
    }

    let instances = match list_all_instances(state.discovery()).await {
        Ok(instances) => instances,
        Err(err) => {
            tracing::warn!(%err, "Failed to fetch instances from discovery");
            vec![]
        }
    };
    let mut endpoints: Vec<String> = instances
        .iter()
        .map(|instance| instance.endpoint_id().as_url())
        .collect();
    endpoints.sort();
    endpoints.dedup();
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "endpoints": endpoints,
            "instances": instances
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::{InvalidReadinessMode, ReadinessMode};

    #[test]
    fn readiness_mode_parses_process_variants() {
        assert_eq!(ReadinessMode::parse("process"), Ok(ReadinessMode::Process));
        assert_eq!(ReadinessMode::parse("PROCESS"), Ok(ReadinessMode::Process));
        assert_eq!(
            ReadinessMode::parse("  process  "),
            Ok(ReadinessMode::Process)
        );
        // Empty string falls back to the default (process).
        assert_eq!(ReadinessMode::parse(""), Ok(ReadinessMode::Process));
    }

    #[test]
    fn readiness_mode_parses_local_worker_variants() {
        assert_eq!(
            ReadinessMode::parse("local-worker"),
            Ok(ReadinessMode::LocalWorker)
        );
        assert_eq!(
            ReadinessMode::parse("local_worker"),
            Ok(ReadinessMode::LocalWorker)
        );
        assert_eq!(
            ReadinessMode::parse("LocalWorker"),
            Ok(ReadinessMode::LocalWorker)
        );
        assert_eq!(
            ReadinessMode::parse("model"),
            Ok(ReadinessMode::LocalWorker)
        );
    }

    #[test]
    fn readiness_mode_unknown_is_rejected() {
        // A set-but-unrecognized value (e.g. a typo) must fail closed rather
        // than silently downgrading to `Process`; the original value is
        // preserved (trimmed) for diagnostics.
        assert_eq!(
            ReadinessMode::parse("bogus"),
            Err(InvalidReadinessMode("bogus".to_string()))
        );
        assert_eq!(
            ReadinessMode::parse("  local-wroker  "),
            Err(InvalidReadinessMode("local-wroker".to_string()))
        );
    }
}
