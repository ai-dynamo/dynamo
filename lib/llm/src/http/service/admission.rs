// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend admission gates (DEP: Request Admission and Rejection Controls).
//!
//! The frontend is the earliest client-facing gate. It rejects before
//! tokenization using only coarse frontend-local state:
//!
//! - **Request concurrency** (`--rejection-frontend-request-concurrency-limit`):
//!   enforced separately for each served model on HTTP inference
//!   request/response endpoints, protecting downstream model capacity. Realtime
//!   WebSocket sessions select their model after the HTTP upgrade and are not
//!   included.
//! - **Runtime tasks** (`--rejection-frontend-runtime-task-limit`): alive tasks
//!   on the frontend tokio runtime, protecting frontend compute capacity.
//! - **Request-plane pressure**
//!   (`--rejection-frontend-request-plane-connection-limit`): process-wide
//!   in-flight request-plane requests/streams to workers. This is an outbound
//!   transport-pressure proxy, not a count of physical TCP connections. The
//!   gate samples current pressure and sheds load on a best-effort basis; it
//!   does not reserve capacity or guarantee a strict upper bound.
//!
//! Every gate is disabled by default and active only when explicitly
//! configured (see [`AdmissionGateConfig`]). A gate rejects when its observed
//! pressure reaches the configured boundary, returning HTTP 503 with a
//! gate-specific message. Frontend rejection is distinct from router and
//! worker rejection: it never inspects token counts, queue state, or SLO
//! predictions.

use std::sync::Arc;

use axum::response::IntoResponse;

use super::openai::{ErrorMessage, ErrorResponse};
use super::service_v2::State;
use crate::discovery::UNKNOWN_METRIC_MODEL;
use crate::frontend_config::AdmissionGateConfig;
use dynamo_runtime::metrics::prometheus_names::frontend_service::admission_gate;
use dynamo_runtime::metrics::request_plane::REQUEST_PLANE_INFLIGHT;

/// State used only by the frontend-local admission middleware.
///
/// `request_plane_exempt_path` names a local inference route that never opens
/// a request-plane stream. It still passes through the runtime-task gate.
pub(crate) struct FrontendLocalGateState {
    state: Arc<State>,
    request_plane_exempt_path: Option<Box<str>>,
}

impl FrontendLocalGateState {
    pub(crate) fn new(state: Arc<State>, request_plane_exempt_path: Option<String>) -> Self {
        Self {
            state,
            request_plane_exempt_path: request_plane_exempt_path.map(String::into_boxed_str),
        }
    }

    fn route_uses_request_plane(&self, path: &str) -> bool {
        self.request_plane_exempt_path.as_deref() != Some(path)
    }
}

/// Per-served-model request concurrency gate, returning the rejection message
/// so non-OpenAI surfaces (e.g. the Anthropic Messages API) can shape their
/// own error body. Rejection metrics and logs are recorded here.
///
/// Must be called AFTER the per-model inflight gauge has been incremented
/// (i.e. after the handler creates its `InflightGuard`): each request then
/// observes a count that includes itself, so concurrent arrivals can
/// over-reject under a race but can never over-admit past the limit.
///
/// `model` is the client-requested name and `metric_model` the resolved
/// metrics label. They differ only when the model is not registered
/// (`metric_model` collapses to the shared unknown-model sentinel); such
/// requests are exempted here so they surface the usual 404 instead of a
/// misleading admission 503.
///
/// The effective limit resolves per the DEP: the dedicated per-model
/// `ModelDeploymentCard::rejection_frontend_request_concurrency_limit` override
/// wins over the frontend-global
/// `--rejection-frontend-request-concurrency-limit`; either alone activates the
/// gate for that model.
pub(crate) fn evaluate_model_concurrency_gate(
    state: &State,
    model: &str,
    metric_model: &str,
) -> Option<String> {
    if model != metric_model {
        return None;
    }

    let global_limit = state.admission_gate_config().request_concurrency_limit();
    let manager = state.manager();
    let has_seen_model_override = manager.has_seen_request_concurrency_limit_override();
    if global_limit.is_none() && !has_seen_model_override {
        return None;
    }

    // `metric_model_for` maps every unregistered name to this sentinel. The
    // ordinary unknown-model case returned above because the strings differ,
    // but a client can request the sentinel literally. Confirm that rare
    // collision is registered so it still reaches the normal 404 path.
    if metric_model == UNKNOWN_METRIC_MODEL && !manager.has_registered_model(model) {
        return None;
    }

    let model_override = if has_seen_model_override {
        manager.request_concurrency_limit_override(model)
    } else {
        None
    };
    let (limit, source) = match model_override {
        Some(limit) => (limit, ConcurrencyLimitSource::ModelDeploymentCard),
        None => (global_limit?, ConcurrencyLimitSource::FrontendGlobal),
    };
    let inflight = state.metrics_clone().get_inflight_count(metric_model);
    if inflight >= 0 && inflight as u64 > limit {
        let configured_limit = match source {
            ConcurrencyLimitSource::ModelDeploymentCard => format!(
                "per-model MDC override rejection_frontend_request_concurrency_limit={limit}"
            ),
            ConcurrencyLimitSource::FrontendGlobal => {
                format!("frontend-global --rejection-frontend-request-concurrency-limit={limit}")
            }
        };
        return Some(reject(
            state,
            admission_gate::REQUEST_CONCURRENCY,
            metric_model,
            format!(
                "Frontend admission gate rejected this request: {inflight} in-flight \
                 requests for model '{metric_model}' exceeds {configured_limit}; retry later"
            ),
        ));
    }
    None
}

#[derive(Clone, Copy)]
enum ConcurrencyLimitSource {
    ModelDeploymentCard,
    FrontendGlobal,
}

/// [`evaluate_model_concurrency_gate`] shaped as an OpenAI-style 503 error.
pub(crate) fn check_model_concurrency_gate(
    state: &State,
    model: &str,
    metric_model: &str,
) -> Result<(), ErrorResponse> {
    match evaluate_model_concurrency_gate(state, model, metric_model) {
        Some(message) => Err(ErrorMessage::service_unavailable_with_body(message)),
        None => Ok(()),
    }
}

/// Frontend-local self-protection gates (runtime tasks and request-plane
/// pressure). Model-independent, evaluated by middleware before the request
/// body is parsed. Rejection metrics and logs are recorded here.
fn check_frontend_local_gates(
    state: &State,
    route_uses_request_plane: bool,
) -> Result<(), ErrorResponse> {
    let config = state.admission_gate_config();

    if let Some(limit) = config.runtime_task_limit() {
        let alive_tasks = tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks() as u64;
        // This measurement already includes the task executing the middleware,
        // so `>` admits exactly `limit` live tasks and rejects the excess.
        if alive_tasks > limit {
            return Err(ErrorMessage::service_unavailable_with_body(reject(
                state,
                admission_gate::RUNTIME_TASK,
                "",
                format!(
                    "Frontend admission gate rejected this request: {alive_tasks} alive \
                     runtime tasks exceeds --rejection-frontend-runtime-task-limit={limit}; \
                     retry later"
                ),
            )));
        }
    }

    if route_uses_request_plane && let Some(limit) = config.request_plane_connection_limit() {
        let inflight_streams = REQUEST_PLANE_INFLIGHT.get();
        // Unlike the model concurrency and runtime-task measurements, this
        // process-wide gauge does not yet include the candidate HTTP request.
        // Reject at equality, but do not reserve capacity: concurrent arrivals
        // may observe the same value and proceed before their outbound streams
        // increment the gauge. This is a best-effort pressure trigger, not a
        // strict concurrency cap.
        if inflight_streams >= limit as f64 {
            return Err(ErrorMessage::service_unavailable_with_body(reject(
                state,
                admission_gate::REQUEST_PLANE_CONNECTION,
                "",
                format!(
                    "Frontend admission gate rejected this request: the in-flight \
                     request-plane stream count ({inflight_streams}) has reached \
                     --rejection-frontend-request-plane-connection-limit={limit}; retry later"
                ),
            )));
        }
    }

    Ok(())
}

/// Middleware enforcing the frontend-local gates on every inference route.
/// Installed only when at least one frontend-local gate is configured.
pub(crate) async fn enforce_frontend_local_gates(
    axum::extract::State(gate_state): axum::extract::State<Arc<FrontendLocalGateState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let route_uses_request_plane = gate_state.route_uses_request_plane(request.uri().path());
    if let Err(err_response) =
        check_frontend_local_gates(&gate_state.state, route_uses_request_plane)
    {
        return err_response.into_response();
    }
    next.run(request).await
}

/// Log enabled gates once at HTTP service construction.
pub(crate) fn announce_enabled_gates(config: &AdmissionGateConfig) {
    let gates = [
        (
            admission_gate::REQUEST_CONCURRENCY,
            config.request_concurrency_limit(),
        ),
        (admission_gate::RUNTIME_TASK, config.runtime_task_limit()),
        (
            admission_gate::REQUEST_PLANE_CONNECTION,
            config.request_plane_connection_limit(),
        ),
    ];
    for (gate, limit) in gates {
        if let Some(limit) = limit {
            tracing::info!(gate, limit, "frontend admission gate enabled");
        }
    }
}

/// Record the rejection (metrics + log) and pass the message through.
fn reject(state: &State, gate: &str, model: &str, message: String) -> String {
    state.metrics_clone().inc_admission_rejection(gate, model);
    tracing::warn!(gate, model, "{message}");
    message
}
