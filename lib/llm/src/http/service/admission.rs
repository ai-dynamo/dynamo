// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend admission gates (DEP: Request Admission and Rejection Controls).
//!
//! The frontend is the earliest client-facing gate. It rejects before
//! tokenization using only coarse frontend-local state:
//!
//! - **Request concurrency** (`--rejection-frontend-request-concurrency-limit`):
//!   enforced separately for each served model, protecting downstream model
//!   capacity.
//! - **Runtime tasks** (`--rejection-frontend-runtime-task-limit`): alive tasks
//!   on the frontend tokio runtime, protecting frontend compute capacity.
//! - **Request-plane pressure**
//!   (`--rejection-frontend-request-plane-connection-limit`): in-flight
//!   request-plane streams to workers, protecting outbound transport resources
//!   (worker connections / TCP source ports in TCP mode).
//!
//! Every gate is disabled by default and active only when explicitly
//! configured (see [`AdmissionGateConfig`]). A gate rejects when the observed
//! value exceeds its configured limit, returning HTTP 503 with a
//! gate-specific message. Frontend rejection is distinct from router and
//! worker rejection: it never inspects token counts, queue state, or SLO
//! predictions.

use std::sync::Arc;

use axum::response::IntoResponse;

use super::openai::{ErrorMessage, ErrorResponse};
use super::service_v2::State;
use crate::frontend_config::AdmissionGateConfig;
use dynamo_runtime::metrics::prometheus_names::frontend_service::admission_gate;
use dynamo_runtime::metrics::request_plane::REQUEST_PLANE_INFLIGHT;

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
/// The effective limit resolves per the DEP: a per-model override supplied
/// at model registration (MDC `router_config`) wins over the frontend-global
/// `--rejection-frontend-request-concurrency-limit`; either alone activates
/// the gate for that model.
pub(crate) fn evaluate_model_concurrency_gate(
    state: &State,
    model: &str,
    metric_model: &str,
) -> Option<String> {
    if model != metric_model {
        return None;
    }
    let limit = state
        .manager()
        .request_concurrency_limit_override(model)
        .or_else(|| state.admission_gate_config().request_concurrency_limit())?;
    let inflight = state.metrics_clone().get_inflight_count(metric_model);
    if inflight >= 0 && inflight as u64 > limit {
        return Some(reject(
            state,
            admission_gate::REQUEST_CONCURRENCY,
            metric_model,
            format!(
                "Frontend admission gate rejected this request: {inflight} in-flight \
                 requests for model '{metric_model}' exceeds \
                 --rejection-frontend-request-concurrency-limit={limit}; retry later"
            ),
        ));
    }
    None
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
pub(crate) fn check_frontend_local_gates(state: &State) -> Result<(), ErrorResponse> {
    let config = state.admission_gate_config();

    if let Some(limit) = config.runtime_task_limit() {
        let alive_tasks = tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks() as u64;
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

    if let Some(limit) = config.request_plane_connection_limit() {
        let inflight_streams = REQUEST_PLANE_INFLIGHT.get();
        if inflight_streams > limit as f64 {
            return Err(ErrorMessage::service_unavailable_with_body(reject(
                state,
                admission_gate::REQUEST_PLANE_CONNECTION,
                "",
                format!(
                    "Frontend admission gate rejected this request: {inflight_streams} \
                     in-flight request-plane streams exceeds \
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
    axum::extract::State(state): axum::extract::State<Arc<State>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if let Err(err_response) = check_frontend_local_gates(&state) {
        return err_response.into_response();
    }
    next.run(request).await
}

/// Log enabled gates and export their configured limits as gauges. Called
/// once at HTTP service construction.
pub(crate) fn announce_enabled_gates(config: &AdmissionGateConfig, metrics: &super::Metrics) {
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
            metrics.set_admission_gate_limit(gate, limit);
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
