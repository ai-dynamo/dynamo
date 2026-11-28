// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The route that serves a Prometheus registry in the text exposition format.

use std::sync::Arc;

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use prometheus::{Encoder, Registry};

/// State for the scrape handler.
/// Optionally holds a reference to the DRT's [`MetricsRegistry`] so that
/// metrics created via `metrics().create*()` anywhere in the process are
/// automatically included in the `/metrics` response.
///
/// [`MetricsRegistry`]: dynamo_runtime::metrics::MetricsRegistry
struct MetricsHandlerState {
    registry: Registry,
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,
}

/// Build the route that serves `registry` in the Prometheus text format, and return it
/// with the path it is mounted at (`/metrics` unless `path` overrides it).
///
/// When `drt_metrics` is provided, the handler serves the DRT's combined registry tree
/// instead, so `registry` must be attached to that tree
/// (`MetricsRegistry::add_prometheus_registry`) for its families to be served.
pub fn scrape_router(
    registry: Registry,
    path: Option<String>,
    drt_metrics: Option<dynamo_runtime::metrics::MetricsRegistry>,
) -> (String, Router) {
    let path = path.unwrap_or_else(|| "/metrics".to_string());

    let metrics_state = MetricsHandlerState {
        registry,
        drt_metrics,
    };

    let route = Router::new()
        .route(&path, get(handler_metrics))
        .with_state(Arc::new(metrics_state));
    (path, route)
}

impl MetricsHandlerState {
    /// The `/metrics` body: with a DRT, its combined output, which includes
    /// this service's registry once it is attached; otherwise this service's
    /// own registry.
    fn render(&self) -> anyhow::Result<String> {
        if let Some(drt_metrics) = &self.drt_metrics {
            return drt_metrics.prometheus_expfmt_combined();
        }
        let mut buffer = vec![];
        prometheus::TextEncoder::new().encode(&self.registry.gather(), &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

/// Renders on the blocking pool: the gather runs update callbacks, some of
/// which take the Python GIL.
async fn handler_metrics(State(state): State<Arc<MetricsHandlerState>>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || state.render()).await {
        Ok(Ok(metrics)) => return (StatusCode::OK, metrics).into_response(),
        Ok(Err(error)) => tracing::warn!(%error, "Failed to encode metrics"),
        Err(error) => tracing::error!(%error, "metrics render panicked"),
    }
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "Failed to encode metrics",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::metrics::{AttachedPrometheusRegistry, MetricsRegistry};

    fn declared(text: &str) -> Vec<&str> {
        text.lines()
            .filter_map(|line| line.strip_prefix("# TYPE "))
            .filter_map(|rest| rest.split_whitespace().next())
            .collect()
    }

    fn counter(registry: &Registry, name: &str) {
        registry
            .register(Box::new(prometheus::IntCounter::new(name, "help").unwrap()))
            .unwrap();
    }

    /// Wire the handler state the way `HttpService::build` does: the local
    /// registry attached to the DRT tree, the handler serving the tree.
    fn attached(
        local: Registry,
        drt: &MetricsRegistry,
    ) -> (MetricsHandlerState, AttachedPrometheusRegistry) {
        let attachment = drt.add_prometheus_registry(local.clone());
        let state = MetricsHandlerState {
            registry: local,
            drt_metrics: Some(drt.clone()),
        };
        (state, attachment)
    }

    /// A frontend family whose type conflicts with a DRT family of the same
    /// name is skipped (#18910): the DRT tree is gathered first, so its
    /// family wins, and every other frontend family is still served.
    #[test]
    fn conflicting_frontend_family_loses_to_drt_and_others_are_served() {
        let drt = MetricsRegistry::new();
        let gauge = prometheus::IntGauge::new("dynamo_conflict", "gauge help").unwrap();
        gauge.set(3);
        drt.get_prometheus_registry()
            .register(Box::new(gauge))
            .unwrap();
        let local = Registry::new();
        counter(&local, "dynamo_conflict");
        counter(&local, "dynamo_frontend_requests_total");
        let (state, _attachment) = attached(local, &drt);

        let text = state.render().unwrap();
        assert!(
            text.lines()
                .any(|line| line == "# TYPE dynamo_conflict gauge"),
            "{text}"
        );
        assert!(
            text.lines().any(|line| line == "dynamo_conflict 3"),
            "{text}"
        );
        assert_eq!(
            declared(&text),
            ["dynamo_conflict", "dynamo_frontend_requests_total"],
            "{text}"
        );
    }
}
