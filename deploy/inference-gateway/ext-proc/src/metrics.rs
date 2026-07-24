// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics exposed by the EPP on its own `/metrics` endpoint.
//!
//! The EPP keeps a private registry instead of the runtime's component
//! registry: it holds no `DistributedRuntime` past startup (see
//! [`crate::epp::Router::from_discovery`]) and these series describe gateway
//! traffic rather than a registered Dynamo component.
//!
//! [`DEFAULT_METRICS_PORT`] matches the port GAIE's Go endpoint picker uses,
//! so existing endpoint-picker scrape configuration keeps working.

use std::sync::LazyLock;

use axum::{
    Router as AxumRouter, http::StatusCode, http::header::CONTENT_TYPE, response::IntoResponse,
    routing::get,
};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use prometheus::{Encoder, HistogramOpts, HistogramVec, Registry, TEXT_FORMAT, TextEncoder};

/// Port the `/metrics` endpoint binds to unless `DYN_EPP_METRICS_PORT` says
/// otherwise. Distinct from the ext_proc gRPC port (9002) and the health port
/// (9003).
pub const DEFAULT_METRICS_PORT: u16 = 9090;

/// Environment variable overriding [`DEFAULT_METRICS_PORT`]. `0` disables the
/// metrics server entirely.
pub const METRICS_PORT_ENV: &str = "DYN_EPP_METRICS_PORT";

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// Prompt tokens the model server reported as prefix-cache hits, read from
/// `usage.prompt_tokens_details.cached_tokens` in the response body.
///
/// This is the *observed* cache hit, as opposed to the overlap estimate the KV
/// router computes at selection time. Buckets mirror the frontend's
/// `dynamo_frontend_cached_tokens` defaults so the two can share a dashboard.
static CACHED_TOKENS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_cached_tokens",
            "Prompt tokens served from the model server's KV cache per request, \
             as reported in usage.prompt_tokens_details.cached_tokens",
        )
        .buckets(generate_log_buckets(50.0, 128_000.0, 12)),
        &["model"],
    )
    .expect("cached_tokens histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("cached_tokens is the only registrant of its name");
    histogram
});

/// Record the model server's reported cache-hit token count for one completed
/// request. `model` is the upstream model that served it.
pub fn observe_cached_tokens(model: &str, cached_tokens: u64) {
    CACHED_TOKENS
        .with_label_values(&[model])
        .observe(cached_tokens as f64);
}

/// Serve `/metrics` until the process exits.
pub async fn serve(port: u16) -> anyhow::Result<()> {
    let app = AxumRouter::new().route("/metrics", get(render));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, "Serving Prometheus metrics on /metrics");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn render() -> impl IntoResponse {
    let mut buf = Vec::new();
    match TextEncoder::new().encode(&REGISTRY.gather(), &mut buf) {
        Ok(()) => (StatusCode::OK, [(CONTENT_TYPE, TEXT_FORMAT)], buf),
        Err(err) => {
            tracing::warn!(%err, "Failed to encode Prometheus metrics");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(CONTENT_TYPE, TEXT_FORMAT)],
                Vec::new(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gathered() -> String {
        let mut buf = Vec::new();
        TextEncoder::new()
            .encode(&REGISTRY.gather(), &mut buf)
            .expect("encode");
        String::from_utf8(buf).expect("utf8")
    }

    #[test]
    fn observed_cached_tokens_are_exported_per_model() {
        observe_cached_tokens("test-model", 128);

        let exposition = gathered();
        assert!(
            exposition.contains("dynamo_epp_cached_tokens_count{model=\"test-model\"} 1"),
            "expected one observation for test-model, got:\n{exposition}"
        );
        assert!(
            exposition.contains("dynamo_epp_cached_tokens_sum{model=\"test-model\"} 128"),
            "expected a sum of 128 for test-model, got:\n{exposition}"
        );
    }

    #[test]
    fn zero_cached_tokens_still_counts_as_an_observation() {
        observe_cached_tokens("cold-model", 0);

        let exposition = gathered();
        assert!(
            exposition.contains("dynamo_epp_cached_tokens_count{model=\"cold-model\"} 1"),
            "a full cache miss must be recorded, not skipped, got:\n{exposition}"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_serves_prometheus_text_format() {
        observe_cached_tokens("served-model", 32);

        let response = render().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .expect("content-type is set"),
            TEXT_FORMAT
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(
            body.contains("# TYPE dynamo_epp_cached_tokens histogram"),
            "expected histogram metadata, got:\n{body}"
        );
    }
}
