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

use std::sync::{LazyLock, OnceLock};

use axum::{
    Router as AxumRouter, http::StatusCode, http::header::CONTENT_TYPE, response::IntoResponse,
    routing::get,
};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntGaugeVec, Opts, Registry, TEXT_FORMAT, TextEncoder,
};

/// Port the `/metrics` endpoint binds to unless `DYN_EPP_METRICS_PORT` says
/// otherwise. Distinct from the ext_proc gRPC port (9002) and the health port
/// (9003).
pub const DEFAULT_METRICS_PORT: u16 = 9090;

/// Environment variable overriding [`DEFAULT_METRICS_PORT`]. `0` disables the
/// metrics server entirely.
pub const METRICS_PORT_ENV: &str = "DYN_EPP_METRICS_PORT";

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// Label value for every series, bound once at startup by [`set_served_model`].
static SERVED_MODEL: OnceLock<String> = OnceLock::new();

/// Used until [`set_served_model`] runs, and in tests that never bind one.
const UNKNOWN_MODEL: &str = "unknown";

/// Bind the `model` label to the model this pool serves: the discovered model
/// card in Dynamo-discovery mode, the configured model name in standalone mode.
///
/// The label deliberately does not come from the request body. That string is
/// unvalidated client input which the router never checks against the served
/// model, so using it would let any caller mint an unbounded number of label
/// values and blow up series cardinality in the scrape backend. The EPP serves
/// one pool, so one process-wide value is also the honest label: it names what
/// actually served the request rather than what the client asked for.
///
/// Only the first call takes effect.
pub fn set_served_model(model: impl Into<String>) {
    if let Err(ignored) = SERVED_MODEL.set(model.into()) {
        tracing::debug!(%ignored, "Served model label already bound; ignoring rebind");
    }
}

fn served_model_label() -> &'static str {
    SERVED_MODEL.get().map_or(UNKNOWN_MODEL, String::as_str)
}

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
/// request, against the model bound by [`set_served_model`].
pub fn observe_cached_tokens(cached_tokens: u64) {
    CACHED_TOKENS
        .with_label_values(&[served_model_label()])
        .observe(cached_tokens as f64);
}

/// Possible startup KV-index recovery outcomes for one EPP replica. The gauge
/// reports which state is current (1 for it, 0 for the others), so a scrape can
/// distinguish "recovered from a peer" from "bootstrapped empty" and from
/// "recovery disabled" — the difference between a normal cold start and a
/// silent full-index loss.
pub const KV_RECOVERY_RECOVERED: &str = "recovered";
pub const KV_RECOVERY_EMPTY_BOOTSTRAP: &str = "empty_bootstrap";
pub const KV_RECOVERY_DISABLED: &str = "recovery_disabled";

/// Current startup KV-index recovery outcome for this replica. Set once at
/// startup. A `1` appears on exactly one `state` label; the others are `0`.
///
/// Alerts consume the *transition* (a replica that previously exported
/// `recovered` flipping to `empty_bootstrap` means a full-index loss, whereas
/// `empty_bootstrap` from first deployment is normal) or a *stuck* `disabled` /
/// `empty_bootstrap` lasting past the expected upgrade window — not the bare
/// state, which is expected during first deploy and rolling upgrades.
static KV_RECOVERY_STATE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let gauge = IntGaugeVec::new(
        Opts::new(
            "dynamo_epp_kv_recovery_state",
            "Startup KV-index peer-recovery outcome for this EPP replica: \
             recovered (restored from a peer dump), empty_bootstrap (no eligible \
             serving peer), or recovery_disabled (peer Service lacks the \
             selection-http port). Exactly one label is 1.",
        ),
        &["state"],
    )
    .expect("kv_recovery_state gauge options are statically valid");
    REGISTRY
        .register(Box::new(gauge.clone()))
        .expect("kv_recovery_state is the only registrant of its name");
    gauge
});

/// Mark this replica's startup KV-index recovery outcome.
pub fn set_kv_recovery_state(state: &str) {
    for candidate in [
        KV_RECOVERY_RECOVERED,
        KV_RECOVERY_EMPTY_BOOTSTRAP,
        KV_RECOVERY_DISABLED,
    ] {
        KV_RECOVERY_STATE
            .with_label_values(&[candidate])
            .set(i64::from(candidate == state));
    }
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
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// The label and the histogram are process-wide, so tests bind the same
    /// value, assert on deltas instead of absolute counts, and serialize so a
    /// concurrent test cannot land an observation mid-delta.
    const TEST_MODEL: &str = "test-model";

    static SERIALIZE: Mutex<()> = Mutex::new(());

    fn bind_test_model() -> MutexGuard<'static, ()> {
        set_served_model(TEST_MODEL);
        SERIALIZE.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `(count, sum)` currently recorded against [`TEST_MODEL`].
    fn recorded() -> (u64, f64) {
        let histogram = CACHED_TOKENS.with_label_values(&[TEST_MODEL]);
        (histogram.get_sample_count(), histogram.get_sample_sum())
    }

    #[test]
    fn observations_land_on_the_bound_served_model() {
        let _guard = bind_test_model();
        let (count_before, sum_before) = recorded();

        observe_cached_tokens(128);

        let (count_after, sum_after) = recorded();
        assert_eq!(count_after, count_before + 1);
        assert_eq!(sum_after, sum_before + 128.0);
    }

    #[test]
    fn zero_cached_tokens_still_counts_as_an_observation() {
        let _guard = bind_test_model();
        let (count_before, _) = recorded();

        observe_cached_tokens(0);

        assert_eq!(
            recorded().0,
            count_before + 1,
            "a full cache miss must be recorded, not skipped"
        );
    }

    /// Regression test: the label used to be the unvalidated `model` string
    /// from the request body, so traffic alone could mint new series.
    #[test]
    fn traffic_volume_cannot_grow_the_label_set() {
        let _guard = bind_test_model();
        for _ in 0..100 {
            observe_cached_tokens(1);
        }

        let series = REGISTRY
            .gather()
            .iter()
            .find(|family| family.name() == "dynamo_epp_cached_tokens")
            .expect("histogram is registered")
            .get_metric()
            .len();
        assert_eq!(series, 1, "the model label must not vary per request");
    }

    #[tokio::test]
    async fn metrics_endpoint_serves_prometheus_text_format() {
        let guard = bind_test_model();
        observe_cached_tokens(32);
        drop(guard);

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
