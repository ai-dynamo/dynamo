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

use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use axum::{
    Router as AxumRouter, http::StatusCode, http::header::CONTENT_TYPE, response::IntoResponse,
    routing::get,
};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use dynamo_llm::kv_router::metrics::RouterRequestMetrics;
use dynamo_runtime::metrics::prometheus_names::{name_prefix, router};
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, Registry, TEXT_FORMAT, TextEncoder,
};

use crate::tokio_runtime::TaskKind;

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

/// Register standalone router metrics on the existing EPP scrape endpoint.
/// Both labels come from startup configuration, never request JSON.
pub(crate) fn register_router_metrics(
    model: &str,
    inference_pool: &str,
) -> anyhow::Result<Arc<RouterRequestMetrics>> {
    RouterRequestMetrics::from_registry_subset(
        &REGISTRY,
        name_prefix::COMPONENT,
        &[("model", model), ("inference_pool", inference_pool)],
        &[
            router::REQUESTS_STARTED_TOTAL,
            router::INPUT_SEQUENCE_TOKENS,
            router::OUTPUT_SEQUENCE_TOKENS,
        ],
    )
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

/// One step of routing a request, timed in `dynamo_epp_stage_duration_seconds`.
///
/// Together the stages split the EPP's share of a request's time to first
/// token: receiving the body, picking a worker (parse, template, tokenize,
/// select, book, all inside pick), rewriting the forwarded body, and then
/// waiting for the first response chunk through the gateway, sidecar and
/// worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Request headers received until the end of the request body: the
    /// gateway streaming the client's body to the EPP.
    ReceiveBody,
    /// Reading the `model` field out of the request body.
    ModelProbe,
    /// Parsing the request body into the typed request.
    Parse,
    /// Chat requests: waiting for a tokenization slot (one per core) and a
    /// blocking-pool thread before the chat template.
    BlockingWait,
    /// Rendering the chat template.
    Template,
    /// Tokenizing the prompt: for chat, the rendered prompt, after
    /// [`Stage::BlockingWait`]; for a completions text prompt, including its
    /// waits for a tokenization slot and a blocking-pool thread.
    Tokenize,
    /// Choosing the prefill and decode workers, including router queueing.
    Select,
    /// Registering the request with the router's load bookkeeping.
    Book,
    /// The whole `pick()` call, including waits between the steps above.
    Pick,
    /// Rewriting the forwarded body (`nvext.token_data`, `nvext` strip,
    /// cache salt).
    Rewrite,
    /// Routing decision ready until the response headers reach the EPP: the
    /// gateway sending the body to the sidecar, and the sidecar reading it
    /// and answering with its headers. Streamed 2xx responses only.
    ResponseHeaders,
    /// Routing decision ready until the first non-empty response body
    /// chunk: the gateway, sidecar and worker producing the first token.
    /// Streamed 2xx responses only.
    FirstResponse,
    /// Request headers received until the first non-empty response body
    /// chunk: the time to first token as the EPP sees it. Client time to
    /// first token minus this is the time spent outside the EPP's view,
    /// without comparing clocks across nodes. Streamed 2xx responses only.
    EppTtft,
}

impl Stage {
    #[cfg(test)]
    const ALL: [Stage; 13] = [
        Stage::ReceiveBody,
        Stage::ModelProbe,
        Stage::Parse,
        Stage::BlockingWait,
        Stage::Template,
        Stage::Tokenize,
        Stage::Select,
        Stage::Book,
        Stage::Pick,
        Stage::Rewrite,
        Stage::ResponseHeaders,
        Stage::FirstResponse,
        Stage::EppTtft,
    ];

    fn label(self) -> &'static str {
        match self {
            Stage::ReceiveBody => "receive_body",
            Stage::ModelProbe => "model_probe",
            Stage::Parse => "parse",
            Stage::BlockingWait => "blocking_wait",
            Stage::Template => "template",
            Stage::Tokenize => "tokenize",
            Stage::Select => "select",
            Stage::Book => "book",
            Stage::Pick => "pick",
            Stage::Rewrite => "rewrite",
            Stage::ResponseHeaders => "response_headers",
            Stage::FirstResponse => "first_response",
            Stage::EppTtft => "epp_ttft",
        }
    }
}

static STAGE_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_stage_duration_seconds",
            "Time one routed request spends in each EPP stage",
        )
        .buckets(generate_log_buckets(0.000_01, 60.0, 26)),
        &["model", "stage"],
    )
    .expect("stage histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("stage_duration_seconds is the only registrant of its name");
    histogram
});

/// Record how long one request spent in `stage`.
pub fn observe_stage(stage: Stage, elapsed: Duration) {
    observe_stage_for(served_model_label(), stage, elapsed);
}

fn observe_stage_for(model: &str, stage: Stage, elapsed: Duration) {
    STAGE_SECONDS
        .with_label_values(&[model, stage.label()])
        .observe(elapsed.as_secs_f64());
}

/// Buckets for the stall histograms, in seconds: 0.1 ms to 10 s.
fn stall_buckets() -> Vec<f64> {
    vec![
        0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0,
    ]
}

static TASK_POLL_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_task_poll_seconds",
            "Task polls that held an async worker thread at least DYN_EPP_TASK_METRICS_MIN_US",
        )
        .buckets(stall_buckets()),
        &["task"],
    )
    .expect("task_poll histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("task_poll_seconds is the only registrant of its name");
    histogram
});

static TASK_SCHEDULE_DELAY_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_task_schedule_delay_seconds",
            "Waits of at least DYN_EPP_TASK_METRICS_MIN_US between a task being \
             woken and being polled",
        )
        .buckets(stall_buckets()),
        &["task"],
    )
    .expect("task_schedule_delay histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("task_schedule_delay_seconds is the only registrant of its name");
    histogram
});

static EVENT_LOOP_DELAY_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    let histogram = Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_epp_event_loop_delay_seconds",
            "How late a 10 ms sleep on the async runtime wakes",
        )
        .buckets(stall_buckets()),
    )
    .expect("event_loop_delay histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("event_loop_delay_seconds is the only registrant of its name");
    histogram
});

// The stall families describe the process's runtime, not a request, so they
// carry no `model` label. Their first observations also come before discovery
// binds the served model.

pub(crate) fn observe_task_poll(task: TaskKind, elapsed: Duration) {
    TASK_POLL_SECONDS
        .with_label_values(&[task.label()])
        .observe(elapsed.as_secs_f64());
}

pub(crate) fn observe_task_schedule_delay(task: TaskKind, delay: Duration) {
    TASK_SCHEDULE_DELAY_SECONDS
        .with_label_values(&[task.label()])
        .observe(delay.as_secs_f64());
}

pub(crate) fn observe_event_loop_delay(delay: Duration) {
    EVENT_LOOP_DELAY_SECONDS.observe(delay.as_secs_f64());
}

#[cfg(test)]
pub(crate) fn task_polls_recorded(task: TaskKind) -> u64 {
    TASK_POLL_SECONDS
        .with_label_values(&[task.label()])
        .get_sample_count()
}

#[cfg(test)]
pub(crate) fn event_loop_delays_recorded() -> u64 {
    EVENT_LOOP_DELAY_SECONDS.get_sample_count()
}

/// Register every family at startup rather than on its first observation. A
/// labeled family shows in a scrape once it has a series.
fn register_all() {
    LazyLock::force(&CACHED_TOKENS);
    LazyLock::force(&STAGE_SECONDS);
    LazyLock::force(&TASK_POLL_SECONDS);
    LazyLock::force(&TASK_SCHEDULE_DELAY_SECONDS);
    LazyLock::force(&EVENT_LOOP_DELAY_SECONDS);
}

/// Serve `/metrics` until the process exits.
pub async fn serve(port: u16) -> anyhow::Result<()> {
    register_all();
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

    #[tokio::test]
    async fn standalone_router_series_use_the_private_scrape_registry() {
        let router = register_router_metrics("router-test-model", "router-test-pool").unwrap();
        router.requests_started_total.inc();
        router
            .input_sequence_tokens
            .with_label_values(&["aggregated", "router-test-model"])
            .observe(19.0);
        observe_cached_tokens(0);

        let response = render().await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let started = body
            .lines()
            .find(|line| line.starts_with("dynamo_component_router_requests_started_total{"))
            .unwrap();
        assert!(started.contains("model=\"router-test-model\""));
        assert!(started.contains("inference_pool=\"router-test-pool\""));
        assert!(started.ends_with("} 1"));
        assert!(body.contains("dynamo_component_router_input_sequence_tokens_sum{"));
        assert!(body.contains("dynamo_component_router_output_sequence_tokens_sum{"));
        assert!(body.contains("# TYPE dynamo_epp_cached_tokens histogram"));
        assert_eq!(
            REGISTRY
                .gather()
                .iter()
                .filter(|family| family.name().starts_with("dynamo_component_router_"))
                .count(),
            3
        );
    }

    // The test below records against a label of its own: the server's tests
    // drive whole streams in this process, which record stages against the
    // served model concurrently.

    #[test]
    fn stages_record_on_a_fixed_label_set() {
        const MODEL: &str = "stage-test";
        for stage in Stage::ALL {
            observe_stage_for(MODEL, stage, Duration::from_millis(3));
        }

        let stages: std::collections::HashSet<String> = REGISTRY
            .gather()
            .iter()
            .find(|family| family.name() == "dynamo_epp_stage_duration_seconds")
            .expect("histogram is registered")
            .get_metric()
            .iter()
            .flat_map(|metric| metric.get_label())
            .filter(|label| label.name() == "stage")
            .map(|label| label.value().to_string())
            .collect();
        assert_eq!(
            stages.len(),
            Stage::ALL.len(),
            "one series per stage: {stages:?}"
        );
    }

    #[tokio::test]
    async fn scrape_lists_the_stage_and_stall_families() {
        register_all();
        // A labeled family appears in a scrape only once it has a series, so
        // record a stage here rather than rely on another test.
        const MODEL: &str = "scrape-test";
        observe_stage_for(MODEL, Stage::Pick, Duration::from_millis(1));
        let response = render().await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body = String::from_utf8(body.to_vec()).expect("utf8");

        assert!(
            body.contains("# TYPE dynamo_epp_stage_duration_seconds histogram"),
            "{body}"
        );
        assert!(
            body.contains("# TYPE dynamo_epp_event_loop_delay_seconds histogram"),
            "{body}"
        );
    }
}
