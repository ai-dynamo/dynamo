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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use axum::{
    Router as AxumRouter, http::StatusCode, http::header::CONTENT_TYPE, response::IntoResponse,
    routing::get,
};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use prometheus::core::Collector;
use prometheus::{
    Counter, Encoder, HistogramOpts, HistogramVec, IntGauge, IntGaugeVec, Opts, Registry,
    TEXT_FORMAT, TextEncoder,
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
    /// Waiting for a blocking-pool thread to start the chat template.
    BlockingWait,
    /// Rendering the chat template.
    Template,
    /// Tokenizing the rendered prompt, excluding any wait for a thread.
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
    /// and answering with its headers.
    ResponseHeaders,
    /// Routing decision ready until the first non-empty response body
    /// chunk: the gateway, sidecar and worker producing the first token.
    FirstResponse,
    /// Request headers received until the first non-empty response body
    /// chunk: the time to first token as the EPP sees it. Client time to
    /// first token minus this is the time spent outside the EPP's view,
    /// without comparing clocks across nodes.
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
        .buckets(generate_log_buckets(0.000_05, 60.0, 24)),
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

/// Which request body [`observe_body_bytes`] measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    /// The body the client sent, as the gateway streamed it to the EPP.
    Received,
    /// The body the EPP hands back to the gateway for the worker, after any
    /// rewrite.
    Forwarded,
}

impl Body {
    fn label(self) -> &'static str {
        match self {
            Body::Received => "received",
            Body::Forwarded => "forwarded",
        }
    }
}

static BODY_BYTES: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_request_body_bytes",
            "Size of each routed request body, as received and as forwarded",
        )
        .buckets(generate_log_buckets(1024.0, 64.0 * 1024.0 * 1024.0, 17)),
        &["model", "body"],
    )
    .expect("body histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("request_body_bytes is the only registrant of its name");
    histogram
});

/// Record the size of one request body.
pub fn observe_body_bytes(body: Body, bytes: usize) {
    observe_body_bytes_for(served_model_label(), body, bytes);
}

fn observe_body_bytes_for(model: &str, body: Body, bytes: usize) {
    BODY_BYTES
        .with_label_values(&[model, body.label()])
        .observe(bytes as f64);
}

static PROMPT_TOKENS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_prompt_tokens",
            "Prompt tokens the EPP routed each request on",
        )
        .buckets(generate_log_buckets(16.0, 1_048_576.0, 17)),
        &["model"],
    )
    .expect("prompt_tokens histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("prompt_tokens is the only registrant of its name");
    histogram
});

/// Record the prompt length one request was routed on.
pub fn observe_prompt_tokens(tokens: usize) {
    observe_prompt_tokens_for(served_model_label(), tokens);
}

fn observe_prompt_tokens_for(model: &str, tokens: usize) {
    PROMPT_TOKENS
        .with_label_values(&[model])
        .observe(tokens as f64);
}

static PICKS_IN_FLIGHT: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let gauge = IntGaugeVec::new(
        Opts::new(
            "dynamo_epp_picks_in_flight",
            "Requests currently inside pick(): parsed, tokenized or waiting for a worker",
        ),
        &["model"],
    )
    .expect("picks_in_flight options are statically valid");
    REGISTRY
        .register(Box::new(gauge.clone()))
        .expect("picks_in_flight is the only registrant of its name");
    gauge
});

/// Counts one request in `dynamo_epp_picks_in_flight` until dropped, so a
/// cancelled or failed pick still leaves the gauge.
pub struct PickInFlight(IntGauge);

impl Drop for PickInFlight {
    fn drop(&mut self) {
        self.0.dec();
    }
}

/// Count one request as inside `pick()` for as long as the guard lives.
pub fn pick_in_flight() -> PickInFlight {
    pick_in_flight_for(served_model_label())
}

fn pick_in_flight_for(model: &str) -> PickInFlight {
    let gauge = PICKS_IN_FLIGHT.with_label_values(&[model]);
    gauge.inc();
    PickInFlight(gauge)
}

/// Tokio runtime gauges, sampled at scrape time. A global queue that stays
/// above zero means every worker thread is busy and ready tasks (other
/// requests' body and token chunks) are waiting for one. A task waiting in a
/// busy worker's LIFO slot does not show here; the task metrics in
/// [`crate::tokio_runtime`] catch that.
struct RuntimeGauges {
    workers: IntGauge,
    alive_tasks: IntGauge,
    global_queue_depth: IntGauge,
    blocking_threads: IntGauge,
    blocking_queue_depth: IntGauge,
    busy_seconds: Counter,
}

static RUNTIME: LazyLock<RuntimeGauges> = LazyLock::new(|| {
    let gauge = |name: &str, help: &str| {
        let gauge = IntGauge::new(name, help).expect("runtime gauge options are statically valid");
        REGISTRY
            .register(Box::new(gauge.clone()))
            .expect("runtime gauges are the only registrants of their names");
        gauge
    };
    let busy_seconds = Counter::new(
        "dynamo_epp_runtime_busy_seconds_total",
        "Time the async runtime's worker threads spent running tasks",
    )
    .expect("runtime counter options are statically valid");
    REGISTRY
        .register(Box::new(busy_seconds.clone()))
        .expect("runtime_busy_seconds_total is the only registrant of its name");
    RuntimeGauges {
        workers: gauge(
            "dynamo_epp_runtime_workers",
            "Worker threads of the EPP's async runtime",
        ),
        alive_tasks: gauge(
            "dynamo_epp_runtime_alive_tasks",
            "Tasks alive in the EPP's async runtime",
        ),
        global_queue_depth: gauge(
            "dynamo_epp_runtime_global_queue_depth",
            "Ready tasks waiting in the async runtime's global queue",
        ),
        blocking_threads: gauge(
            "dynamo_epp_runtime_blocking_threads",
            "Threads in the async runtime's blocking pool, busy or idle",
        ),
        blocking_queue_depth: gauge(
            "dynamo_epp_runtime_blocking_queue_depth",
            "Blocking-pool tasks, such as chat template renders, waiting for a thread",
        ),
        busy_seconds,
    }
});

/// Worker busy time already added to `dynamo_epp_runtime_busy_seconds_total`.
static BUSY_COUNTED: Mutex<Duration> = Mutex::new(Duration::ZERO);

fn sample_runtime() {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let metrics = handle.metrics();
    let to_i64 = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    RUNTIME.workers.set(to_i64(metrics.num_workers()));
    RUNTIME.alive_tasks.set(to_i64(metrics.num_alive_tasks()));
    RUNTIME
        .global_queue_depth
        .set(to_i64(metrics.global_queue_depth()));
    RUNTIME
        .blocking_threads
        .set(to_i64(metrics.num_blocking_threads()));
    RUNTIME
        .blocking_queue_depth
        .set(to_i64(metrics.blocking_queue_depth()));

    let busy: Duration = (0..metrics.num_workers())
        .map(|worker| metrics.worker_total_busy_duration(worker))
        .sum();
    let mut counted = BUSY_COUNTED.lock().unwrap_or_else(PoisonError::into_inner);
    if busy > *counted {
        RUNTIME.busy_seconds.inc_by((busy - *counted).as_secs_f64());
        *counted = busy;
    }
}

static EXT_PROC_CONNECTIONS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let gauge = IntGaugeVec::new(
        Opts::new(
            "dynamo_epp_ext_proc_connections",
            "Gateway connections to the ext_proc TLS port",
        ),
        &["model"],
    )
    .expect("ext_proc_connections options are statically valid");
    REGISTRY
        .register(Box::new(gauge.clone()))
        .expect("ext_proc_connections is the only registrant of its name");
    gauge
});

static EXT_PROC_STREAMS: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    let gauge = IntGaugeVec::new(
        Opts::new(
            "dynamo_epp_ext_proc_streams",
            "Open ext_proc streams: requests the gateway is still relaying through the EPP",
        ),
        &["model"],
    )
    .expect("ext_proc_streams options are statically valid");
    REGISTRY
        .register(Box::new(gauge.clone()))
        .expect("ext_proc_streams is the only registrant of its name");
    gauge
});

static CONNECTION_STREAMS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_ext_proc_connection_streams",
            "Streams open on a gateway connection when one more starts on it, \
             that one included",
        )
        .buckets(vec![
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 100.0, 128.0, 192.0, 256.0,
        ]),
        &["model"],
    )
    .expect("connection_streams histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("ext_proc_connection_streams is the only registrant of its name");
    histogram
});

static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

/// One gateway connection to the ext_proc TLS port, shared by the task that
/// serves it and every ext_proc stream on it. Counted in
/// `dynamo_epp_ext_proc_connections` until the last of them drops it.
#[derive(Debug)]
pub struct Connection {
    id: u64,
    open_streams: AtomicUsize,
    streams_served: AtomicU64,
    peak_streams: AtomicUsize,
    gauge: IntGauge,
}

impl Connection {
    pub fn open() -> Arc<Self> {
        let gauge = EXT_PROC_CONNECTIONS.with_label_values(&[served_model_label()]);
        gauge.inc();
        Arc::new(Self {
            id: NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed),
            open_streams: AtomicUsize::new(0),
            streams_served: AtomicU64::new(0),
            peak_streams: AtomicUsize::new(0),
            gauge,
        })
    }

    /// Process-unique id, for joining a connection's log lines with its
    /// streams' lines.
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn streams_served(&self) -> u64 {
        self.streams_served.load(Ordering::Relaxed)
    }

    pub fn peak_streams(&self) -> usize {
        self.peak_streams.load(Ordering::Relaxed)
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

/// One ext_proc stream, counted in `dynamo_epp_ext_proc_streams` and on its
/// connection until dropped.
#[derive(Debug)]
pub struct OpenStream {
    connection: Option<Arc<Connection>>,
    connection_streams: Option<usize>,
    gauge: IntGauge,
}

impl OpenStream {
    /// Start counting a stream. `connection` is `None` on the plaintext
    /// listener, which does not track connections.
    pub fn open(connection: Option<Arc<Connection>>) -> Self {
        let model = served_model_label();
        let gauge = EXT_PROC_STREAMS.with_label_values(&[model]);
        gauge.inc();
        let connection_streams = connection.as_ref().map(|connection| {
            let open = connection.open_streams.fetch_add(1, Ordering::Relaxed) + 1;
            connection.streams_served.fetch_add(1, Ordering::Relaxed);
            connection.peak_streams.fetch_max(open, Ordering::Relaxed);
            CONNECTION_STREAMS
                .with_label_values(&[model])
                .observe(open as f64);
            open
        });
        Self {
            connection,
            connection_streams,
            gauge,
        }
    }

    pub fn connection_id(&self) -> Option<u64> {
        self.connection.as_ref().map(|connection| connection.id())
    }

    /// Streams open on this stream's connection when it started, itself
    /// included: how many requests one stall of that connection holds.
    pub fn connection_streams(&self) -> Option<usize> {
        self.connection_streams
    }
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        self.gauge.dec();
        if let Some(connection) = &self.connection {
            connection.open_streams.fetch_sub(1, Ordering::Relaxed);
        }
    }
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
        &["model", "task"],
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
        &["model", "task"],
    )
    .expect("task_schedule_delay histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("task_schedule_delay_seconds is the only registrant of its name");
    histogram
});

static EVENT_LOOP_DELAY_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let histogram = HistogramVec::new(
        HistogramOpts::new(
            "dynamo_epp_event_loop_delay_seconds",
            "How late a 10 ms sleep on the async runtime wakes",
        )
        .buckets(stall_buckets()),
        &["model"],
    )
    .expect("event_loop_delay histogram options are statically valid");
    REGISTRY
        .register(Box::new(histogram.clone()))
        .expect("event_loop_delay_seconds is the only registrant of its name");
    histogram
});

pub(crate) fn observe_task_poll(task: TaskKind, elapsed: Duration) {
    TASK_POLL_SECONDS
        .with_label_values(&[served_model_label(), task.label()])
        .observe(elapsed.as_secs_f64());
}

pub(crate) fn observe_task_schedule_delay(task: TaskKind, delay: Duration) {
    TASK_SCHEDULE_DELAY_SECONDS
        .with_label_values(&[served_model_label(), task.label()])
        .observe(delay.as_secs_f64());
}

pub(crate) fn observe_event_loop_delay(delay: Duration) {
    EVENT_LOOP_DELAY_SECONDS
        .with_label_values(&[served_model_label()])
        .observe(delay.as_secs_f64());
}

#[cfg(test)]
pub(crate) fn task_polls_recorded(task: TaskKind) -> u64 {
    TASK_POLL_SECONDS
        .with_label_values(&[served_model_label(), task.label()])
        .get_sample_count()
}

#[cfg(test)]
pub(crate) fn event_loop_delays_recorded() -> u64 {
    EVENT_LOOP_DELAY_SECONDS
        .with_label_values(&[served_model_label()])
        .get_sample_count()
}

/// The tokenizer's prefix-cache counters. The EPP's tokenizer increments
/// them as the frontend's does; registering them here exports them, so the
/// cache's hit rate can be compared between the two at each load.
static TOKENIZER_CACHE: LazyLock<()> = LazyLock::new(|| {
    use dynamo_runtime::metrics::frontend_perf::{
        TOKENIZER_CACHE_CACHED_TOKENS_TOTAL, TOKENIZER_CACHE_HITS_TOTAL,
        TOKENIZER_CACHE_MISSES_TOTAL, TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL,
    };
    let collectors: [Box<dyn Collector>; 4] = [
        Box::new(TOKENIZER_CACHE_HITS_TOTAL.clone()),
        Box::new(TOKENIZER_CACHE_MISSES_TOTAL.clone()),
        Box::new(TOKENIZER_CACHE_CACHED_TOKENS_TOTAL.clone()),
        Box::new(TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL.clone()),
    ];
    for collector in collectors {
        REGISTRY
            .register(collector)
            .expect("the EPP registers the tokenizer cache counters once");
    }
});

/// Register every family up front, so a scrape shows them before the first
/// request.
fn register_all() {
    LazyLock::force(&CACHED_TOKENS);
    LazyLock::force(&STAGE_SECONDS);
    LazyLock::force(&BODY_BYTES);
    LazyLock::force(&PROMPT_TOKENS);
    LazyLock::force(&PICKS_IN_FLIGHT);
    LazyLock::force(&RUNTIME);
    LazyLock::force(&EXT_PROC_CONNECTIONS);
    LazyLock::force(&EXT_PROC_STREAMS);
    LazyLock::force(&CONNECTION_STREAMS);
    LazyLock::force(&TASK_POLL_SECONDS);
    LazyLock::force(&TASK_SCHEDULE_DELAY_SECONDS);
    LazyLock::force(&EVENT_LOOP_DELAY_SECONDS);
    LazyLock::force(&TOKENIZER_CACHE);
}

/// Aborts the event-loop watcher when the metrics server stops.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Serve `/metrics` until the process exits, and watch the async runtime for
/// stalls (see [`crate::tokio_runtime`]) for as long as it does.
pub async fn serve(port: u16) -> anyhow::Result<()> {
    register_all();
    let _watcher = AbortOnDrop(tokio::spawn(crate::tokio_runtime::watch_event_loop()));
    let app = AxumRouter::new().route("/metrics", get(render));
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, "Serving Prometheus metrics on /metrics");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn render() -> impl IntoResponse {
    sample_runtime();
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

    // The tests below record against labels of their own: the server's tests
    // drive whole streams in this process, which record stages, body sizes and
    // in-flight picks against the served model concurrently.

    #[test]
    fn stages_record_on_a_fixed_label_set() {
        const MODEL: &str = "stage-test";
        for stage in Stage::ALL {
            observe_stage_for(MODEL, stage, Duration::from_millis(3));
            observe_stage_for(MODEL, stage, Duration::from_millis(3));
            let recorded = STAGE_SECONDS.with_label_values(&[MODEL, stage.label()]);
            assert_eq!(recorded.get_sample_count(), 2, "{stage:?}");
            assert!(
                (recorded.get_sample_sum() - 0.006).abs() < 1e-9,
                "{stage:?}"
            );
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

    #[test]
    fn body_sizes_and_prompt_tokens_are_recorded() {
        const MODEL: &str = "body-test";
        observe_body_bytes_for(MODEL, Body::Received, 271_000);
        observe_body_bytes_for(MODEL, Body::Forwarded, 663_000);
        observe_prompt_tokens_for(MODEL, 64_000);

        let sum = |body: &str| {
            BODY_BYTES
                .with_label_values(&[MODEL, body])
                .get_sample_sum()
        };
        assert_eq!(sum("received"), 271_000.0);
        assert_eq!(sum("forwarded"), 663_000.0);
        assert_eq!(
            PROMPT_TOKENS.with_label_values(&[MODEL]).get_sample_sum(),
            64_000.0
        );
    }

    #[test]
    fn pick_in_flight_counts_until_the_guard_drops() {
        const MODEL: &str = "in-flight-test";
        let gauge = PICKS_IN_FLIGHT.with_label_values(&[MODEL]);

        let first = pick_in_flight_for(MODEL);
        let second = pick_in_flight_for(MODEL);
        assert_eq!(gauge.get(), 2);
        drop(first);
        assert_eq!(gauge.get(), 1);
        drop(second);
        assert_eq!(gauge.get(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scrape_reports_the_async_runtime() {
        register_all();
        let response = render().await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body = String::from_utf8(body.to_vec()).expect("utf8");

        assert!(body.contains("dynamo_epp_runtime_workers 2"), "{body}");
        assert!(body.contains("# TYPE dynamo_epp_runtime_global_queue_depth gauge"));
        assert!(body.contains("# TYPE dynamo_epp_runtime_blocking_threads gauge"));
        assert!(body.contains("# TYPE dynamo_epp_runtime_blocking_queue_depth gauge"));
        assert!(body.contains("# TYPE dynamo_epp_runtime_busy_seconds_total counter"));
        assert!(body.contains("# TYPE dynamo_epp_stage_duration_seconds histogram"));
        assert!(body.contains("# TYPE dynamo_epp_picks_in_flight gauge"));
        assert!(body.contains("# TYPE dynamo_frontend_tokenizer_cache_hits_total counter"));
    }

    #[test]
    fn streams_are_counted_on_their_connection_until_dropped() {
        let _guard = bind_test_model();
        let connection = Connection::open();
        let first = OpenStream::open(Some(Arc::clone(&connection)));
        let second = OpenStream::open(Some(Arc::clone(&connection)));
        let plaintext = OpenStream::open(None);

        assert_eq!(first.connection_id(), Some(connection.id()));
        assert_eq!(first.connection_streams(), Some(1));
        assert_eq!(second.connection_streams(), Some(2));
        assert_eq!(plaintext.connection_id(), None);
        assert_eq!(plaintext.connection_streams(), None);

        drop(first);
        let third = OpenStream::open(Some(Arc::clone(&connection)));
        assert_eq!(
            third.connection_streams(),
            Some(2),
            "a closed stream frees its slot"
        );
        assert_eq!(connection.streams_served(), 3);
        assert_eq!(connection.peak_streams(), 2);
    }
}
