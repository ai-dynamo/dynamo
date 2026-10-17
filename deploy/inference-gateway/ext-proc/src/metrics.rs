// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics exposed by the EPP on its own `/metrics` endpoint.
//!
//! The EPP keeps a private registry instead of the runtime's component registry:
//! it holds no `DistributedRuntime` past startup (see
//! [`crate::epp::Router::from_discovery`]) and these series describe gateway
//! traffic rather than a registered Dynamo component.
//!
//! # Metric glossary
//!
//! Label values are closed sets. `model` is the process-wide label bound by
//! [`set_served_model`]; it is never read from a request body.
//!
//! | Metric | Type | Labels | Observed at |
//! |---|---|---|---|
//! | `dynamo_epp_requests_total` | counter | `model`, `outcome` | Once per ext_proc attempt, at its terminal state |
//! | `dynamo_epp_request_duration_seconds` | histogram | `model`, `outcome` | Same instant as `requests_total`, measured from attempt start |
//! | `dynamo_epp_streams_inflight` | gauge | `model` | Incremented when an attempt starts, decremented at its terminal state |
//! | `dynamo_epp_phase_duration_seconds` | histogram | `model`, `phase`, `outcome` | When a phase the attempt entered returns |
//! | `dynamo_epp_first_response_body_seconds` | histogram | `model` | On the first non-empty response body chunk of an attempt |
//! | `dynamo_epp_lifecycle_callbacks_total` | counter | `model`, `operation`, `result` | When a picker lifecycle callback returns |
//! | `dynamo_epp_cached_tokens` | histogram | `model` | Unchanged: when the backend response carries `usage.prompt_tokens_details.cached_tokens` |
//!
//! `outcome` is one of `response_eos`, `upstream_http_error`, `early_reject`,
//! `ext_proc_error`, `incomplete`; `phase` is `render_tokenize` or `selection`;
//! `operation` is `prefill_complete` or `request_complete`; `result` is `ok` or
//! `error`.
//!
//! Every family is a process-local observation of one EPP replica, so a gauge
//! such as `streams_inflight` must not be summed across replicas as if it were a
//! global count.
//!
//! These readings are the ones a dashboard is most likely to get wrong:
//!
//! * `request_duration_seconds` spans the whole attempt, including selection,
//!   renderer wait, and the backend response. It is not generation-only latency.
//! * `phase="selection"` includes queue wait inside the selection service, and
//!   `phase="render_tokenize"` includes the renderer's own latency. Neither can
//!   be split into compute alone at this layer, so neither is named for compute.
//! * `first_response_body_seconds` is not time to first token: the first
//!   non-empty chunk may be SSE metadata, a role chunk, or buffered content.
//! * `lifecycle_callbacks_total` counts invocations, not state changes.
//!   `free_reservation` and `prefill_complete` are idempotent and report success
//!   for an unknown id, so this counter must not drive an active gauge.
//! * `cached_tokens` is what the backend reported, not the KV router's overlap
//!   estimate. A reported zero is a real observation; a response with no usage
//!   information records nothing rather than a fabricated zero. The same rule
//!   applies to a phase that never started and an attempt that saw no body.
//!
//! Not observable here: real TTFT/ITL and per-token progress, KV transfer success
//! in a disaggregated pair, authoritative reservation transitions, and replica
//! connection or sync-lag state.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use axum::{
    Router as AxumRouter, http::StatusCode, http::header::CONTENT_TYPE, response::IntoResponse,
    routing::get,
};
use dynamo_llm::http::service::metrics::generate_log_buckets;
use prometheus::{
    CounterVec, Encoder, GaugeVec, HistogramOpts, HistogramVec, Registry, TEXT_FORMAT, TextEncoder,
};

/// Port the `/metrics` endpoint binds to unless `DYN_EPP_METRICS_PORT` says
/// otherwise. Distinct from the ext_proc gRPC port (9002) and the health port
/// (9003).
pub const DEFAULT_METRICS_PORT: u16 = 9090;

/// Environment variable overriding [`DEFAULT_METRICS_PORT`]. `0` disables the
/// metrics server entirely.
pub const METRICS_PORT_ENV: &str = "DYN_EPP_METRICS_PORT";

/// Label value for every series, bound once at startup by [`set_served_model`].
static SERVED_MODEL: OnceLock<String> = OnceLock::new();

/// Used until [`set_served_model`] runs, and in tests that never bind one.
const UNKNOWN_MODEL: &str = "unknown";

/// The process-wide registry backing the `/metrics` listener.
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

/// Terminal classification of one ext-proc request attempt.
///
/// Exactly one of these is recorded per attempt, by the single
/// [`RequestObservation`] that owns the attempt. The variants form a priority
/// chain applied once at terminal time (see [`RequestObservation::finish`]); no
/// call site races another to write it. The set is closed: request input cannot
/// extend it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The response body or trailers reached end-of-stream and no more specific
    /// terminal error was observed. Says nothing about the correctness of the
    /// generated content.
    ResponseEos,
    /// The upstream model server returned an HTTP error status and no more
    /// specific terminal error was observed.
    UpstreamHttpError,
    /// This layer itself rejected the request (selection failed, flow control
    /// evicted it) without waiting for the backend.
    EarlyReject,
    /// The ext-proc stream failed on a protocol or transport error before a
    /// backend terminal was observed. Does not prove the backend stopped.
    ExtProcError,
    /// The stream ended without a provable backend terminal: client or gateway
    /// disconnect, force shutdown. Deliberately not called a client
    /// cancellation, which this layer cannot distinguish.
    Incomplete,
}

impl Outcome {
    /// Label value for every variant. Stable wire contract.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ResponseEos => "response_eos",
            Self::UpstreamHttpError => "upstream_http_error",
            Self::EarlyReject => "early_reject",
            Self::ExtProcError => "ext_proc_error",
            Self::Incomplete => "incomplete",
        }
    }

    /// Every variant, so tests can assert the label set is exactly this.
    pub const ALL: [Self; 5] = [
        Self::ResponseEos,
        Self::UpstreamHttpError,
        Self::EarlyReject,
        Self::ExtProcError,
        Self::Incomplete,
    ];
}

/// Finite phase names for [`Metrics::phase_duration_seconds`].
///
/// A phase sample exists only for a phase the request actually entered; a
/// request rejected before a phase starts contributes no sample for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The render/tokenize call and its wait. Includes the upstream renderer's
    /// own latency and the connection wait, so it is not pure tokenizer CPU
    /// time.
    RenderTokenize,
    /// The selection call and its wait, including the wait inside the selection
    /// service. Deliberately not named a compute-only metric: the queue wait
    /// cannot be separated from the scoring work at this layer.
    Selection,
}

impl Phase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RenderTokenize => "render_tokenize",
            Self::Selection => "selection",
        }
    }
}

/// Finite results for [`Metrics::phase_duration_seconds`] and
/// [`Metrics::lifecycle_callbacks_total`]. `cancelled` is deliberately absent:
/// the call sites cannot reliably distinguish a cancelled call from a failed
/// one, so cancelled calls are reported as errors rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageResult {
    Ok,
    Error,
}

impl StageResult {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

/// Finite operations for [`Metrics::lifecycle_callbacks_total`]. These count
/// invocations of the picker's lifecycle callbacks, not authoritative
/// reservation state transitions: a callback that returns `Ok` may be an
/// idempotent no-op for an unknown id, so these counters must never be used to
/// derive an active-reservation gauge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleOperation {
    /// `EndpointPicker::on_prefill_complete`.
    PrefillComplete,
    /// `EndpointPicker::on_request_complete`.
    RequestComplete,
}

impl LifecycleOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrefillComplete => "prefill_complete",
            Self::RequestComplete => "request_complete",
        }
    }
}

/// Finite routing stages for the `epp.routing_failed` event.
///
/// The stage names where the request stopped, not which component is at fault:
/// `RenderTokenize` covers a malformed client body and an unreachable renderer
/// alike, which the paired `code` field separates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStage {
    /// Ready-worker catalog not yet populated.
    Catalog,
    /// Selection requested workers from a subset that matched none.
    EndpointSubset,
    /// In-flight pick limit saturated; the request was shed, not queued.
    Capacity,
    /// The render/tokenize call, including the client-body parse.
    RenderTokenize,
    /// Worker selection and its reservation.
    Selection,
    /// The selected worker could not be resolved to a routable address.
    EndpointResolve,
}

impl RoutingStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::EndpointSubset => "endpoint_subset",
            Self::Capacity => "capacity",
            Self::RenderTokenize => "render_tokenize",
            Self::Selection => "selection",
            Self::EndpointResolve => "endpoint_resolve",
        }
    }
}

/// Finite error code for the `epp.routing_failed` event.
///
/// Same closed set as the client-visible [`crate::picker::PickError`] variants,
/// so a log line and a 4xx/5xx status always agree, and no string error message
/// ever becomes a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingErrorCode {
    NoEndpoints,
    RoutingFailed,
    InvalidRequest,
    MetadataHeadersTooLarge,
    TokenizerUnavailable,
    TokenizerTimeout,
    TokenizerUpstreamError,
    Overloaded,
}

impl RoutingErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoEndpoints => "no_endpoints",
            Self::RoutingFailed => "routing_failed",
            Self::InvalidRequest => "invalid_request",
            Self::MetadataHeadersTooLarge => "metadata_headers_too_large",
            Self::TokenizerUnavailable => "tokenizer_unavailable",
            Self::TokenizerTimeout => "tokenizer_timeout",
            Self::TokenizerUpstreamError => "tokenizer_upstream_error",
            Self::Overloaded => "overloaded",
        }
    }
}

/// Emit the `epp.routing_failed` event for a request that stopped before it was
/// dispatched.
///
/// Fields are finite and bounded: stage, error code, and whether a reservation
/// had already been taken when the request failed. No prompt, header, worker
/// address, or error message text is logged.
pub fn record_routing_failed(
    request_id: &str,
    stage: RoutingStage,
    code: RoutingErrorCode,
    reservation_taken: bool,
) {
    tracing::debug!(
        request_id,
        stage = stage.as_str(),
        code = code.as_str(),
        reservation_taken,
        "epp.routing_failed"
    );
}

/// Emit the `epp.lifecycle_callback` event for one picker callback invocation.
pub fn record_lifecycle_callback(
    operation: LifecycleOperation,
    result: StageResult,
    booking_id: &str,
) {
    tracing::debug!(
        booking_id,
        operation = operation.as_str(),
        result = result.as_str(),
        "epp.lifecycle_callback"
    );
}

/// The EPP's metric families plus the registry they are registered in.
///
/// Families are constructed by [`Metrics::with_registry`], so tests get an
/// isolated registry while production shares the process-wide one behind
/// [`Metrics::global`]. The handle is cheap to clone per request; the
/// underlying time series are shared.
#[derive(Debug)]
pub struct Metrics {
    /// Retained so an isolated recorder owns the registry its families live in.
    /// Production uses the process-wide registry through [`Metrics::global`].
    registry: Registry,
    request_count: CounterVec,
    request_duration_seconds: HistogramVec,
    streams_inflight: GaugeVec,
    phase_duration_seconds: HistogramVec,
    first_response_body_seconds: HistogramVec,
    lifecycle_callbacks_total: CounterVec,
}

impl Metrics {
    /// Build and register every family in `registry`.
    ///
    /// Registering the same family twice in one registry panics, so each
    /// registry must be used for exactly one `Metrics`.
    pub fn with_registry(registry: Registry) -> Self {
        let register = |collector: Box<dyn prometheus::core::Collector>| {
            registry
                .register(collector)
                .expect("metric family names are unique within one registry");
        };

        let request_count = CounterVec::new(
            prometheus::Opts::new(
                "dynamo_epp_requests_total",
                "EPP ext_proc request attempts that reached a terminal state, by terminal outcome",
            ),
            &["model", "outcome"],
        )
        .expect("request_count options are statically valid");
        register(Box::new(request_count.clone()));

        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "dynamo_epp_request_duration_seconds",
                "Wall time from the start of an EPP request attempt to its terminal state",
            )
            .buckets(generate_log_buckets(0.001, 64.0, 14)),
            &["model", "outcome"],
        )
        .expect("request_duration options are statically valid");
        register(Box::new(request_duration_seconds.clone()));

        let streams_inflight = GaugeVec::new(
            prometheus::Opts::new(
                "dynamo_epp_streams_inflight",
                "ext_proc streams that entered processing and have not yet terminated",
            ),
            &["model"],
        )
        .expect("streams_inflight options are statically valid");
        register(Box::new(streams_inflight.clone()));

        let phase_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "dynamo_epp_phase_duration_seconds",
                "Wall time of the routing phases an EPP request attempt actually entered",
            )
            .buckets(generate_log_buckets(0.001, 2.0, 12)),
            &["model", "phase", "outcome"],
        )
        .expect("phase_duration options are statically valid");
        register(Box::new(phase_duration_seconds.clone()));

        let first_response_body_seconds = HistogramVec::new(
            HistogramOpts::new(
                "dynamo_epp_first_response_body_seconds",
                "Time from the start of an EPP request attempt to the first non-empty \
                 response body chunk. This is not time to first token: the chunk may be \
                 SSE metadata, a role chunk, or buffered content",
            )
            .buckets(generate_log_buckets(0.001, 1.0, 12)),
            &["model"],
        )
        .expect("first_response_body options are statically valid");
        register(Box::new(first_response_body_seconds.clone()));

        let lifecycle_callbacks_total = CounterVec::new(
            prometheus::Opts::new(
                "dynamo_epp_lifecycle_callbacks_total",
                "Invocations of the picker's reservation lifecycle callbacks and their \
                 return, not authoritative reservation state transitions",
            ),
            &["model", "operation", "result"],
        )
        .expect("lifecycle_callbacks options are statically valid");
        register(Box::new(lifecycle_callbacks_total.clone()));

        Self {
            registry,
            request_count,
            request_duration_seconds,
            streams_inflight,
            phase_duration_seconds,
            first_response_body_seconds,
            lifecycle_callbacks_total,
        }
    }

    /// The process-wide instance backing `/metrics`.
    pub fn global() -> &'static Arc<Self> {
        static GLOBAL: LazyLock<Arc<Metrics>> =
            LazyLock::new(|| Arc::new(Metrics::with_registry(Registry::new())));
        &GLOBAL
    }

    /// Begin observing one request attempt.
    pub fn start_request(self: &Arc<Self>) -> RequestObservation {
        self.streams_inflight
            .with_label_values(&[served_model_label()])
            .inc();
        RequestObservation {
            metrics: self.clone(),
            started: Instant::now(),
            first_body_seen: AtomicBool::new(false),
            finished: false,
        }
    }

    /// Record one lifecycle callback invocation, for call sites that do not
    /// carry a [`RequestObservation`] (the standalone picker observes its own
    /// callbacks; the ext-proc server observes them through the attempt).
    pub fn observe_lifecycle_callback(&self, operation: LifecycleOperation, result: StageResult) {
        self.lifecycle_callbacks_total
            .with_label_values(&[served_model_label(), operation.as_str(), result.as_str()])
            .inc();
    }

    /// Record the wall time of one phase.
    pub fn observe_phase(&self, phase: Phase, result: StageResult, elapsed: Duration) {
        self.phase_duration_seconds
            .with_label_values(&[served_model_label(), phase.as_str(), result.as_str()])
            .observe(elapsed.as_secs_f64());
    }

    /// Registry the families are registered in.
    ///
    /// Exposed so a caller can render an isolated recorder's exposition (tests)
    /// and so the ownership of the registry each family lives in stays visible.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

/// Request-scoped observer for one ext-proc attempt.
///
/// Owns the attempt's terminal accounting: [`Self::finish`] is consuming, so an
/// attempt records exactly one `requests_total` increment, one duration sample,
/// and one `streams_inflight` decrement. If the owning future is dropped before
/// `finish` runs, [`Drop`] performs the same accounting as
/// [`Outcome::Incomplete`], so a disconnect cannot leak the inflight gauge or
/// lose the attempt from the counters.
///
/// Observation methods only update bounded local label sets. None of them touch
/// the network, serialize a log line, or take a lock.
#[derive(Debug)]
pub struct RequestObservation {
    metrics: Arc<Metrics>,
    started: Instant,
    first_body_seen: AtomicBool,
    /// Set by [`Self::finish`] so [`Drop`] does not record a second terminal.
    finished: bool,
}

impl RequestObservation {
    /// Time since this attempt started.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Record the wall time of a phase this attempt actually entered.
    pub fn observe_phase(&self, phase: Phase, result: StageResult, elapsed: Duration) {
        self.metrics.observe_phase(phase, result, elapsed);
    }

    /// Record the first non-empty response body chunk for this attempt.
    ///
    /// Idempotent: only the first call per attempt records a sample, and an
    /// attempt that never sees a body records none. Not reported as time to
    /// first token, which needs a token-boundary signal this layer lacks.
    pub fn observe_first_response_body(&self) {
        if self
            .first_body_seen
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        self.metrics
            .first_response_body_seconds
            .with_label_values(&[served_model_label()])
            .observe(self.started.elapsed().as_secs_f64());
    }

    /// Record one lifecycle callback invocation and its return.
    ///
    /// Counts calls, not state changes: repeated callbacks for the same booking
    /// legitimately increment this, and an `Ok` for an unknown id is not proof
    /// that anything was released.
    pub fn observe_lifecycle_callback(&self, operation: LifecycleOperation, result: StageResult) {
        self.metrics.observe_lifecycle_callback(operation, result);
    }

    /// Close the attempt with `outcome` and release the inflight gauge.
    pub fn finish(mut self, outcome: Outcome) {
        self.finished = true;
        self.record(outcome);
    }

    fn record(&self, outcome: Outcome) {
        let model = served_model_label();
        self.metrics
            .request_count
            .with_label_values(&[model, outcome.as_str()])
            .inc();
        let elapsed = self.started.elapsed();
        self.metrics
            .request_duration_seconds
            .with_label_values(&[model, outcome.as_str()])
            .observe(elapsed.as_secs_f64());
        self.metrics
            .streams_inflight
            .with_label_values(&[model])
            .dec();
        // One terminal event per attempt, carrying only finite fields: the
        // outcome taxonomy and the elapsed time. No prompt, response, header,
        // request id, or booking id is logged here.
        tracing::debug!(
            outcome = outcome.as_str(),
            duration_ms = elapsed.as_secs_f64() * 1000.0,
            "epp.request_finished"
        );
    }
}

impl Drop for RequestObservation {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // An attempt that never reached a terminal state is incomplete, which is
        // exactly the disconnect / force-shutdown case. It deliberately does not
        // claim the client cancelled.
        self.record(Outcome::Incomplete);
    }
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

/// Record the model server's reported cache-hit token count for one completed
/// request, against the model bound by [`set_served_model`].
///
/// Only called when the response actually carried `cached_tokens`: a response
/// with no usage information is unknown, not a zero observation.
pub fn observe_cached_tokens(cached_tokens: u64) {
    CACHED_TOKENS
        .with_label_values(&[served_model_label()])
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
    let (status, body) = encode(&REGISTRY);
    (status, [(CONTENT_TYPE, TEXT_FORMAT)], body)
}

/// Encode `registry` in Prometheus text exposition format.
///
/// Takes the registry so a test can assert on an isolated one instead of the
/// process-wide instance.
fn encode(registry: &Registry) -> (StatusCode, Vec<u8>) {
    let mut buf = Vec::new();
    match TextEncoder::new().encode(&registry.gather(), &mut buf) {
        Ok(()) => (StatusCode::OK, buf),
        Err(err) => {
            tracing::warn!(%err, "Failed to encode Prometheus metrics");
            (StatusCode::INTERNAL_SERVER_ERROR, Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// The model label is process-wide, so tests bind the same value and assert
    /// on isolated registries rather than on absolute process-wide counts. The
    /// mutex only guards tests that still read the shared `CACHED_TOKENS`.
    const TEST_MODEL: &str = "test-model";

    static SERIALIZE: Mutex<()> = Mutex::new(());

    fn bind_test_model() -> MutexGuard<'static, ()> {
        set_served_model(TEST_MODEL);
        SERIALIZE.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// An isolated recorder, so a test never depends on samples another test
    /// left in the process-wide registry.
    fn isolated() -> Arc<Metrics> {
        set_served_model(TEST_MODEL);
        Arc::new(Metrics::with_registry(Registry::new()))
    }

    /// `(count, sum)` currently recorded against [`TEST_MODEL`].
    fn recorded() -> (u64, f64) {
        let histogram = CACHED_TOKENS.with_label_values(&[TEST_MODEL]);
        (histogram.get_sample_count(), histogram.get_sample_sum())
    }

    fn series_count(registry: &Registry, family: &str) -> usize {
        registry
            .gather()
            .iter()
            .find(|f| f.name() == family)
            .map_or(0, |f| f.get_metric().len())
    }

    /// Value of one gathered series: counter and gauge report their value, a
    /// histogram reports its sample count.
    fn series_value(metric: &prometheus::proto::Metric) -> f64 {
        if let Some(counter) = metric.counter.as_ref() {
            counter.value()
        } else if let Some(gauge) = metric.gauge.as_ref() {
            gauge.value()
        } else {
            metric
                .histogram
                .as_ref()
                .map_or(0.0, |h| h.get_sample_count() as f64)
        }
    }

    /// Sum of every series in `family`.
    fn family_total(registry: &Registry, family: &str) -> f64 {
        registry
            .gather()
            .iter()
            .find(|f| f.name() == family)
            .map_or(0.0, |f| f.get_metric().iter().map(series_value).sum())
    }

    /// Value of the series in `family` carrying exactly `labels`, or `None` when
    /// the family or that series is absent. A histogram reports its sample count.
    fn sample_value(registry: &Registry, family: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let families = registry.gather();
        let family = families.iter().find(|f| f.name() == family)?;
        let metric = family.get_metric().iter().find(|m| {
            let carried: Vec<&prometheus::proto::LabelPair> = m
                .get_label()
                .iter()
                .filter(|l| labels.iter().any(|(name, _)| l.name() == *name))
                .collect();
            carried.len() == labels.len()
                && labels.iter().all(|(name, value)| {
                    carried
                        .iter()
                        .any(|l| l.name() == *name && l.value() == *value)
                })
        })?;
        if let Some(counter) = metric.counter.as_ref() {
            Some(counter.value())
        } else if let Some(gauge) = metric.gauge.as_ref() {
            Some(gauge.value())
        } else {
            metric
                .histogram
                .as_ref()
                .map(|h| h.get_sample_count() as f64)
        }
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

    #[test]
    fn finish_records_exactly_one_terminal_sample() {
        let metrics = isolated();
        metrics.start_request().finish(Outcome::ResponseEos);

        let registry = metrics.registry();
        assert_eq!(
            sample_value(
                registry,
                "dynamo_epp_requests_total",
                &[("outcome", "response_eos")]
            ),
            Some(1.0),
            "finish followed by Drop must not record the attempt twice"
        );
        assert_eq!(
            sample_value(registry, "dynamo_epp_streams_inflight", &[]),
            Some(0.0),
            "the inflight gauge must return to zero"
        );
    }

    #[test]
    fn dropping_without_finish_still_closes_the_attempt() {
        let metrics = isolated();
        drop(metrics.start_request());

        let registry = metrics.registry();
        assert_eq!(
            sample_value(
                registry,
                "dynamo_epp_requests_total",
                &[("outcome", "incomplete")]
            ),
            Some(1.0),
            "a dropped attempt is incomplete, not lost"
        );
        assert_eq!(
            sample_value(registry, "dynamo_epp_streams_inflight", &[]),
            Some(0.0)
        );
    }

    #[test]
    fn first_response_body_is_recorded_at_most_once() {
        let metrics = isolated();
        let observation = metrics.start_request();
        observation.observe_first_response_body();
        observation.observe_first_response_body();
        observation.observe_first_response_body();
        observation.finish(Outcome::ResponseEos);

        assert_eq!(
            sample_value(
                metrics.registry(),
                "dynamo_epp_first_response_body_seconds",
                &[]
            ),
            Some(1.0),
            "only the first non-empty body chunk is a sample"
        );
    }

    #[test]
    fn attempts_without_a_body_record_no_first_body_sample() {
        let metrics = isolated();
        metrics.start_request().finish(Outcome::EarlyReject);

        assert_eq!(
            sample_value(
                metrics.registry(),
                "dynamo_epp_first_response_body_seconds",
                &[]
            ),
            None,
            "a rejection that never sees a body must not add a sample"
        );
    }

    #[test]
    fn lifecycle_callback_counts_calls_not_state() {
        let metrics = isolated();
        let observation = metrics.start_request();
        for _ in 0..3 {
            observation
                .observe_lifecycle_callback(LifecycleOperation::RequestComplete, StageResult::Ok);
        }
        observation.finish(Outcome::ResponseEos);

        assert_eq!(
            sample_value(
                metrics.registry(),
                "dynamo_epp_lifecycle_callbacks_total",
                &[("operation", "request_complete"), ("result", "ok")]
            ),
            Some(3.0),
            "repeated idempotent callbacks are three calls, not one state change"
        );
    }

    /// The cardinality budget: 5,000 observations spread over the closed
    /// enumerations produce exactly one series per (label combination actually
    /// observed), and no more. Nothing request-scoped becomes a label, so more
    /// traffic can only add series up to the enumeration product, never past it.
    #[test]
    fn label_cardinality_is_bounded_by_the_enumerations() {
        let metrics = isolated();
        let outcomes = Outcome::ALL;
        let phases = [Phase::RenderTokenize, Phase::Selection];
        let operations = [
            LifecycleOperation::PrefillComplete,
            LifecycleOperation::RequestComplete,
        ];
        let results = [StageResult::Ok, StageResult::Error];

        // Drive every combination once, so every series the label schema allows
        // is created and each is observed a known number of times.
        for (index, outcome) in outcomes.iter().enumerate() {
            let observation = metrics.start_request();
            observation.observe_first_response_body();
            for phase in phases {
                // Results alternate by index so both values reach every phase.
                observation.observe_phase(
                    phase,
                    results[index % results.len()],
                    Duration::from_millis(1),
                );
            }
            for operation in operations {
                observation.observe_lifecycle_callback(operation, results[index % results.len()]);
            }
            observation.finish(*outcome);
        }

        let registry = metrics.registry();
        assert_eq!(
            series_count(registry, "dynamo_epp_requests_total"),
            outcomes.len(),
            "one series per outcome"
        );
        assert_eq!(
            family_total(registry, "dynamo_epp_requests_total"),
            outcomes.len() as f64,
            "one terminal per attempt"
        );
        assert_eq!(series_count(registry, "dynamo_epp_streams_inflight"), 1);
        assert_eq!(
            series_count(registry, "dynamo_epp_phase_duration_seconds"),
            phases.len() * results.len(),
            "one series per (phase, result) pair"
        );
        assert_eq!(
            series_count(registry, "dynamo_epp_lifecycle_callbacks_total"),
            operations.len() * results.len(),
            "one series per (operation, result) pair"
        );
        assert_eq!(
            series_count(registry, "dynamo_epp_first_response_body_seconds"),
            1
        );
    }

    /// The exposition a scrape sees carries HELP/TYPE for every new family.
    #[test]
    fn exposition_declares_every_family() {
        let metrics = isolated();
        let observation = metrics.start_request();
        observation.observe_first_response_body();
        observation.observe_phase(Phase::Selection, StageResult::Ok, Duration::from_millis(2));
        observation
            .observe_lifecycle_callback(LifecycleOperation::PrefillComplete, StageResult::Ok);
        observation.finish(Outcome::ResponseEos);

        let (status, body) = encode(metrics.registry());
        assert_eq!(status, StatusCode::OK);
        let body = String::from_utf8(body).expect("utf8");
        for expected in [
            "# TYPE dynamo_epp_requests_total counter",
            "# TYPE dynamo_epp_request_duration_seconds histogram",
            "# TYPE dynamo_epp_streams_inflight gauge",
            "# TYPE dynamo_epp_phase_duration_seconds histogram",
            "# TYPE dynamo_epp_first_response_body_seconds histogram",
            "# TYPE dynamo_epp_lifecycle_callbacks_total counter",
        ] {
            assert!(body.contains(expected), "missing {expected} in:\n{body}");
        }
    }
}
