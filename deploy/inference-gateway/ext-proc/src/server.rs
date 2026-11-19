// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy `ExternalProcessor.Process` bidirectional streaming implementation.
//!
//! Handles the ext-proc protocol and delegates endpoint selection to an
//! `EndpointPicker` implementation.
//!
//! The state machine enforces ordered responses:
//! `RequestHeaders → RequestBody → RequestTrailers → ResponseHeaders → ResponseBody → ResponseTrailers`

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

use crate::envoy_helpers::{self, metadata};
use crate::log_budget::LogBudget;
use crate::metrics::{self, Stage};
use crate::picker::{
    CacheSaltForwarding, Endpoint, EndpointPicker, PickError, PickTimings, RequestInfo,
    ResponseUsage,
};
use crate::proto::envoy::service::ext_proc::v3::{
    self as ext_proc, ProcessingRequest, ProcessingResponse,
    external_processor_server::{ExternalProcessor, ExternalProcessorServer},
    processing_request,
};
use crate::proto::envoy::r#type::v3::StatusCode;
use crate::tokio_runtime::{self, TaskKind};
use dynamo_kv_router::zmq_wire::DYNAMO_CACHE_SALT_PREFIX;
use serde_json::value::RawValue;

/// State machine phases for the ext_proc stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    RequestReceived,
    HeaderRequestResponseComplete,
    BodyRequestResponsesComplete,
    TrailerRequestResponsesComplete,
    ResponseReceived,
    HeaderResponseResponseComplete,
    BodyResponseResponsesComplete,
    RequestEvicted,
}

/// Per-request context carried across the lifetime of one HTTP stream.
struct RequestContext {
    state: StreamState,
    target_endpoint: String,
    incoming_model_name: String,
    target_model_name: String,
    request_id: String,
    /// Booking id the picker returned from `pick()` for this stream's load
    /// reservation (`None` if it booked nothing). Handed back to the lifecycle
    /// callbacks so the picker frees the exact reservation — no shared map.
    booking_id: Option<String>,
    request_size: usize,
    response_size: usize,
    response_complete: bool,
    model_server_streaming: bool,
    body_routed: bool,
    prefill_complete_signaled: bool,

    /// Set once we've validated the gateway's `ProtocolConfiguration`.
    /// `protocol_config` may appear on every `ProcessingRequest`; we only
    /// check it once per stream to keep the hot path cheap.
    protocol_validated: bool,

    request_headers: Vec<(String, String)>,
    request_metadata: HashMap<String, prost_types::Struct>,
    response_headers: HashMap<String, String>,

    req_header_resp: Option<ProcessingResponse>,
    req_body_resp: Vec<ProcessingResponse>,
    req_trailer_resp: Option<ProcessingResponse>,

    resp_header_resp: Option<ProcessingResponse>,
    resp_body_resp: Vec<ProcessingResponse>,
    resp_trailer_resp: Option<ProcessingResponse>,

    /// Parsed response `usage`, passed to the picker on completion.
    parsed_usage: Option<ResponseUsage>,

    /// Incomplete trailing SSE bytes awaiting the next chunk / EOS.
    sse_usage_buf: Vec<u8>,

    /// When the request headers arrived, for [`Stage::ReceiveBody`].
    headers_received_at: Option<Instant>,
    /// When the routing decision was ready, for [`Stage::FirstResponse`].
    routed_at: Option<Instant>,

    /// The gateway connection this stream arrived on, and how many streams
    /// it carried at the time, for the routing and summary log lines.
    connection_id: Option<u64>,
    connection_streams: Option<usize>,
    /// What happened on this stream, for the stream summary log line.
    trace: StreamTrace,
}

impl RequestContext {
    fn new() -> Self {
        Self {
            state: StreamState::RequestReceived,
            target_endpoint: String::new(),
            incoming_model_name: String::new(),
            target_model_name: String::new(),
            request_id: String::new(),
            booking_id: None,
            request_size: 0,
            response_size: 0,
            response_complete: false,
            model_server_streaming: false,
            body_routed: false,
            prefill_complete_signaled: false,
            protocol_validated: false,
            request_headers: Vec::new(),
            request_metadata: HashMap::new(),
            response_headers: HashMap::new(),
            req_header_resp: None,
            req_body_resp: Vec::new(),
            req_trailer_resp: None,
            resp_header_resp: None,
            resp_body_resp: Vec::new(),
            resp_trailer_resp: None,
            parsed_usage: None,
            sse_usage_buf: Vec::new(),
            headers_received_at: None,
            routed_at: None,
            connection_id: None,
            connection_streams: None,
            trace: StreamTrace::default(),
        }
    }

    /// Time since the request headers arrived: the offsets in the stream
    /// summary.
    fn since_headers(&self) -> Option<Duration> {
        self.headers_received_at.map(|received| received.elapsed())
    }

    /// Whether the response status is 2xx, or unknown because the gateway
    /// sent no `:status`.
    fn response_succeeded(&self) -> bool {
        self.response_headers
            .get(":status")
            .is_none_or(|status| status.starts_with('2'))
    }

    /// Advance the state machine and collect responses that are ready to send.
    fn drain_pending_responses(&mut self) -> Vec<ProcessingResponse> {
        let mut out = Vec::new();

        if self.state == StreamState::RequestEvicted {
            out.push(envoy_helpers::build_eviction_response());
            return out;
        }

        if self.state == StreamState::RequestReceived
            && let Some(resp) = self.req_header_resp.take()
        {
            if let Some(crate::proto::envoy::service::ext_proc::v3::processing_response::Response::RequestHeaders(ref hr)) = resp.response
                && let Some(ref common) = hr.response
                && let Some(ref hm) = common.header_mutation
            {
                tracing::debug!(
                    set_headers_count = hm.set_headers.len(),
                    clear_route_cache = common.clear_route_cache,
                    has_dynamic_metadata = resp.dynamic_metadata.is_some(),
                    "[WIRE] Sending RequestHeaders response to Envoy"
                );
                for h in &hm.set_headers {
                    if let Some(ref hv) = h.header {
                        tracing::debug!(
                            key = %hv.key,
                            value = %String::from_utf8_lossy(&hv.raw_value),
                            "[WIRE] set_header"
                        );
                    }
                }
            }
            out.push(resp);
            self.state = StreamState::HeaderRequestResponseComplete;
        }

        if self.state == StreamState::HeaderRequestResponseComplete
            && !self.req_body_resp.is_empty()
        {
            tracing::debug!(
                count = self.req_body_resp.len(),
                "[WIRE] Sending req_body_resp to Envoy"
            );
            out.append(&mut self.req_body_resp);
            self.state = StreamState::BodyRequestResponsesComplete;
        }

        if self.state == StreamState::BodyRequestResponsesComplete
            && let Some(resp) = self.req_trailer_resp.take()
        {
            out.push(resp);
            self.state = StreamState::TrailerRequestResponsesComplete;
        }

        if self.state == StreamState::ResponseReceived
            && let Some(resp) = self.resp_header_resp.take()
        {
            out.push(resp);
            self.state = StreamState::HeaderResponseResponseComplete;
        }

        if self.state == StreamState::HeaderResponseResponseComplete {
            out.append(&mut self.resp_body_resp);
            if self.response_complete {
                self.state = StreamState::BodyResponseResponsesComplete;
            }
        }

        if self.state == StreamState::BodyResponseResponsesComplete
            && let Some(resp) = self.resp_trailer_resp.take()
        {
            out.push(resp);
        }

        out
    }
}

/// How each stream ended, in the stream summary log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// The response finished.
    Complete,
    /// The gateway ended the stream before the response finished.
    ClosedEarly,
    /// The gateway stopped reading the EPP's responses, so the request was
    /// cancelled.
    Cancelled,
    /// No worker was picked; the EPP answered the request itself.
    PickError,
    /// A receive error or an unsupported protocol configuration.
    StreamError,
    /// The EPP evicted the request.
    Evicted,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Outcome::Complete => "complete",
            Outcome::ClosedEarly => "closed_early",
            Outcome::Cancelled => "cancelled",
            Outcome::PickError => "pick_error",
            Outcome::StreamError => "stream_error",
            Outcome::Evicted => "evicted",
        }
    }
}

/// A response chunk gap at least this long is counted in the stream summary.
const NOTABLE_CHUNK_GAP: Duration = Duration::from_millis(40);
/// A stream with a response chunk gap, or a routed-to-response-headers time,
/// at least this long is slow.
const SLOW_GAP: Duration = Duration::from_millis(100);
/// A stream that took at least this long to receive its body, or to pick a
/// worker, is slow.
const SLOW_PHASE: Duration = Duration::from_secs(1);

/// What happened on one ext_proc stream, on the EPP's clock, for the stream
/// summary log line. Offsets are from the arrival of the request headers.
#[derive(Default)]
struct StreamTrace {
    outcome: Option<Outcome>,
    /// Request body messages received.
    body_msgs: u32,
    /// The end of the request body.
    body_eos: Option<Duration>,
    /// The routing decision and the forwarded body ready.
    routed: Option<Duration>,
    /// The response headers' arrival.
    response_headers: Option<Duration>,
    /// Whether the response streams. A unary response's headers wait for
    /// the whole generation.
    streamed: bool,
    /// The first non-empty response chunk's arrival.
    first_chunk: Option<Duration>,
    rewrite: Option<Duration>,
    body_bytes_out: usize,
    timings: Option<PickTimings>,
    /// Response body messages received, and the gaps between them.
    chunks: u32,
    last_chunk_at: Option<Instant>,
    max_chunk_gap: Duration,
    notable_chunk_gaps: u32,
}

impl StreamTrace {
    fn record_chunk(&mut self, now: Instant) {
        self.chunks += 1;
        if let Some(last) = self.last_chunk_at.replace(now) {
            let gap = now.saturating_duration_since(last);
            self.max_chunk_gap = self.max_chunk_gap.max(gap);
            if gap >= NOTABLE_CHUNK_GAP {
                self.notable_chunk_gaps += 1;
            }
        }
    }

    /// Whether the stream stalled somewhere the EPP can see: a response chunk
    /// held back, a slow trip from the routing decision to a streamed
    /// response's headers, or a slow body or pick. `end` is the stream's end,
    /// which bounds a pick that never finished, because the request was
    /// cancelled or failed while it waited.
    fn is_slow(&self, end: Option<Duration>) -> bool {
        let elapsed = |from: Option<Duration>, to: Option<Duration>| {
            from.zip(to).map(|(from, to)| to.saturating_sub(from))
        };
        let pick = elapsed(self.body_eos, self.routed.or(end));
        self.max_chunk_gap >= SLOW_GAP
            || (self.streamed
                && elapsed(self.routed, self.response_headers).is_some_and(|d| d >= SLOW_GAP))
            || self.body_eos.is_some_and(|d| d >= SLOW_PHASE)
            || pick.is_some_and(|d| d >= SLOW_PHASE)
    }
}

static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

/// One gateway connection to the ext_proc TLS port, shared by the task that
/// serves it and every ext_proc stream on it. Its id and stream counts go into
/// the connection's log lines and its streams' routing and summary lines.
#[derive(Debug)]
pub(crate) struct Connection {
    id: u64,
    open_streams: AtomicUsize,
    streams_served: AtomicU64,
    peak_streams: AtomicUsize,
}

impl Connection {
    pub(crate) fn open() -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed),
            open_streams: AtomicUsize::new(0),
            streams_served: AtomicU64::new(0),
            peak_streams: AtomicUsize::new(0),
        })
    }

    /// Process-unique id, for joining a connection's log lines with its
    /// streams' lines.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn streams_served(&self) -> u64 {
        self.streams_served.load(Ordering::Relaxed)
    }

    pub(crate) fn peak_streams(&self) -> usize {
        self.peak_streams.load(Ordering::Relaxed)
    }
}

/// One ext_proc stream, counted on its gateway connection until dropped.
struct OpenStream {
    connection: Option<Arc<Connection>>,
    connection_streams: Option<usize>,
}

impl OpenStream {
    /// Start counting a stream. `connection` is `None` on the plaintext
    /// listener, which does not track connections.
    fn open(connection: Option<Arc<Connection>>) -> Self {
        let connection_streams = connection.as_ref().map(|connection| {
            let open = connection.open_streams.fetch_add(1, Ordering::Relaxed) + 1;
            connection.streams_served.fetch_add(1, Ordering::Relaxed);
            connection.peak_streams.fetch_max(open, Ordering::Relaxed);
            open
        });
        Self {
            connection,
            connection_streams,
        }
    }

    fn connection_id(&self) -> Option<u64> {
        self.connection.as_ref().map(|connection| connection.id())
    }

    /// Streams open on this stream's connection when it started, itself
    /// included: how many requests one stall of that connection holds.
    fn connection_streams(&self) -> Option<usize> {
        self.connection_streams
    }
}

impl Drop for OpenStream {
    fn drop(&mut self) {
        if let Some(connection) = &self.connection {
            connection.open_streams.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// Environment variable choosing which streams get an INFO summary line:
/// `slow` (default) for slow streams only, at most 20 lines per second;
/// `all` for every stream, as a benchmark wants; `off` for none. Other
/// streams' summaries are logged at DEBUG.
pub const STREAM_SUMMARY_ENV: &str = "DYN_EPP_STREAM_SUMMARY";

#[derive(Clone, Copy, PartialEq, Eq)]
enum SummaryMode {
    All,
    Slow,
    Off,
}

static SUMMARY_MODE: LazyLock<SummaryMode> = LazyLock::new(|| {
    match std::env::var(STREAM_SUMMARY_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "all" => SummaryMode::All,
        "off" => SummaryMode::Off,
        _ => SummaryMode::Slow,
    }
});

/// Slow-stream summary lines per second, at most.
static SLOW_SUMMARY_BUDGET: LogBudget = LogBudget::new(20);

/// Log what happened on one stream, after its last response is queued.
fn log_stream_summary(ctx: &RequestContext) {
    let trace = &ctx.trace;
    let outcome = trace.outcome.unwrap_or(if ctx.response_complete {
        Outcome::Complete
    } else {
        Outcome::ClosedEarly
    });
    let end = ctx.since_headers();
    let slow = trace.is_slow(end);
    macro_rules! summary {
        ($level:ident, $suppressed:expr) => {
            tracing::$level!(
                request_id = %ctx.request_id,
                conn_id = ctx.connection_id,
                conn_streams = ctx.connection_streams,
                outcome = outcome.label(),
                slow,
                prompt_tokens = trace.timings.map(|t| t.prompt_tokens),
                body_bytes_in = ctx.request_size,
                body_bytes_out = trace.body_bytes_out,
                body_msgs_in = trace.body_msgs,
                body_eos_us = trace.body_eos.map(micros),
                routed_us = trace.routed.map(micros),
                resp_headers_us = trace.response_headers.map(micros),
                streamed = trace.streamed,
                first_chunk_us = trace.first_chunk.map(micros),
                end_us = end.map(micros),
                rewrite_us = trace.rewrite.map(micros),
                chunks = trace.chunks,
                max_chunk_gap_us = micros(trace.max_chunk_gap),
                chunk_gaps_over_40ms = trace.notable_chunk_gaps,
                suppressed = $suppressed,
                "ext_proc stream summary"
            )
        };
    }
    match *SUMMARY_MODE {
        SummaryMode::All => summary!(info, 0u64),
        SummaryMode::Slow if slow => {
            if let Some(suppressed) = SLOW_SUMMARY_BUDGET.take() {
                summary!(info, suppressed);
            }
        }
        _ => summary!(debug, 0u64),
    }
}

/// The ext_proc gRPC server.
///
/// Takes an `EndpointPicker` for endpoint selection, decoupling the ext-proc
/// protocol handling from the routing decision.
///
/// Endpoints are resolved internally by the picker (the `Router` uses a K8s
/// pod reflector), so pickers always receive an empty endpoint slice.
pub struct ExtProcServer<P: EndpointPicker> {
    picker: Arc<P>,
}

impl<P: EndpointPicker> ExtProcServer<P> {
    pub fn new(picker: Arc<P>) -> Self {
        Self { picker }
    }

    /// Create a `tonic` service ready for registration on a gRPC server.
    pub fn into_service(self) -> ExternalProcessorServer<Self> {
        ExternalProcessorServer::new(self)
    }

    /// Handle request headers phase.
    fn handle_request_headers(ctx: &mut RequestContext, hdr: &ext_proc::HttpHeaders) {
        ctx.headers_received_at = Some(Instant::now());
        // Collect headers and resolve the request ID for every request,
        // including header-only (end_of_stream) requests such as GET /v1/models.
        // The body-less case is routed later via `handle_header_only_request`,
        // but it still relies on `ctx.request_headers` / `ctx.request_id` being
        // populated here — both for the `RequestInfo` contract passed to the
        // picker and for the stream-end bookkeeping keyed on the request ID.
        if let Some(header_map) = &hdr.headers {
            ctx.request_headers = envoy_helpers::collect_headers(header_map);
            // Client-owned routing metadata must not influence selection. A
            // trusted replacement can only come back through `PickResult`.
            ctx.request_headers
                .retain(|(key, _)| !envoy_helpers::is_prefiller_host_port_header(key));

            if let Some(id) =
                envoy_helpers::extract_header_value(header_map, metadata::REQUEST_ID_HEADER_KEY)
                && !id.is_empty()
            {
                ctx.request_id = id;
            }
        }

        if ctx.request_id.is_empty() {
            ctx.request_id = uuid::Uuid::new_v4().to_string();
            ctx.request_headers.push((
                metadata::REQUEST_ID_HEADER_KEY.to_string(),
                ctx.request_id.clone(),
            ));
        }
    }

    /// Handle a header-only request (EndOfStream on headers, no body).
    async fn handle_header_only_request(
        picker: &P,
        ctx: &mut RequestContext,
        endpoints: &[Endpoint],
    ) -> Result<(), ExtProcError> {
        let req_info = RequestInfo {
            request_id: ctx.request_id.clone(),
            headers: ctx.request_headers.clone(),
            body: Bytes::new(),
            model: String::new(),
            candidate_subset: vec![],
        };

        let result = picker
            .pick(&req_info, endpoints)
            .await
            .map_err(ExtProcError::from_pick_error)?;

        ctx.booking_id = result.reservation_id.clone();
        ctx.target_endpoint = result.endpoint.clone();
        ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
            &result.endpoint,
            None,
            &result.headers,
            result.selected_prefill_endpoint.as_deref(),
        ));
        Ok(())
    }

    /// Handle request body phase: extract model, call picker.
    async fn handle_request_body(
        picker: &P,
        ctx: &mut RequestContext,
        raw_body: Bytes,
        endpoints: &[Endpoint],
    ) -> Result<(), ExtProcError> {
        ctx.request_size = raw_body.len();

        let probe_started = Instant::now();
        let model = extract_model_from_body(&raw_body);
        let model_probe = probe_started.elapsed();
        metrics::observe_stage(Stage::ModelProbe, model_probe);
        let candidate_subset = extract_candidate_subset(&ctx.request_metadata);

        let req_info = RequestInfo {
            request_id: ctx.request_id.clone(),
            headers: ctx.request_headers.clone(),
            // Cheap refcount clone; the picker/renderer share this buffer.
            body: raw_body.clone(),
            model: model.clone(),
            candidate_subset,
        };

        let picked = {
            let started = Instant::now();
            let picked = picker.pick(&req_info, endpoints).await;
            metrics::observe_stage(Stage::Pick, started.elapsed());
            picked
        };
        let result = picked.map_err(ExtProcError::from_pick_error)?;

        ctx.body_routed = true;
        ctx.booking_id = result.reservation_id.clone();
        ctx.target_endpoint = result.endpoint.clone();
        ctx.incoming_model_name = model;
        ctx.target_model_name = ctx.incoming_model_name.clone();
        let timings = result.timings;
        ctx.trace.timings = timings;
        // One line per request, joinable with the client's records by
        // `request_id`: which step of the pick took the time, and how many
        // requests share this request's gateway connection.
        tracing::info!(
            request_id = %ctx.request_id,
            endpoint = %result.endpoint,
            picker_header_count = result.headers.len(),
            conn_id = ctx.connection_id,
            conn_streams = ctx.connection_streams,
            prompt_tokens = timings.map(|t| t.prompt_tokens),
            body_bytes_in = raw_body.len(),
            receive_body_us = ctx.trace.body_eos.map(micros),
            model_probe_us = micros(model_probe),
            parse_us = timings.map(|t| micros(t.parse)),
            blocking_wait_us = timings.and_then(|t| t.blocking_wait).map(micros),
            template_us = timings.and_then(|t| t.template).map(micros),
            tokenize_us = timings.and_then(|t| t.tokenize).map(micros),
            select_us = timings.map(|t| micros(t.select)),
            book_us = timings.map(|t| micros(t.book)),
            "Request routed"
        );
        for (k, v) in &result.headers {
            tracing::debug!(key = %k, value = %v, "[MUTATION] Routing header going into ext_proc set_headers");
        }

        // Only send NEW headers (routing headers from the picker) in the
        // ext_proc header mutation. Do NOT re-send original request headers —
        // they already exist on the request and Envoy rejects mutations that
        // try to set restricted headers (x-envoy-*, x-forwarded-*, pseudo-headers).
        ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
            &result.endpoint,
            Some(ctx.request_size),
            &result.headers,
            result.selected_prefill_endpoint.as_deref(),
        ));

        // Only the rewrite path allocates a new body; otherwise the unchanged
        // body is a cheap `Bytes` clone.
        let cache_salt = match result.cache_salt_forwarding {
            CacheSaltForwarding::RawEngine => result.cache_namespace.as_deref(),
            CacheSaltForwarding::Preserve => None,
        };
        let forwarded_body: Bytes = if result.strip_nvext
            || result.token_ids.is_some()
            || cache_salt.is_some()
        {
            let started = Instant::now();
            // Scales with the prompt, so it runs off the async worker.
            let rewritten = tokio_runtime::off_runtime(|| {
                rewrite_forwarded_body(
                    &raw_body,
                    result.strip_nvext,
                    result.token_ids.as_deref(),
                    cache_salt,
                )
            });
            let rewrite = started.elapsed();
            metrics::observe_stage(Stage::Rewrite, rewrite);
            ctx.trace.rewrite = Some(rewrite);
            match rewritten {
                Ok(modified) => {
                    tracing::trace!(
                        request_id = %ctx.request_id,
                        strip_nvext = result.strip_nvext,
                        has_cache_namespace = result.cache_namespace.is_some(),
                        cache_salt_forwarding = ?result.cache_salt_forwarding,
                        "Forwarded body with routing extensions"
                    );
                    Bytes::from(modified)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to rewrite request body, forwarding original body");
                    raw_body.clone()
                }
            }
        } else {
            raw_body.clone()
        };

        ctx.trace.body_bytes_out = forwarded_body.len();
        ctx.req_body_resp = envoy_helpers::build_request_body_responses(&forwarded_body);
        tracing::debug!(
            has_header_resp = ctx.req_header_resp.is_some(),
            body_resp_count = ctx.req_body_resp.len(),
            "[MUTATION] Responses prepared, waiting for drain"
        );
        ctx.routed_at = Some(Instant::now());
        ctx.trace.routed = ctx.since_headers();

        Ok(())
    }

    /// Handle response headers from the upstream model server.
    fn handle_response_headers(ctx: &mut RequestContext, hdr: &ext_proc::HttpHeaders) {
        if let Some(header_map) = &hdr.headers {
            for h in &header_map.headers {
                let key = h.key.to_ascii_lowercase();
                let value = envoy_helpers::get_header_value(h);
                if key == "content-type" && value.contains("text/event-stream") {
                    ctx.model_server_streaming = true;
                }
                ctx.response_headers.insert(key, value);
            }
        }

        ctx.state = StreamState::ResponseReceived;
        ctx.resp_header_resp = Some(envoy_helpers::build_response_header_response());
    }

    /// Handle response body from the upstream model server.
    fn handle_response_body(ctx: &mut RequestContext, body: &ext_proc::HttpBody) {
        let end_of_stream = body.end_of_stream;
        let chunk = &body.body;
        ctx.response_size += chunk.len();

        if ctx.model_server_streaming {
            // Reassemble SSE across Envoy chunk boundaries before parsing usage.
            ingest_streaming_usage(ctx, chunk, end_of_stream);
            if end_of_stream {
                ctx.response_complete = true;
            }
            let rewritten = envoy_helpers::rewrite_model_name(
                chunk,
                &ctx.target_model_name,
                &ctx.incoming_model_name,
            );
            ctx.resp_body_resp =
                envoy_helpers::build_response_body_responses(&rewritten, end_of_stream, None);
        } else if end_of_stream {
            ctx.response_complete = true;
            // Non-streaming: `chunk` is the fully buffered JSON body.
            if let Some(usage) = parse_unary_usage(chunk) {
                ctx.parsed_usage = Some(usage);
            }
            let rewritten = envoy_helpers::rewrite_model_name(
                chunk,
                &ctx.target_model_name,
                &ctx.incoming_model_name,
            );
            ctx.resp_body_resp =
                envoy_helpers::build_response_body_responses(&rewritten, true, None);
        }
    }
}

/// Extract [`ResponseUsage`] from a JSON `usage` object.
fn usage_from_json(value: &serde_json::Value) -> Option<ResponseUsage> {
    let usage = value.get("usage")?;
    if usage.is_null() {
        return None;
    }
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(serde_json::Value::as_u64);
    Some(ResponseUsage {
        prompt_tokens: usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64),
        completion_tokens: usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64),
        total_tokens: usage
            .get("total_tokens")
            .and_then(serde_json::Value::as_u64),
        cached_tokens,
    })
}

/// Parse `usage` from a buffered non-streaming JSON body.
fn parse_unary_usage(body: &[u8]) -> Option<ResponseUsage> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    usage_from_json(&value)
}

/// Buffer SSE bytes across chunks; parse complete lines, keep the incomplete suffix.
fn ingest_streaming_usage(ctx: &mut RequestContext, chunk: &[u8], end_of_stream: bool) {
    if end_of_stream {
        if ctx.sse_usage_buf.is_empty() {
            update_streaming_usage(ctx, chunk);
        } else {
            ctx.sse_usage_buf.extend_from_slice(chunk);
            if let Some(usage) = parse_streaming_usage(&ctx.sse_usage_buf) {
                ctx.parsed_usage = Some(usage);
            }
            ctx.sse_usage_buf.clear();
        }
        return;
    }

    let Some(complete_len) = chunk.iter().rposition(|&b| b == b'\n').map(|i| i + 1) else {
        buffer_incomplete_sse(&mut ctx.sse_usage_buf, chunk);
        return;
    };

    if ctx.sse_usage_buf.is_empty() {
        update_streaming_usage(ctx, &chunk[..complete_len]);
    } else {
        ctx.sse_usage_buf.extend_from_slice(&chunk[..complete_len]);
        if let Some(usage) = parse_streaming_usage(&ctx.sse_usage_buf) {
            ctx.parsed_usage = Some(usage);
        }
        ctx.sse_usage_buf.clear();
    }
    buffer_incomplete_sse(&mut ctx.sse_usage_buf, &chunk[complete_len..]);
}

fn update_streaming_usage(ctx: &mut RequestContext, complete: &[u8]) {
    if let Some(usage) = parse_streaming_usage(complete) {
        ctx.parsed_usage = Some(usage);
    }
}

/// Append an incomplete SSE line, bounding memory. A `usage` event is small, so
/// a partial line larger than the cap can't be one; drop it (self-corrects once
/// the next newline lands).
fn buffer_incomplete_sse(buf: &mut Vec<u8>, bytes: &[u8]) {
    const MAX_SSE_USAGE_BUF: usize = 64 * 1024;
    if buf.len() + bytes.len() > MAX_SSE_USAGE_BUF {
        buf.clear();
        return;
    }
    buf.extend_from_slice(bytes);
}

/// Parse `usage` from complete SSE `data:` lines (last wins).
fn parse_streaming_usage(chunk: &[u8]) -> Option<ResponseUsage> {
    // Skip JSON work unless the terminal `"usage"` field is present.
    const USAGE_FIELD: &str = "\"usage\"";
    if !chunk
        .windows(USAGE_FIELD.len())
        .any(|window| window == USAGE_FIELD.as_bytes())
    {
        return None;
    }
    let text = std::str::from_utf8(chunk).ok()?;
    let mut latest = None;
    for line in text.lines() {
        let line = line.trim_start();
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" || !payload.contains(USAGE_FIELD) {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload)
            && let Some(usage) = usage_from_json(&value)
        {
            latest = Some(usage);
        }
    }
    latest
}

#[tonic::async_trait]
impl<P: EndpointPicker> ExternalProcessor for ExtProcServer<P> {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ProcessingResponse, Status>> + Send + 'static>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        // Set by the TLS listener, which tracks gateway connections.
        let connection = request.extensions().get::<Arc<Connection>>().cloned();
        let mut inbound = request.into_inner();
        let picker = self.picker.clone();

        let (tx, rx) = mpsc::channel::<Result<ProcessingResponse, Status>>(32);
        let output_stream = ReceiverStream::new(rx);

        tokio_runtime::spawn(TaskKind::ExtProcStream, async move {
            let stream = OpenStream::open(connection);
            let mut ctx = RequestContext::new();
            ctx.connection_id = stream.connection_id();
            ctx.connection_streams = stream.connection_streams();
            let mut body_buf: Vec<u8> = Vec::new();
            let mut resp_body_buf: Vec<u8> = Vec::new();

            let result: Result<(), Status> = async {
                while let Some(req_result) = inbound.next().await {
                    let mut req = req_result.map_err(|e| {
                        Status::unknown(format!("Cannot receive stream request: {e}"))
                    })?;

                    // Only the pick reads the routing metadata. The gateway can
                    // resend it with every message, so take it from the request
                    // phase only, by move, rather than copy it per token chunk.
                    if matches!(
                        req.request,
                        Some(
                            processing_request::Request::RequestHeaders(_)
                                | processing_request::Request::RequestBody(_)
                        )
                    ) {
                        ctx.request_metadata = req
                            .metadata_context
                            .take()
                            .map(|metadata| metadata.filter_metadata)
                            .unwrap_or_default();
                    }

                    if let Some(ref pc) = req.protocol_config {
                        tracing::debug!(
                            request_body_mode = pc.request_body_mode,
                            response_body_mode = pc.response_body_mode,
                            send_body_without_waiting =
                                pc.send_body_without_waiting_for_header_response,
                            "[PROTOCOL] ProtocolConfiguration from Envoy"
                        );
                        if !ctx.protocol_validated {
                            validate_protocol_config(pc)?;
                            ctx.protocol_validated = true;
                        }
                    }

                    match req.request {
                        Some(processing_request::Request::RequestHeaders(ref hdr)) => {
                            tracing::debug!(
                                eos = hdr.end_of_stream,
                                "[MSG-ORDER] Received RequestHeaders from Envoy"
                            );
                            ExtProcServer::<P>::handle_request_headers(&mut ctx, hdr);

                            if hdr.end_of_stream {
                                // Same cancellation race as the body path below:
                                // if the stream closes while the header-only pick
                                // is queued, drop the pick future to cancel it.
                                let routed = tokio::select! {
                                    biased;
                                    _ = tx.closed() => {
                                        tracing::debug!(
                                            request_id = %ctx.request_id,
                                            "ext_proc stream closed during selection; cancelling"
                                        );
                                        ctx.trace.outcome = Some(Outcome::Cancelled);
                                        return Ok(());
                                    }
                                    result = ExtProcServer::handle_header_only_request(
                                        &*picker,
                                        &mut ctx,
                                        &[],
                                    ) => result,
                                };
                                if let Err(e) = routed {
                                    ctx.trace.outcome = Some(Outcome::PickError);
                                    let resp = e.into_processing_response();
                                    let _ = tx.send(Ok(resp)).await;
                                    return Ok(());
                                }
                            }
                        }
                        Some(processing_request::Request::RequestBody(ref body)) => {
                            tracing::debug!(
                                eos = body.end_of_stream,
                                body_len = body.body.len(),
                                "[MSG-ORDER] Received RequestBody from Envoy"
                            );
                            body_buf.extend_from_slice(&body.body);
                            ctx.trace.body_msgs += 1;

                            if body.end_of_stream {
                                if let Some(received) = ctx.headers_received_at {
                                    metrics::observe_stage(Stage::ReceiveBody, received.elapsed());
                                }
                                ctx.trace.body_eos = ctx.since_headers();
                                // Freeze the accumulated body once into `Bytes`
                                // (moves the Vec, no copy); downstream sharing is
                                // by cheap refcount clone.
                                let raw_body = Bytes::from(std::mem::take(&mut body_buf));
                                // Race selection against the client dropping the
                                // stream. If Envoy closes it while `pick()` is
                                // queued in the selector, dropping the pick future
                                // closes the scheduler's response receiver, so the
                                // queued request is cancelled (its capacity
                                // reservation is skipped/released) instead of
                                // booking for a request that is already gone.
                                // Biased: check closure before polling the pick.
                                let routed = tokio::select! {
                                    biased;
                                    _ = tx.closed() => {
                                        tracing::debug!(
                                            request_id = %ctx.request_id,
                                            "ext_proc stream closed during selection; cancelling"
                                        );
                                        ctx.trace.outcome = Some(Outcome::Cancelled);
                                        return Ok(());
                                    }
                                    result = ExtProcServer::handle_request_body(
                                        &*picker,
                                        &mut ctx,
                                        raw_body,
                                        &[],
                                    ) => result,
                                };
                                if let Err(e) = routed {
                                    ctx.trace.outcome = Some(Outcome::PickError);
                                    let resp = e.into_processing_response();
                                    let _ = tx.send(Ok(resp)).await;
                                    return Ok(());
                                }
                            }
                        }
                        Some(processing_request::Request::RequestTrailers(_)) => {}
                        Some(processing_request::Request::ResponseHeaders(ref hdr)) => {
                            ExtProcServer::<P>::handle_response_headers(&mut ctx, hdr);
                            // Like the first-token stages: a unary response's
                            // headers wait for the whole generation.
                            if ctx.model_server_streaming
                                && ctx.response_succeeded()
                                && let Some(routed) = ctx.routed_at
                            {
                                metrics::observe_stage(Stage::ResponseHeaders, routed.elapsed());
                            }
                            ctx.trace.response_headers = ctx.since_headers();
                            ctx.trace.streamed = ctx.model_server_streaming;
                        }
                        Some(processing_request::Request::ResponseBody(ref body)) => {
                            ctx.trace.record_chunk(Instant::now());
                            // Signal prefill completion on the first non-empty
                            // response body chunk (the first generated token).
                            // In streaming mode the upstream flushes HTTP
                            // response headers before producing any token, so
                            // signaling on ResponseHeaders would release prefill
                            // bookkeeping before decode actually starts. The
                            // first non-empty body chunk is the earliest signal
                            // that prefill produced output and decode is underway.
                            if ctx.body_routed
                                && !ctx.prefill_complete_signaled
                                && !body.body.is_empty()
                            {
                                ctx.prefill_complete_signaled = true;
                                // A first token only in a streamed 2xx response:
                                // a unary response's first chunk is the whole
                                // answer, and an error body is no token.
                                let routed = ctx.routed_at.take();
                                if ctx.model_server_streaming && ctx.response_succeeded() {
                                    if let Some(routed) = routed {
                                        metrics::observe_stage(
                                            Stage::FirstResponse,
                                            routed.elapsed(),
                                        );
                                    }
                                    if let Some(received) = ctx.headers_received_at {
                                        metrics::observe_stage(Stage::EppTtft, received.elapsed());
                                    }
                                }
                                ctx.trace.first_chunk = ctx.since_headers();
                                // Hand back the picker's own booking id (falls
                                // back to request_id if it booked nothing).
                                let booking_id = ctx
                                    .booking_id
                                    .clone()
                                    .unwrap_or_else(|| ctx.request_id.clone());
                                // Detach: prefill-completion is idempotent,
                                // best-effort load bookkeeping. Awaiting it inline
                                // would gate first-token forwarding on the router's
                                // admission actor (an unbounded, untimed send+ack
                                // when queueing is enabled) — a TTFT stall. Spawn it
                                // so the token forwards immediately; the callback
                                // logs its own errors.
                                let picker = picker.clone();
                                tokio::spawn(async move {
                                    picker.on_prefill_complete(&booking_id).await;
                                });
                            }

                            // TODO(epp-output-tracking): Parse generated-token progress and
                            // update router output blocks instead of tracking only phase changes.
                            if ctx.model_server_streaming {
                                ExtProcServer::<P>::handle_response_body(&mut ctx, body);
                            } else {
                                resp_body_buf.extend_from_slice(&body.body);
                                if body.end_of_stream {
                                    let full_body = std::mem::take(&mut resp_body_buf);
                                    let synthetic = ext_proc::HttpBody {
                                        body: full_body,
                                        end_of_stream: true,
                                    };
                                    ExtProcServer::<P>::handle_response_body(&mut ctx, &synthetic);
                                }
                            }
                        }
                        Some(processing_request::Request::ResponseTrailers(_)) => {
                            if !ctx.response_complete {
                                ctx.response_complete = true;
                                if !resp_body_buf.is_empty() {
                                    let full_body = std::mem::take(&mut resp_body_buf);
                                    let synthetic = ext_proc::HttpBody {
                                        body: full_body,
                                        end_of_stream: true,
                                    };
                                    ExtProcServer::<P>::handle_response_body(&mut ctx, &synthetic);
                                }
                            }
                            ctx.resp_trailer_resp =
                                Some(envoy_helpers::build_response_trailer_response());
                        }
                        None => {
                            tracing::warn!("Received ProcessingRequest with no request variant");
                        }
                    }

                    let responses = ctx.drain_pending_responses();
                    for resp in responses {
                        if tx.send(Ok(resp)).await.is_err() {
                            ctx.trace.outcome = Some(Outcome::Cancelled);
                            return Ok(());
                        }
                    }

                    if ctx.state == StreamState::RequestEvicted {
                        ctx.trace.outcome = Some(Outcome::Evicted);
                        break;
                    }
                }

                Ok(())
            }
            .await;

            if let Err(e) = result {
                ctx.trace.outcome = Some(Outcome::StreamError);
                let _ = tx.send(Err(e)).await;
            }
            log_stream_summary(&ctx);

            // TODO(epp-disconnect-semantics): Define how Envoy retries and backend
            // work continuing after an ext_proc disconnect affect booking ownership.
            // Notify the picker that this request is complete so it can free
            // router bookkeeping state.
            if ctx.body_routed && !ctx.request_id.is_empty() {
                let booking_id = ctx
                    .booking_id
                    .clone()
                    .unwrap_or_else(|| ctx.request_id.clone());
                let usage = ctx.parsed_usage.take();
                if let Some(cached_tokens) = usage.as_ref().and_then(|u| u.cached_tokens) {
                    metrics::observe_cached_tokens(cached_tokens);
                }
                picker
                    .on_request_complete_with_usage(&booking_id, usage)
                    .await;
            }
        });

        Ok(Response::new(Box::pin(output_stream)))
    }
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------

/// Validate the gateway's `ProtocolConfiguration` against the protocol
/// contract this EPP requires: `FULL_DUPLEX_STREAMED` on both body
/// directions.
///
/// **Request direction.** We build the `RequestHeaders` response only after
/// receiving the request body, because:
///   * The body holds the chat-completion prompt.
///   * We tokenize it.
///   * We feed those tokens to the KV-aware router to choose a worker.
///   * The chosen worker becomes the value of `x-dynamo-worker-instance-id` /
///     `x-gateway-destination-endpoint` in the `RequestHeaders` response.
///
/// That ordering — header response *after* body — needs the data plane to
/// send the body without waiting for our header response.
/// `BodySendMode::FULL_DUPLEX_STREAMED` always does: the data plane streams
/// the body as it arrives, and the server may hold every response until it
/// has the whole body. In the other modes the data plane can wait for our
/// header response before sending body chunks while we wait for body chunks
/// before producing the header response, which silently deadlocks until the
/// ext_proc timeout fires.
///
/// `send_body_without_waiting_for_header_response` is not checked. Envoy
/// applies it only to `STREAMED` mode, which this EPP rejects anyway, and
/// ignores it in every other mode. agentgateway 1.3 and later report it as
/// `false` alongside `FULL_DUPLEX_STREAMED`.
///
/// **Response direction.** `response_body_mode` defaults to `NONE` in Envoy,
/// which delivers no `ResponseBody` messages at all. Three behaviours depend
/// on receiving them, and all three fail silently under `NONE`:
///   * `on_prefill_complete` fires on the first non-empty body chunk, so
///     disaggregated prefill bookkeeping would stay held for the whole stream.
///   * Token usage (`cached_tokens`) is parsed out of the terminal chunk.
///   * Model-name rewriting mutates body bytes on their way to the client.
///
/// Streaming the response also requires `FULL_DUPLEX_STREAMED` specifically:
/// the buffering modes hold the whole body before handing it over, which
/// would break SSE token streaming. This matches the llm-d router, which
/// documents `FULL_DUPLEX_STREAMED` as the only supported mode for both
/// directions.
///
/// Failing fast with `Status::failed_precondition` here turns a multi-second
/// hidden timeout (or silently absent telemetry) into an immediate,
/// self-explaining error visible in Envoy logs the first time the EPP is
/// wired up behind a misconfigured gateway.
///
/// Older Envoy versions (pre-1.32) do not send `ProtocolConfiguration`; in
/// that case the caller skips this validation entirely and trusts the
/// operator to have configured the filter correctly.
// `Status` is the tonic-mandated error type for this stream, so we can't box
// it without rewriting the return path. The function is called once per
// stream, so the size of the `Err` variant is not a hot-path concern.
#[allow(clippy::result_large_err)]
fn validate_protocol_config(
    pc: &crate::proto::envoy::service::ext_proc::v3::ProtocolConfiguration,
) -> Result<(), Status> {
    use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

    let request_mode = BodySendMode::try_from(pc.request_body_mode).ok();
    let response_mode = BodySendMode::try_from(pc.response_body_mode).ok();
    let full_duplex = Some(BodySendMode::FullDuplexStreamed);

    if request_mode == full_duplex && response_mode == full_duplex {
        return Ok(());
    }

    let detail = format!(
        "ext_proc filter must be configured with request_body_mode=FULL_DUPLEX_STREAMED \
         and response_body_mode=FULL_DUPLEX_STREAMED; got \
         request_body_mode={request_mode:?}, response_body_mode={response_mode:?}. \
         The Rust EPP defers its RequestHeaders response until after it has tokenized \
         the body and selected a worker, and it reads response bodies to signal prefill \
         completion, parse token usage, and rewrite the model name."
    );
    tracing::error!(
        request_body_mode = pc.request_body_mode,
        response_body_mode = pc.response_body_mode,
        "ProtocolConfiguration mismatch — failing stream"
    );
    Err(Status::failed_precondition(detail))
}

/// Rewrite the request body for the backend: drop the client's `nvext`
/// (`strip_nvext`), add `nvext.token_data` (`token_ids`), and set the tagged
/// top-level `cache_salt` (`cache_salt`), in that order.
///
/// The body is never parsed into a `Value` tree. Untouched top-level members
/// are borrowed and re-emitted byte for byte, only `nvext` is parsed when
/// tokens are injected, and the token ids are written straight from the
/// slice. A tree would cost an allocation per message part and per token.
fn rewrite_forwarded_body(
    body: &[u8],
    strip_nvext: bool,
    token_ids: Option<&[u32]>,
    cache_salt: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    let mut members: BTreeMap<String, &RawValue> = serde_json::from_slice(body)?;
    if strip_nvext {
        members.remove("nvext");
    }
    let nvext = token_ids
        .map(|ids| nvext_with_token_data(members.get("nvext").copied(), ids))
        .transpose()?;
    if let Some(nvext) = &nvext {
        members.insert("nvext".to_string(), nvext);
    }
    let salt = cache_salt
        .map(|salt| serde_json::value::to_raw_value(&tagged_cache_salt(salt)))
        .transpose()?;
    if let Some(salt) = &salt {
        members.insert("cache_salt".to_string(), salt);
    }
    Ok(serde_json::to_vec(&members)?)
}

/// `nvext` with `token_data` set to `token_ids`, keeping its other fields
/// byte for byte.
fn nvext_with_token_data(
    nvext: Option<&RawValue>,
    token_ids: &[u32],
) -> anyhow::Result<Box<RawValue>> {
    let mut fields: BTreeMap<String, Box<RawValue>> = match nvext {
        Some(raw) => serde_json::from_str(raw.get())
            .map_err(|_| anyhow::anyhow!("nvext is not a JSON object"))?,
        None => BTreeMap::new(),
    };
    fields.insert(
        "token_data".to_string(),
        serde_json::value::to_raw_value(token_ids)?,
    );
    Ok(serde_json::value::to_raw_value(&fields)?)
}

fn tagged_cache_salt(cache_salt: &str) -> String {
    format!("{DYNAMO_CACHE_SALT_PREFIX}{cache_salt}")
}

/// Extract the "model" field from a JSON request body.
fn extract_model_from_body(body: &[u8]) -> String {
    #[derive(serde::Deserialize)]
    struct ModelField {
        model: Option<String>,
    }

    serde_json::from_slice::<ModelField>(body)
        .ok()
        .and_then(|m| m.model)
        .unwrap_or_default()
}

/// Extract the candidate endpoint subset from ext-proc request metadata.
fn extract_candidate_subset(
    request_metadata: &HashMap<String, prost_types::Struct>,
) -> Vec<String> {
    let ns = match request_metadata.get(metadata::SUBSET_FILTER_NAMESPACE) {
        Some(s) => s,
        None => return vec![],
    };

    let subset_val = match ns.fields.get(metadata::SUBSET_FILTER_KEY) {
        Some(v) => v,
        None => return vec![],
    };

    if let Some(prost_types::value::Kind::StringValue(s)) = &subset_val.kind {
        if s.is_empty() {
            return vec![];
        }
        return s.split(',').map(|s| s.to_string()).collect();
    }

    if let Some(prost_types::value::Kind::ListValue(list)) = &subset_val.kind {
        return list
            .values
            .iter()
            .filter_map(|v| {
                if let Some(prost_types::value::Kind::StringValue(s)) = &v.kind {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
    }

    vec![]
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

struct ExtProcError {
    status_code: StatusCode,
    message: String,
}

impl ExtProcError {
    fn from_pick_error(e: PickError) -> Self {
        match e {
            PickError::NoEndpoints => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: e.to_string(),
            },
            PickError::RoutingFailed(msg) => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: msg,
            },
            PickError::InvalidRequest(msg) => Self {
                status_code: StatusCode::BadRequest,
                message: msg,
            },
            PickError::MetadataHeadersTooLarge(err) => Self {
                status_code: StatusCode::RequestHeaderFieldsTooLarge,
                message: err.to_string(),
            },
            // Upstream tokenizer failures are not client errors: preserve their
            // semantics so clients retry appropriately. `e.to_string()` is the
            // client-safe variant message; the detailed cause is logged upstream.
            PickError::TokenizerUnavailable => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: e.to_string(),
            },
            PickError::TokenizerTimeout => Self {
                status_code: StatusCode::GatewayTimeout,
                message: e.to_string(),
            },
            PickError::TokenizerUpstreamError => Self {
                status_code: StatusCode::BadGateway,
                message: e.to_string(),
            },
            // In-flight limit saturated: shed as retryable backpressure. The
            // variant message ("endpoint picker overloaded") is client-safe.
            PickError::Overloaded => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: e.to_string(),
            },
        }
    }

    fn into_processing_response(self) -> ProcessingResponse {
        envoy_helpers::build_error_response(self.status_code, Some(&self.message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_counted_on_their_connection_until_dropped() {
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
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Duration;

    use tokio::sync::Notify;

    use crate::picker::{PickError, PickResult};
    use crate::proto::envoy::config::core::v3::{HeaderMap, HeaderValue};
    use crate::proto::envoy::service::ext_proc::v3::{
        HttpBody, HttpHeaders, ProcessingRequest,
        external_processor_client::ExternalProcessorClient, processing_request::Request as ProcReq,
    };

    fn protocol_config(
        request_body_mode: i32,
        response_body_mode: i32,
        send_body_without_waiting_for_header_response: bool,
    ) -> crate::proto::envoy::service::ext_proc::v3::ProtocolConfiguration {
        crate::proto::envoy::service::ext_proc::v3::ProtocolConfiguration {
            request_body_mode,
            response_body_mode,
            send_body_without_waiting_for_header_response,
        }
    }

    #[test]
    fn protocol_config_accepts_full_duplex_on_both_directions() {
        use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        let full_duplex = BodySendMode::FullDuplexStreamed as i32;
        assert!(validate_protocol_config(&protocol_config(full_duplex, full_duplex, true)).is_ok());
        // Envoy ignores send_body_without_waiting_for_header_response outside
        // STREAMED mode, and agentgateway 1.3+ reports it as false.
        assert!(
            validate_protocol_config(&protocol_config(full_duplex, full_duplex, false)).is_ok()
        );
    }

    #[test]
    fn protocol_config_rejects_response_body_mode_none() {
        use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        // Envoy's default. Without response bodies the EPP never signals
        // prefill completion, parses usage, or rewrites the model name.
        let err = validate_protocol_config(&protocol_config(
            BodySendMode::FullDuplexStreamed as i32,
            BodySendMode::None as i32,
            true,
        ))
        .expect_err("response_body_mode=NONE must be rejected");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("response_body_mode"));
    }

    #[test]
    fn protocol_config_rejects_buffered_response_body_mode() {
        use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        // Buffering holds the whole body, which would break SSE streaming.
        let err = validate_protocol_config(&protocol_config(
            BodySendMode::FullDuplexStreamed as i32,
            BodySendMode::Buffered as i32,
            true,
        ))
        .expect_err("response_body_mode=BUFFERED must be rejected");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn protocol_config_still_rejects_request_side_misconfiguration() {
        use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        // In these modes the data plane can hold the body until our header
        // response, which we only send once the body has arrived.
        let full_duplex = BodySendMode::FullDuplexStreamed as i32;
        for request_mode in [
            BodySendMode::None,
            BodySendMode::Streamed,
            BodySendMode::Buffered,
            BodySendMode::BufferedPartial,
        ] {
            let pc = protocol_config(request_mode as i32, full_duplex, true);
            assert!(
                validate_protocol_config(&pc).is_err(),
                "request_body_mode={request_mode:?} must be rejected"
            );
        }
    }

    #[test]
    fn parse_unary_usage_extracts_cached_tokens() {
        let body = br#"{
            "id": "cmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 64}
            }
        }"#;
        let usage = parse_unary_usage(body).expect("usage present");
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(120));
        assert_eq!(usage.cached_tokens, Some(64));
    }

    #[test]
    fn parse_unary_usage_without_details_has_no_cached_tokens() {
        let body = br#"{"usage": {"prompt_tokens": 10, "total_tokens": 10}}"#;
        let usage = parse_unary_usage(body).expect("usage present");
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.cached_tokens, None);
    }

    #[test]
    fn parse_unary_usage_none_when_absent_or_invalid() {
        assert!(parse_unary_usage(br#"{"choices": []}"#).is_none());
        assert!(parse_unary_usage(br#"{"usage": null}"#).is_none());
        assert!(parse_unary_usage(b"not json").is_none());
    }

    #[test]
    fn parse_streaming_usage_returns_last_event_with_usage() {
        let chunk = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":2,\"total_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n",
            "data: [DONE]\n\n"
        );
        let usage = parse_streaming_usage(chunk.as_bytes()).expect("usage present");
        assert_eq!(usage.total_tokens, Some(10));
        assert_eq!(usage.cached_tokens, Some(4));
    }

    #[test]
    fn parse_streaming_usage_none_without_usage_event() {
        let chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
        assert!(parse_streaming_usage(chunk.as_bytes()).is_none());
    }

    #[test]
    fn ingest_streaming_usage_does_not_buffer_complete_chunks() {
        let mut ctx = RequestContext::new();
        let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";

        ingest_streaming_usage(&mut ctx, chunk, false);

        assert!(ctx.sse_usage_buf.is_empty());
        assert!(ctx.parsed_usage.is_none());
    }

    #[test]
    fn ingest_streaming_usage_survives_split_data_event() {
        let mut ctx = RequestContext::new();
        let part1 = br#"data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":"#;
        let part2 = br#"4}}}"#;
        let part3 = b"\n\ndata: [DONE]\n\n";

        ingest_streaming_usage(&mut ctx, part1, false);
        assert!(ctx.parsed_usage.is_none());
        assert!(!ctx.sse_usage_buf.is_empty());

        ingest_streaming_usage(&mut ctx, part2, false);
        assert!(ctx.parsed_usage.is_none());
        assert!(!ctx.sse_usage_buf.is_empty());

        ingest_streaming_usage(&mut ctx, part3, true);
        let usage = ctx.parsed_usage.expect("usage across split chunks");
        assert_eq!(usage.total_tokens, Some(10));
        assert_eq!(usage.cached_tokens, Some(4));
        assert!(ctx.sse_usage_buf.is_empty());
    }

    #[test]
    fn ingest_streaming_usage_keeps_incomplete_suffix_only() {
        let mut ctx = RequestContext::new();
        let chunk = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1"
        );
        ingest_streaming_usage(&mut ctx, chunk.as_bytes(), false);
        assert!(ctx.parsed_usage.is_none());
        assert_eq!(
            std::str::from_utf8(&ctx.sse_usage_buf).unwrap(),
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":1"
        );

        ingest_streaming_usage(&mut ctx, b"}}}\n\n", true);
        let usage = ctx.parsed_usage.expect("usage after completing suffix");
        assert_eq!(usage.cached_tokens, Some(1));
        assert!(ctx.sse_usage_buf.is_empty());
    }

    #[test]
    fn ingest_streaming_usage_bounds_incomplete_buffer() {
        let mut ctx = RequestContext::new();
        // A very long newline-less run must not grow the buffer unboundedly.
        let huge = vec![b'x'; 128 * 1024];
        ingest_streaming_usage(&mut ctx, &huge, false);
        assert!(ctx.sse_usage_buf.len() <= 64 * 1024);

        // Once a real usage event completes afterward, it is still parsed.
        let tail = concat!(
            "\n\n",
            "data: {\"choices\":[],\"usage\":{\"total_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":4}}}\n\n"
        );
        ingest_streaming_usage(&mut ctx, tail.as_bytes(), true);
        let usage = ctx.parsed_usage.expect("usage after oversized run");
        assert_eq!(usage.cached_tokens, Some(4));
    }

    /// RAII probe that records, on drop, that a `pick` future was torn down. A
    /// blocked `pick` that never resolves only drops when the server cancels it,
    /// so this observes the disconnect-cancellation path.
    struct CancelProbe(Arc<AtomicBool>);
    impl Drop for CancelProbe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Mock `EndpointPicker` with per-callback counters. Extended beyond the
    /// simple 3-counter form to support the reservation-lifecycle tests:
    ///
    /// * `mint_reservations` makes `pick` return a fresh, distinct
    ///   `reservation_id` per call and records the booking ids handed back to the
    ///   lifecycle callbacks, so a test can prove each stream frees its own
    ///   booking (keyed by the EPP-minted id, not `x-request-id`).
    /// * `block` makes `pick` park until cancelled, so a test can close the
    ///   stream while a pick is in flight and observe the server cancel it.
    struct Tracker {
        add: AtomicU32,
        prefill_complete: AtomicU32,
        free: AtomicU32,
        disagg: bool,
        /// When true, `pick` mints `res-{n}` reservation ids.
        mint_reservations: bool,
        next_reservation: AtomicU32,
        /// Booking ids passed to `on_request_complete` / `on_prefill_complete`.
        freed: Mutex<Vec<String>>,
        prefilled: Mutex<Vec<String>>,
        /// When set, `pick` waits on this (never-notified) `Notify`, so it only
        /// resolves by being dropped (cancelled).
        block: Option<Arc<Notify>>,
        /// Notified once `pick` has entered its blocking wait.
        pick_started: Arc<Notify>,
        /// Set when a blocked `pick` future is dropped (cancelled).
        pick_cancelled: Arc<AtomicBool>,
    }

    impl Tracker {
        fn new(disagg: bool) -> Self {
            Self {
                add: 0.into(),
                prefill_complete: 0.into(),
                free: 0.into(),
                disagg,
                mint_reservations: false,
                next_reservation: 0.into(),
                freed: Mutex::new(Vec::new()),
                prefilled: Mutex::new(Vec::new()),
                block: None,
                pick_started: Arc::new(Notify::new()),
                pick_cancelled: Arc::new(AtomicBool::new(false)),
            }
        }
        fn agg() -> Self {
            Self::new(false)
        }
        fn disagg() -> Self {
            Self::new(true)
        }
        /// Mint a distinct `reservation_id` per `pick`, so bookings are keyed by
        /// the picker's id rather than the request's `x-request-id`.
        fn minting(mut self) -> Self {
            self.mint_reservations = true;
            self
        }
        /// Park `pick` on `block` until its future is dropped (cancelled).
        fn blocking(mut self, block: Arc<Notify>) -> Self {
            self.block = Some(block);
            self
        }
    }

    #[tonic::async_trait]
    impl EndpointPicker for Tracker {
        async fn pick(&self, _: &RequestInfo, _: &[Endpoint]) -> Result<PickResult, PickError> {
            self.add.fetch_add(1, Ordering::SeqCst);
            if let Some(block) = &self.block {
                // Signal that the pick is in flight, then park until cancelled.
                // The probe records the cancellation when this future is dropped.
                let _probe = CancelProbe(self.pick_cancelled.clone());
                self.pick_started.notify_one();
                block.notified().await;
            }
            let mode = if self.disagg {
                "disaggregated"
            } else {
                "aggregated"
            };
            let reservation_id = self.mint_reservations.then(|| {
                let n = self.next_reservation.fetch_add(1, Ordering::SeqCst);
                format!("res-{n}")
            });
            Ok(PickResult {
                endpoint: "1.2.3.4:80".into(),
                headers: vec![("x-dynamo-routing-mode".into(), mode.into())],
                reservation_id,
                ..Default::default()
            })
        }
        async fn on_prefill_complete(&self, booking_id: &str) {
            self.prefill_complete.fetch_add(1, Ordering::SeqCst);
            self.prefilled.lock().unwrap().push(booking_id.to_string());
        }
        async fn on_request_complete(&self, booking_id: &str) {
            self.free.fetch_add(1, Ordering::SeqCst);
            self.freed.lock().unwrap().push(booking_id.to_string());
        }
    }

    // Spin up a GRPC server and create a gRPC bi-directional stream
    async fn connect(t: Arc<Tracker>) -> ExternalProcessorClient<tonic::transport::Channel> {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        let svc = ExtProcServer::new(t).into_service();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(svc)
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(l)),
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        ExternalProcessorClient::new(
            tonic::transport::Channel::from_shared(format!("http://{addr}"))
                .unwrap()
                .connect()
                .await
                .unwrap(),
        )
    }

    fn stream() -> Vec<ProcessingRequest> {
        stream_with_request_id("r1")
    }

    /// A full request/response stream carrying an explicit `x-request-id`, so
    /// tests can drive two streams that share the same client-controlled id.
    fn stream_with_request_id(request_id: &str) -> Vec<ProcessingRequest> {
        vec![
            ProcessingRequest {
                request: Some(ProcReq::RequestHeaders(HttpHeaders {
                    headers: Some(HeaderMap {
                        headers: vec![HeaderValue {
                            key: "x-request-id".into(),
                            value: request_id.into(),
                            raw_value: vec![],
                        }],
                    }),
                    end_of_stream: false,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::RequestBody(HttpBody {
                    body: br#"{"model":"m","messages":[]}"#.to_vec(),
                    end_of_stream: true,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::ResponseHeaders(HttpHeaders {
                    headers: Some(HeaderMap { headers: vec![] }),
                    end_of_stream: false,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::ResponseBody(HttpBody {
                    body: b"{".to_vec(),
                    end_of_stream: false,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::ResponseBody(HttpBody {
                    body: b"}".to_vec(),
                    end_of_stream: true,
                })),
                ..Default::default()
            },
        ]
    }

    /// Just the request headers + end-of-stream body, with no response phase, so
    /// the picker's `pick` is invoked and left in flight (used to drive the
    /// disconnect-cancellation path).
    fn request_only_stream(request_id: &str) -> Vec<ProcessingRequest> {
        vec![
            ProcessingRequest {
                request: Some(ProcReq::RequestHeaders(HttpHeaders {
                    headers: Some(HeaderMap {
                        headers: vec![HeaderValue {
                            key: "x-request-id".into(),
                            value: request_id.into(),
                            raw_value: vec![],
                        }],
                    }),
                    end_of_stream: false,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::RequestBody(HttpBody {
                    body: br#"{"model":"m","messages":[]}"#.to_vec(),
                    end_of_stream: true,
                })),
                ..Default::default()
            },
        ]
    }

    fn header_only_stream() -> Vec<ProcessingRequest> {
        vec![
            ProcessingRequest {
                request: Some(ProcReq::RequestHeaders(HttpHeaders {
                    headers: Some(HeaderMap {
                        headers: vec![HeaderValue {
                            key: "x-request-id".into(),
                            value: "r1".into(),
                            raw_value: vec![],
                        }],
                    }),
                    end_of_stream: true,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::ResponseHeaders(HttpHeaders {
                    headers: Some(HeaderMap { headers: vec![] }),
                    end_of_stream: false,
                })),
                ..Default::default()
            },
            ProcessingRequest {
                request: Some(ProcReq::ResponseBody(HttpBody {
                    body: b"{}".to_vec(),
                    end_of_stream: true,
                })),
                ..Default::default()
            },
        ]
    }

    async fn run_stream(
        c: &mut ExternalProcessorClient<tonic::transport::Channel>,
        requests: Vec<ProcessingRequest>,
    ) {
        let mut r = c
            .process(tokio_stream::iter(requests))
            .await
            .unwrap()
            .into_inner();
        while r.message().await.unwrap().is_some() {}
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    async fn run(c: &mut ExternalProcessorClient<tonic::transport::Channel>) {
        run_stream(c, stream()).await;
    }

    /// add_request: pick() is invoked → registers request with the slot tracker.
    #[tokio::test]
    async fn test_add_request_called() {
        let t = Arc::new(Tracker::agg());
        run(&mut connect(t.clone()).await).await;
        assert_eq!(t.add.load(Ordering::SeqCst), 1);
    }

    /// mark_prefill_complete: on_prefill_complete() fires exactly once on the first
    /// non-empty ResponseBody chunk (the first generated token) in both routing modes.
    #[tokio::test]
    async fn test_mark_prefill_complete_called_once_for_both_routing_modes() {
        for tracker in [Tracker::agg(), Tracker::disagg()] {
            let tracker = Arc::new(tracker);
            run(&mut connect(tracker.clone()).await).await;
            // `on_prefill_complete` is dispatched off-path (a detached task) so it
            // can't stall first-token forwarding, so wait for it to land.
            tokio::time::timeout(Duration::from_secs(5), async {
                while tracker.prefill_complete.load(Ordering::SeqCst) == 0 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .expect("on_prefill_complete should fire");
            assert_eq!(tracker.prefill_complete.load(Ordering::SeqCst), 1);
        }
    }

    /// free_request: on_request_complete() fires when the stream ends.
    #[tokio::test]
    async fn test_free_request_called() {
        let t = Arc::new(Tracker::agg());
        run(&mut connect(t.clone()).await).await;
        assert_eq!(t.free.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_header_only_response_skips_router_lifecycle_callbacks() {
        let t = Arc::new(Tracker::agg());
        run_stream(&mut connect(t.clone()).await, header_only_stream()).await;

        assert_eq!(t.add.load(Ordering::SeqCst), 1);
        assert_eq!(t.prefill_complete.load(Ordering::SeqCst), 0);
        assert_eq!(t.free.load(Ordering::SeqCst), 0);
    }

    /// agentgateway 1.3 and later put this `ProtocolConfiguration` on the first
    /// message of every stream. The EPP must route the request rather than fail
    /// the stream with FAILED_PRECONDITION.
    #[tokio::test]
    async fn test_agentgateway_protocol_config_is_routed() {
        use crate::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        let t = Arc::new(Tracker::agg());
        let full_duplex = BodySendMode::FullDuplexStreamed as i32;
        let mut requests = stream();
        requests[0].protocol_config = Some(protocol_config(full_duplex, full_duplex, false));
        run_stream(&mut connect(t.clone()).await, requests).await;

        assert_eq!(t.add.load(Ordering::SeqCst), 1);
        assert_eq!(t.free.load(Ordering::SeqCst), 1);
    }

    /// Item 4 (queued disconnect / cancellation): when the ext-proc stream closes
    /// while a `pick` is still in flight, the server's biased `select!` on
    /// `tx.closed()` must drop the pick future (cancelling it) instead of
    /// blocking on it, and it must not run the completion callback for a booking
    /// that was never adopted (no leak).
    #[tokio::test]
    async fn test_queued_pick_is_cancelled_when_stream_closes() {
        // `block` is never notified, so `pick` only resolves by being dropped.
        let block = Arc::new(Notify::new());
        let t = Arc::new(Tracker::agg().blocking(block));
        let mut c = connect(t.clone()).await;

        // Send headers + body(eos); the request stream then half-closes, and the
        // server enters the pick and parks awaiting the (blocked) result.
        let response = c
            .process(tokio_stream::iter(request_only_stream("r1")))
            .await
            .unwrap()
            .into_inner();

        // Wait until the pick is actually in flight before disconnecting.
        tokio::time::timeout(Duration::from_secs(5), t.pick_started.notified())
            .await
            .expect("pick should start");

        // Client goes away: dropping the response stream and the client tears down
        // the gRPC call, which the biased `select!` observes as `tx.closed()`.
        drop(response);
        drop(c);

        // The server must promptly cancel the in-flight pick (drop its future).
        let cancelled = tokio::time::timeout(Duration::from_secs(5), async {
            while !t.pick_cancelled.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            cancelled.is_ok(),
            "server must cancel the in-flight pick on stream close (no hang)"
        );

        // No booking was adopted, so no lifecycle callback leaks.
        assert_eq!(
            t.free.load(Ordering::SeqCst),
            0,
            "a cancelled pick must not trigger on_request_complete"
        );
        assert_eq!(
            t.prefill_complete.load(Ordering::SeqCst),
            0,
            "a cancelled pick must not trigger on_prefill_complete"
        );
    }

    /// Item 6 (duplicate `x-request-id` isolation): two concurrent streams that
    /// reuse the same client-controlled `x-request-id` must each free only their
    /// own booking. Bookings are keyed by the EPP-minted `reservation_id` carried
    /// on the per-stream context, so the picker sees two distinct booking ids
    /// freed — never the shared `x-request-id`, and never one stream freeing the
    /// other's reservation.
    #[tokio::test]
    async fn test_duplicate_request_id_streams_free_their_own_bookings() {
        let t = Arc::new(Tracker::agg().minting());

        // Run both streams concurrently, both carrying x-request-id "dup".
        let (mut c1, mut c2) = tokio::join!(connect(t.clone()), connect(t.clone()));
        tokio::join!(
            run_stream(&mut c1, stream_with_request_id("dup")),
            run_stream(&mut c2, stream_with_request_id("dup")),
        );

        let freed = t.freed.lock().unwrap().clone();
        assert_eq!(freed.len(), 2, "each stream completes and frees once");
        assert!(
            freed.iter().all(|id| id != "dup"),
            "bookings are freed by the minted reservation_id, not the shared x-request-id"
        );
        let unique: HashSet<&String> = freed.iter().collect();
        assert_eq!(
            unique.len(),
            2,
            "each stream frees its own distinct booking (no cross-free)"
        );

        // Prefill completion is likewise keyed per-stream: two distinct ids.
        let prefilled = t.prefilled.lock().unwrap().clone();
        assert_eq!(prefilled.len(), 2);
        let unique_prefilled: HashSet<&String> = prefilled.iter().collect();
        assert_eq!(unique_prefilled, unique);
    }

    /// A shed `PickError::Overloaded` maps to a retryable 503 (not a 4xx), so
    /// clients back off and retry rather than treating the load-shed as their
    /// own error.
    #[test]
    fn overloaded_pick_error_maps_to_503() {
        let err = ExtProcError::from_pick_error(PickError::Overloaded);
        assert_eq!(err.status_code, StatusCode::ServiceUnavailable);
    }

    #[test]
    fn metadata_headers_too_large_maps_to_431() {
        let err = ExtProcError::from_pick_error(PickError::MetadataHeadersTooLarge(
            dynamo_llm::http::service::metadata::MetadataHeaderError::TooManyEntries { limit: 64 },
        ));
        assert_eq!(err.status_code, StatusCode::RequestHeaderFieldsTooLarge);
    }

    /// Cache salt is injected as a top-level field and tagged with the Dynamo
    /// cache-salt prefix so the backend's KV-event extra_keys carry an
    /// unambiguous namespace marker.
    #[test]
    fn inject_body_extensions_adds_cache_salt() {
        let body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let modified = rewrite_forwarded_body(body, false, None, Some("salt-a")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&modified).unwrap();
        assert_eq!(
            parsed.get("cache_salt").and_then(|v| v.as_str()),
            Some("dynamo-cache-salt:salt-a")
        );
    }

    /// Injecting both token_data and cache_salt preserves existing nvext fields.
    #[test]
    fn inject_body_extensions_preserves_existing_fields() {
        let body = br#"{"model":"m","nvext":{"extra_fields":["engine_data"]}}"#;
        let modified =
            rewrite_forwarded_body(body, false, Some(&[10, 20, 30]), Some("salt-b")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&modified).unwrap();

        assert_eq!(
            parsed.get("cache_salt").and_then(|v| v.as_str()),
            Some("dynamo-cache-salt:salt-b")
        );

        let nvext = parsed.get("nvext").expect("nvext preserved");
        assert_eq!(
            nvext
                .get("extra_fields")
                .and_then(|v| v.as_array())
                .map(|a| a.len()),
            Some(1)
        );
        let token_data: Vec<u64> = nvext
            .get("token_data")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
        assert_eq!(token_data, vec![10, 20, 30]);
    }

    /// Cross-salt isolation: different salts produce different body values.
    #[test]
    fn inject_body_extensions_isolates_salts() {
        let body = br#"{}"#;
        let a = rewrite_forwarded_body(body, false, None, Some("salt-a")).unwrap();
        let b = rewrite_forwarded_body(body, false, None, Some("salt-b")).unwrap();
        let parsed_a: serde_json::Value = serde_json::from_slice(&a).unwrap();
        let parsed_b: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(parsed_a["cache_salt"], "dynamo-cache-salt:salt-a");
        assert_eq!(parsed_b["cache_salt"], "dynamo-cache-salt:salt-b");
        assert_ne!(parsed_a["cache_salt"], parsed_b["cache_salt"]);
    }

    /// Raw-engine forwarding writes only the top-level `cache_salt`; nvext is
    /// not read by a raw engine, so a body `nvext.cache_salt` is left untouched
    /// (Dynamo-runtime backends use `Preserve`, which skips the salt rewrite).
    #[test]
    fn inject_body_extensions_leaves_nvext_cache_salt_untouched() {
        let body = br#"{"nvext":{"cache_salt":"body-salt","extra_fields":["engine_data"]}}"#;
        let modified =
            rewrite_forwarded_body(body, false, Some(&[10, 20, 30]), Some("header-salt")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&modified).unwrap();

        assert_eq!(parsed["cache_salt"], "dynamo-cache-salt:header-salt");

        let nvext = parsed.get("nvext").expect("nvext preserved");
        assert_eq!(nvext["cache_salt"], "body-salt");

        let extra_fields = nvext
            .get("extra_fields")
            .and_then(|v| v.as_array())
            .expect("extra_fields preserved");
        assert_eq!(extra_fields.len(), 1);
    }

    /// Malformed bodies are rejected rather than silently forwarded unchanged.
    #[test]
    fn inject_body_extensions_rejects_non_object_body() {
        let body = br#"["not", "an", "object"]"#;
        assert!(rewrite_forwarded_body(body, false, Some(&[1]), Some("salt")).is_err());
    }

    /// A body with a non-object `nvext` cannot accept token_data.
    #[test]
    fn inject_body_extensions_rejects_non_object_nvext() {
        let body = br#"{"nvext": "bad"}"#;
        assert!(rewrite_forwarded_body(body, false, Some(&[1]), None).is_err());
    }

    /// `nvext` as an object or an explicit `null` is removed; the salt is still written.
    #[test]
    fn rewrite_forwarded_body_strips_nvext_for_raw_engines() {
        for body in [
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"nvext":{"agent_hints":{"osl":32},"cache_salt":"tenant-a"}}"#.as_slice(),
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"nvext":null}"#.as_slice(),
        ] {
            let modified = rewrite_forwarded_body(body, true, None, Some("tenant-a")).unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&modified).unwrap();
            let obj = parsed.as_object().unwrap();

            assert!(!obj.contains_key("nvext"), "{parsed}");
            assert_eq!(parsed["model"], "m");
            assert_eq!(parsed["messages"][0]["content"], "hi");
            assert_eq!(parsed["max_tokens"], 8);
            assert_eq!(parsed["cache_salt"], "dynamo-cache-salt:tenant-a");
        }
    }

    #[test]
    fn rewrite_forwarded_body_strip_without_nvext_is_a_no_op() {
        let body = br#"{"model":"m","messages":[]}"#;
        let modified = rewrite_forwarded_body(body, true, None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&modified).unwrap();
        assert_eq!(parsed, serde_json::json!({"model":"m","messages":[]}));
    }

    /// Untouched values keep their exact bytes (whitespace, escapes, number
    /// formatting) and escaped keys survive; members come out in key order.
    #[test]
    fn rewrite_forwarded_body_copies_other_members_verbatim() {
        let body = r#"{ "model" : "m", "messages": [ {"role":"user","content":"café \"q\""} ],"nvext":{"agent_hints":{"osl":32}}, "temperature": 1.0e0, "k\"y": true , "cache_salt": "client"}"#;

        let out = rewrite_forwarded_body(body.as_bytes(), true, None, Some("tenant-a")).unwrap();

        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{"cache_salt":"dynamo-cache-salt:tenant-a","k\"y":true,"messages":[ {"role":"user","content":"café \"q\""} ],"model":"m","temperature":1.0e0}"#
        );
    }

    /// Injecting tokens keeps the same guarantee: other top-level members and
    /// the other `nvext` fields keep their exact bytes, and a client
    /// `token_data` is replaced.
    #[test]
    fn rewrite_forwarded_body_injects_tokens_without_reformatting() {
        let body = r#"{ "model" : "m", "messages": [ {"role":"user","content":"café \"q\""} ], "nvext": { "agent_hints": {"osl": 32}, "token_data": [9, 9] }, "seed": 18446744073709551616 }"#;

        let out =
            rewrite_forwarded_body(body.as_bytes(), false, Some(&[1, 22, 333]), None).unwrap();

        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{"messages":[ {"role":"user","content":"café \"q\""} ],"model":"m","nvext":{"agent_hints":{"osl": 32},"token_data":[1,22,333]},"seed":18446744073709551616}"#
        );
    }

    /// The rewrite produces the same JSON as a `Value`-tree rewrite, for
    /// every combination of inputs it serves.
    #[test]
    fn rewrite_forwarded_body_matches_a_value_tree_rewrite() {
        fn by_tree(
            body: &[u8],
            strip_nvext: bool,
            token_ids: Option<&[u32]>,
            cache_salt: Option<&str>,
        ) -> Option<serde_json::Value> {
            let mut parsed: serde_json::Value = serde_json::from_slice(body).ok()?;
            let obj = parsed.as_object_mut()?;
            if strip_nvext {
                obj.remove("nvext");
            }
            if let Some(ids) = token_ids {
                let nvext = obj
                    .entry("nvext")
                    .or_insert_with(|| serde_json::json!({}))
                    .as_object_mut()?;
                nvext.insert("token_data".to_string(), serde_json::json!(ids));
            }
            if let Some(salt) = cache_salt {
                obj.insert("cache_salt".to_string(), tagged_cache_salt(salt).into());
            }
            Some(parsed)
        }

        let bodies: [&[u8]; 7] = [
            br#"{}"#,
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            br#"{"model":"m","nvext":{"extra_fields":["engine_data"],"cache_salt":"body"}}"#,
            br#"{"model":"m","nvext":{"token_data":[7]},"cache_salt":"client"}"#,
            br#"{"model":"m","nvext":null}"#,
            br#"{"model":"m","nvext":"bad"}"#,
            br#"["not","an","object"]"#,
        ];
        let token_choices: [Option<&[u32]>; 3] = [None, Some(&[]), Some(&[5, 151_000, 0])];
        for body in bodies {
            for strip_nvext in [false, true] {
                for token_ids in token_choices {
                    for cache_salt in [None, Some("tenant")] {
                        let ours = rewrite_forwarded_body(body, strip_nvext, token_ids, cache_salt)
                            .ok()
                            .map(|out| serde_json::from_slice::<serde_json::Value>(&out).unwrap());
                        let tree = by_tree(body, strip_nvext, token_ids, cache_salt);
                        assert_eq!(
                            ours,
                            tree,
                            "body={} strip={strip_nvext} tokens={token_ids:?} salt={cache_salt:?}",
                            String::from_utf8_lossy(body)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rewrite_forwarded_body_rejects_non_object_body() {
        for body in [br#"["not", "an", "object"]"#.as_slice(), br#"not json"#] {
            assert!(rewrite_forwarded_body(body, true, None, None).is_err());
            assert!(rewrite_forwarded_body(body, false, Some(&[1]), None).is_err());
        }
    }

    #[test]
    fn stream_trace_counts_response_chunk_gaps() {
        let ms = Duration::from_millis;
        let start = Instant::now();
        let mut trace = StreamTrace::default();
        for at in [0, 10, 60] {
            trace.record_chunk(start + ms(at));
        }
        assert_eq!(trace.chunks, 3);
        assert_eq!(trace.max_chunk_gap, ms(50));
        assert_eq!(trace.notable_chunk_gaps, 1);
        assert!(!trace.is_slow(None));

        trace.record_chunk(start + ms(170));
        assert!(
            trace.is_slow(None),
            "a 110 ms hold between chunks is a stall"
        );
    }

    #[test]
    fn slow_forward_path_or_pick_marks_the_stream_slow() {
        let ms = Duration::from_millis;
        let trace = |body_eos, routed, response_headers| StreamTrace {
            body_eos: Some(ms(body_eos)),
            routed: Some(ms(routed)),
            response_headers: Some(ms(response_headers)),
            streamed: true,
            ..Default::default()
        };
        assert!(!trace(5, 20, 30).is_slow(None));
        assert!(
            trace(5, 20, 150).is_slow(None),
            "slow trip to the sidecar and back"
        );
        let unary = StreamTrace {
            streamed: false,
            ..trace(5, 20, 5_000)
        };
        assert!(
            !unary.is_slow(None),
            "a unary response's headers wait for the generation"
        );
        assert!(trace(5, 1_200, 1_210).is_slow(None), "slow pick");
        assert!(
            trace(1_100, 1_110, 1_120).is_slow(None),
            "slow request body"
        );
    }

    /// A pick that never finishes, because the gateway gave up on it or it
    /// failed, is as slow as the time it ran before the stream ended.
    #[test]
    fn an_unfinished_pick_marks_the_stream_slow() {
        let ms = Duration::from_millis;
        let waiting = StreamTrace {
            body_eos: Some(ms(5)),
            ..Default::default()
        };
        assert!(!waiting.is_slow(Some(ms(300))));
        assert!(
            waiting.is_slow(Some(ms(4_000))),
            "a pick cancelled after 4 s is slow"
        );
    }

    /// The first-token stages count only a 2xx response, or one whose status
    /// the gateway did not send.
    #[test]
    fn only_a_2xx_or_missing_status_is_a_success() {
        let mut ctx = RequestContext::new();
        assert!(ctx.response_succeeded());
        ctx.response_headers
            .insert(":status".to_string(), "200".to_string());
        assert!(ctx.response_succeeded());
        ctx.response_headers
            .insert(":status".to_string(), "503".to_string());
        assert!(!ctx.response_succeeded());
    }
}
