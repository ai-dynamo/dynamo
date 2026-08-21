// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the `EndpointPicker` trait and its associated types (`Endpoint`,
//! `RequestInfo`, `PickResult`, `PickError`). This mirrors the Go LW-EPP's
//! `EndpointPicker` interface from GAIE #2834. The ext_proc server is generic
//! over this trait — it handles the Envoy protocol, the picker handles the
//! routing decision.

use std::collections::HashMap;

use bytes::Bytes;

/// Endpoint represents a model server pod endpoint available for serving requests.
/// Mirrors Go `epplight.Endpoint` in pkg/lwepp/datastore/datastore.go
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Pod name
    pub pod_name: String,
    /// Pod IP address
    pub address: String,
    /// Target port
    pub port: String,
    /// Pod labels
    pub labels: HashMap<String, String>,
}

impl Endpoint {
    /// Returns the endpoint in "ip:port" format, bracketing IPv6 literals.
    pub fn address_port(&self) -> String {
        join_host_port(&self.address, &self.port)
    }
}

/// Join a host and a port into socket-address syntax.
///
/// An IPv6 literal has to be bracketed or the result is ambiguous: address
/// `fd00::10` with port `8000` concatenates to `fd00::10:8000`, which reads as
/// another IPv6 address rather than as a host and a port. Hosts that are
/// already bracketed are left alone, and IPv4 addresses and DNS names are
/// unaffected.
pub fn join_host_port(host: &str, port: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Extract the host from `host:port`, `[v6]:port`, or a bare host.
///
/// `split(':').next()` cannot do this: it truncates a bare IPv6 literal at its
/// first group, so `fd00::10` becomes `fd00`.
pub fn host_of(addr: &str) -> &str {
    if let Some(rest) = addr.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    // More than one colon and no brackets means a bare IPv6 literal, which
    // carries no port to strip.
    if addr.matches(':').count() > 1 {
        return addr;
    }
    addr.split(':').next().unwrap_or(addr)
}

/// RequestInfo contains metadata about the incoming HTTP request.
/// Mirrors Go `epplight.RequestInfo`.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Unique request ID (from `x-request-id` header or generated UUID).
    /// Used for router bookkeeping (add_request / free_request).
    pub request_id: String,
    /// HTTP request headers, preserved as ordered pairs so that repeated
    /// header keys (valid in HTTP) are not silently collapsed.
    pub headers: Vec<(String, String)>,
    /// Raw request body (empty for GET). `Bytes` so it can be shared with the
    /// tokenizer/renderer and the forwarding path by cheap refcount clones rather
    /// than copied; a fresh allocation is only needed when the body is mutated.
    pub body: Bytes,
    /// Model name extracted from the request body
    pub model: String,
    /// From x-gateway-destination-endpoint-subset metadata
    pub candidate_subset: Vec<String>,
}

/// PickResult contains the endpoint selection result.
/// Mirrors Go `epplight.PickResult`, extended with Dynamo-specific
/// routing headers that the backend workers need.
#[derive(Debug, Clone, Default)]
pub struct PickResult {
    /// Primary endpoint in "ip:port" format
    pub endpoint: String,
    /// Optional fallback endpoints in "ip:port" format
    pub fallbacks: Vec<String>,
    /// Extra headers to inject into the forwarded request.
    /// Used by Dynamo for routing metadata (worker IDs, DP ranks, routing mode).
    pub headers: Vec<(String, String)>,
    /// Pre-computed token IDs from the picker's tokenization.
    /// Injected into the request body as `nvext.token_data` so the backend
    /// skips redundant tokenization. Mirrors Go EPP's `setTokenizedPrompt`.
    pub token_ids: Option<Vec<u32>>,
    /// Booking id the picker recorded for this request's load reservation, if
    /// any. The server carries it on the per-stream context and hands it back to
    /// [`EndpointPicker::on_prefill_complete`] / [`EndpointPicker::on_request_complete`]
    /// so the picker can free the exact reservation for this stream without a
    /// shared, request-id-keyed lookup. `None` when the picker booked nothing.
    pub reservation_id: Option<String>,
}

/// EndpointPicker is the central abstraction for endpoint selection.
/// Mirrors Go `epplight.EndpointPicker` interface.
///
/// Implementations receive request metadata and a list of available endpoints,
/// and return the chosen endpoint(s). The ext_proc server handles all Envoy
/// protocol details, subset filtering, and pod discovery.
#[tonic::async_trait]
pub trait EndpointPicker: Send + Sync + 'static {
    async fn pick(
        &self,
        req: &RequestInfo,
        endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError>;

    /// Called when the first response body arrives from the backend, signalling
    /// prefill is done and decode has started. `booking_id` is the
    /// [`PickResult::reservation_id`] this request returned, or its request id if
    /// the picker booked nothing. Mirrors Go EPP's PostResponse → MarkPrefillComplete.
    async fn on_prefill_complete(&self, _booking_id: &str) {}

    /// Called when a request's response is fully complete (end-of-stream on the
    /// response body or trailers). Lets the picker free bookkeeping state.
    /// `booking_id` is as in [`Self::on_prefill_complete`]. Mirrors Go EPP's
    /// PostResponse → FreeRequest. Prefer
    /// [`Self::on_request_complete_with_usage`] when usage is needed.
    async fn on_request_complete(&self, _booking_id: &str) {}

    /// Like [`Self::on_request_complete`], with optional parsed token usage.
    /// Defaults to forwarding to [`Self::on_request_complete`].
    async fn on_request_complete_with_usage(
        &self,
        booking_id: &str,
        _usage: Option<ResponseUsage>,
    ) {
        self.on_request_complete(booking_id).await;
    }
}

/// Token usage from the terminal response (`None` fields may be omitted).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.cached_tokens`
    pub cached_tokens: Option<u64>,
}

/// Error from an endpoint picker. Variants map to distinct HTTP statuses at the
/// ext_proc boundary (see `server.rs::from_pick_error`), so upstream failures are
/// not mislabelled as client errors. Messages are client-safe; detailed causes
/// (which may include upstream internals) are logged, not returned to the client.
#[derive(Debug, thiserror::Error)]
pub enum PickError {
    #[error("no endpoints available")]
    NoEndpoints,
    #[error("routing failed: {0}")]
    RoutingFailed(String),
    /// Malformed client input (unparseable body, or a 4xx from the tokenizer) → 400.
    #[error("tokenization failed: {0}")]
    TokenizationFailed(String),
    /// The upstream tokenization service could not be reached → 503.
    #[error("tokenization service unavailable")]
    TokenizerUnavailable,
    /// The upstream tokenization service did not respond in time → 504.
    #[error("tokenization service timed out")]
    TokenizerTimeout,
    /// The upstream tokenization service returned a 5xx or invalid response → 502.
    #[error("tokenization service error")]
    TokenizerUpstreamError,
    /// The in-flight-request limit is saturated: the request is shed (not queued)
    /// as retryable backpressure → 503. A load-shed guardrail, since HTTP/2 stream
    /// multiplexing means the connection cap does not bound concurrent requests.
    #[error("endpoint picker overloaded")]
    Overloaded,
}

#[cfg(test)]
mod addr_tests {
    use super::{host_of, join_host_port};

    #[test]
    fn ipv6_endpoints_are_bracketed() {
        // Without brackets this is "fd00::10:8000", which reads as another
        // IPv6 address rather than a host and a port.
        assert_eq!(join_host_port("fd00::10", "8000"), "[fd00::10]:8000");
    }

    #[test]
    fn ipv4_and_hostnames_are_unchanged() {
        assert_eq!(join_host_port("10.0.0.1", "8000"), "10.0.0.1:8000");
        assert_eq!(join_host_port("worker.svc", "8000"), "worker.svc:8000");
    }

    #[test]
    fn an_already_bracketed_host_is_not_double_bracketed() {
        assert_eq!(join_host_port("[fd00::10]", "8000"), "[fd00::10]:8000");
    }

    #[test]
    fn host_of_handles_every_subset_form() {
        // The two forms a subset hint can carry for one IPv6 worker.
        assert_eq!(host_of("[fd00::10]:8000"), "fd00::10");
        assert_eq!(host_of("fd00::10"), "fd00::10");
        // And the forms that already worked.
        assert_eq!(host_of("10.0.0.1:8000"), "10.0.0.1");
        assert_eq!(host_of("10.0.0.1"), "10.0.0.1");
        assert_eq!(host_of("worker.svc:8000"), "worker.svc");
    }

    #[test]
    fn round_trip_recovers_the_host() {
        for host in ["fd00::10", "10.0.0.1", "worker.svc"] {
            assert_eq!(host_of(&join_host_port(host, "8000")), host);
        }
    }
}
