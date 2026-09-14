// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines the `EndpointPicker` trait and its associated types (`Endpoint`,
//! `RequestInfo`, `PickResult`, `PickError`). The ext_proc server is generic
//! over this trait — it handles the Envoy protocol, the picker handles the
//! routing decision.

use std::collections::HashMap;

use bytes::Bytes;
use dynamo_llm::http::service::metadata::MetadataHeaderError;

use dynamo_llm::protocols::common::extensions::{HEADER_TENANT_ID, last_non_empty_trimmed_value};

/// A model server pod endpoint available for serving requests.
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
    /// Returns the endpoint in `host:port` format, bracketing IPv6 addresses.
    pub fn address_port(&self) -> String {
        match self.address.parse::<std::net::IpAddr>() {
            Ok(std::net::IpAddr::V6(address)) => format!("[{address}]:{}", self.port),
            _ => format!("{}:{}", self.address, self.port),
        }
    }
}

/// Metadata about the incoming HTTP request.
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

/// How `PickResult.cache_namespace` is written into the forwarded request body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheSaltForwarding {
    /// Leave the body's cache salt untouched. Dynamo-runtime backends encode
    /// it themselves.
    #[default]
    Preserve,
    /// Write `cache_salt = "dynamo-cache-salt:" + namespace` at the top level.
    /// Raw native vLLM has no Dynamo code in the request path to add the
    /// marker, so the EPP must add it.
    NativeVllm,
}

/// The endpoint selection result, with the Dynamo-specific routing headers
/// the backend workers need.
#[derive(Debug, Clone, Default)]
pub struct PickResult {
    /// Primary endpoint in "ip:port" format
    pub endpoint: String,
    /// Optional fallback endpoints in "ip:port" format
    pub fallbacks: Vec<String>,
    /// Extra headers to inject into the forwarded request.
    /// Used by Dynamo for routing metadata (worker IDs, DP ranks, routing mode).
    pub headers: Vec<(String, String)>,
    /// Callable prefill endpoint selected by standalone EPP.
    ///
    /// Runtime EPP carries its authoritative selection as worker identity and
    /// rank routing headers. Standalone EPP instead needs a callable host and
    /// port, so ext-proc emits `x-prefiller-host-port` from this field. Neither
    /// form accepts client input.
    pub selected_prefill_endpoint: Option<String>,
    /// Pre-computed token IDs from the picker's tokenization.
    /// Injected into the request body as `nvext.token_data` so the backend
    /// skips redundant tokenization.
    pub token_ids: Option<Vec<u32>>,
    /// Cache namespace used for selection. Whether and how it is written into
    /// the forwarded body is decided by `cache_salt_forwarding`.
    pub cache_namespace: Option<String>,
    /// Body-encoding policy for `cache_namespace`.
    pub cache_salt_forwarding: CacheSaltForwarding,
    /// Booking id the picker recorded for this request's load reservation, if
    /// any. The server carries it on the per-stream context and hands it back to
    /// [`EndpointPicker::on_prefill_complete`] / [`EndpointPicker::on_request_complete`]
    /// so the picker can free the exact reservation for this stream without a
    /// shared, request-id-keyed lookup. `None` when the picker booked nothing.
    pub reservation_id: Option<String>,
}

/// Resolve the request's cache namespace with the canonical precedence:
/// non-empty `x-tenant-id` header, then `nvext.cache_salt`, then top-level
/// `cache_salt`. Empty values count as absent.
pub fn resolve_cache_namespace(
    headers: &[(String, String)],
    nvext_cache_salt: Option<&str>,
    top_level_cache_salt: Option<&str>,
) -> Option<String> {
    last_non_empty_trimmed_value(
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case(HEADER_TENANT_ID))
            .map(|(_, value)| value.as_str()),
    )
    .map(str::to_owned)
    .or_else(|| non_empty_owned(nvext_cache_salt))
    .or_else(|| non_empty_owned(top_level_cache_salt))
}

fn non_empty_owned(value: Option<&str>) -> Option<String> {
    value.filter(|v| !v.is_empty()).map(str::to_owned)
}

/// The central abstraction for endpoint selection.
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
    /// the picker booked nothing.
    async fn on_prefill_complete(&self, _booking_id: &str) {}

    /// Called when a request's response is fully complete (end-of-stream on the
    /// response body or trailers). Lets the picker free bookkeeping state.
    /// `booking_id` is as in [`Self::on_prefill_complete`]. Prefer
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
    /// Malformed client input (unparseable body, or a 4xx from the renderer) → 400.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// Metadata headers exceeded the frontend's entry/size limits → 431, the
    /// same status the frontend returns.
    #[error("metadata headers too large: {0}")]
    MetadataHeadersTooLarge(MetadataHeaderError),
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
mod tests {
    use super::*;

    #[test]
    fn address_port_brackets_ipv6() {
        let endpoint = Endpoint {
            pod_name: "prefill-0".to_string(),
            address: "2001:db8::10".to_string(),
            port: "8001".to_string(),
            labels: Default::default(),
        };

        assert_eq!(endpoint.address_port(), "[2001:db8::10]:8001");
    }

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn resolve_cache_namespace_precedence() {
        for (name, request_headers, nvext_cache_salt, top_level_cache_salt, expected) in [
            (
                "tenant header wins over both body sources",
                headers(&[("X-Tenant-ID", "tenant-header")]),
                Some("nvext-salt"),
                Some("top-level-salt"),
                Some("tenant-header"),
            ),
            (
                "last non-empty trimmed tenant header wins",
                headers(&[
                    ("x-tenant-id", "tenant-client"),
                    ("X-Tenant-ID", "   "),
                    ("x-tenant-id", " tenant-gateway "),
                ]),
                Some("nvext-salt"),
                None,
                Some("tenant-gateway"),
            ),
            (
                "nvext cache salt wins over top level",
                vec![],
                Some("nvext-salt"),
                Some("top-level-salt"),
                Some("nvext-salt"),
            ),
            (
                "top-level cache salt is the fallback",
                vec![],
                None,
                Some("top-level-salt"),
                Some("top-level-salt"),
            ),
            (
                "empty values are absent",
                headers(&[("x-tenant-id", ""), ("X-Tenant-ID", "   ")]),
                Some(""),
                Some("top-level-salt"),
                Some("top-level-salt"),
            ),
            ("all sources absent", vec![], None, Some(""), None),
        ] {
            assert_eq!(
                resolve_cache_namespace(&request_headers, nvext_cache_salt, top_level_cache_salt,)
                    .as_deref(),
                expected,
                "{name}"
            );
        }
    }
}
