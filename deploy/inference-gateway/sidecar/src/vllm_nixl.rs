// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw-vLLM NIXL disaggregated P/D adapter.
//!
//! Implements the [`PdAdapter`] contract for the standalone decode sidecar when
//! the prefill and decode workers are plain vLLM OpenAI servers running the
//! NIXL pull connector. The adapter orchestrates three steps and nothing else:
//!
//! ```text
//! original request
//!   -> derived prefill-only request  -> selected prefill worker
//!   <- kv_transfer_params handoff
//!   -> original request + handoff    -> fixed local decode worker
//! ```
//!
//! KV bytes never pass through here. The two workers transfer them over NIXL
//! directly; the adapter only relays the connector metadata that tells the
//! decode worker where to pull from.
//!
//! # Supported protocol
//!
//! The handoff shape is pinned to **vLLM v0.29.0**, where the prefill worker
//! returns `kv_transfer_params` from
//! `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py`
//! (`NixlPullConnectorScheduler::request_finished`):
//!
//! | Field | Type | Notes |
//! |---|---|---|
//! | `do_remote_prefill` | bool | `true` on the handoff, meaning "the decode side pulls" |
//! | `do_remote_decode` | bool | `false` on the handoff |
//! | `remote_block_ids` | array of arrays | per KV cache group; may contain empty groups |
//! | `remote_engine_id` | string | producer engine id |
//! | `remote_request_id` | string | producer-side request id |
//! | `remote_host` | string | side-channel host, **not** an HTTP destination |
//! | `remote_port` | int | side-channel port |
//! | `tp_size`, `dcp_size`, `pp_size` | int | producer parallel sizes |
//! | `remote_num_tokens` | int | tokens the producer actually computed |
//! | `remote_blocks_expiry_time` | float or null | block lease expiry |
//! | `transfer_mode` | string | connector transfer mode |
//!
//! Unknown fields are preserved verbatim: the connector is the authority on its
//! own metadata, so a newer vLLM adding a field must still work. An empty
//! `remote_block_ids` (including empty inner groups) is a **valid** handoff — it
//! is what a full local prefix-cache hit looks like — and is not treated as a
//! missing handoff. See the module tests for the fixtures.
//!
//! # Deliberate limits
//!
//! * The decode destination is the locally configured engine. `remote_host` and
//!   `remote_port` are connector side-channel information and never become the
//!   HTTP destination.
//! * `n != 1` is rejected before the prefill leg rather than silently
//!   rewritten, because the protocol does not define how multiple handoffs
//!   would be matched to the returned choices.
//! * A client-supplied non-empty `kv_transfer_params` is rejected: the adapter
//!   owns that field on this path.
//! * Neither leg is retried. A retry could duplicate generation or leak the
//!   producer's block lease, so retries need their own idempotency design.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::error::SidecarError;
use crate::metadata::PrefillEndpoint;
use crate::proxy::target_url;
use crate::server::PdAdapter;

/// Environment variable selecting the P/D adapter. `none` (the default) keeps
/// the historical `UnavailablePdAdapter`, so a request carrying EPP P/D
/// metadata still fails with `pd_adapter_unavailable` until an operator opts in.
pub const ADAPTER_ENV: &str = "DYN_SIDECAR_PD_ADAPTER";

/// Environment variable overriding the assumed vLLM NIXL protocol revision.
///
/// The adapter asserts this against the revision it was compiled for and
/// refuses to start on a mismatch, so an operator cannot silently run a
/// different protocol than the one the fixtures describe.
pub const PROTOCOL_VERSION_ENV: &str = "DYN_VLLM_NIXL_PROTOCOL_VERSION";

/// vLLM release whose NIXL pull handoff shape this adapter implements.
pub const SUPPORTED_VLLM_VERSION: &str = "v0.29.0";

/// Protocol revision string asserted at startup. Kept separate from
/// [`SUPPORTED_VLLM_VERSION`] so a compatible patch release can be allowed
/// without pretending the fixtures were revalidated.
pub const SUPPORTED_PROTOCOL_VERSION: &str = "vllm-v0.29.0-nixl-pull";

/// Environment variable overriding the maximum accepted request body size.
pub const MAX_REQUEST_BYTES_ENV: &str = "DYN_SIDECAR_MAX_REQUEST_BYTES";

/// Environment variable overriding the maximum accepted prefill response size.
pub const MAX_PREFILL_RESPONSE_BYTES_ENV: &str = "DYN_SIDECAR_MAX_PREFILL_RESPONSE_BYTES";

/// Default request body cap: 32 MiB, generous enough for an inline multimodal
/// request while still bounding what one stream can buffer.
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;

/// Default prefill response cap: 1 MiB. The response is a one-token completion
/// plus the handoff, so this is orders of magnitude of headroom.
pub const DEFAULT_MAX_PREFILL_RESPONSE_BYTES: usize = 1024 * 1024;

/// Finite error codes this adapter returns. Each maps to a stable HTTP status
/// in [`SidecarError::adapter`].
pub mod code {
    /// The PD request body was not a JSON object.
    pub const INVALID_PD_REQUEST: &str = "invalid_pd_request";
    /// The request asked for something the protocol cannot express.
    pub const UNSUPPORTED_PD_VARIANT: &str = "unsupported_pd_variant";
    /// The request body exceeded the configured cap.
    pub const PD_REQUEST_TOO_LARGE: &str = "pd_request_too_large";
    /// The prefill worker returned a successful HTTP response without a usable
    /// handoff.
    pub const INVALID_PREFILL_HANDOFF: &str = "invalid_prefill_handoff";
    /// The prefill response exceeded the configured cap.
    pub const PREFILL_RESPONSE_TOO_LARGE: &str = "prefill_response_too_large";
    /// The prefill worker could not be reached.
    pub const PREFILL_UPSTREAM_UNAVAILABLE: &str = "prefill_upstream_unavailable";
    /// The prefill worker timed out.
    pub const PREFILL_UPSTREAM_TIMEOUT: &str = "prefill_upstream_timeout";
    /// The prefill worker returned an HTTP error.
    pub const PREFILL_UPSTREAM_ERROR: &str = "prefill_upstream_error";
}

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Model name the workers serve. Recorded for the startup log and used to
    /// assert that both legs describe the same model.
    pub model: String,
    /// Maximum accepted request body size in bytes.
    pub max_request_bytes: usize,
    /// Maximum accepted prefill response size in bytes.
    pub max_prefill_response_bytes: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_prefill_response_bytes: DEFAULT_MAX_PREFILL_RESPONSE_BYTES,
        }
    }
}

/// The adapter. Holds the HTTP client, the fixed decode destination, and the
/// protocol configuration.
pub struct VllmNixlAdapter {
    client: Client,
    decode_engine_url: Url,
    config: Config,
}

impl VllmNixlAdapter {
    /// Build an adapter that always dispatches its decode leg to
    /// `decode_engine_url`.
    ///
    /// The client mirrors the sidecar's existing forwarding configuration:
    /// environment proxies and redirects are disabled so a request is never
    /// re-routed by ambient configuration.
    pub fn new(
        decode_engine_url: Url,
        connect_timeout: Duration,
        read_timeout: Duration,
        config: Config,
    ) -> Result<Arc<Self>, reqwest::Error> {
        Ok(Arc::new(Self {
            client: Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(connect_timeout)
                .read_timeout(read_timeout)
                .build()?,
            decode_engine_url,
            config,
        }))
    }

    /// The protocol revision this build implements.
    pub fn protocol_version() -> &'static str {
        SUPPORTED_PROTOCOL_VERSION
    }

    /// Assert an operator-provided protocol version against this build.
    pub fn check_protocol_version(configured: &str) -> Result<(), String> {
        if configured == SUPPORTED_PROTOCOL_VERSION {
            Ok(())
        } else {
            Err(format!(
                "{PROTOCOL_VERSION_ENV}={configured} does not match the protocol this build \
                 implements ({SUPPORTED_PROTOCOL_VERSION}, vLLM {SUPPORTED_VLLM_VERSION}). \
                 Regenerate the protocol fixtures before changing it."
            ))
        }
    }
}

#[async_trait]
impl PdAdapter for VllmNixlAdapter {
    async fn execute(
        &self,
        request: Request<Body>,
        prefill_endpoint: PrefillEndpoint,
        cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        let (parts, body) = request.into_parts();
        let body = read_bounded(body, self.config.max_request_bytes)
            .await
            .map_err(|error| match error {
                BoundedRead::TooLarge => SidecarError::adapter(
                    axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                    code::PD_REQUEST_TOO_LARGE,
                    format!(
                        "Request body exceeds the configured {} byte limit",
                        self.config.max_request_bytes
                    ),
                ),
                BoundedRead::Read => SidecarError::adapter(
                    axum::http::StatusCode::BAD_REQUEST,
                    code::INVALID_PD_REQUEST,
                    "Could not read the request body",
                ),
            })?;

        let original: Value = serde_json::from_slice(&body).map_err(|_| {
            SidecarError::adapter(
                axum::http::StatusCode::BAD_REQUEST,
                code::INVALID_PD_REQUEST,
                "The P/D request body must be a JSON object",
            )
        })?;

        let prefill_request = prepare_prefill_request(&original)?;

        let prefill_url = prefill_url(&prefill_endpoint, &parts.uri)?;
        let prefill_response = tokio::select! {
            response = self.send_prefill(&prefill_url, &prefill_request, &parts.headers) => response?,
            () = cancellation.cancelled() => return Err(SidecarError::Cancelled),
        };

        let handoff = validate_prefill_handoff(prefill_response)?;
        let decode_request = prepare_decode_request(&original, handoff)?;

        let decode_url = target_url(&self.decode_engine_url, &parts.uri);
        self.send_decode(&decode_url, &decode_request, &parts.headers, &cancellation)
            .await
    }
}

impl VllmNixlAdapter {
    /// Send the derived prefill request and return the handoff object.
    async fn send_prefill(
        &self,
        url: &Url,
        prefill_request: &Value,
        client_headers: &axum::http::HeaderMap,
    ) -> Result<Value, SidecarError> {
        let mut headers = client_headers.clone();
        crate::proxy::strip_proxy_headers(&mut headers);
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );

        let response = self
            .client
            .post(url.clone())
            .headers(headers)
            .json(prefill_request)
            .send()
            .await
            .map_err(|error| prefill_transport_error(&error))?;

        let status = response.status();
        if !status.is_success() {
            // The prefill stage failed, so the decode leg must not run. The
            // upstream status is preserved, but its body is not forwarded: it
            // may quote the request.
            return Err(SidecarError::adapter(
                status,
                code::PREFILL_UPSTREAM_ERROR,
                "The prefill worker rejected the request",
            ));
        }

        let body = read_bounded_response(response, self.config.max_prefill_response_bytes)
            .await
            .map_err(|error| match error {
                BoundedRead::TooLarge => SidecarError::adapter(
                    axum::http::StatusCode::BAD_GATEWAY,
                    code::PREFILL_RESPONSE_TOO_LARGE,
                    format!(
                        "Prefill response exceeds the configured {} byte limit",
                        self.config.max_prefill_response_bytes
                    ),
                ),
                BoundedRead::Read => SidecarError::adapter(
                    axum::http::StatusCode::BAD_GATEWAY,
                    code::INVALID_PREFILL_HANDOFF,
                    "Could not read the prefill response",
                ),
            })?;

        serde_json::from_slice(&body).map_err(|_| {
            SidecarError::adapter(
                axum::http::StatusCode::BAD_GATEWAY,
                code::INVALID_PREFILL_HANDOFF,
                "The prefill worker returned a non-JSON response",
            )
        })
    }

    /// Forward the original request plus the validated handoff to the fixed
    /// local decode engine.
    async fn send_decode(
        &self,
        url: &Url,
        decode_request: &Value,
        client_headers: &axum::http::HeaderMap,
        cancellation: &CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        let mut headers = client_headers.clone();
        crate::proxy::strip_proxy_headers(&mut headers);
        headers.remove(axum::http::header::CONTENT_LENGTH);
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );

        let request = self
            .client
            .post(url.clone())
            .headers(headers)
            .json(decode_request);

        let upstream = tokio::select! {
            response = request.send() => response.map_err(SidecarError::DecodeUpstream)?,
            () = cancellation.cancelled() => return Err(SidecarError::Cancelled),
        };

        let status = upstream.status();
        let mut response_headers = upstream.headers().clone();
        crate::proxy::strip_proxy_headers(&mut response_headers);
        let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
        *response.status_mut() = status;
        *response.headers_mut() = response_headers;
        Ok(response)
    }
}

/// Outcome of a bounded body read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedRead {
    TooLarge,
    Read,
}

/// Read a request body with a hard cap, so one stream cannot buffer without
/// limit. The cap is checked while reading, not after.
async fn read_bounded(body: Body, limit: usize) -> Result<Bytes, BoundedRead> {
    use futures::StreamExt;

    let mut stream = body.into_data_stream();
    let mut collected: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| BoundedRead::Read)?;
        if collected.len().saturating_add(chunk.len()) > limit {
            return Err(BoundedRead::TooLarge);
        }
        collected.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(collected))
}

/// Read an upstream response with a hard cap. `Content-Length` is checked first
/// so an obviously oversized response is rejected without buffering it.
async fn read_bounded_response(
    response: reqwest::Response,
    limit: usize,
) -> Result<Bytes, BoundedRead> {
    if let Some(length) = response.content_length()
        && length > limit as u64
    {
        return Err(BoundedRead::TooLarge);
    }
    read_bounded(Body::from_stream(response.bytes_stream()), limit).await
}

/// Map a prefill transport failure to a prefill-stage error, so it is never
/// reported as a decode failure.
fn prefill_transport_error(error: &reqwest::Error) -> SidecarError {
    if error.is_timeout() {
        SidecarError::adapter(
            axum::http::StatusCode::GATEWAY_TIMEOUT,
            code::PREFILL_UPSTREAM_TIMEOUT,
            "The prefill worker timed out",
        )
    } else {
        SidecarError::adapter(
            axum::http::StatusCode::BAD_GATEWAY,
            code::PREFILL_UPSTREAM_UNAVAILABLE,
            "The prefill worker could not be reached",
        )
    }
}

/// Build the prefill leg URL from the EPP-selected endpoint and the original
/// request URI. The request path and query are preserved, matching the
/// passthrough contract for the decode leg.
pub fn prefill_url(
    prefill_endpoint: &PrefillEndpoint,
    request_uri: &axum::http::Uri,
) -> Result<Url, SidecarError> {
    let base = Url::parse(&format!("http://{}", prefill_endpoint.authority())).map_err(|_| {
        SidecarError::adapter(
            axum::http::StatusCode::BAD_GATEWAY,
            code::INVALID_PD_REQUEST,
            "The selected prefill endpoint is not a usable HTTP authority",
        )
    })?;
    Ok(target_url(&base, request_uri))
}

/// Derive the prefill-only request from the original request.
///
/// The original JSON is preserved field for field except for the documented
/// prefill-leg rewrites, so `tools`, `response_format`, sampling parameters,
/// and unknown vendor extensions all reach the prefill worker unchanged.
pub fn prepare_prefill_request(original: &Value) -> Result<Value, SidecarError> {
    let request = original.as_object().ok_or_else(|| {
        SidecarError::adapter(
            axum::http::StatusCode::BAD_REQUEST,
            code::INVALID_PD_REQUEST,
            "The P/D request body must be a JSON object",
        )
    })?;

    reject_client_handoff(request)?;
    reject_unsupported_variants(request)?;

    let mut prefill = request.clone();
    // The prefill worker must not stream, must emit exactly one token, and must
    // be told this is the producer side of a disaggregated pair.
    prefill.insert("stream".to_string(), Value::Bool(false));
    prefill.insert("max_tokens".to_string(), json!(1));
    if prefill.contains_key("max_completion_tokens") {
        prefill.insert("max_completion_tokens".to_string(), json!(1));
    }
    // Streaming options and minimum-length constraints describe the caller's
    // streaming response and would contradict a one-token prefill.
    prefill.remove("stream_options");
    prefill.remove("min_tokens");
    prefill.remove("min_completion_tokens");

    prefill.insert(
        "kv_transfer_params".to_string(),
        json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": null,
            "remote_block_ids": null,
            "remote_host": null,
            "remote_port": null,
        }),
    );
    Ok(Value::Object(prefill))
}

/// Reject a client that tries to drive the internal handoff field.
fn reject_client_handoff(request: &Map<String, Value>) -> Result<(), SidecarError> {
    match request.get("kv_transfer_params") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(fields)) if fields.is_empty() => Ok(()),
        Some(_) => Err(SidecarError::adapter(
            axum::http::StatusCode::BAD_REQUEST,
            code::INVALID_PD_REQUEST,
            "kv_transfer_params is reserved for the P/D adapter and must not be set by a client",
        )),
    }
}

/// Reject request variants whose P/D behaviour the pinned protocol does not
/// define, before any prefill work is dispatched.
fn reject_unsupported_variants(request: &Map<String, Value>) -> Result<(), SidecarError> {
    match request.get("n") {
        None => Ok(()),
        Some(Value::Number(number)) if number.as_i64() == Some(1) => Ok(()),
        Some(_) => Err(SidecarError::adapter(
            axum::http::StatusCode::BAD_REQUEST,
            code::UNSUPPORTED_PD_VARIANT,
            "n must be 1: the P/D protocol defines a single handoff per request",
        )),
    }
}

/// Validate the prefill worker's handoff and return it for the decode leg.
///
/// The whole `kv_transfer_params` object is returned, not a rebuilt subset, so
/// connector fields this build does not know about still reach the decode
/// worker.
pub fn validate_prefill_handoff(response: Value) -> Result<Value, SidecarError> {
    let response = response
        .as_object()
        .ok_or_else(|| invalid_handoff("response is not an object"))?;

    // A prefill response is a completion, but it is generated only to produce
    // the handoff. It must never be shown to the caller.
    let handoff = response
        .get("kv_transfer_params")
        .ok_or_else(|| invalid_handoff("response has no kv_transfer_params"))?;
    let handoff = handoff
        .as_object()
        .ok_or_else(|| invalid_handoff("kv_transfer_params is not an object"))?;

    let do_remote_prefill = handoff
        .get("do_remote_prefill")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid_handoff("do_remote_prefill is missing or not a boolean"))?;
    if !do_remote_prefill {
        return Err(invalid_handoff(
            "do_remote_prefill is false, so the producer did not stage a handoff for this side",
        ));
    }

    match handoff.get("do_remote_decode").and_then(Value::as_bool) {
        Some(false) => {}
        Some(true) => {
            return Err(invalid_handoff(
                "do_remote_decode is true, which is the producer-side flag",
            ));
        }
        None => {
            return Err(invalid_handoff(
                "do_remote_decode is missing or not a boolean",
            ));
        }
    }

    // Present-and-null is a missing handoff. An empty array is not: a
    // full prefix-cache hit on the decode side legitimately transfers no
    // blocks, and the connector still needs the notification.
    let blocks = handoff
        .get("remote_block_ids")
        .ok_or_else(|| invalid_handoff("remote_block_ids is missing"))?;
    validate_block_ids(blocks)?;

    for field in ["remote_engine_id", "remote_request_id"] {
        match handoff.get(field) {
            Some(Value::String(value)) if !value.is_empty() => {}
            _ => return Err(invalid_handoff(&format!("{field} is missing or empty"))),
        }
    }
    match handoff.get("remote_host") {
        Some(Value::String(value)) if !value.is_empty() => {}
        _ => return Err(invalid_handoff("remote_host is missing or empty")),
    }
    match handoff.get("remote_port").and_then(Value::as_u64) {
        Some(port) if port > 0 && port <= u64::from(u16::MAX) => {}
        _ => {
            return Err(invalid_handoff(
                "remote_port is missing or not a valid port",
            ));
        }
    }

    Ok(Value::Object(handoff.clone()))
}

/// `remote_block_ids` is a per-KV-cache-group list of lists. The nesting is
/// preserved exactly; flattening it would make the decode worker pull from the
/// wrong groups.
fn validate_block_ids(blocks: &Value) -> Result<(), SidecarError> {
    let groups = blocks
        .as_array()
        .ok_or_else(|| invalid_handoff("remote_block_ids is not an array"))?;
    for group in groups {
        let ids = group
            .as_array()
            .ok_or_else(|| invalid_handoff("a remote_block_ids group is not an array"))?;
        for id in ids {
            if !id.is_u64() {
                return Err(invalid_handoff(
                    "a remote_block_ids entry is not a non-negative integer",
                ));
            }
        }
    }
    Ok(())
}

fn invalid_handoff(reason: &str) -> SidecarError {
    // The reason is a fixed phrase about the shape of the metadata, never
    // request content, so it is safe to return to the caller.
    SidecarError::adapter(
        axum::http::StatusCode::BAD_GATEWAY,
        code::INVALID_PREFILL_HANDOFF,
        format!("Invalid prefill handoff: {reason}"),
    )
}

/// Build the decode request: the untouched original request plus the validated
/// backend handoff.
///
/// The derived prefill request is deliberately not the base here — it carries
/// `stream = false`, `max_tokens = 1`, and none of the caller's
/// `stream_options`, all of which the decode leg must not inherit.
pub fn prepare_decode_request(original: &Value, handoff: Value) -> Result<Value, SidecarError> {
    let mut decode = original.as_object().cloned().ok_or_else(|| {
        SidecarError::adapter(
            axum::http::StatusCode::BAD_REQUEST,
            code::INVALID_PD_REQUEST,
            "The P/D request body must be a JSON object",
        )
    })?;
    decode.insert("kv_transfer_params".to_string(), handoff);
    Ok(Value::Object(decode))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handoff as vLLM v0.29.0 `NixlPullConnectorScheduler::request_finished`
    /// emits it, with two KV cache groups.
    fn valid_handoff() -> Value {
        json!({
            "do_remote_prefill": true,
            "do_remote_decode": false,
            "remote_block_ids": [[1, 2, 3], [7]],
            "remote_engine_id": "engine-abc",
            "remote_request_id": "req-123",
            "remote_host": "10.0.0.5",
            "remote_port": 5600,
            "tp_size": 1,
            "dcp_size": 1,
            "pp_size": 1,
            "remote_num_tokens": 128,
            "remote_blocks_expiry_time": null,
            "transfer_mode": "pull"
        })
    }

    fn prefill_response_with(handoff: Value) -> Value {
        json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "Qwen/Qwen3-0.6B",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 128, "completion_tokens": 1, "total_tokens": 129},
            "kv_transfer_params": handoff
        })
    }

    fn original_request() -> Value {
        json!({
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 64,
            "min_tokens": 4,
            "temperature": 0.7,
            "tools": [{"type": "function", "function": {"name": "get_time"}}]
        })
    }

    #[test]
    fn prefill_request_is_non_streaming_and_one_token() {
        let prefill = prepare_prefill_request(&original_request()).unwrap();
        assert_eq!(prefill["stream"], json!(false));
        assert_eq!(prefill["max_tokens"], json!(1));
        assert!(prefill.get("stream_options").is_none());
        assert!(prefill.get("min_tokens").is_none());
    }

    #[test]
    fn prefill_request_sets_producer_side_flags() {
        let prefill = prepare_prefill_request(&original_request()).unwrap();
        let params = &prefill["kv_transfer_params"];
        assert_eq!(params["do_remote_decode"], json!(true));
        assert_eq!(params["do_remote_prefill"], json!(false));
        assert!(params["remote_block_ids"].is_null());
    }

    #[test]
    fn prefill_request_preserves_everything_else() {
        let original = original_request();
        let prefill = prepare_prefill_request(&original).unwrap();
        assert_eq!(prefill["model"], original["model"]);
        assert_eq!(prefill["messages"], original["messages"]);
        assert_eq!(prefill["tools"], original["tools"]);
        assert_eq!(prefill["temperature"], original["temperature"]);
    }

    #[test]
    fn prefill_request_handles_max_completion_tokens() {
        let original = json!({"model": "m", "max_completion_tokens": 99});
        let prefill = prepare_prefill_request(&original).unwrap();
        assert_eq!(prefill["max_completion_tokens"], json!(1));

        let original_without = json!({"model": "m"});
        let prefill = prepare_prefill_request(&original_without).unwrap();
        assert!(prefill.get("max_completion_tokens").is_none());
    }

    #[test]
    fn decode_request_restores_the_original_generation_parameters() {
        let original = original_request();
        let handoff = validate_prefill_handoff(prefill_response_with(valid_handoff())).unwrap();
        let decode = prepare_decode_request(&original, handoff).unwrap();

        assert_eq!(decode["stream"], json!(true));
        assert_eq!(decode["max_tokens"], json!(64));
        assert_eq!(decode["min_tokens"], json!(4));
        assert_eq!(decode["stream_options"], original["stream_options"]);
        assert_eq!(decode["tools"], original["tools"]);
    }

    #[test]
    fn decode_request_carries_the_backend_handoff_verbatim() {
        let handoff = valid_handoff();
        let decoded = validate_prefill_handoff(prefill_response_with(handoff.clone())).unwrap();
        let decode = prepare_decode_request(&original_request(), decoded).unwrap();
        assert_eq!(decode["kv_transfer_params"], handoff);
    }

    #[test]
    fn handoff_preserves_unknown_and_extended_fields() {
        let mut handoff = valid_handoff();
        handoff["future_field"] = json!({"nested": [1, 2]});
        let decoded = validate_prefill_handoff(prefill_response_with(handoff.clone())).unwrap();
        assert_eq!(decoded["future_field"], json!({"nested": [1, 2]}));
        assert_eq!(decoded["tp_size"], json!(1));
        assert_eq!(decoded["transfer_mode"], json!("pull"));
    }

    #[test]
    fn handoff_preserves_grouped_block_structure() {
        let mut handoff = valid_handoff();
        handoff["remote_block_ids"] = json!([[1, 2], [], [9, 10, 11]]);
        let decoded = validate_prefill_handoff(prefill_response_with(handoff)).unwrap();
        assert_eq!(
            decoded["remote_block_ids"],
            json!([[1, 2], [], [9, 10, 11]])
        );
    }

    #[test]
    fn empty_block_groups_are_a_valid_handoff() {
        for blocks in [json!([]), json!([[]]), json!([[], []])] {
            let mut handoff = valid_handoff();
            handoff["remote_block_ids"] = blocks.clone();
            assert!(
                validate_prefill_handoff(prefill_response_with(handoff)).is_ok(),
                "{blocks} must be accepted: a full local cache hit transfers no blocks"
            );
        }
    }

    #[test]
    fn missing_handoff_is_rejected() {
        let response = json!({"choices": [], "usage": {}});
        assert!(validate_prefill_handoff(response).is_err());
    }

    #[test]
    fn null_handoff_is_rejected() {
        let response = json!({"choices": [], "kv_transfer_params": null});
        assert!(validate_prefill_handoff(response).is_err());
    }

    #[test]
    fn non_object_handoff_is_rejected() {
        for handoff in [json!("pull"), json!(7), json!([1, 2])] {
            assert!(validate_prefill_handoff(prefill_response_with(handoff)).is_err());
        }
    }

    #[test]
    fn producer_side_direction_flags_are_rejected() {
        let mut handoff = valid_handoff();
        handoff["do_remote_prefill"] = json!(false);
        assert!(validate_prefill_handoff(prefill_response_with(handoff)).is_err());

        let mut handoff = valid_handoff();
        handoff["do_remote_decode"] = json!(true);
        assert!(validate_prefill_handoff(prefill_response_with(handoff)).is_err());

        let mut handoff = valid_handoff();
        handoff.as_object_mut().unwrap().remove("do_remote_decode");
        assert!(validate_prefill_handoff(prefill_response_with(handoff)).is_err());
    }

    #[test]
    fn malformed_block_ids_are_rejected() {
        for blocks in [
            json!(1),
            json!("1,2"),
            json!([[1, "2"]]),
            json!([[1], 3]),
            json!([null]),
        ] {
            let mut handoff = valid_handoff();
            handoff["remote_block_ids"] = blocks.clone();
            assert!(
                validate_prefill_handoff(prefill_response_with(handoff)).is_err(),
                "{blocks} must be rejected"
            );
        }
    }

    #[test]
    fn missing_connector_identity_is_rejected() {
        for field in [
            "remote_engine_id",
            "remote_request_id",
            "remote_host",
            "remote_port",
        ] {
            let mut handoff = valid_handoff();
            handoff.as_object_mut().unwrap().remove(field);
            assert!(
                validate_prefill_handoff(prefill_response_with(handoff)).is_err(),
                "missing {field} must be rejected"
            );
        }
    }

    #[test]
    fn invalid_ports_are_rejected() {
        for port in [json!(0), json!(70000), json!("5600"), json!(-1)] {
            let mut handoff = valid_handoff();
            handoff["remote_port"] = port.clone();
            assert!(
                validate_prefill_handoff(prefill_response_with(handoff)).is_err(),
                "port {port} must be rejected"
            );
        }
    }

    #[test]
    fn client_supplied_handoff_is_rejected() {
        for params in [json!({"do_remote_decode": true}), json!([1]), json!("x")] {
            let mut original = original_request();
            original["kv_transfer_params"] = params;
            assert!(prepare_prefill_request(&original).is_err());
        }
    }

    #[test]
    fn absent_or_null_or_empty_client_handoff_is_allowed() {
        for params in [Value::Null, json!({})] {
            let mut original = original_request();
            original["kv_transfer_params"] = params;
            assert!(prepare_prefill_request(&original).is_ok());
        }
        assert!(prepare_prefill_request(&original_request()).is_ok());
    }

    #[test]
    fn n_greater_than_one_is_rejected_not_rewritten() {
        let mut original = original_request();
        original["n"] = json!(2);
        let error = prepare_prefill_request(&original).unwrap_err();
        assert!(error.to_string().contains("n must be 1"));
    }

    #[test]
    fn n_equal_to_one_is_accepted() {
        let mut original = original_request();
        original["n"] = json!(1);
        let prefill = prepare_prefill_request(&original).unwrap();
        assert_eq!(prefill["n"], json!(1));
    }

    #[test]
    fn non_object_body_is_rejected() {
        assert!(prepare_prefill_request(&json!([1, 2, 3])).is_err());
        assert!(prepare_prefill_request(&json!("text")).is_err());
    }

    #[test]
    fn prefill_url_preserves_path_and_query() {
        let endpoint = "prefill.default.svc:8001"
            .parse::<axum::http::HeaderValue>()
            .unwrap();
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(crate::metadata::PREFILLER_HOST_PORT, endpoint);
        let endpoint = PrefillEndpoint::parse_headers(&headers).unwrap().unwrap();
        let uri: axum::http::Uri = "/v1/chat/completions?trace=true".parse().unwrap();

        assert_eq!(
            prefill_url(&endpoint, &uri).unwrap().as_str(),
            "http://prefill.default.svc:8001/v1/chat/completions?trace=true"
        );
    }

    #[test]
    fn prefill_url_brackets_ipv6() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            crate::metadata::PREFILLER_HOST_PORT,
            axum::http::HeaderValue::from_static("[2001:db8::10]:8001"),
        );
        let endpoint = PrefillEndpoint::parse_headers(&headers).unwrap().unwrap();
        let uri: axum::http::Uri = "/v1/chat/completions".parse().unwrap();

        assert_eq!(
            prefill_url(&endpoint, &uri).unwrap().as_str(),
            "http://[2001:db8::10]:8001/v1/chat/completions"
        );
    }

    #[test]
    fn protocol_version_mismatch_is_rejected() {
        assert!(VllmNixlAdapter::check_protocol_version(SUPPORTED_PROTOCOL_VERSION).is_ok());
        assert!(VllmNixlAdapter::check_protocol_version("vllm-v0.28.0-nixl-pull").is_err());
    }

    #[tokio::test]
    async fn request_body_cap_is_enforced_while_reading() {
        let body = Body::from(vec![b'x'; 64]);
        assert_eq!(read_bounded(body, 64).await.unwrap().len(), 64);

        let body = Body::from(vec![b'x'; 65]);
        assert_eq!(
            read_bounded(body, 64).await.unwrap_err(),
            BoundedRead::TooLarge
        );
    }
}
