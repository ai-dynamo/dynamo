// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Decode-side P/D routing sidecar for **vanilla vLLM** disaggregation.
//!
//! The Dynamo EPP (router-only mode) makes the KV-aware prefill+decode decision
//! and emits the selected prefill pod's address as `x-prefiller-host-port`; the
//! gateway then forwards the request to the **decode** pod. This sidecar runs in
//! that decode pod, in front of the local vLLM, and performs vLLM's NIXL
//! `kv_transfer_params` handshake so stock `vllm serve` gets disaggregated
//! serving with no Dynamo worker/runtime:
//!
//! ```text
//! gateway ─▶ sidecar(:8000) ──(1) prefill, do_remote_decode──▶ prefiller vLLM (x-prefiller-host-port)
//!                           ◀── kv_transfer_params ───────────
//!                           ──(2) decode, reuse those params ─▶ local vLLM (:8001) ──▶ stream back
//! ```
//!
//! Pull-based NIXL only: the prefill request carries
//! `{do_remote_decode:true, ...}` and the decode request reuses the
//! `kv_transfer_params` returned on the prefill response (mirrors the
//! `NixlConnectorProtocol` in `components/src/dynamo/vllm/kv_connector_protocols.py`).
//!
//! Requests without `x-prefiller-host-port` (and all non-completion paths such
//! as `GET /v1/models`, `/health`) are transparently proxied to the local vLLM,
//! so the sidecar is a safe drop-in even for aggregated traffic.
//!
//! ## Configuration (env)
//! | var | default | meaning |
//! |-----|---------|---------|
//! | `DYN_PD_LISTEN_ADDR` | `0.0.0.0:8000` | address the sidecar binds (the pool targetPort) |
//! | `DYN_PD_DECODE_URL` | `http://127.0.0.1:8001` | local vLLM OpenAI base URL |
//! | `DYN_PD_PREFILLER_HEADER` | `x-prefiller-host-port` | header carrying the prefiller `host:port` |
//! | `DYN_PD_PREFILL_SCHEME` | `http` | scheme used to dial the prefiller |
//! | `DYN_PD_TIMEOUT_SECS` | `300` | per-request upstream timeout |
//!
//! NOTE: this is **vLLM-specific** and coupled to vLLM's `kv_transfer_params`
//! wire shape; pin the sidecar image to a compatible vLLM release.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

/// Cap request bodies the sidecar will buffer for the prefill/decode rewrite.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    /// Local vLLM OpenAI base URL (no trailing slash).
    decode_url: String,
    /// Lower-cased header name carrying the prefiller `host:port`.
    prefiller_header: String,
    /// Scheme used to dial the prefiller.
    prefill_scheme: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let listen_addr = env_or("DYN_PD_LISTEN_ADDR", "0.0.0.0:8000");
    let decode_url = env_or("DYN_PD_DECODE_URL", "http://127.0.0.1:8001")
        .trim_end_matches('/')
        .to_string();
    let prefiller_header =
        env_or("DYN_PD_PREFILLER_HEADER", "x-prefiller-host-port").to_lowercase();
    let prefill_scheme = env_or("DYN_PD_PREFILL_SCHEME", "http");
    let timeout_secs: u64 = env_or("DYN_PD_TIMEOUT_SECS", "300").parse().unwrap_or(300);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;

    let state = AppState {
        client,
        decode_url: decode_url.clone(),
        prefiller_header: prefiller_header.clone(),
        prefill_scheme: prefill_scheme.clone(),
    };

    tracing::info!(
        listen_addr,
        decode_url,
        prefiller_header,
        prefill_scheme,
        timeout_secs,
        "Starting Dynamo P/D routing sidecar (vanilla vLLM disaggregation)"
    );

    let app = Router::new().fallback(handle).with_state(Arc::new(state));
    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Route every request: disaggregate completion requests that carry the
/// prefiller header, otherwise transparently proxy to the local vLLM.
async fn handle(State(st): State<Arc<AppState>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();

    let bytes = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, format!("failed to read body: {e}"));
        }
    };

    let path = parts.uri.path().to_string();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| path.clone());

    let is_completion = parts.method == Method::POST
        && (path == "/v1/chat/completions" || path == "/v1/completions");
    let prefiller = parts
        .headers
        .get(&st.prefiller_header)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    match (is_completion, prefiller) {
        (true, Some(prefiller)) => disaggregate(&st, &path, &bytes, &prefiller).await,
        _ => {
            proxy_to_decode(
                &st,
                &parts.method,
                &path_and_query,
                &parts.headers,
                bytes.to_vec(),
            )
            .await
        }
    }
}

/// Run the NIXL prefill→decode handshake and stream the decode response back.
async fn disaggregate(st: &AppState, path: &str, body: &[u8], prefiller: &str) -> Response {
    let base: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}"));
        }
    };

    // (1) Prefill: do_remote_decode, single-token, non-streaming.
    let mut prefill = base.clone();
    if let Some(obj) = prefill.as_object_mut() {
        obj.insert(
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
        obj.insert("max_tokens".to_string(), json!(1));
        obj.insert("max_completion_tokens".to_string(), json!(1));
        obj.insert("stream".to_string(), json!(false));
    }

    let prefill_url = format!("{}://{}{}", st.prefill_scheme, prefiller, path);
    let prefill_resp = match st.client.post(&prefill_url).json(&prefill).send().await {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("prefill request to {prefiller} failed: {e}"),
            );
        }
    };
    if !prefill_resp.status().is_success() {
        let status = prefill_resp.status();
        let detail = prefill_resp.text().await.unwrap_or_default();
        return error_response(
            StatusCode::BAD_GATEWAY,
            format!("prefiller {prefiller} returned {status}: {detail}"),
        );
    }
    let prefill_json: Value = match prefill_resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("could not decode prefill response: {e}"),
            );
        }
    };

    let kv_transfer_params = prefill_json.get("kv_transfer_params").cloned();
    if kv_transfer_params.is_none() {
        // Without transfer params the decode worker can't pull the prefill KV;
        // surface it rather than silently running a full (non-disagg) decode.
        return error_response(
            StatusCode::BAD_GATEWAY,
            "prefill response carried no kv_transfer_params; \
             ensure the prefiller runs vLLM with a NixlConnector kv-transfer-config"
                .to_string(),
        );
    }

    // (2) Decode: reuse the prefill's kv_transfer_params; preserve the client's
    // original streaming preference.
    let mut decode = base;
    if let (Some(obj), Some(ktp)) = (decode.as_object_mut(), kv_transfer_params) {
        obj.insert("kv_transfer_params".to_string(), ktp);
    }

    let decode_url = format!("{}{}", st.decode_url, path);
    match st.client.post(&decode_url).json(&decode).send().await {
        Ok(resp) => stream_back(resp),
        Err(e) => error_response(
            StatusCode::BAD_GATEWAY,
            format!("decode request to local vLLM failed: {e}"),
        ),
    }
}

/// Transparently forward a request to the local vLLM and stream the response.
async fn proxy_to_decode(
    st: &AppState,
    method: &Method,
    path_and_query: &str,
    headers: &HeaderMap,
    body: Vec<u8>,
) -> Response {
    let url = format!("{}{}", st.decode_url, path_and_query);
    let reqwest_method =
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);

    let mut builder = st.client.request(reqwest_method, &url);
    if let Some(ct) = headers.get(header::CONTENT_TYPE) {
        builder = builder.header(reqwest::header::CONTENT_TYPE, ct.as_bytes());
    }
    if !body.is_empty() {
        builder = builder.body(body);
    }

    match builder.send().await {
        Ok(resp) => stream_back(resp),
        Err(e) => error_response(
            StatusCode::BAD_GATEWAY,
            format!("proxy to local vLLM failed: {e}"),
        ),
    }
}

/// Convert an upstream `reqwest` response into a streaming axum response,
/// preserving status and content-type (so SSE streams flow through unbuffered).
fn stream_back(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .map(|v| v.as_bytes().to_vec());

    let mut builder = Response::builder().status(status);
    if let Some(ct) = content_type {
        builder = builder.header(header::CONTENT_TYPE, ct);
    }
    match builder.body(Body::from_stream(resp.bytes_stream())) {
        Ok(response) => response,
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to build streamed response: {e}"),
        ),
    }
}

/// Build an OpenAI-shaped JSON error response.
fn error_response(status: StatusCode, message: String) -> Response {
    tracing::warn!(%message, status = %status, "P/D sidecar error");
    let body = serde_json::to_vec(&json!({
        "error": { "message": message, "type": "pd_sidecar_error" }
    }))
    .unwrap_or_default();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}
