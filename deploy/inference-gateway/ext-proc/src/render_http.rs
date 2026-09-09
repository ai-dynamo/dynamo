// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared HTTP transport helpers for render clients.
//!
//! Both the vLLM and sglang render clients use the same transport protocol,
//! error categories, and body-reading logic. This module holds the shared
//! implementation so neither client duplicates it.

use std::time::Duration;

use anyhow::Context;
use futures::StreamExt;
use reqwest::{StatusCode, Url};
use thiserror::Error;

const MAX_ERROR_BODY_BYTES: usize = 1024;

/// Unified error type for both vLLM and sglang render clients.
#[derive(Debug, Error)]
pub enum RenderError {
    /// The renderer could not be reached or the connection failed.
    #[error("render request failed: {source}")]
    Unavailable {
        #[source]
        source: reqwest::Error,
    },
    /// The renderer did not complete the request before the configured deadline.
    #[error("render request timed out after {timeout:?}: {source}")]
    Timeout {
        timeout: Duration,
        #[source]
        source: reqwest::Error,
    },
    /// The renderer returned an HTTP error response.
    #[error("renderer returned {status}: {body}")]
    UpstreamStatus { status: StatusCode, body: String },
    /// The renderer returned a successful response that did not match its contract.
    #[error("renderer returned an invalid response: {source}")]
    InvalidResponse {
        #[source]
        source: serde_json::Error,
    },
    /// The renderer returned a successful response larger than the configured limit.
    #[error("renderer response is too large: {received} bytes exceeds the {limit}-byte limit")]
    ResponseTooLarge { limit: usize, received: u64 },
}

/// Parse and validate a render service base URL.
pub(crate) fn parse_render_base_url(base_url: &str) -> anyhow::Result<Url> {
    let url = Url::parse(base_url)
        .with_context(|| format!("invalid render service base URL {base_url:?}"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "render service base URL must be an absolute HTTP(S) URL"
    );
    Ok(url)
}

pub(crate) fn classify_transport_error(source: reqwest::Error, timeout: Duration) -> RenderError {
    if source.is_timeout() {
        RenderError::Timeout { timeout, source }
    } else {
        RenderError::Unavailable { source }
    }
}

/// Reject immediately if the declared Content-Length exceeds the limit.
///
/// Both vLLM and sglang-renderer set Content-Length on their success responses
/// (the full JSON body is buffered before sending), so this fast-path is
/// reliable for these backends. The streaming check in [`read_success_body`]
/// enforces the limit for chunked or undeclared responses.
pub(crate) fn check_content_length(
    response: &reqwest::Response,
    max_bytes: usize,
) -> Result<(), RenderError> {
    if let Some(received) = response.content_length()
        && received > max_bytes as u64
    {
        return Err(RenderError::ResponseTooLarge {
            limit: max_bytes,
            received,
        });
    }
    Ok(())
}

pub(crate) async fn read_success_body(
    response: reqwest::Response,
    max_bytes: usize,
    timeout: Duration,
) -> Result<Vec<u8>, RenderError> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| classify_transport_error(e, timeout))?;
        let received = body.len().saturating_add(chunk.len());
        if received > max_bytes {
            return Err(RenderError::ResponseTooLarge {
                limit: max_bytes,
                received: received as u64,
            });
        }
        body.extend_from_slice(&chunk);
    }

    Ok(body)
}

pub(crate) async fn read_error_body(response: reqwest::Response) -> String {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();

    while body.len() < MAX_ERROR_BODY_BYTES {
        let Some(chunk) = stream.next().await else {
            break;
        };
        let Ok(chunk) = chunk else {
            break;
        };
        let remaining = MAX_ERROR_BODY_BYTES - body.len();
        body.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
    }

    String::from_utf8_lossy(&body).into_owned()
}
