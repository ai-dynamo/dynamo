// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use dynamo_runtime::config::environment_names::llm as env_llm;
use futures::StreamExt;
use http_body_util::{BodyExt, Limited};
use reqwest::Url;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use super::write_timeout::ConnectionDeadline;
use super::{RouteDoc, validate_extension_route_path};

#[derive(Clone)]
struct ProxyState {
    client: reqwest::Client,
    gateway: Url,
    files_path: String,
    batches_path: String,
    max_body: usize,
    max_upload_body: usize,
    file_transfers: Arc<Semaphore>,
    file_transfer_timeout: Duration,
    cancel: CancellationToken,
}

pub(super) fn upload_body_limit() -> usize {
    std::env::var(env_llm::DYN_BATCH_UPLOAD_BODY_LIMIT_MB)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .and_then(|value| value.checked_mul(1024 * 1024))
        .unwrap_or(202 * 1024 * 1024)
}

fn max_file_transfers() -> usize {
    std::env::var(env_llm::DYN_BATCH_MAX_FILE_TRANSFERS)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=Semaphore::MAX_PERMITS).contains(value))
        .unwrap_or(4)
}

fn file_transfer_timeout() -> Duration {
    std::env::var(env_llm::DYN_BATCH_FILE_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .filter(|duration| tokio::time::Instant::now().checked_add(*duration).is_some())
        .unwrap_or(Duration::from_secs(600))
}

pub(super) fn router(
    gateway: &str,
    files_path: Option<String>,
    batches_path: Option<String>,
    max_body: usize,
    max_upload_body: usize,
    cancel: CancellationToken,
) -> anyhow::Result<(Vec<RouteDoc>, Router)> {
    let gateway = Url::parse(gateway)?;
    anyhow::ensure!(
        matches!(gateway.scheme(), "http" | "https")
            && gateway.host_str().is_some()
            && gateway.username().is_empty()
            && gateway.password().is_none()
            && gateway.path() == "/"
            && gateway.query().is_none()
            && gateway.fragment().is_none(),
        "batch gateway URL must be an HTTP(S) origin without credentials, path, query, or fragment"
    );
    let files_path = files_path.unwrap_or_else(|| "/v1/files".to_string());
    let batches_path = batches_path.unwrap_or_else(|| "/v1/batches".to_string());
    for path in [&files_path, &batches_path] {
        validate_extension_route_path(path)?;
        anyhow::ensure!(
            path != "/" && !path.ends_with('/'),
            "batch endpoint paths must not be root or end with '/'"
        );
    }
    anyhow::ensure!(
        files_path != batches_path,
        "batch endpoint paths must differ"
    );
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .build()?;

    let mut docs = Vec::new();
    let mut router = Router::new();
    for (path, methods) in [
        (
            files_path.clone(),
            vec![axum::http::Method::GET, axum::http::Method::POST],
        ),
        (
            format!("{files_path}/{{file_id}}"),
            vec![axum::http::Method::GET, axum::http::Method::DELETE],
        ),
        (
            format!("{files_path}/{{file_id}}/content"),
            vec![axum::http::Method::GET],
        ),
        (
            batches_path.clone(),
            vec![axum::http::Method::GET, axum::http::Method::POST],
        ),
        (
            format!("{batches_path}/{{batch_id}}"),
            vec![axum::http::Method::GET],
        ),
        (
            format!("{batches_path}/{{batch_id}}/cancel"),
            vec![axum::http::Method::POST],
        ),
    ] {
        let mut route = axum::routing::MethodRouter::new();
        for method in methods {
            route = route.on(
                axum::routing::MethodFilter::try_from(method.clone())?,
                forward,
            );
            docs.push(RouteDoc::new(method, &path));
        }
        router = router.route(&path, route);
    }
    Ok((
        docs,
        router.with_state(ProxyState {
            client,
            gateway,
            files_path,
            batches_path,
            max_body,
            max_upload_body,
            file_transfers: Arc::new(Semaphore::new(max_file_transfers())),
            file_transfer_timeout: file_transfer_timeout(),
            cancel,
        }),
    ))
}

// RFC 9110: Connection can name additional hop-by-hop headers on either leg.
fn remove_hop_headers(headers: &mut HeaderMap) {
    let nominated: Vec<_> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|name| header::HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in nominated {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ] {
        headers.remove(name);
    }
}

fn proxy_error(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {
            "message": message,
            "type": if status.is_client_error() { "invalid_request_error" } else { "server_error" },
            "code": status.as_u16()
        }})),
    )
        .into_response()
}

async fn forward(State(state): State<ProxyState>, request: Request) -> Response {
    let (mut parts, body) = request.into_parts();
    let upload = parts.method == axum::http::Method::POST && parts.uri.path() == state.files_path;
    let max_body = if upload {
        state.max_upload_body
    } else {
        state.max_body
    };
    if parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > max_body as u64)
    {
        return proxy_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Batch request body exceeds the frontend limit",
        );
    }
    let path = parts.uri.path();
    let upstream_path = [
        (state.files_path.as_str(), "/v1/files"),
        (state.batches_path.as_str(), "/v1/batches"),
    ]
    .into_iter()
    .filter_map(|(mount, canonical)| {
        path.strip_prefix(mount)
            .filter(|suffix| suffix.is_empty() || suffix.starts_with('/'))
            .map(|suffix| (mount.len(), format!("{canonical}{suffix}")))
    })
    .max_by_key(|(length, _)| *length)
    .map(|(_, path)| path);
    let Some(upstream_path) = upstream_path else {
        return proxy_error(StatusCode::NOT_FOUND, "Unknown Batch API path");
    };
    let file_transfer = upload
        || (parts.method == axum::http::Method::GET
            && upstream_path.starts_with("/v1/files/")
            && upstream_path.ends_with("/content"));
    let permit = if file_transfer {
        match state.file_transfers.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                return proxy_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Batch file transfer capacity exhausted; retry later",
                );
            }
        }
    } else {
        None
    };
    // The connection owner enforces this even when HTTP/2 flow control stops
    // polling the response body. It closes the connection when a transfer expires.
    let deadline = if file_transfer {
        parts
            .extensions
            .get::<ConnectionDeadline>()
            .map(|connection| connection.start(state.file_transfer_timeout))
    } else {
        None
    };
    // Only trusted configuration constructs the outbound destination. Client input
    // can update path/query components, never the scheme, authority or port.
    let mut request = reqwest::Request::new(parts.method, state.gateway.clone());
    let url = request.url_mut();
    url.set_path(&upstream_path);
    // Reject encoded dot segments rather than letting URL normalization escape the Batch API.
    if url.path() != upstream_path {
        return proxy_error(StatusCode::BAD_REQUEST, "Invalid Batch API path");
    }
    url.set_query(parts.uri.query());
    // Enforce that forwarding never leaves the configured gateway origin.
    if url.scheme() != state.gateway.scheme()
        || url.host_str() != state.gateway.host_str()
        || url.port_or_known_default() != state.gateway.port_or_known_default()
    {
        return proxy_error(StatusCode::BAD_REQUEST, "Invalid Batch API path");
    }
    remove_hop_headers(&mut parts.headers);
    parts.headers.remove(header::HOST);
    parts.headers.remove(header::CONTENT_LENGTH);

    let upload_failure = Arc::new(AtomicU8::new(0));
    let failure = upload_failure.clone();
    let upload = Limited::new(body, max_body)
        .into_data_stream()
        .map(move |item| {
            if let Err(error) = &item {
                failure.store(
                    if error.is::<http_body_util::LengthLimitError>() {
                        1
                    } else {
                        2
                    },
                    Ordering::Relaxed,
                );
            }
            item
        });
    *request.headers_mut() = parts.headers;
    *request.body_mut() = Some(reqwest::Body::wrap_stream(upload));
    let upstream = tokio::select! {
        biased;
        _ = state.cancel.cancelled() => return proxy_error(StatusCode::SERVICE_UNAVAILABLE, "Frontend is shutting down"),
        // Includes the streamed upload and response headers, not the output download.
        response = tokio::time::timeout(Duration::from_secs(600), state.client.execute(request)) => {
            match response {
                Ok(response) => response,
                Err(_) => return proxy_error(StatusCode::GATEWAY_TIMEOUT, "Batch gateway request timed out"),
            }
        },
    };
    match upload_failure.load(Ordering::Relaxed) {
        1 => {
            return proxy_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Batch request body exceeds the frontend limit",
            );
        }
        2 => return proxy_error(StatusCode::BAD_REQUEST, "Unable to read Batch request body"),
        _ => {}
    }
    let upstream = match upstream {
        Ok(response) => response,
        Err(error) => {
            let status = if error.is_timeout() {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::BAD_GATEWAY
            };
            tracing::warn!(
                is_timeout = error.is_timeout(),
                is_connect = error.is_connect(),
                is_body = error.is_body(),
                is_request = error.is_request(),
                "Batch gateway request failed"
            );
            return proxy_error(status, "Unable to reach Batch gateway");
        }
    };
    let status = upstream.status();
    let mut headers = upstream.headers().clone();
    remove_hop_headers(&mut headers);
    let mut chunks = upstream.bytes_stream();
    let output = async_stream::stream! {
        // Captured even before the stream's first poll. Dropping an unread body
        // releases admission; returning headers alone must not release it.
        let _permit = permit;
        let _deadline = deadline;
        loop {
            let chunk = tokio::select! {
                biased;
                _ = state.cancel.cancelled() => {
                    yield Err::<Bytes, std::io::Error>(std::io::Error::new(std::io::ErrorKind::Interrupted, "Frontend is shutting down"));
                    break;
                },
                chunk = tokio::time::timeout(Duration::from_secs(120), chunks.next()) => {
                    match chunk {
                        Ok(chunk) => chunk,
                        Err(_) => {
                            yield Err::<Bytes, std::io::Error>(std::io::Error::new(std::io::ErrorKind::TimedOut, "Batch gateway response stalled"));
                            break;
                        }
                    }
                },
            };
            match chunk {
                Some(chunk) => yield chunk.map_err(std::io::Error::other),
                None => break,
            }
        }
    };
    let mut response = Response::new(Body::from_stream(output));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

#[cfg(test)]
#[path = "batch_proxy_tests.rs"]
mod tests;
