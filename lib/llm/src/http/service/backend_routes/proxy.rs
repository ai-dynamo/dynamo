// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a backend route reaches its worker: the request goes with its method,
//! path, query, end-to-end headers and body over HTTP/1.1, and the response
//! streams back. WebSocket upgrades are tunnelled: on the worker's `101`, both
//! connections are spliced together. Other upgrade requests are forwarded as
//! plain requests.

use std::time::Duration;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::Request;
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::{StreamExt, TryFutureExt};
use hyper::upgrade::OnUpgrade;
use tokio_util::sync::CancellationToken;

use super::super::openai;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Below common upstream keep-alive timeouts, so a pooled connection is not
/// reused just as the worker closes it.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(15);

/// Headers that describe one connection, not the message (RFC 9110 §7.6.1),
/// plus `host`, which must name the worker rather than the frontend.
const HOP_BY_HOP_HEADERS: [header::HeaderName; 9] = [
    header::CONNECTION,
    header::HOST,
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
    header::HeaderName::from_static("keep-alive"),
];

/// The client requests go to workers with: HTTP/1.1 only, since an `https`
/// worker could otherwise negotiate HTTP/2, which has no `Upgrade`. Redirects
/// are passed back, not followed.
pub(super) fn client() -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .http1_only()
        .build()
}

/// Sends `request` to `url`: the worker's response, with the client's half of
/// a WebSocket upgrade, or the response to send instead.
pub(super) async fn send(
    client: &reqwest::Client,
    request: Request,
    url: &reqwest::Url,
) -> Result<(reqwest::Response, Option<OnUpgrade>), Response> {
    let (mut parts, body) = request.into_parts();
    let body = prepare_upload(&mut parts, body).await?;
    let (request, client_upgrade) = upstream_request(client, parts, url, body);
    match request.send().await {
        Ok(upstream) => Ok((upstream, client_upgrade)),
        Err(err) => {
            tracing::warn!(%url, error = %err, "backend route request failed");
            Err(error_response(
                StatusCode::BAD_GATEWAY,
                "the worker serving this route is unavailable",
            ))
        }
    }
}

/// The worker's request for `parts`, with end-to-end headers only. Only a
/// WebSocket upgrade on a connection that can be spliced is passed on, and its
/// client half is returned beside the request: another protocol (e.g. `h2c`)
/// would reach worker paths unchecked, so any other upgrade request is
/// forwarded as a plain request.
fn upstream_request(
    client: &reqwest::Client,
    mut parts: Parts,
    url: &reqwest::Url,
    body: Option<reqwest::Body>,
) -> (reqwest::RequestBuilder, Option<OnUpgrade>) {
    let upgrade = parts
        .headers
        .get(header::UPGRADE)
        .filter(|value| value.as_bytes().eq_ignore_ascii_case(b"websocket"))
        .cloned()
        .and_then(|upgrade| Some((upgrade, parts.extensions.remove::<OnUpgrade>()?)));
    let mut headers = strip_hop_by_hop(parts.headers);
    let client_upgrade = upgrade.map(|(upgrade, client_upgrade)| {
        headers.insert(
            header::CONNECTION,
            header::HeaderValue::from_static("upgrade"),
        );
        headers.insert(header::UPGRADE, upgrade);
        client_upgrade
    });
    let mut request = client.request(parts.method, url.clone()).headers(headers);
    // No body at all for a bodiless request, not an empty chunked one.
    if let Some(body) = body {
        request = request.body(body);
    }
    (request, client_upgrade)
}

/// Checks the request body against the body limit and readies it for the
/// worker, or returns the response to send instead. A known length is
/// enforced by the server, so that body streams. An unknown one is read
/// within the limit first, so the worker never sees a partial body or answers
/// before the limit is decided.
async fn prepare_upload(parts: &mut Parts, body: Body) -> Result<Option<reqwest::Body>, Response> {
    let exact = HttpBody::size_hint(&body).exact();
    if over_body_limit(parts, exact) {
        return Err(openai::payload_too_large_error().into_response());
    }
    match exact {
        Some(0) => Ok(None),
        Some(_) => Ok(Some(reqwest::Body::wrap_stream(body.into_data_stream()))),
        None => {
            let collected = collect_upload(body).await?;
            // The length now comes from the bytes read, not from a
            // Content-Length that disagreed with the chunked framing.
            parts.headers.remove(header::CONTENT_LENGTH);
            Ok(Some(reqwest::Body::from(collected)))
        }
    }
}

/// The whole request body, read within the body limit, or the response to
/// send instead.
pub(super) async fn read_upload(parts: &Parts, body: Body) -> Result<Bytes, Response> {
    if over_body_limit(parts, HttpBody::size_hint(&body).exact()) {
        return Err(openai::payload_too_large_error().into_response());
    }
    collect_upload(body).await
}

/// Whether the declared or `exact` body length exceeds the body limit.
fn over_body_limit(parts: &Parts, exact: Option<u64>) -> bool {
    let declared = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok()?.parse::<u64>().ok());
    declared
        .or(exact)
        .is_some_and(|length| length > openai::get_body_limit() as u64)
}

/// Reads `body` within the body limit.
async fn collect_upload(body: Body) -> Result<Bytes, Response> {
    let limit = openai::get_body_limit();
    match http_body_util::BodyExt::collect(http_body_util::Limited::new(body, limit)).await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(err) if err.is::<http_body_util::LengthLimitError>() => {
            Err(openai::payload_too_large_error().into_response())
        }
        Err(_) => Err(openai::failed_to_read_request_body_error().into_response()),
    }
}

/// The client's response to the worker's. `permit` moves into whichever part
/// of the exchange outlives this call: the tunnel after a WebSocket `101`,
/// otherwise the response body. A tunnel ends when `cancel` fires.
pub(super) fn client_response<P: Send + 'static>(
    upstream: reqwest::Response,
    client_upgrade: Option<OnUpgrade>,
    url: reqwest::Url,
    cancel: CancellationToken,
    permit: P,
) -> Response {
    let status = upstream.status();
    let mut headers = strip_hop_by_hop(upstream.headers().clone());
    // A chunked response is re-streamed; a Content-Length sent beside it
    // would misframe that stream for the client.
    if upstream.headers().contains_key(header::TRANSFER_ENCODING) {
        headers.remove(header::CONTENT_LENGTH);
    }
    let body = match client_upgrade {
        Some(client_upgrade) if status == StatusCode::SWITCHING_PROTOCOLS => {
            for name in [header::CONNECTION, header::UPGRADE] {
                if let Some(value) = upstream.headers().get(&name) {
                    headers.insert(name, value.clone());
                }
            }
            tokio::spawn(tunnel(client_upgrade, upstream, url, cancel, permit));
            Body::empty()
        }
        _ => streamed_body(upstream, permit),
    };
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// `upstream`'s body, streamed to the client, holding `permit` until it ends
/// or is dropped.
fn streamed_body<P: Send + 'static>(upstream: reqwest::Response, permit: P) -> Body {
    Body::from_stream(upstream.bytes_stream().map(move |chunk| {
        let _held = &permit;
        chunk.map_err(std::io::Error::other)
    }))
}

/// Splices the client's upgraded connection to the worker's until either side
/// closes or `cancel` fires, holding `_permit` until then.
async fn tunnel<P>(
    client: OnUpgrade,
    upstream: reqwest::Response,
    url: reqwest::Url,
    cancel: CancellationToken,
    _permit: P,
) {
    let upgraded = tokio::try_join!(
        client.map_err(anyhow::Error::from),
        upstream.upgrade().map_err(anyhow::Error::from),
    );
    let (client, mut upstream) = match upgraded {
        Ok(upgraded) => upgraded,
        Err(err) => {
            tracing::warn!(%url, error = %err, "backend route upgrade failed");
            return;
        }
    };
    let mut client = hyper_util::rt::TokioIo::new(client);
    tokio::select! {
        result = tokio::io::copy_bidirectional(&mut client, &mut upstream) => {
            if let Err(err) = result {
                tracing::debug!(%url, error = %err, "backend route tunnel closed");
            }
        }
        _ = cancel.cancelled() => {}
    }
}

/// Removes the fixed hop-by-hop headers and any header the `Connection`
/// header names (RFC 9110 §7.6.1).
fn strip_hop_by_hop(mut headers: HeaderMap) -> HeaderMap {
    let nominated: Vec<header::HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .flat_map(|value| value.as_bytes().split(|byte| *byte == b','))
        .filter_map(|token| header::HeaderName::from_bytes(token.trim_ascii()).ok())
        .collect();
    for name in &nominated {
        headers.remove(name);
    }
    for name in &HOP_BY_HOP_HEADERS {
        headers.remove(name);
    }
    headers
}

/// A worker's base URL, checked: http(s), without credentials, query or
/// fragment. Errors never echo the URL, which may hold credentials.
pub(super) fn upstream_base(url: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(url).map_err(|e| format!("is invalid: {e}"))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("must not contain credentials".to_string());
    }
    if !matches!(url.scheme(), "http" | "https")
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("must be http(s) without query or fragment".to_string());
    }
    Ok(url)
}

/// `path` and `query` under `base`, or `None` when normalization took the
/// result out from under `prefix`.
pub(super) fn scoped_url(
    base: &reqwest::Url,
    prefix: &str,
    path: &str,
    query: Option<&str>,
) -> Option<reqwest::Url> {
    let mut url = base.clone();
    let base = url.path().trim_end_matches('/').to_string();
    url.set_path(&format!("{base}{path}"));
    url.set_query(query);
    // Defence in depth: whatever normalization did, stay under the prefix.
    super::under(url.path(), &format!("{base}{prefix}")).then_some(url)
}

pub(super) fn error_response(code: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "message": message,
        "type": code.canonical_reason().unwrap_or_default(),
        "code": code.as_u16(),
    });
    (code, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests;
