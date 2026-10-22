// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reverse proxy for operator-configured path prefixes (`--forward-route`, or
//! `DYN_HTTP_FORWARD_ROUTES` when none is configured), for backend-specific
//! HTTP APIs the frontend does not implement. Built-in routes always win: a request is forwarded only
//! when no frontend route matched it. A route forwards to one fixed upstream,
//! or, as `PREFIX=session-owner`, to the worker holding the session the path
//! names (see [`sessions`]). WebSocket upgrades are tunnelled: on the
//! upstream's `101`, both connections are spliced together. Other upgrade
//! requests are forwarded as plain requests.

mod sessions;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::Request;
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use dynamo_runtime::config::environment_names::llm as env_llm;
use futures::{StreamExt, TryFutureExt};
use hyper::upgrade::OnUpgrade;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub(crate) use sessions::WorkerDiscovery;
use sessions::{SessionOwners, WorkerUrls};

/// The upstream of a route whose requests go to the worker holding their
/// session.
const SESSION_OWNER: &str = "session-owner";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Below common upstream keep-alive timeouts, so a pooled connection is not
/// reused just as the upstream closes it.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(15);

/// Headers that describe one connection, not the message (RFC 9110 §7.6.1),
/// plus `host`, which must name the upstream rather than the frontend.
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

struct ForwardRoute {
    prefix: String,
    upstream: Upstream,
}

enum Upstream {
    Fixed(reqwest::Url),
    /// The worker holding the session the path names.
    SessionOwner,
}

/// Path prefixes forwarded to an upstream HTTP server with the request's
/// method, path, query, end-to-end headers and body.
pub(crate) struct ForwardRoutes {
    routes: Vec<ForwardRoute>,
    /// HTTP/1.1 only: an `https` upstream could otherwise negotiate HTTP/2,
    /// which has no `Upgrade`.
    client: reqwest::Client,
    /// Present when a route forwards to session owners.
    sessions: Option<SessionOwners>,
}

/// Where a request is forwarded. Only [`ForwardRoutes::resolve`] makes one.
pub(crate) struct ForwardTarget(Target);

enum Target {
    /// The upstream URL, query included, checked to stay under the route that
    /// covers it.
    Url(reqwest::Url),
    /// The owner of the session the path names, under the route at this index.
    SessionOwner(usize),
}

impl ForwardRoutes {
    /// Parses `routes` when given, one `PREFIX=URL` or `PREFIX=session-owner`
    /// entry each (an empty list disables forwarding); otherwise falls back to
    /// the whitespace-separated entries of `DYN_HTTP_FORWARD_ROUTES` (direct
    /// Rust entrypoints). `None` when the chosen source has no entries.
    /// Session-owner routes find workers through `workers`, and are refused
    /// without it.
    pub(crate) fn from_config(
        routes: Option<&[String]>,
        workers: Option<WorkerDiscovery>,
    ) -> anyhow::Result<Option<Self>> {
        let parsed = match routes {
            Some(routes) => Self::parse(routes)?,
            None => match std::env::var(env_llm::DYN_HTTP_FORWARD_ROUTES) {
                Ok(spec) => Self::parse(spec.split_whitespace())?,
                Err(_) => None,
            },
        };
        let Some(mut parsed) = parsed else {
            return Ok(None);
        };
        if let Some(route) = parsed
            .routes
            .iter()
            .find(|r| matches!(r.upstream, Upstream::SessionOwner))
        {
            let Some(workers) = workers else {
                anyhow::bail!(
                    "forward route {:?} needs worker discovery to find session owners",
                    route.prefix
                );
            };
            parsed.sessions = Some(SessionOwners::new(WorkerUrls::watch(workers)));
        }
        Ok(Some(parsed))
    }

    fn parse(entries: impl IntoIterator<Item = impl AsRef<str>>) -> anyhow::Result<Option<Self>> {
        let mut routes: Vec<ForwardRoute> = Vec::new();
        // Errors name the entry or its prefix, never the URL: a URL may hold
        // credentials, and startup errors reach the logs.
        for (n, entry) in entries.into_iter().enumerate().map(|(i, e)| (i + 1, e)) {
            let Some((prefix, url)) = entry.as_ref().split_once('=') else {
                anyhow::bail!("forward route {n} must be PREFIX=URL");
            };
            if !prefix.starts_with('/') {
                anyhow::bail!("forward route {n}: prefix must start with '/'");
            }
            let prefix = prefix.trim_end_matches('/').to_string();
            let upstream = if url == SESSION_OWNER {
                Upstream::SessionOwner
            } else {
                Upstream::Fixed(
                    upstream_base(url)
                        .map_err(|e| anyhow::anyhow!("forward route URL for {prefix:?} {e}"))?,
                )
            };
            if routes.iter().any(|r| r.prefix == prefix) {
                anyhow::bail!("duplicate forward route prefix {prefix:?}");
            }
            routes.push(ForwardRoute { prefix, upstream });
        }
        if routes.is_empty() {
            return Ok(None);
        }
        // Longest prefix first, so the most specific route matches.
        routes.sort_by_key(|r| std::cmp::Reverse(r.prefix.len()));
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .http1_only()
            .build()?;
        for route in &routes {
            match &route.upstream {
                Upstream::Fixed(url) => {
                    tracing::info!(prefix = %route.prefix, upstream = %url, "forwarding HTTP route");
                }
                Upstream::SessionOwner => {
                    tracing::info!(prefix = %route.prefix, "forwarding HTTP route to session owners");
                }
            }
        }
        Ok(Some(Self {
            routes,
            client,
            sessions: None,
        }))
    }

    /// Where a request for `uri` is forwarded, or `None` when no prefix covers
    /// its path; see [`Self::route`].
    pub(crate) fn resolve(&self, uri: &Uri) -> Option<ForwardTarget> {
        let (index, route) = self.route(uri.path())?;
        let target = match &route.upstream {
            Upstream::Fixed(base) => {
                Target::Url(scoped_url(base, &route.prefix, uri.path(), uri.query())?)
            }
            Upstream::SessionOwner => Target::SessionOwner(index),
        };
        Some(ForwardTarget(target))
    }

    /// The upstream URL of a fixed route for `path` and `query`; see
    /// [`Self::route`].
    #[cfg(test)]
    fn upstream_url(&self, path: &str, query: Option<&str>) -> Option<reqwest::Url> {
        let (_, route) = self.route(path)?;
        let Upstream::Fixed(base) = &route.upstream else {
            return None;
        };
        scoped_url(base, &route.prefix, path, query)
    }

    /// The route covering `path` and its index, or `None` when no prefix
    /// covers it. Prefixes match whole segments: `/v1/custom` covers
    /// `/v1/custom` and `/v1/custom/abc`, not `/v1/customx`. Paths with
    /// backslashes or dot segments, including those separated by an encoded
    /// `/` or `\`, are never forwarded: normalization here or upstream would
    /// let them escape the prefix. Other encoded bytes are forwarded as sent.
    fn route(&self, path: &str) -> Option<(usize, &ForwardRoute)> {
        let separated = path
            .to_ascii_lowercase()
            .replace("%2f", "/")
            .replace("%5c", "/");
        if path.contains('\\')
            || separated
                .split('/')
                .any(|segment| matches!(segment, "." | ".." | "%2e" | "%2e%2e" | ".%2e" | "%2e."))
        {
            return None;
        }
        self.routes.iter().enumerate().find(|(_, r)| {
            path.strip_prefix(r.prefix.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        })
    }

    /// Forwards `request` to `target`. `permit` lives as long as the forwarded
    /// exchange (the response body, or the tunnel), so callers can count it
    /// as inflight. Forwarded responses and tunnels end when `cancel` fires.
    pub(crate) async fn forward<P: Send + Sync + 'static>(
        &self,
        request: Request,
        target: ForwardTarget,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        match target.0 {
            Target::Url(url) => self.forward_to(request, url, cancel, permit).await,
            Target::SessionOwner(index) => match &self.sessions {
                Some(owners) => {
                    let prefix = &self.routes[index].prefix;
                    self.forward_session(owners, prefix, request, cancel, permit)
                        .await
                }
                None => sessions::no_workers(),
            },
        }
    }

    /// Forwards `request` to `url`; see [`Self::forward`].
    async fn forward_to<P: Send + Sync + 'static>(
        &self,
        request: Request,
        url: reqwest::Url,
        cancel: CancellationToken,
        permit: P,
    ) -> Response {
        let (mut parts, body) = request.into_parts();
        let body = match prepare_upload(&mut parts, body, &cancel).await {
            Ok(body) => body,
            Err(refused) => return refused,
        };
        let (request, client_upgrade) = self.upstream_request(parts, &url, body);
        // An upstream that never answers must not outlive the drain window.
        let sent = tokio::select! {
            sent = request.send() => sent,
            _ = cancel.cancelled() => {
                return super::openai::ErrorMessage::_service_unavailable().into_response();
            }
        };
        match sent {
            Ok(upstream) => client_response(upstream, client_upgrade, url, cancel, permit),
            Err(err) => {
                tracing::warn!(%url, error = %err, "forward route upstream request failed");
                error_response(
                    StatusCode::BAD_GATEWAY,
                    "upstream for forwarded route is unavailable",
                )
            }
        }
    }

    /// The upstream request for `parts`, with end-to-end headers only. Only a
    /// WebSocket upgrade on a connection that can be spliced is passed on, and
    /// its client half is returned beside the request: another protocol (e.g.
    /// `h2c`) would reach upstream paths unchecked, so any other upgrade
    /// request is forwarded as a plain request.
    fn upstream_request(
        &self,
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
        let mut request = self
            .client
            .request(parts.method, url.clone())
            .headers(headers);
        // No body at all for a bodiless request, not an empty chunked one.
        if let Some(body) = body {
            request = request.body(body);
        }
        (request, client_upgrade)
    }
}

/// Checks the request body against the body limit and readies it for the
/// upstream, or returns the response to send instead. A known length is
/// enforced by the server, so that body streams. An unknown one is read
/// within the limit first, so the upstream never sees a partial body or
/// answers before the limit is decided.
async fn prepare_upload(
    parts: &mut Parts,
    body: Body,
    cancel: &CancellationToken,
) -> Result<Option<reqwest::Body>, Response> {
    let exact = HttpBody::size_hint(&body).exact();
    if over_body_limit(parts, exact) {
        return Err(super::openai::payload_too_large_error().into_response());
    }
    match exact {
        Some(0) => Ok(None),
        Some(_) => Ok(Some(reqwest::Body::wrap_stream(body.into_data_stream()))),
        None => {
            let collected = collect_upload(body, cancel).await?;
            // The length now comes from the bytes read, not from a
            // Content-Length that disagreed with the chunked framing.
            parts.headers.remove(header::CONTENT_LENGTH);
            Ok(Some(reqwest::Body::from(collected)))
        }
    }
}

/// The whole request body, read within the body limit, or the response to
/// send instead.
async fn read_upload(
    parts: &Parts,
    body: Body,
    cancel: &CancellationToken,
) -> Result<Bytes, Response> {
    if over_body_limit(parts, HttpBody::size_hint(&body).exact()) {
        return Err(super::openai::payload_too_large_error().into_response());
    }
    collect_upload(body, cancel).await
}

/// Whether the declared or `exact` body length exceeds the body limit.
fn over_body_limit(parts: &Parts, exact: Option<u64>) -> bool {
    let declared = parts
        .headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok()?.parse::<u64>().ok());
    declared
        .or(exact)
        .is_some_and(|length| length > super::openai::get_body_limit() as u64)
}

/// Reads `body` within the body limit. A stalled upload must not outlive the
/// drain window.
async fn collect_upload(body: Body, cancel: &CancellationToken) -> Result<Bytes, Response> {
    let limit = super::openai::get_body_limit();
    let collected = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            return Err(super::openai::ErrorMessage::_service_unavailable().into_response());
        }
        collected = http_body_util::BodyExt::collect(http_body_util::Limited::new(body, limit)) => collected,
    };
    match collected {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(err) if err.is::<http_body_util::LengthLimitError>() => {
            Err(super::openai::payload_too_large_error().into_response())
        }
        Err(_) => Err(super::openai::failed_to_read_request_body_error().into_response()),
    }
}

/// The client's response to `upstream`'s. `permit` moves into whichever part
/// of the exchange outlives this call: the tunnel after a WebSocket `101`,
/// otherwise the response body.
fn client_response<P: Send + Sync + 'static>(
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
        _ => streamed_body(upstream, cancel, permit),
    };
    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

/// `upstream`'s body, streamed to the client, holding `permit` until it ends
/// or is dropped. Cut off by `cancel`, it ends in an error rather than a clean
/// end, so the client can tell it was truncated.
fn streamed_body<P: Send + Sync + 'static>(
    upstream: reqwest::Response,
    cancel: CancellationToken,
    permit: P,
) -> Body {
    Body::from_stream(futures::stream::unfold(
        Some((Box::pin(upstream.bytes_stream()), cancel, permit)),
        |state| async move {
            let (mut chunks, cancel, permit) = state?;
            tokio::select! {
                biased;
                _ = cancel.cancelled() => Some((
                    Err(std::io::Error::other("frontend shut down mid-response")),
                    None,
                )),
                chunk = chunks.next() => chunk.map(|chunk| {
                    (chunk.map_err(std::io::Error::other), Some((chunks, cancel, permit)))
                }),
            }
        },
    ))
}

/// Splices the client's upgraded connection to the upstream's until either
/// side closes or `cancel` fires, holding `_permit` until then.
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
            tracing::warn!(%url, error = %err, "forward route upgrade failed");
            return;
        }
    };
    let mut client = hyper_util::rt::TokioIo::new(client);
    tokio::select! {
        result = tokio::io::copy_bidirectional(&mut client, &mut upstream) => {
            if let Err(err) = result {
                tracing::debug!(%url, error = %err, "forward route tunnel closed");
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

/// A forwarded route's base URL, checked: http(s), without credentials,
/// query or fragment. Errors never echo the URL, which may hold credentials.
fn upstream_base(url: &str) -> Result<reqwest::Url, String> {
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
/// result out from under the route at `prefix`.
fn scoped_url(
    base: &reqwest::Url,
    prefix: &str,
    path: &str,
    query: Option<&str>,
) -> Option<reqwest::Url> {
    let mut url = base.clone();
    let base = url.path().trim_end_matches('/').to_string();
    url.set_path(&format!("{base}{path}"));
    url.set_query(query);
    // Defence in depth: whatever normalization did, stay under the route.
    let scope = format!("{base}{prefix}");
    url.path()
        .strip_prefix(scope.as_str())
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        .then_some(url)
}

fn error_response(code: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "message": message,
        "type": code.canonical_reason().unwrap_or_default(),
        "code": code.as_u16(),
    });
    (code, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests;
