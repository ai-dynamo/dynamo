// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reverse proxy for operator-configured path prefixes
//! (`DYN_HTTP_FORWARD_ROUTES`), for backend-specific HTTP APIs the frontend
//! does not implement. Built-in routes always win: a request is forwarded only
//! when no frontend route matched it. Protocol upgrades (e.g. WebSocket) are
//! tunnelled: on the upstream's `101`, both connections are spliced together.

use axum::body::{Body, HttpBody};
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use dynamo_runtime::config::environment_names::llm as env_llm;
use futures::StreamExt;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

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
    upstream: reqwest::Url,
}

/// Path prefixes forwarded verbatim (method, path, query, headers, streamed
/// body) to an upstream HTTP server.
pub(crate) struct ForwardRoutes {
    routes: Vec<ForwardRoute>,
    client: reqwest::Client,
}

impl ForwardRoutes {
    /// Reads `DYN_HTTP_FORWARD_ROUTES`; `None` when unset or empty.
    pub(crate) fn from_env() -> anyhow::Result<Option<Self>> {
        match std::env::var(env_llm::DYN_HTTP_FORWARD_ROUTES) {
            Ok(spec) => Self::parse(&spec),
            Err(_) => Ok(None),
        }
    }

    /// Parses whitespace-separated `PREFIX=URL` entries, e.g.
    /// `/v1/custom=http://127.0.0.1:8080`.
    fn parse(spec: &str) -> anyhow::Result<Option<Self>> {
        let mut routes: Vec<ForwardRoute> = Vec::new();
        for entry in spec.split_whitespace() {
            let Some((prefix, url)) = entry.split_once('=') else {
                anyhow::bail!("forward route must be PREFIX=URL: {entry:?}");
            };
            if !prefix.starts_with('/') {
                anyhow::bail!("forward route prefix must start with '/': {entry:?}");
            }
            let prefix = prefix.trim_end_matches('/').to_string();
            let upstream = reqwest::Url::parse(url)
                .map_err(|e| anyhow::anyhow!("invalid forward route URL {url:?}: {e}"))?;
            if !matches!(upstream.scheme(), "http" | "https")
                || upstream.query().is_some()
                || upstream.fragment().is_some()
            {
                anyhow::bail!(
                    "forward route URL must be http(s) without query or fragment: {url:?}"
                );
            }
            if routes.iter().any(|r| r.prefix == prefix) {
                anyhow::bail!("duplicate forward route prefix: {entry:?}");
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
            .build()?;
        for route in &routes {
            tracing::info!(prefix = %route.prefix, upstream = %route.upstream, "forwarding HTTP route");
        }
        Ok(Some(Self { routes, client }))
    }

    /// The upstream URL for `path` and `query`, or `None` when no prefix
    /// covers `path`. Prefixes match whole segments: `/v1/custom` covers
    /// `/v1/custom` and `/v1/custom/abc`, not `/v1/customx`. Paths with dot
    /// segments are never forwarded: URL normalization would let them escape
    /// the prefix.
    fn upstream_url(&self, path: &str, query: Option<&str>) -> Option<reqwest::Url> {
        if path.split('/').any(|segment| {
            matches!(
                segment.to_ascii_lowercase().as_str(),
                "." | ".." | "%2e" | "%2e%2e" | ".%2e" | "%2e."
            )
        }) {
            return None;
        }
        let route = self.routes.iter().find(|r| {
            path.strip_prefix(r.prefix.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        })?;
        let mut url = route.upstream.clone();
        let base = url.path().trim_end_matches('/').to_string();
        url.set_path(&format!("{base}{path}"));
        url.set_query(query);
        Some(url)
    }

    /// Forwards `request` when a prefix covers its path; otherwise hands it
    /// back unchanged. Forwarded responses and tunnels end when `cancel`
    /// fires, so they cannot hold up shutdown.
    pub(crate) async fn forward(
        &self,
        request: Request,
        cancel: CancellationToken,
    ) -> Result<Response, Request> {
        let Some(url) = self.upstream_url(request.uri().path(), request.uri().query()) else {
            return Err(request);
        };
        let (mut parts, body) = request.into_parts();
        let upgrade = parts.headers.get(header::UPGRADE).cloned();
        let client_upgrade = upgrade
            .as_ref()
            .and_then(|_| parts.extensions.remove::<hyper::upgrade::OnUpgrade>());
        let mut headers = strip_hop_by_hop(parts.headers);
        if let Some(upgrade) = &upgrade {
            headers.insert(
                header::CONNECTION,
                header::HeaderValue::from_static("upgrade"),
            );
            headers.insert(header::UPGRADE, upgrade.clone());
        }
        let mut builder = self
            .client
            .request(parts.method, url.clone())
            .headers(headers);
        // Send a body only when there is one, so a bodiless GET is not
        // re-sent with `transfer-encoding: chunked`.
        if HttpBody::size_hint(&body).exact() != Some(0) {
            let body = http_body_util::Limited::new(body, super::openai::get_body_limit());
            builder = builder.body(reqwest::Body::wrap_stream(
                http_body_util::BodyDataStream::new(body),
            ));
        }
        let upstream = match builder.send().await {
            Ok(upstream) => upstream,
            Err(err) => {
                tracing::warn!(%url, error = %err, "forward route upstream request failed");
                return Ok(bad_gateway());
            }
        };
        let status = upstream.status();
        let mut headers = strip_hop_by_hop(upstream.headers().clone());
        let body = match client_upgrade {
            Some(client_upgrade) if status == StatusCode::SWITCHING_PROTOCOLS => {
                for name in [header::CONNECTION, header::UPGRADE] {
                    if let Some(value) = upstream.headers().get(&name) {
                        headers.insert(name, value.clone());
                    }
                }
                tokio::spawn(tunnel(client_upgrade, upstream, url, cancel));
                Body::empty()
            }
            _ => Body::from_stream(upstream.bytes_stream().take_until(cancel.cancelled_owned())),
        };
        let mut response = Response::new(body);
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        Ok(response)
    }
}

/// Splices the client's upgraded connection to the upstream's until either
/// side closes or `cancel` fires.
async fn tunnel(
    client: hyper::upgrade::OnUpgrade,
    upstream: reqwest::Response,
    url: reqwest::Url,
    cancel: CancellationToken,
) {
    let (client, mut upstream) =
        match tokio::try_join!(async { client.await.map_err(anyhow::Error::from) }, async {
            upstream.upgrade().await.map_err(anyhow::Error::from)
        },)
        {
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

fn strip_hop_by_hop(mut headers: HeaderMap) -> HeaderMap {
    for name in &HOP_BY_HOP_HEADERS {
        headers.remove(name);
    }
    headers
}

fn bad_gateway() -> Response {
    let code = StatusCode::BAD_GATEWAY;
    let body = serde_json::json!({
        "message": "upstream for forwarded route is unavailable",
        "type": "Bad Gateway",
        "code": code.as_u16(),
    });
    (code, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(routes: &ForwardRoutes, path: &str, query: Option<&str>) -> Option<String> {
        routes.upstream_url(path, query).map(|u| u.to_string())
    }

    #[test]
    fn parse_rejects_invalid_entries() {
        for bad in [
            "/v1/custom",
            "v1/custom=http://h:1",
            "/a=not-a-url",
            "/a=ftp://h:1",
            "/a=http://h:1/?q=1",
            "/a=http://h:1 /a/=http://h:2",
        ] {
            assert!(
                ForwardRoutes::parse(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
        assert!(ForwardRoutes::parse("  ").unwrap().is_none());
    }

    #[tokio::test]
    async fn forward_proxies_request_and_streams_response() {
        use axum::routing::post;

        let upstream = axum::Router::new().route(
            "/v1/custom/{id}/push",
            post(
                |axum::extract::Path(id): axum::extract::Path<String>,
                 uri: axum::http::Uri,
                 headers: HeaderMap,
                 body: String| async move {
                    let auth = headers[header::AUTHORIZATION].to_str().unwrap().to_string();
                    let chunks = [id, uri.query().unwrap_or("").to_string(), auth, body];
                    let stream = futures::stream::iter(
                        chunks.map(|c| Ok::<_, std::io::Error>(format!("{c};"))),
                    );
                    (StatusCode::ACCEPTED, Body::from_stream(stream))
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let routes = ForwardRoutes::parse(&format!("/v1/custom=http://{addr}"))
            .unwrap()
            .unwrap();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/custom/s1/push?x=1")
            .header(header::AUTHORIZATION, "Bearer k")
            .header(header::CONNECTION, "close")
            .body(Body::from("hello"))
            .unwrap();
        let response = routes
            .forward(request, CancellationToken::new())
            .await
            .ok()
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"s1;x=1;Bearer k;hello;");

        let unmatched = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        assert!(
            routes
                .forward(unmatched, CancellationToken::new())
                .await
                .is_err()
        );

        // A dead upstream answers 502 rather than failing the request.
        let dead = ForwardRoutes::parse("/v1/custom=http://127.0.0.1:1")
            .unwrap()
            .unwrap();
        let request = Request::builder()
            .uri("/v1/custom")
            .body(Body::empty())
            .unwrap();
        let response = dead
            .forward(request, CancellationToken::new())
            .await
            .ok()
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn forward_tunnels_protocol_upgrades() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Upstream: accept one upgrade, answer 101, then echo bytes.
        let upstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut conn, _) = upstream.accept().await.unwrap();
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                head.push(conn.read_u8().await.unwrap());
            }
            let head = String::from_utf8(head).unwrap().to_ascii_lowercase();
            assert!(head.starts_with("get /v1/custom/s1/ws "), "{head}");
            assert!(head.contains("upgrade: echo") && head.contains("connection: upgrade"));
            conn.write_all(
                b"HTTP/1.1 101 Switching Protocols\r\nconnection: upgrade\r\nupgrade: echo\r\n\r\n",
            )
            .await
            .unwrap();
            let (mut read, mut write) = conn.split();
            tokio::io::copy(&mut read, &mut write).await.unwrap();
        });

        // Frontend: forwards everything through the fallback.
        let routes = std::sync::Arc::new(
            ForwardRoutes::parse(&format!("/v1/custom=http://{upstream_addr}"))
                .unwrap()
                .unwrap(),
        );
        let frontend = axum::Router::new().fallback(move |request: Request| {
            let routes = routes.clone();
            async move {
                routes
                    .forward(request, CancellationToken::new())
                    .await
                    .unwrap()
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, frontend).await.unwrap() });

        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        client
            .write_all(b"GET /v1/custom/s1/ws HTTP/1.1\r\nhost: x\r\nconnection: upgrade\r\nupgrade: echo\r\n\r\n")
            .await
            .unwrap();
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            head.push(client.read_u8().await.unwrap());
        }
        assert!(
            head.starts_with(b"HTTP/1.1 101"),
            "{}",
            String::from_utf8_lossy(&head)
        );
        client.write_all(b"ping").await.unwrap();
        let mut echoed = [0u8; 4];
        client.read_exact(&mut echoed).await.unwrap();
        assert_eq!(&echoed, b"ping");
    }

    #[test]
    fn upstream_url_rejects_dot_segments() {
        let routes = ForwardRoutes::parse("/v1/custom=http://up:8080/base")
            .unwrap()
            .unwrap();
        for path in [
            "/v1/custom/../metrics",
            "/v1/custom/%2E%2e/x",
            "/v1/custom/./x",
        ] {
            assert_eq!(
                url(&routes, path, None),
                None,
                "{path} must not be forwarded"
            );
        }
    }

    #[test]
    fn upstream_url_matches_whole_segments_longest_first() {
        let routes = ForwardRoutes::parse(
            "/v1/custom=http://up:8080 /v1/custom/admin=http://admin:9000/base/",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            url(&routes, "/v1/custom", None).as_deref(),
            Some("http://up:8080/v1/custom")
        );
        assert_eq!(
            url(&routes, "/v1/custom/abc/events", Some("x=1")).as_deref(),
            Some("http://up:8080/v1/custom/abc/events?x=1")
        );
        assert_eq!(
            url(&routes, "/v1/custom/admin/x", None).as_deref(),
            Some("http://admin:9000/base/v1/custom/admin/x")
        );
        assert_eq!(url(&routes, "/v1/customx", None), None);
        assert_eq!(url(&routes, "/v1/models", None), None);
    }
}
