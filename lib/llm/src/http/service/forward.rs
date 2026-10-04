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
    /// The explicitly configured `PREFIX=URL` entries, or `DYN_HTTP_FORWARD_ROUTES`
    /// when none were given (direct Rust entrypoints); `None` when neither has any.
    pub(crate) fn from_config(routes: Option<&[String]>) -> anyhow::Result<Option<Self>> {
        match routes {
            Some(routes) => Self::parse(&routes.join(" ")),
            None => match std::env::var(env_llm::DYN_HTTP_FORWARD_ROUTES) {
                Ok(spec) => Self::parse(&spec),
                Err(_) => Ok(None),
            },
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
    pub(crate) fn covers(&self, path: &str) -> bool {
        self.upstream_url(path, None).is_some()
    }

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
    /// back unchanged. `guard` lives as long as the forwarded exchange (the
    /// response body, or the tunnel), so callers can count it as inflight.
    /// Forwarded responses and tunnels end when `cancel` fires.
    pub(crate) async fn forward<G: Send + Sync + 'static>(
        &self,
        request: Request,
        cancel: CancellationToken,
        guard: G,
    ) -> Result<Response, Request> {
        let Some(url) = self.upstream_url(request.uri().path(), request.uri().query()) else {
            return Err(request);
        };
        let limit = super::openai::get_body_limit();
        let declared = request
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok()?.parse::<usize>().ok());
        if declared.is_some_and(|length| length > limit) {
            return Ok(super::openai::payload_too_large_error().into_response());
        }
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
            let body = http_body_util::Limited::new(body, limit);
            builder = builder.body(reqwest::Body::wrap_stream(
                http_body_util::BodyDataStream::new(body),
            ));
        }
        // An upstream that never answers must not outlive the drain window.
        let sent = tokio::select! {
            sent = builder.send() => sent,
            _ = cancel.cancelled() => {
                return Ok(super::openai::ErrorMessage::_service_unavailable().into_response());
            }
        };
        let upstream = match sent {
            Ok(upstream) => upstream,
            Err(err) if exceeded_body_limit(&err) => {
                return Ok(super::openai::payload_too_large_error().into_response());
            }
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
                tokio::spawn(async move {
                    tunnel(client_upgrade, upstream, url, cancel).await;
                    drop(guard);
                });
                Body::empty()
            }
            _ => Body::from_stream(
                upstream
                    .bytes_stream()
                    .take_until(cancel.cancelled_owned())
                    .map(move |chunk| {
                        let _guard = &guard;
                        chunk
                    }),
            ),
        };
        let mut response = Response::new(body);
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        Ok(response)
    }
}

/// Whether a failed upstream send was cut short by the client body exceeding
/// `DYN_HTTP_BODY_LIMIT_MB`, rather than by the upstream.
fn exceeded_body_limit(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut source = Some(err);
    while let Some(err) = source {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        source = err.source();
    }
    false
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

/// Removes the fixed hop-by-hop headers and any header the `Connection`
/// header names (RFC 9110 §7.6.1).
fn strip_hop_by_hop(mut headers: HeaderMap) -> HeaderMap {
    let nominated: Vec<header::HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|token| header::HeaderName::from_bytes(token.trim().as_bytes()).ok())
        .collect();
    for name in &nominated {
        headers.remove(name);
    }
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
            .forward(request, CancellationToken::new(), ())
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
                .forward(unmatched, CancellationToken::new(), ())
                .await
                .is_err()
        );

        let dead = ForwardRoutes::parse("/v1/custom=http://127.0.0.1:1")
            .unwrap()
            .unwrap();
        let request = Request::builder()
            .uri("/v1/custom")
            .body(Body::empty())
            .unwrap();
        let response = dead
            .forward(request, CancellationToken::new(), ())
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

        let routes = std::sync::Arc::new(
            ForwardRoutes::parse(&format!("/v1/custom=http://{upstream_addr}"))
                .unwrap()
                .unwrap(),
        );
        // Stands in for the inflight permit, which must outlive the 101.
        struct Guard(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let live = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let held = live.clone();
        let frontend = axum::Router::new().fallback(move |request: Request| {
            let (routes, live) = (routes.clone(), live.clone());
            async move {
                live.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                routes
                    .forward(request, CancellationToken::new(), Guard(live))
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
        let live = || held.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(live(), 1, "open tunnel holds its guard");

        drop(client);
        tokio::time::timeout(Duration::from_secs(5), async {
            while live() > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("closed tunnel releases its guard");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn forward_rejects_oversized_bodies_with_413() {
        temp_env::async_with_vars([(env_llm::DYN_HTTP_BODY_LIMIT_MB, Some("1"))], async move {
            // Reads everything but never answers: only the body limit can end a request.
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                while let Ok((mut conn, _)) = listener.accept().await {
                    tokio::spawn(async move {
                        let _ = tokio::io::copy(&mut conn, &mut tokio::io::sink()).await;
                    });
                }
            });
            let routes = ForwardRoutes::parse(&format!("/v1/custom=http://{addr}"))
                .unwrap()
                .unwrap();
            let oversized = vec![b'x'; 2 * 1024 * 1024];

            // Declared length: refused before anything is sent.
            let declared = Request::builder()
                .method("POST")
                .uri("/v1/custom/push")
                .header(header::CONTENT_LENGTH, oversized.len())
                .body(Body::from(oversized.clone()))
                .unwrap();
            let response = routes
                .forward(declared, CancellationToken::new(), ())
                .await
                .ok()
                .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

            // Chunked: cut off while streaming.
            let chunks = oversized
                .chunks(64 * 1024)
                .map(|c| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(c)))
                .collect::<Vec<_>>();
            let chunked = Request::builder()
                .method("POST")
                .uri("/v1/custom/push")
                .body(Body::from_stream(futures::stream::iter(chunks)))
                .unwrap();
            let response = tokio::time::timeout(
                Duration::from_secs(10),
                routes.forward(chunked, CancellationToken::new(), ()),
            )
            .await
            .expect("an oversized chunked body must not hang")
            .ok()
            .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        })
        .await;
    }

    #[tokio::test]
    async fn forward_send_ends_when_cancelled() {
        // Accepts connections but never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((conn, _)) = listener.accept().await {
                held.push(conn);
            }
        });
        let routes = ForwardRoutes::parse(&format!("/v1/custom=http://{addr}"))
            .unwrap()
            .unwrap();
        let cancel = CancellationToken::new();
        tokio::spawn({
            let cancel = cancel.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                cancel.cancel();
            }
        });
        let request = Request::builder()
            .uri("/v1/custom/stuck")
            .body(Body::empty())
            .unwrap();
        let response =
            tokio::time::timeout(Duration::from_secs(5), routes.forward(request, cancel, ()))
                .await
                .expect("a stuck upstream must not outlive cancellation")
                .ok()
                .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn strip_hop_by_hop_removes_connection_nominated_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONNECTION,
            "x-internal, keep-alive".parse().unwrap(),
        );
        headers.insert("x-internal", "secret".parse().unwrap());
        headers.insert(header::AUTHORIZATION, "Bearer k".parse().unwrap());
        let headers = strip_hop_by_hop(headers);
        assert!(!headers.contains_key("x-internal"));
        assert!(!headers.contains_key(header::CONNECTION));
        assert!(headers.contains_key(header::AUTHORIZATION));
    }

    #[test]
    #[serial_test::serial]
    fn from_config_prefers_explicit_routes() {
        temp_env::with_var(
            env_llm::DYN_HTTP_FORWARD_ROUTES,
            Some("/v1/env=http://env:9000"),
            || {
                let explicit = ["/v1/custom=http://up:8080".to_string()];
                let routes = ForwardRoutes::from_config(Some(&explicit))
                    .unwrap()
                    .unwrap();
                assert!(url(&routes, "/v1/custom/x", None).is_some());
                assert_eq!(url(&routes, "/v1/env/x", None), None);
                // An explicit empty list disables forwarding despite the environment.
                assert!(ForwardRoutes::from_config(Some(&[])).unwrap().is_none());
                // Without explicit routes, the environment applies.
                let routes = ForwardRoutes::from_config(None).unwrap().unwrap();
                assert!(url(&routes, "/v1/env/x", None).is_some());
            },
        );
    }

    #[test]
    fn upstream_url_keeps_percent_encoded_path() {
        let routes = ForwardRoutes::parse("/v1/custom=http://up:8080/base")
            .unwrap()
            .unwrap();
        assert_eq!(
            url(&routes, "/v1/custom/files/a%2Fb%20c", Some("q=a%26b")).as_deref(),
            Some("http://up:8080/base/v1/custom/files/a%2Fb%20c?q=a%26b")
        );
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
