// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn url(routes: &ForwardRoutes, path: &str, query: Option<&str>) -> Option<String> {
    routes.upstream_url(path, query).map(|u| u.to_string())
}

/// Forwards `request`, which a route must cover.
async fn forward<P: Send + Sync + 'static>(
    routes: &ForwardRoutes,
    request: Request,
    cancel: CancellationToken,
    permit: P,
) -> Response {
    let target = routes
        .resolve(request.uri())
        .expect("a route covers the request");
    routes.forward(request, target, cancel, permit).await
}

#[test]
fn parse_rejects_invalid_entries() {
    let invalid: &[&[&str]] = &[
        &["/v1/custom"],
        &["v1/custom=http://h:1"],
        &["/a=not-a-url"],
        &["/a=ftp://h:1"],
        &["/a=http://h:1/?q=1"],
        &["/a=http://h:1/#frag"],
        &["/a=http://h:1", "/a/=http://h:2"],
    ];
    for &bad in invalid {
        assert!(
            ForwardRoutes::parse(bad).is_err(),
            "expected {bad:?} to be rejected"
        );
    }
    assert!(
        ForwardRoutes::parse("  ".split_whitespace())
            .unwrap()
            .is_none()
    );
    for secret in [
        "/a=http://user:hunter2@h:1",
        "/a=http://user@h:1",
        "/a=http://user:hunter2@%",
        "http://user:hunter2@h:1",
        "a=http://user:hunter2@h:1",
        "/a=ftp://user:hunter2@h:1",
        "/a=http://user:hunter2@h:1/?q=1",
    ] {
        let err = ForwardRoutes::parse([secret]).err().unwrap().to_string();
        assert!(!err.contains("hunter2") && !err.contains("user"), "{err}");
    }
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
                let header = |name| {
                    headers
                        .get(name)
                        .map_or("-", |v| v.to_str().unwrap())
                        .to_string()
                };
                let chunks = [
                    id,
                    uri.query().unwrap_or("").to_string(),
                    header("authorization"),
                    header("host"),
                    header("x-internal"),
                    body,
                ];
                let stream =
                    futures::stream::iter(chunks.map(|c| Ok::<_, std::io::Error>(format!("{c};"))));
                (StatusCode::ACCEPTED, Body::from_stream(stream))
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

    let routes = ForwardRoutes::parse([format!("/v1/custom=http://{addr}")])
        .unwrap()
        .unwrap();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/custom/s1/push?x=1")
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::HOST, "frontend.example")
        .header(header::CONNECTION, "close, x-internal")
        .header("x-internal", "secret")
        .body(Body::from("hello"))
        .unwrap();
    let response = forward(&routes, request, CancellationToken::new(), ()).await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&body),
        format!("s1;x=1;Bearer k;{addr};-;hello;")
    );

    assert!(routes.resolve(&Uri::from_static("/v1/models")).is_none());

    let dead = ForwardRoutes::parse(["/v1/custom=http://127.0.0.1:1"])
        .unwrap()
        .unwrap();
    let request = Request::builder()
        .uri("/v1/custom")
        .body(Body::empty())
        .unwrap();
    let response = forward(&dead, request, CancellationToken::new(), ()).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn forward_tunnels_protocol_upgrades() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Upstream: answer each upgrade with 101, then echo bytes.
    let upstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_addr = upstream.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut conn, _)) = upstream.accept().await {
            tokio::spawn(async move {
                let mut head = Vec::new();
                while !head.ends_with(b"\r\n\r\n") {
                    head.push(conn.read_u8().await.unwrap());
                }
                let head = String::from_utf8(head).unwrap().to_ascii_lowercase();
                assert!(head.starts_with("get /v1/custom/s1/ws "), "{head}");
                assert!(
                    head.contains("upgrade: websocket") && head.contains("connection: upgrade")
                );
                conn.write_all(
                    b"HTTP/1.1 101 Switching Protocols\r\nconnection: upgrade\r\nupgrade: websocket\r\n\r\n",
                )
                .await
                .unwrap();
                let (mut read, mut write) = conn.split();
                let _ = tokio::io::copy(&mut read, &mut write).await;
            });
        }
    });

    let routes = std::sync::Arc::new(
        ForwardRoutes::parse([format!("/v1/custom=http://{upstream_addr}")])
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
    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    let frontend = axum::Router::new().fallback(move |request: Request| {
        let (routes, live, cancel) = (routes.clone(), live.clone(), cancel.clone());
        async move {
            live.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            forward(&routes, request, cancel, Guard(live)).await
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, frontend).await.unwrap() });

    let open_tunnel = || async {
        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        client
            .write_all(b"GET /v1/custom/s1/ws HTTP/1.1\r\nhost: x\r\nconnection: upgrade\r\nupgrade: websocket\r\n\r\n")
            .await
            .unwrap();
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            head.push(client.read_u8().await.unwrap());
        }
        let head = String::from_utf8_lossy(&head).to_ascii_lowercase();
        assert!(head.starts_with("http/1.1 101"), "{head}");
        assert!(head.contains("upgrade: websocket") && head.contains("connection: upgrade"));
        client.write_all(b"ping").await.unwrap();
        let mut echoed = [0u8; 4];
        client.read_exact(&mut echoed).await.unwrap();
        assert_eq!(&echoed, b"ping");
        client
    };
    let live = || held.load(std::sync::atomic::Ordering::SeqCst);
    let released = |what: &'static str| async move {
        tokio::time::timeout(Duration::from_secs(5), async {
            while live() > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect(what);
    };

    let client = open_tunnel().await;
    assert_eq!(live(), 1, "open tunnel holds its guard");
    drop(client);
    released("closed tunnel releases its guard").await;

    let mut client = open_tunnel().await;
    assert_eq!(live(), 1, "open tunnel holds its guard");
    shutdown.cancel();
    released("cancelled tunnel releases its guard").await;
    let mut rest = Vec::new();
    let closed = tokio::time::timeout(Duration::from_secs(5), client.read_to_end(&mut rest))
        .await
        .expect("cancelled tunnel closes the client connection");
    assert!(closed.is_err() || rest.is_empty());
}

#[tokio::test]
async fn forward_reports_shutdown_truncation_as_body_error() {
    let upstream = axum::Router::new().route(
        "/v1/custom/events",
        axum::routing::get(|| async {
            let first = futures::stream::once(async { Ok::<_, std::io::Error>("first") });
            Body::from_stream(first.chain(futures::stream::pending()))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });
    let routes = ForwardRoutes::parse([format!("/v1/custom=http://{addr}")])
        .unwrap()
        .unwrap();
    let cancel = CancellationToken::new();
    let request = Request::builder()
        .uri("/v1/custom/events")
        .body(Body::empty())
        .unwrap();
    let response = forward(&routes, request, cancel.clone(), ()).await;
    let mut body = response.into_body().into_data_stream();
    assert_eq!(&body.next().await.unwrap().unwrap()[..], b"first");
    cancel.cancel();
    let end = tokio::time::timeout(Duration::from_secs(5), body.next())
        .await
        .expect("cancellation ends the body");
    assert!(
        matches!(end, Some(Err(_))),
        "truncation must not look like a clean end"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn forward_checks_body_limit_before_the_upstream_answers() {
    temp_env::async_with_vars([(env_llm::DYN_HTTP_BODY_LIMIT_MB, Some("1"))], async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Answers with the request head's framing as soon as the head
        // arrives, without waiting for the body.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    while !head.ends_with(b"\r\n\r\n") {
                        head.push(conn.read_u8().await.unwrap());
                    }
                    let head = String::from_utf8(head).unwrap().to_ascii_lowercase();
                    let framing = head
                        .lines()
                        .find(|l| l.starts_with("content-length:") || l.starts_with("transfer-encoding:"))
                        .unwrap_or("none")
                        .to_string();
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{framing}",
                        framing.len()
                    );
                    let _ = conn.write_all(reply.as_bytes()).await;
                });
            }
        });
        let routes = ForwardRoutes::parse([format!("/v1/custom=http://{addr}")])
            .unwrap()
            .unwrap();
        let chunked = |size: usize| {
            let chunks = vec![b'x'; size]
                .chunks(64 * 1024)
                .map(|c| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(c)))
                .collect::<Vec<_>>();
            Request::builder()
                .method("POST")
                .uri("/v1/custom/push")
                .body(Body::from_stream(futures::stream::iter(chunks)))
                .unwrap()
        };

        // Declared length: the body is tiny, so only the Content-Length
        // check can refuse it.
        let declared = Request::builder()
            .method("POST")
            .uri("/v1/custom/push")
            .header(header::CONTENT_LENGTH, 2 * 1024 * 1024)
            .body(Body::from("x"))
            .unwrap();
        let response = forward(&routes, declared, CancellationToken::new(), ()).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // Unknown length: refused once more than the limit has been read.
        let response = forward(
            &routes,
            chunked(2 * 1024 * 1024),
            CancellationToken::new(),
            (),
        )
        .await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // Within the limit, the whole body is sent with its real length,
        // even when the client also sent a Content-Length that disagrees.
        let mut stale = chunked(100 * 1024);
        stale
            .headers_mut()
            .insert(header::CONTENT_LENGTH, header::HeaderValue::from_static("5"));
        let response = forward(&routes, stale, CancellationToken::new(), ()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], format!("content-length: {}", 100 * 1024).as_bytes());
    })
    .await;
}

#[tokio::test]
async fn forward_drops_content_length_sent_beside_chunked_response() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // A misbehaving upstream: both framings, and the chunked body is longer.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut conn, _) = listener.accept().await.unwrap();
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            head.push(conn.read_u8().await.unwrap());
        }
        conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\ntransfer-encoding: chunked\r\n\r\nb\r\nhello world\r\n0\r\n\r\n")
            .await
            .unwrap();
    });
    let routes = ForwardRoutes::parse([format!("/v1/custom=http://{addr}")])
        .unwrap()
        .unwrap();
    let request = Request::builder()
        .uri("/v1/custom/x")
        .body(Body::empty())
        .unwrap();
    let response = forward(&routes, request, CancellationToken::new(), ()).await;
    assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], b"hello world");
}

#[tokio::test]
async fn forward_stalled_upload_ends_when_cancelled() {
    let routes = ForwardRoutes::parse(["/v1/custom=http://127.0.0.1:1"])
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
    // Unknown length, one chunk, then nothing more.
    let first = futures::stream::once(async {
        Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"partial"))
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/custom/push")
        .body(Body::from_stream(first.chain(futures::stream::pending())))
        .unwrap();
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        forward(&routes, request, cancel, ()),
    )
    .await
    .expect("a stalled upload must not outlive cancellation");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn forward_does_not_tunnel_non_websocket_upgrades() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let upstream = axum::Router::new().route(
        "/v1/custom/x",
        axum::routing::get(|headers: HeaderMap| async move {
            let seen = ["upgrade", "http2-settings"]
                .map(|name| format!("{name}={}", headers.contains_key(name)));
            seen.join(";")
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream_addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

    let routes = std::sync::Arc::new(
        ForwardRoutes::parse([format!("/v1/custom=http://{upstream_addr}")])
            .unwrap()
            .unwrap(),
    );
    let frontend = axum::Router::new().fallback(move |request: Request| {
        let routes = routes.clone();
        async move { forward(&routes, request, CancellationToken::new(), ()).await }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, frontend).await.unwrap() });

    let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
    client
        .write_all(b"GET /v1/custom/x HTTP/1.1\r\nhost: x\r\nconnection: Upgrade, HTTP2-Settings\r\nupgrade: h2c\r\nhttp2-settings: AAMAAABkAAQAAP__\r\n\r\n")
        .await
        .unwrap();
    let mut response = vec![0u8; 4096];
    let n = tokio::time::timeout(Duration::from_secs(5), client.read(&mut response))
        .await
        .unwrap()
        .unwrap();
    let response = String::from_utf8_lossy(&response[..n]).to_ascii_lowercase();
    assert!(response.starts_with("http/1.1 200"), "{response}");
    assert!(
        response.ends_with("upgrade=false;http2-settings=false"),
        "{response}"
    );
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
    let routes = ForwardRoutes::parse([format!("/v1/custom=http://{addr}")])
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
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        forward(&routes, request, cancel, ()),
    )
    .await
    .expect("a stuck upstream must not outlive cancellation");
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
    headers.append(
        header::CONNECTION,
        header::HeaderValue::from_bytes(b"x-other, \x80").unwrap(),
    );
    headers.insert("x-other", "secret".parse().unwrap());
    let headers = strip_hop_by_hop(headers);
    assert!(!headers.contains_key("x-internal"));
    assert!(
        !headers.contains_key("x-other"),
        "non-ASCII bytes must not hide a token"
    );
    assert!(!headers.contains_key(header::CONNECTION));
    assert!(headers.contains_key(header::AUTHORIZATION));
}

#[test]
#[serial_test::serial]
fn from_config_prefers_explicit_routes() {
    temp_env::with_var(
        env_llm::DYN_HTTP_FORWARD_ROUTES,
        Some("/v1/env=http://env:9000 /v1/env2=http://env:9001"),
        || {
            let explicit = ["/v1/custom=http://up:8080".to_string()];
            let routes = ForwardRoutes::from_config(Some(&explicit))
                .unwrap()
                .unwrap();
            assert!(url(&routes, "/v1/custom/x", None).is_some());
            assert_eq!(url(&routes, "/v1/env/x", None), None);
            // An explicit empty list disables forwarding despite the environment.
            assert!(ForwardRoutes::from_config(Some(&[])).unwrap().is_none());
            // An explicit entry is one route: it is never split on whitespace.
            let joined = ["/v1/custom=http://up:8080 /v1/other=http://up:8081".to_string()];
            assert!(ForwardRoutes::from_config(Some(&joined)).is_err());
            // Without explicit routes, the environment's entries apply.
            let routes = ForwardRoutes::from_config(None).unwrap().unwrap();
            assert!(url(&routes, "/v1/env/x", None).is_some());
            assert!(url(&routes, "/v1/env2/x", None).is_some());
        },
    );
}

#[test]
fn upstream_url_keeps_percent_encoded_path() {
    let routes = ForwardRoutes::parse(["/v1/custom=http://up:8080/base"])
        .unwrap()
        .unwrap();
    assert_eq!(
        url(&routes, "/v1/custom/files/a%2Fb%20c", Some("q=a%26b")).as_deref(),
        Some("http://up:8080/base/v1/custom/files/a%2Fb%20c?q=a%26b")
    );
}

#[test]
fn upstream_url_rejects_dot_segments() {
    let routes = ForwardRoutes::parse(["/v1/custom=http://up:8080/base"])
        .unwrap()
        .unwrap();
    for path in [
        "/v1/custom/../metrics",
        "/v1/custom/%2E%2e/x",
        "/v1/custom/./x",
        "/v1/custom/.%2e/x",
        "/v1/custom/%2e./x",
        "/v1/custom/%2e/x",
        "/v1/custom/..\\..\\metrics",
        "/v1/custom/a\\b",
        "/v1/custom/a%2f..%2fmetrics",
        "/v1/custom/a%5C..%5Cmetrics",
        "/v1/custom/a%2F%2e%2E%2Fmetrics",
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
    let routes = ForwardRoutes::parse([
        "/v1/custom=http://up:8080",
        "/v1/custom/admin=http://admin:9000/base/",
    ])
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
