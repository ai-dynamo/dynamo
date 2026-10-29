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
            let routes = ForwardRoutes::from_config(Some(&explicit), None)
                .unwrap()
                .unwrap();
            assert!(url(&routes, "/v1/custom/x", None).is_some());
            assert_eq!(url(&routes, "/v1/env/x", None), None);
            // An explicit empty list disables forwarding despite the environment.
            assert!(
                ForwardRoutes::from_config(Some(&[]), None)
                    .unwrap()
                    .is_none()
            );
            // An explicit entry is one route: it is never split on whitespace.
            let joined = ["/v1/custom=http://up:8080 /v1/other=http://up:8081".to_string()];
            assert!(ForwardRoutes::from_config(Some(&joined), None).is_err());
            // Without explicit routes, the environment's entries apply.
            let routes = ForwardRoutes::from_config(None, None).unwrap().unwrap();
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

mod owner {
    use super::*;
    use crate::namespace::NamespaceFilter;
    use dynamo_runtime::discovery::{
        Discovery, DiscoveryInstance, DiscoverySpec, MockDiscovery, SharedMockRegistry,
    };
    use std::collections::HashSet;
    use std::sync::Arc;

    /// A worker's session API: sessions are created under the body's `id`,
    /// and every answer names the worker.
    async fn session_worker(name: &'static str) -> String {
        use axum::extract::Path;
        use axum::routing::{get, post};

        let sessions = Arc::new(parking_lot::Mutex::new(HashSet::<String>::new()));
        let held = move |sessions: &parking_lot::Mutex<HashSet<String>>, id: &str| {
            if sessions.lock().contains(id) {
                (
                    StatusCode::OK,
                    axum::Json(serde_json::json!({"id": id, "worker": name})),
                )
            } else {
                (
                    StatusCode::NOT_FOUND,
                    axum::Json(serde_json::json!({"worker": name})),
                )
            }
        };
        let app = axum::Router::new()
            .route(
                "/v1/sessions",
                post({
                    let sessions = sessions.clone();
                    move |body: Bytes| async move {
                        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                        let id = body["id"]
                            .as_str()
                            .expect("the frontend names every session");
                        if !sessions.lock().insert(id.to_string()) {
                            return (
                                StatusCode::CONFLICT,
                                axum::Json(serde_json::json!({"worker": name})),
                            );
                        }
                        (
                            StatusCode::OK,
                            axum::Json(serde_json::json!({"id": id, "worker": name})),
                        )
                    }
                }),
            )
            .route(
                "/v1/sessions/{id}",
                get({
                    let sessions = sessions.clone();
                    move |Path(id): Path<String>| async move { held(&sessions, &id) }
                })
                .delete({
                    let sessions = sessions.clone();
                    move |Path(id): Path<String>| async move {
                        let answer = held(&sessions, &id);
                        sessions.lock().remove(&id);
                        answer
                    }
                }),
            )
            .route(
                "/v1/sessions/{id}/warm",
                post({
                    let sessions = sessions.clone();
                    move |Path(id): Path<String>| async move { held(&sessions, &id) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// Registers worker `id`'s model card in `namespace`, advertising `url`.
    async fn register(
        registry: &SharedMockRegistry,
        id: u64,
        namespace: &str,
        url: &str,
    ) -> (MockDiscovery, DiscoveryInstance) {
        let discovery = MockDiscovery::new(Some(id), registry.clone());
        let card = serde_json::json!({
            "display_name": "model",
            "runtime_config": {"runtime_data": {"forward_url": url}},
        });
        let spec = DiscoverySpec::from_model(
            namespace.to_string(),
            "backend".to_string(),
            "generate".to_string(),
            &card,
        )
        .unwrap();
        let instance = discovery.register(spec).await.unwrap();
        (discovery, instance)
    }

    /// A frontend's owner route at `/v1/sessions`, once it sees
    /// `workers` workers.
    async fn frontend(
        registry: &SharedMockRegistry,
        namespaces: NamespaceFilter,
        workers: usize,
    ) -> ForwardRoutes {
        let routes = ForwardRoutes::from_config(
            Some(&["/v1/sessions=owner".to_string()]),
            Some(WorkerDiscovery {
                discovery: Arc::new(MockDiscovery::new(None, registry.clone())),
                namespaces,
                cancel: CancellationToken::new(),
            }),
        )
        .unwrap()
        .unwrap();
        wait_for_workers(&routes, workers).await;
        routes
    }

    async fn wait_for_workers(routes: &ForwardRoutes, count: usize) {
        let owners = routes.owners.as_ref().unwrap();
        tokio::time::timeout(
            Duration::from_secs(5),
            owners.workers().wait_for_count(count),
        )
        .await
        .expect("the frontend sees the workers");
    }

    async fn call(
        routes: &ForwardRoutes,
        method: &str,
        path: &str,
        body: &str,
    ) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = forward(routes, request, CancellationToken::new(), ()).await;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap_or_default())
    }

    #[test]
    fn owner_routes_need_worker_discovery() {
        let routes = ["/v1/sessions=owner".to_string()];
        assert!(ForwardRoutes::parse(&routes).unwrap().is_some());
        let err = ForwardRoutes::from_config(Some(&routes), None)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("worker discovery"), "{err}");
    }

    #[tokio::test]
    async fn requests_follow_the_session_to_its_worker() {
        let registry = SharedMockRegistry::new();
        register(&registry, 1, "ns", &session_worker("a").await).await;
        register(&registry, 2, "ns", &session_worker("b").await).await;
        let first = frontend(&registry, NamespaceFilter::Global, 2).await;
        let second = frontend(&registry, NamespaceFilter::Global, 2).await;

        // Created without an id: the frontend mints one, so it knows the owner.
        let (status, created) = call(&first, "POST", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::OK);
        let id = created["id"].as_str().unwrap().to_string();
        assert!(
            id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "{id}"
        );
        let owner = created["worker"].clone();
        for _ in 0..3 {
            let (status, warmed) =
                call(&first, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
            assert_eq!((status, &warmed["worker"]), (StatusCode::OK, &owner));
        }
        // Another frontend asks the workers.
        let (status, warmed) = call(&second, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
        assert_eq!((status, &warmed["worker"]), (StatusCode::OK, &owner));

        // A chosen id lands on one worker whichever frontend creates it.
        let (status, chosen) = call(&first, "POST", "/v1/sessions", r#"{"id": "agent-1"}"#).await;
        assert_eq!(status, StatusCode::OK);
        let (status, again) = call(&second, "POST", "/v1/sessions", r#"{"id": "agent-1"}"#).await;
        assert_eq!(
            (status, &again["worker"]),
            (StatusCode::CONFLICT, &chosen["worker"])
        );

        // Deleted through one frontend, gone through the other, which held its owner.
        let (status, _) = call(&second, "DELETE", &format!("/v1/sessions/{id}"), "").await;
        assert_eq!(status, StatusCode::OK);
        for _ in 0..2 {
            let (status, _) = call(&first, "GET", &format!("/v1/sessions/{id}"), "").await;
            assert_eq!(status, StatusCode::NOT_FOUND);
        }
        let (status, missing) = call(&second, "GET", "/v1/sessions/nope", "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(missing["message"], "no worker holds this resource");

        let (status, _) = call(&first, "GET", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        for bad in [r#"{"id": 7}"#, r#"{"id": "a/b"}"#, "[]"] {
            let (status, _) = call(&first, "POST", "/v1/sessions", bad).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
        }
    }

    #[tokio::test]
    async fn a_session_whose_worker_left_is_not_found() {
        let registry = SharedMockRegistry::new();
        let a = register(&registry, 1, "ns", &session_worker("a").await).await;
        let b = register(&registry, 2, "ns", &session_worker("b").await).await;
        let routes = frontend(&registry, NamespaceFilter::Global, 2).await;

        let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::OK);
        let id = created["id"].as_str().unwrap().to_string();
        let (discovery, card) = if created["worker"] == "a" { a } else { b };
        discovery.unregister(card).await.unwrap();
        wait_for_workers(&routes, 1).await;

        let (status, _) = call(&routes, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_worker_that_cannot_answer_makes_a_lookup_unavailable() {
        let registry = SharedMockRegistry::new();
        register(&registry, 1, "ns", &session_worker("a").await).await;
        let routes = frontend(&registry, NamespaceFilter::Global, 1).await;
        let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::OK);
        let id = created["id"].as_str().unwrap().to_string();

        register(&registry, 2, "ns", "http://127.0.0.1:1").await;
        let other = frontend(&registry, NamespaceFilter::Global, 2).await;
        // The owner answers, so the silent worker does not matter.
        let (status, warmed) = call(&other, "POST", &format!("/v1/sessions/{id}/warm"), "").await;
        assert_eq!(
            (status, &warmed["worker"]),
            (StatusCode::OK, &created["worker"])
        );
        // Without an owner, the silent worker might hold it: not a 404.
        let (status, _) = call(&other, "GET", "/v1/sessions/nope", "").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn only_workers_in_the_served_namespaces_take_sessions() {
        let registry = SharedMockRegistry::new();
        register(&registry, 1, "ns", &session_worker("a").await).await;
        register(&registry, 2, "other", &session_worker("b").await).await;
        let routes = frontend(&registry, NamespaceFilter::Exact("ns".to_string()), 1).await;
        for _ in 0..8 {
            let (status, created) = call(&routes, "POST", "/v1/sessions", "").await;
            assert_eq!(
                (status, &created["worker"]),
                (StatusCode::OK, &serde_json::json!("a"))
            );
        }

        let empty = frontend(&SharedMockRegistry::new(), NamespaceFilter::Global, 0).await;
        let (status, _) = call(&empty, "POST", "/v1/sessions", "").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
