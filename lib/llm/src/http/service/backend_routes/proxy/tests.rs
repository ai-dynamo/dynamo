// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::config::environment_names::llm as env_llm;

/// Sends `request` to the worker at `base`, under the same path, and answers
/// as the frontend does.
async fn exchange<P: Send + 'static>(
    base: &str,
    request: Request,
    cancel: CancellationToken,
    permit: P,
) -> Response {
    let url = reqwest::Url::parse(&format!("{base}{}", request.uri())).unwrap();
    match send(&client().unwrap(), request, &url).await {
        Ok((upstream, upgrade)) => client_response(upstream, upgrade, url, cancel, permit),
        Err(refused) => refused,
    }
}

/// Reads one request head from `conn`.
async fn read_head(conn: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        head.push(conn.read_u8().await.unwrap());
    }
    String::from_utf8_lossy(&head).to_ascii_lowercase()
}

#[tokio::test]
async fn proxies_request_and_streams_response() {
    use axum::routing::post;

    let worker = axum::Router::new().route(
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
    tokio::spawn(async move { axum::serve(listener, worker).await.unwrap() });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/custom/s1/push?x=1")
        .header(header::AUTHORIZATION, "Bearer k")
        .header(header::HOST, "frontend.example")
        .header(header::CONNECTION, "close, x-internal")
        .header("x-internal", "secret")
        .body(Body::from("hello"))
        .unwrap();
    let response = exchange(
        &format!("http://{addr}"),
        request,
        CancellationToken::new(),
        (),
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&body),
        format!("s1;x=1;Bearer k;{addr};-;hello;")
    );

    let request = Request::builder()
        .uri("/v1/custom/s1")
        .body(Body::empty())
        .unwrap();
    let response = exchange("http://127.0.0.1:1", request, CancellationToken::new(), ()).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn tunnels_websocket_upgrades() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Answers each upgrade with 101, then echoes bytes.
    let worker = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", worker.local_addr().unwrap());
    tokio::spawn(async move {
        while let Ok((mut conn, _)) = worker.accept().await {
            tokio::spawn(async move {
                let head = read_head(&mut conn).await;
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

    // Stands in for the inflight permit, which must outlive the 101.
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let live = Arc::new(AtomicUsize::new(0));
    let held = live.clone();
    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    let frontend = axum::Router::new().fallback(move |request: Request| {
        let (base, live, cancel) = (base.clone(), live.clone(), cancel.clone());
        async move {
            live.fetch_add(1, Ordering::SeqCst);
            exchange(&base, request, cancel, Guard(live)).await
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
        let head = read_head(&mut client).await;
        assert!(head.starts_with("http/1.1 101"), "{head}");
        assert!(head.contains("upgrade: websocket") && head.contains("connection: upgrade"));
        client.write_all(b"ping").await.unwrap();
        let mut echoed = [0u8; 4];
        client.read_exact(&mut echoed).await.unwrap();
        assert_eq!(&echoed, b"ping");
        client
    };
    let live = || held.load(Ordering::SeqCst);
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
async fn does_not_tunnel_non_websocket_upgrades() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let worker = axum::Router::new().route(
        "/v1/custom/x",
        axum::routing::get(|headers: HeaderMap| async move {
            let seen = ["upgrade", "http2-settings"]
                .map(|name| format!("{name}={}", headers.contains_key(name)));
            seen.join(";")
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, worker).await.unwrap() });

    let frontend = axum::Router::new().fallback(move |request: Request| {
        let base = base.clone();
        async move { exchange(&base, request, CancellationToken::new(), ()).await }
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
#[serial_test::serial]
async fn checks_body_limit_before_the_worker_answers() {
    temp_env::async_with_vars([(env_llm::DYN_HTTP_BODY_LIMIT_MB, Some("1"))], async move {
        use tokio::io::AsyncWriteExt;

        // Answers with the request head's framing as soon as the head
        // arrives, without waiting for the body.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            while let Ok((mut conn, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let head = read_head(&mut conn).await;
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
        let chunked = |size: usize| {
            let chunks = vec![b'x'; size]
                .chunks(64 * 1024)
                .map(|c| Ok::<_, std::io::Error>(Bytes::copy_from_slice(c)))
                .collect::<Vec<_>>();
            Request::builder()
                .method("POST")
                .uri("/v1/custom/s1/push")
                .body(Body::from_stream(futures::stream::iter(chunks)))
                .unwrap()
        };

        // Declared length: the body is tiny, so only the Content-Length
        // check can refuse it.
        let declared = Request::builder()
            .method("POST")
            .uri("/v1/custom/s1/push")
            .header(header::CONTENT_LENGTH, 2 * 1024 * 1024)
            .body(Body::from("x"))
            .unwrap();
        let response = exchange(&base, declared, CancellationToken::new(), ()).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);

        // Unknown length: refused once more than the limit has been read.
        let response = exchange(
            &base,
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
        let response = exchange(&base, stale, CancellationToken::new(), ()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], format!("content-length: {}", 100 * 1024).as_bytes());
    })
    .await;
}

#[tokio::test]
async fn drops_content_length_sent_beside_chunked_response() {
    use tokio::io::AsyncWriteExt;

    // A misbehaving worker: both framings, and the chunked body is longer.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut conn, _) = listener.accept().await.unwrap();
        read_head(&mut conn).await;
        conn.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 5\r\ntransfer-encoding: chunked\r\n\r\nb\r\nhello world\r\n0\r\n\r\n")
            .await
            .unwrap();
    });
    let request = Request::builder()
        .uri("/v1/custom/x")
        .body(Body::empty())
        .unwrap();
    let response = exchange(&base, request, CancellationToken::new(), ()).await;
    assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], b"hello world");
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
fn scoped_url_keeps_the_path_under_the_base() {
    let base = reqwest::Url::parse("http://worker:8080/base/").unwrap();
    let url = |path, query| scoped_url(&base, "/v1/custom", path, query).map(|u| u.to_string());
    assert_eq!(
        url("/v1/custom/files/a%2Fb%20c", Some("q=a%26b")).as_deref(),
        Some("http://worker:8080/base/v1/custom/files/a%2Fb%20c?q=a%26b")
    );
    assert_eq!(url("/v1/custom/../metrics", None), None);
}

#[test]
fn upstream_base_refuses_credentials_without_echoing_them() {
    for bad in [
        "http://user:hunter2@worker:1",
        "http://user@worker:1",
        "http://user:hunter2@%",
        "ftp://user:hunter2@worker:1",
        "http://user:hunter2@worker:1/?q=1",
    ] {
        let err = upstream_base(bad).unwrap_err();
        assert!(!err.contains("hunter2") && !err.contains("user"), "{err}");
    }
    for bad in [
        "not-a-url",
        "ftp://worker:1",
        "http://worker:1/?q=1",
        "http://worker:1/#f",
    ] {
        assert!(upstream_base(bad).is_err(), "{bad}");
    }
    assert!(upstream_base("https://worker:1/base").is_ok());
}
