// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::{Method, Uri};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;
use tracing::instrument::WithSubscriber;

struct Server {
    origin: String,
    cancel: CancellationToken,
    task: Option<JoinHandle<anyhow::Result<()>>>,
}

impl Server {
    async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let cancel = CancellationToken::new();
        let shutdown = cancel.clone();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown.cancelled_owned())
                .await?;
            Ok(())
        });
        Self {
            origin,
            cancel,
            task: Some(task),
        }
    }

    async fn stop(mut self) {
        self.cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), self.task.as_mut().unwrap())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let _ = self.task.take();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

struct Captured {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
}

async fn echo_request(State(sender): State<mpsc::Sender<Captured>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(body) => body,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let response_body = body.clone();
    sender
        .send(Captured {
            method: parts.method,
            uri: parts.uri,
            headers: parts.headers,
            body,
        })
        .await
        .unwrap();
    (
        StatusCode::MULTI_STATUS,
        [
            ("content-type", "application/x-ndjson"),
            ("connection", "x-upstream-hop"),
            ("x-upstream-hop", "remove-me"),
            ("x-upstream-trace", "keep-me"),
        ],
        response_body,
    )
        .into_response()
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

#[derive(Clone, Default)]
struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CapturedLogs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn failed_requests_log_bounded_classification_without_client_values() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let unavailable = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let state = ProxyState {
            client: client(),
            gateway: Url::parse(&unavailable).unwrap(),
            files_path: "/v1/files".to_string(),
            batches_path: "/v1/batches".to_string(),
            max_body: 1024,
            cancel: CancellationToken::new(),
        };
        let logs = CapturedLogs::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_target(false)
            .with_max_level(tracing::Level::WARN)
            .with_writer(move || writer.clone())
            .finish();
        let subscriber = tracing::Dispatch::new(subscriber);
        let mut log_length = None;
        for length in [8, 32 * 1024] {
            logs.0.lock().unwrap().clear();
            let request = Request::builder()
                .uri(format!(
                    "/v1/batches/batch-UNTRUSTED?secret={}&forged=%0AERROR%20UNTRUSTED%1B",
                    "x".repeat(length)
                ))
                .header("authorization", "Bearer UNTRUSTED")
                .body(Body::empty())
                .unwrap();
            let response = forward(State(state.clone()), request)
                .with_subscriber(subscriber.clone())
                .await;
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            let body = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(error["error"]["message"], "Unable to reach Batch gateway");
            let recorded = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
            assert_eq!(recorded.matches("Batch gateway request failed").count(), 1);
            assert!(recorded.contains("is_connect=true"));
            assert!(recorded.contains("is_timeout=false"));
            assert!(!recorded.contains("UNTRUSTED"));
            assert!(!recorded.contains("secret="));
            assert!(recorded.len() < 256);
            assert_eq!(*log_length.get_or_insert(recorded.len()), recorded.len());
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn keeps_client_controlled_uris_and_headers_on_the_configured_origin() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let (canary_sender, mut canary_requests) = mpsc::channel(1);
        let canary =
            Server::start(Router::new().fallback(echo_request).with_state(canary_sender)).await;
        let (sender, mut captured) = mpsc::channel(1);
        let upstream = Server::start(Router::new().fallback(echo_request).with_state(sender)).await;
        let (_, proxy) =
            router(&upstream.origin, None, None, 1024, CancellationToken::new()).unwrap();
        let proxy = Server::start(proxy).await;
        let canary_url = Url::parse(&canary.origin).unwrap();
        let upstream_url = Url::parse(&upstream.origin).unwrap();
        for (target, status) in [
            (
                format!("{}/v1/files/file-1?next={}", canary.origin, canary.origin),
                StatusCode::MULTI_STATUS,
            ),
            (
                format!("/v1/files/http%3A%2F%2F{}", canary_url.authority()),
                StatusCode::MULTI_STATUS,
            ),
            (
                "/v1/files/%2e%2e/content".to_string(),
                StatusCode::BAD_REQUEST,
            ),
            (
                format!("/v1/files/file-1?next={}&next=//{}&empty=&encoded=%23", canary.origin, canary_url.authority()),
                StatusCode::MULTI_STATUS,
            ),
            (
                "/v1/files/.%2E/content".to_string(),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/files/%2E./content".to_string(),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            // Exercise absolute-form and encoded paths without client-side rewriting.
            let mut connection = TcpStream::connect(Url::parse(&proxy.origin).unwrap().authority())
                .await
                .unwrap();
            connection
                .write_all(
                    format!(
                        "GET {target} HTTP/1.1\r\nHost: {}\r\nX-Forwarded-Host: {}\r\nConnection: close\r\n\r\n",
                        canary_url.authority(),
                        canary_url.authority(),
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            let mut response = Vec::new();
            connection.read_to_end(&mut response).await.unwrap();
            assert!(response.starts_with(format!("HTTP/1.1 {}", status.as_u16()).as_bytes()));
            assert!(canary_requests.try_recv().is_err());
            if status == StatusCode::BAD_REQUEST {
                assert!(captured.try_recv().is_err());
                continue;
            }
            let forwarded = captured.recv().await.unwrap();
            assert_eq!(forwarded.headers[header::HOST], upstream_url.authority());
            let target: Uri = target.parse().unwrap();
            assert_eq!(forwarded.uri.path(), target.path());
            assert_eq!(forwarded.uri.query(), target.query());
        }
        proxy.stop().await;
        upstream.stop().await;
        canary.stop().await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn proxies_full_batch_api_and_nested_mounts_without_changing_requests() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let (sender, mut captured) = mpsc::channel(1);
        let upstream = Server::start(Router::new().fallback(echo_request).with_state(sender)).await;
        let (docs, proxy) = router(
            &upstream.origin,
            Some("/api".to_string()),
            Some("/api/batches".to_string()),
            1024,
            CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(docs.len(), 9);
        let proxy = Server::start(proxy).await;
        let client = client();
        let multipart = "--batch-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.jsonl\"\r\nContent-Type: application/jsonl\r\n\r\n{\"custom_id\":\"one\"}\n\r\n--batch-boundary--\r\n";
        for (method, path, upstream_path) in [
            (Method::GET, "/api", "/v1/files"),
            (Method::POST, "/api", "/v1/files"),
            (Method::GET, "/api/file-1", "/v1/files/file-1"),
            (Method::DELETE, "/api/file-1", "/v1/files/file-1"),
            (Method::GET, "/api/file-1/content", "/v1/files/file-1/content"),
            (Method::GET, "/api/batches", "/v1/batches"),
            (Method::POST, "/api/batches", "/v1/batches"),
            (Method::GET, "/api/batches/batch-1", "/v1/batches/batch-1"),
            (Method::POST, "/api/batches/batch-1/cancel", "/v1/batches/batch-1/cancel"),
        ] {
            let body = if method == Method::POST && path == "/api" {
                multipart
            } else {
                ""
            };
            let response = client
                .request(method.clone(), format!("{}{path}?after=a%2Fb&limit=2", proxy.origin))
                .header("authorization", "Bearer test-token")
                .header("x-tenant-id", "tenant-a")
                .header("content-type", "multipart/form-data; boundary=batch-boundary")
                .header("connection", "x-client-hop")
                .header("x-client-hop", "remove-me")
                .body(body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::MULTI_STATUS);
            assert_eq!(response.headers()["x-upstream-trace"], "keep-me");
            assert_eq!(response.headers()["content-type"], "application/x-ndjson");
            assert!(!response.headers().contains_key("x-upstream-hop"));
            assert!(!response.headers().contains_key("connection"));
            assert_eq!(response.bytes().await.unwrap(), body.as_bytes());
            let request = captured.recv().await.unwrap();
            assert_eq!(request.method, method);
            assert_eq!(request.uri.path(), upstream_path);
            assert_eq!(request.uri.query(), Some("after=a%2Fb&limit=2"));
            assert_eq!(request.body, body.as_bytes());
            assert_eq!(request.headers["authorization"], "Bearer test-token");
            assert_eq!(request.headers["x-tenant-id"], "tenant-a");
            assert_eq!(request.headers["content-type"], "multipart/form-data; boundary=batch-boundary");
            assert!(!request.headers.contains_key("x-client-hop"));
        }
        proxy.stop().await;
        upstream.stop().await;
    }).await.unwrap();
}

#[tokio::test]
async fn rejects_invalid_configuration_and_oversize_uploads() {
    tokio::time::timeout(Duration::from_secs(15), async {
        for origin in [
            "ftp://example.test",
            "http://user@example.test",
            "http://example.test/path",
            "http://example.test/?query=x",
            "http://example.test/#fragment",
        ] {
            assert!(router(origin, None, None, 8, CancellationToken::new()).is_err());
        }
        for path in ["/", "/files/", "files", "/files/{id}"] {
            assert!(
                router(
                    "http://example.test",
                    Some(path.to_string()),
                    None,
                    8,
                    CancellationToken::new()
                )
                .is_err()
            );
        }
        assert!(
            router(
                "http://example.test",
                Some("/same".to_string()),
                Some("/same".to_string()),
                8,
                CancellationToken::new()
            )
            .is_err()
        );
        let (sender, mut captured) = mpsc::channel(1);
        let upstream = Server::start(Router::new().fallback(echo_request).with_state(sender)).await;
        let (_, proxy) = router(&upstream.origin, None, None, 8, CancellationToken::new()).unwrap();
        let proxy = Server::start(proxy).await;
        let client = client();
        let response = client
            .post(format!("{}/v1/files", proxy.origin))
            .body("123456789")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(captured.try_recv().is_err());
        let chunks = futures::stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"12345678")),
            Ok(Bytes::from_static(b"9")),
        ]);
        let response = client
            .post(format!("{}/v1/files", proxy.origin))
            .body(reqwest::Body::wrap_stream(chunks))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        proxy.stop().await;
        upstream.stop().await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn streams_output_and_preserves_redirects_failures_and_cancellation() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let release = Arc::new(Notify::new());
        let stream_release = release.clone();
        let upstream = Server::start(Router::new().fallback(move || {
            let release = stream_release.clone();
            async move {
                let body = async_stream::stream! {
                    yield Ok::<_, std::io::Error>(Bytes::from_static(b"first\n"));
                    release.notified().await;
                    yield Ok(Bytes::from_static(b"second\n"));
                };
                Body::from_stream(body)
            }
        }))
        .await;
        let cancel = CancellationToken::new();
        let (_, proxy) = router(&upstream.origin, None, None, 1024, cancel.clone()).unwrap();
        let proxy = Server::start(proxy).await;
        let client = client();
        let mut response = client
            .get(format!("{}/v1/files/file-1/content", proxy.origin))
            .send()
            .await
            .unwrap();
        let first = tokio::time::timeout(Duration::from_secs(2), response.chunk())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first, "first\n");
        release.notify_one();
        assert_eq!(response.bytes().await.unwrap(), "second\n");
        cancel.cancel();
        assert_eq!(
            client
                .get(format!("{}/v1/batches", proxy.origin))
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        proxy.stop().await;
        upstream.stop().await;

        let upstream = Server::start(Router::new().fallback(|request: Request| async move {
            if request.uri().path() == "/v1/batches/bad" {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("retry-after", "7")],
                    Json(serde_json::json!({"error": {"message": "queue full"}})),
                )
                    .into_response();
            }
            (
                StatusCode::TEMPORARY_REDIRECT,
                [("location", "http://must-not-follow.invalid")],
                "redirect-body",
            )
                .into_response()
        }))
        .await;
        let (_, proxy) =
            router(&upstream.origin, None, None, 1024, CancellationToken::new()).unwrap();
        let proxy = Server::start(proxy).await;
        let response = client
            .get(format!("{}/v1/batches", proxy.origin))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(
            response.headers()["location"],
            "http://must-not-follow.invalid"
        );
        assert_eq!(response.text().await.unwrap(), "redirect-body");
        let response = client
            .get(format!("{}/v1/batches/bad", proxy.origin))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "7");
        assert_eq!(
            response.json::<serde_json::Value>().await.unwrap(),
            serde_json::json!({"error": {"message": "queue full"}})
        );
        proxy.stop().await;
        upstream.stop().await;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let unavailable = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let (_, proxy) = router(&unavailable, None, None, 1024, CancellationToken::new()).unwrap();
        let proxy = Server::start(proxy).await;
        let response = client
            .post(format!("{}/v1/batches", proxy.origin))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            response.json::<serde_json::Value>().await.unwrap()["error"]["message"],
            "Unable to reach Batch gateway"
        );
        proxy.stop().await;
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn does_not_follow_gateway_redirects_to_another_origin() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let (sender, mut captured) = mpsc::channel(1);
        let canary = Server::start(Router::new().fallback(echo_request).with_state(sender)).await;
        let destination = format!("{}/private", canary.origin);
        let location = destination.clone();
        let upstream = Server::start(Router::new().fallback(move || {
            let location = location.clone();
            async move {
                (
                    StatusCode::TEMPORARY_REDIRECT,
                    [(header::LOCATION, location)],
                )
            }
        }))
        .await;
        let (_, proxy) =
            router(&upstream.origin, None, None, 1024, CancellationToken::new()).unwrap();
        let proxy = Server::start(proxy).await;
        let response = client()
            .get(format!("{}/v1/batches", proxy.origin))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(response.headers()[header::LOCATION], destination);
        // A live listener distinguishes redirect blocking from a failed DNS lookup.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), captured.recv())
                .await
                .is_err()
        );
        proxy.stop().await;
        upstream.stop().await;
        canary.stop().await;
    })
    .await
    .unwrap();
}
