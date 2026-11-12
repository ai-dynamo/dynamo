// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Router,
    body::{Body, Bytes},
    routing::get,
};
use dynamo_http_server::server::serve;
use std::{future::Future, io, net::SocketAddr, sync::Arc, time::Duration};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio_rustls::TlsAcceptor;
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};
// Actual HTTP and TLS connections with a client that stops consuming data.
// Runs the same accept loop used by the frontend, without GPU/model infrastructure.
// Serial: the HTTP/2 test below shortens the transfer deadline through the environment.
#[tokio::test]
#[serial_test::serial]
async fn stalled_download_is_bounded_for_http_and_https() {
    tokio::time::timeout(Duration::from_secs(15), async {
        for tls in [false, true] {
            for shutdown in [false, true] {
                let (upstream, released, _upstream) = spawn_upstream().await;
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
                let idle = if shutdown {
                    Duration::from_secs(30)
                } else {
                    Duration::from_millis(100)
                };
                let stop = CancellationToken::new();
                let stopped = stop.clone().cancelled_owned();
                let (addr, frontend) = spawn_frontend(
                    upstream,
                    tls.then(|| tls_acceptor(&cert)),
                    idle,
                    async move {
                        stopped.await;
                        Duration::from_millis(100)
                    },
                )
                .await;
                let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
                let mut client: Box<dyn TestConnection> = if tls {
                    let mut roots = rustls::RootCertStore::empty();
                    roots.add(cert.cert.der().clone()).unwrap();
                    let provider = Arc::new(rustls::crypto::ring::default_provider());
                    let config = rustls::ClientConfig::builder_with_provider(provider)
                        .with_safe_default_protocol_versions()
                        .unwrap()
                        .with_root_certificates(roots)
                        .with_no_client_auth();
                    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
                    let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
                    Box::new(connector.connect(name, stream).await.unwrap())
                } else {
                    Box::new(stream)
                };
                client
                    .write_all(b"GET /v1/files/file-1/content HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .await
                    .unwrap();
                client.flush().await.unwrap();
                let mut headers = Vec::new();
                while !headers.ends_with(b"\r\n\r\n") {
                    assert!(headers.len() < 8192);
                    headers.push(client.read_u8().await.unwrap());
                }
                assert!(headers.starts_with(b"HTTP/1.1 200"));
                // Keep the socket open but stop reading, filling downstream buffers.
                if shutdown {
                    stop.cancel();
                    tokio::time::timeout(Duration::from_secs(3), frontend)
                        .await
                        .expect("shutdown waited on the stalled connection")
                        .unwrap();
                }
                tokio::time::timeout(Duration::from_secs(3), released.cancelled())
                    .await
                    .expect("stalled connection retained the upstream response");
            }
        }
    })
    .await
    .unwrap();
}

trait TestConnection: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> TestConnection for T {}

#[tokio::test]
#[serial_test::serial]
async fn http2_transfer_deadline_closes_unconsumed_response() {
    use dynamo_runtime::config::environment_names::llm::DYN_BATCH_FILE_TRANSFER_TIMEOUT_SECS;
    temp_env::async_with_vars(
        [(DYN_BATCH_FILE_TRANSFER_TIMEOUT_SECS, Some("1"))],
        tokio::time::timeout(Duration::from_secs(10), async {
            for tls in [false, true] {
                let (upstream, released, _upstream) = spawn_upstream().await;
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
                let (addr, _frontend) = spawn_frontend(
                    upstream,
                    tls.then(|| tls_acceptor(&cert)),
                    Duration::from_secs(30),
                    std::future::pending(),
                )
                .await;
                let client = reqwest::Client::builder()
                    .no_proxy()
                    .timeout(Duration::from_secs(5))
                    .add_root_certificate(reqwest::Certificate::from_der(cert.cert.der()).unwrap());
                let client = if tls {
                    client
                } else {
                    client.http2_prior_knowledge()
                };
                let scheme = if tls { "https" } else { "http" };
                let response = client
                    .build()
                    .unwrap()
                    .get(format!(
                        "{scheme}://localhost:{}/v1/files/file-1/content",
                        addr.port()
                    ))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.version(), reqwest::Version::HTTP_2);
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                // Keep the response unread. No socket-write stall or shutdown can make this pass.
                tokio::time::timeout(Duration::from_secs(3), released.cancelled())
                    .await
                    .expect("HTTP/2 transfer deadline retained upstream response");
            }
        }),
    )
    .await
    .unwrap();
}

/// Serves an endless file download; the token is cancelled once its body is dropped.
async fn spawn_upstream() -> (SocketAddr, CancellationToken, AbortOnDropHandle<()>) {
    let released = CancellationToken::new();
    let body_released = released.clone();
    let router = Router::new().route(
        "/v1/files/file-1/content",
        get(move || {
            let guard = body_released.clone().drop_guard();
            async move {
                let chunk = Bytes::from(vec![b'x'; 65536]);
                Body::from_stream(async_stream::stream! {
                    let _guard = guard;
                    loop { yield Ok::<_, io::Error>(chunk.clone()); }
                })
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let idle = Duration::from_secs(30);
    let server = serve(listener, None, router, idle, std::future::pending());
    (addr, released, AbortOnDropHandle::new(tokio::spawn(server)))
}

/// Serves the batch proxy for `upstream` on the frontend's accept loop.
async fn spawn_frontend(
    upstream: SocketAddr,
    tls: Option<TlsAcceptor>,
    write_idle_timeout: Duration,
    shutdown: impl Future<Output = Duration> + Send + 'static,
) -> (SocketAddr, AbortOnDropHandle<()>) {
    let (_, router) = super::batch_proxy::router(
        &format!("http://{upstream}"),
        None,
        None,
        1024,
        1024,
        CancellationToken::new(),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = serve(listener, tls, router, write_idle_timeout, shutdown);
    (addr, AbortOnDropHandle::new(tokio::spawn(server)))
}

fn tls_acceptor(cert: &rcgen::CertifiedKey) -> TlsAcceptor {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let key = rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.der().clone()], key.into())
        .unwrap();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    TlsAcceptor::from(Arc::new(config))
}
