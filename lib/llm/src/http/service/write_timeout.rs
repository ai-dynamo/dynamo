// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connection-level write and batch-transfer bounds, independent of body polling.

use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use std::time::Duration;

use axum_server::accept::Accept;
use futures::task::AtomicWaker;
use parking_lot::Mutex;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::time::Instant;
use tokio::time::Sleep;

#[derive(Clone, Default)]
pub(super) struct ConnectionDeadline(Arc<Deadlines>);

#[derive(Default)]
struct Deadlines {
    transfers: Mutex<BTreeMap<Instant, usize>>,
    is_active: AtomicBool,
    waker: AtomicWaker,
}

impl ConnectionDeadline {
    pub(super) fn start(&self, timeout: Duration) -> TransferDeadline {
        let now = Instant::now();
        let deadline = now
            .checked_add(timeout)
            .unwrap_or_else(|| now + Duration::from_secs(600));
        *self.0.transfers.lock().entry(deadline).or_default() += 1;
        self.0.is_active.store(true, Ordering::Release);
        self.0.waker.wake();
        TransferDeadline {
            connection: self.clone(),
            deadline,
        }
    }

    fn next(&self, cx: &mut Context<'_>) -> Option<Instant> {
        self.0.waker.register(cx.waker());
        if !self.0.is_active.load(Ordering::Acquire) {
            return None;
        }
        self.0
            .transfers
            .lock()
            .first_key_value()
            .map(|(deadline, _)| *deadline)
    }
}

pub(super) struct TransferDeadline {
    connection: ConnectionDeadline,
    deadline: Instant,
}

impl Drop for TransferDeadline {
    fn drop(&mut self) {
        let mut transfers = self.connection.0.transfers.lock();
        if let Some(count) = transfers.get_mut(&self.deadline) {
            *count -= 1;
            if *count == 0 {
                transfers.remove(&self.deadline);
            }
        }
        self.connection
            .0
            .is_active
            .store(!transfers.is_empty(), Ordering::Release);
        drop(transfers);
        self.connection.0.waker.wake();
    }
}

#[derive(Clone)]
pub(super) struct ConnectionService<S> {
    inner: S,
    deadline: ConnectionDeadline,
}

impl<S, B> tower::Service<axum::http::Request<B>> for ConnectionService<S>
where
    S: tower::Service<axum::http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: axum::http::Request<B>) -> Self::Future {
        request.extensions_mut().insert(self.deadline.clone());
        self.inner.call(request)
    }
}

#[derive(Clone)]
pub(super) struct WriteTimeoutAcceptor<A> {
    inner: A,
    timeout: Duration,
}

impl<A> WriteTimeoutAcceptor<A> {
    pub(super) fn new(inner: A, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

impl<A, I, S> Accept<I, S> for WriteTimeoutAcceptor<A>
where
    A: Accept<I, S>,
    A::Future: Send + 'static,
    A::Stream: Send + 'static,
    A::Service: Send + 'static,
{
    type Stream = WriteTimeoutIo<A::Stream>;
    type Service = ConnectionService<A::Service>;
    type Future = Pin<Box<dyn Future<Output = io::Result<(Self::Stream, Self::Service)>> + Send>>;

    fn accept(&self, stream: I, service: S) -> Self::Future {
        let accepted = self.inner.accept(stream, service);
        let timeout = self.timeout;
        Box::pin(async move {
            let (inner, service) = accepted.await?;
            let deadline = ConnectionDeadline::default();
            Ok((
                WriteTimeoutIo {
                    inner,
                    timeout,
                    stalled: None,
                    is_write_pending: false,
                    expired: false,
                    connection_deadline: deadline.clone(),
                    transfer_timer: None,
                },
                ConnectionService {
                    inner: service,
                    deadline,
                },
            ))
        })
    }
}

pub(super) struct WriteTimeoutIo<I> {
    inner: I,
    timeout: Duration,
    stalled: Option<Pin<Box<Sleep>>>,
    is_write_pending: bool,
    expired: bool,
    connection_deadline: ConnectionDeadline,
    transfer_timer: Option<Pin<Box<Sleep>>>,
}

impl<I> WriteTimeoutIo<I> {
    fn check_timeout(&mut self, cx: &mut Context<'_>) -> io::Result<()> {
        if let Some(deadline) = self.connection_deadline.next(cx) {
            let timer = self
                .transfer_timer
                .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(deadline)));
            if timer.deadline() != deadline {
                timer.as_mut().reset(deadline);
            }
            if timer.as_mut().poll(cx).is_ready() {
                self.expired = true;
            }
        } else {
            self.transfer_timer = None;
        }
        if self
            .stalled
            .as_mut()
            .is_some_and(|timer| timer.as_mut().poll(cx).is_ready())
        {
            self.expired = true;
        }
        if self.expired {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Frontend connection transfer deadline expired",
            ));
        }
        Ok(())
    }

    fn start_timer(&mut self, cx: &mut Context<'_>) {
        let timer = self
            .stalled
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(self.timeout)));
        // Wake the connection task even when Hyper stops polling the body.
        let _ = timer.as_mut().poll(cx);
    }

    fn track_write(&mut self, cx: &mut Context<'_>, result: &Poll<io::Result<usize>>) {
        match result {
            Poll::Pending => {
                self.is_write_pending = true;
                self.start_timer(cx);
            }
            Poll::Ready(Ok(written)) if *written > 0 => {
                self.is_write_pending = false;
                self.stalled = None;
            }
            _ => {}
        }
    }

    fn track_flush(&mut self, cx: &mut Context<'_>, result: &Poll<io::Result<()>>) {
        match result {
            Poll::Pending => self.start_timer(cx),
            // A no-op flush must not reset a blocked write's deadline.
            Poll::Ready(Ok(())) if !self.is_write_pending => self.stalled = None,
            _ => {}
        }
    }
}

impl<I: AsyncRead + Unpin> AsyncRead for WriteTimeoutIo<I> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.check_timeout(cx)?;
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<I: AsyncWrite + Unpin> AsyncWrite for WriteTimeoutIo<I> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.check_timeout(cx)?;
        let result = Pin::new(&mut self.inner).poll_write(cx, buf);
        self.track_write(cx, &result);
        result
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.check_timeout(cx)?;
        let result = Pin::new(&mut self.inner).poll_write_vectored(cx, bufs);
        self.track_write(cx, &result);
        result
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.check_timeout(cx)?;
        let result = Pin::new(&mut self.inner).poll_flush(cx);
        self.track_flush(cx, &result);
        result
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.check_timeout(cx)?;
        let result = Pin::new(&mut self.inner).poll_shutdown(cx);
        self.track_flush(cx, &result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, Bytes},
        routing::get,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test(start_paused = true)]
    async fn completed_transfers_remove_their_connection_deadline() {
        let (inner, _) = tokio::io::duplex(1);
        let connection = ConnectionDeadline::default();
        let first = connection.start(Duration::from_secs(1));
        let second = connection.start(Duration::from_secs(5));
        let mut writer = WriteTimeoutIo {
            inner,
            timeout: Duration::from_secs(30),
            stalled: None,
            is_write_pending: false,
            expired: false,
            connection_deadline: connection.clone(),
            transfer_timer: None,
        };
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        writer.check_timeout(&mut cx).unwrap();
        assert_eq!(
            writer.transfer_timer.as_ref().unwrap().deadline(),
            first.deadline
        );
        drop(first);
        writer.check_timeout(&mut cx).unwrap();
        assert_eq!(
            writer.transfer_timer.as_ref().unwrap().deadline(),
            second.deadline
        );
        tokio::time::advance(Duration::from_secs(2)).await;
        writer.check_timeout(&mut cx).unwrap();
        drop(second);
        tokio::time::advance(Duration::from_secs(10)).await;
        writer.check_timeout(&mut cx).unwrap();
        assert!(writer.transfer_timer.is_none());
        let _expired = connection.start(Duration::from_secs(1));
        writer.check_timeout(&mut cx).unwrap();
        tokio::time::advance(Duration::from_secs(2)).await;
        assert_eq!(
            writer.check_timeout(&mut cx).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
    }

    #[tokio::test]
    async fn write_progress_resets_deadline_and_stalled_write_expires() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let (inner, mut reader) = tokio::io::duplex(1);
            let mut writer = WriteTimeoutIo {
                inner,
                timeout: Duration::from_millis(100),
                stalled: None,
                is_write_pending: false,
                expired: false,
                connection_deadline: ConnectionDeadline::default(),
                transfer_timer: None,
            };
            writer.write_all(b"a").await.unwrap();
            let write = writer.write_all(b"b");
            tokio::pin!(write);
            // Start a stalled write, then free capacity before its idle deadline.
            assert!(
                tokio::time::timeout(Duration::from_millis(5), &mut write)
                    .await
                    .is_err()
            );
            let mut byte = [0];
            reader.read_exact(&mut byte).await.unwrap();
            write.await.unwrap();
            assert_eq!(byte, *b"a");
            {
                let bufs = [io::IoSlice::new(b"c")];
                let write = writer.write_vectored(&bufs);
                tokio::pin!(write);
                assert!(
                    tokio::time::timeout(Duration::from_millis(5), &mut write)
                        .await
                        .is_err()
                );
            }
            // DuplexStream's flush is a no-op: it cannot renew a blocked write.
            writer.flush().await.unwrap();
            let error = writer.write_all(b"c").await.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::TimedOut);
            // An expired connection must not become usable again.
            reader.read_exact(&mut byte).await.unwrap();
            assert_eq!(
                writer.write_all(b"d").await.unwrap_err().kind(),
                io::ErrorKind::TimedOut
            );
        })
        .await
        .unwrap();
    }

    // Actual HTTP and TLS connections with a client that stops consuming data.
    // Runs the same acceptor used by the frontend, without GPU/model infrastructure.
    #[tokio::test]
    async fn stalled_download_is_bounded_for_http_and_https() {
        tokio::time::timeout(Duration::from_secs(15), async {
            for tls in [false, true] {
                for shutdown in [false, true] {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    let addr = listener.local_addr().unwrap();
                    let handle = axum_server::Handle::new();
                    let idle = if shutdown {
                        Duration::from_secs(30)
                    } else {
                        Duration::from_millis(100)
                    };
                    let upstream_listener =
                        tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    let upstream_addr = upstream_listener.local_addr().unwrap();
                    let upstream_handle = axum_server::Handle::new();
                    let dropped = std::sync::Arc::new(tokio::sync::Notify::new());
                    let body_dropped = dropped.clone();
                    let upstream_router = Router::new().route(
                        "/v1/files/file-1/content",
                        get(move || {
                            let notice = DropNotice(body_dropped.clone());
                            async move {
                                let chunk = Bytes::from(vec![b'x'; 65536]);
                                Body::from_stream(async_stream::stream! {
                                    let _notice = notice;
                                    loop { yield Ok::<_, io::Error>(chunk.clone()); }
                                })
                            }
                        }),
                    );
                    let upstream = TestServer {
                        task: tokio::spawn(
                            axum_server::from_tcp(upstream_listener.into_std().unwrap())
                                .handle(upstream_handle.clone())
                                .serve(upstream_router.into_make_service()),
                        ),
                        handle: upstream_handle,
                    };
                    let (_, router) = super::super::batch_proxy::router(
                        &format!("http://{upstream_addr}"),
                        None,
                        None,
                        1024,
                        1024,
                        tokio_util::sync::CancellationToken::new(),
                    )
                    .unwrap();
                    let mut client: Box<dyn TestConnection>;
                    let task;
                    if tls {
                        let cert =
                            rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
                        let provider =
                            std::sync::Arc::new(rustls::crypto::ring::default_provider());
                        let config = rustls::ServerConfig::builder_with_provider(provider.clone())
                            .with_safe_default_protocol_versions()
                            .unwrap()
                            .with_no_client_auth()
                            .with_single_cert(
                                vec![cert.cert.der().clone()],
                                rustls::pki_types::PrivatePkcs8KeyDer::from(
                                    cert.key_pair.serialize_der(),
                                )
                                .into(),
                            )
                            .unwrap();
                        let config = axum_server::tls_rustls::RustlsConfig::from_config(
                            std::sync::Arc::new(config),
                        );
                        let server =
                            axum_server::from_tcp_rustls(listener.into_std().unwrap(), config)
                                .map(|inner| WriteTimeoutAcceptor::new(inner, idle))
                                .handle(handle.clone())
                                .serve(router.into_make_service());
                        task = tokio::spawn(server);
                        let mut roots = rustls::RootCertStore::empty();
                        roots.add(cert.cert.der().clone()).unwrap();
                        let config = rustls::ClientConfig::builder_with_provider(provider)
                            .with_safe_default_protocol_versions()
                            .unwrap()
                            .with_root_certificates(roots)
                            .with_no_client_auth();
                        let connector =
                            tokio_rustls::TlsConnector::from(std::sync::Arc::new(config));
                        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
                        client = Box::new(
                            connector
                                .connect(
                                    rustls::pki_types::ServerName::try_from("localhost").unwrap(),
                                    stream,
                                )
                                .await
                                .unwrap(),
                        );
                    } else {
                        let server = axum_server::from_tcp(listener.into_std().unwrap())
                            .map(|inner| WriteTimeoutAcceptor::new(inner, idle))
                            .handle(handle.clone())
                            .serve(router.into_make_service());
                        task = tokio::spawn(server);
                        client = Box::new(tokio::net::TcpStream::connect(addr).await.unwrap());
                    }
                    let mut frontend = TestServer {
                        handle: handle.clone(),
                        task,
                    };
                    client
                        .write_all(
                            b"GET /v1/files/file-1/content HTTP/1.1\r\nHost: localhost\r\n\r\n",
                        )
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
                        handle.graceful_shutdown(Some(Duration::from_millis(100)));
                    }
                    tokio::time::timeout(Duration::from_secs(3), async {
                        while handle.connection_count() != 0 {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                    })
                    .await
                    .expect("stalled connection did not close independently of body polling");
                    tokio::time::timeout(Duration::from_secs(1), dropped.notified())
                        .await
                        .expect("proxy retained the upstream response after downstream closure");
                    handle.shutdown();
                    tokio::time::timeout(Duration::from_secs(1), &mut frontend.task)
                        .await
                        .unwrap()
                        .unwrap()
                        .unwrap();
                    drop(client);
                    drop(upstream);
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
        temp_env::async_with_vars([
            (dynamo_runtime::config::environment_names::llm::DYN_BATCH_FILE_TRANSFER_TIMEOUT_SECS, Some("1")),
        ], tokio::time::timeout(Duration::from_secs(10), async {
            for is_tls in [false, true] {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let upstream_addr = upstream_listener.local_addr().unwrap();
                let upstream_handle = axum_server::Handle::new();
                let dropped = Arc::new(tokio::sync::Notify::new());
                let body_dropped = dropped.clone();
                let upstream_router = Router::new().route("/v1/files/file-1/content", get(move || {
                    let notice = DropNotice(body_dropped.clone());
                    async move {
                        let chunk = Bytes::from(vec![b'x'; 65536]);
                        Body::from_stream(async_stream::stream! {
                            let _notice = notice;
                            loop { yield Ok::<_, io::Error>(chunk.clone()); }
                        })
                    }
                }));
                let upstream = TestServer {
                    task: tokio::spawn(axum_server::from_tcp(upstream_listener.into_std().unwrap())
                        .handle(upstream_handle.clone()).serve(upstream_router.into_make_service())),
                    handle: upstream_handle,
                };
                let (_, proxy) = super::super::batch_proxy::router(
                    &format!("http://{upstream_addr}"), None, None, 1024, 1024,
                    tokio_util::sync::CancellationToken::new(),
                ).unwrap();
                let handle = axum_server::Handle::new();
                let idle = Duration::from_secs(30);
                let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
                let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(5))
                    .add_root_certificate(reqwest::Certificate::from_der(cert.cert.der()).unwrap());
                let task = if is_tls {
                    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
                        .with_safe_default_protocol_versions().unwrap().with_no_client_auth()
                        .with_single_cert(vec![cert.cert.der().clone()], rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()).into()).unwrap();
                    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
                    let config = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(config));
                    tokio::spawn(axum_server::from_tcp_rustls(listener.into_std().unwrap(), config)
                        .map(|inner| WriteTimeoutAcceptor::new(inner, idle)).handle(handle.clone())
                        .serve(proxy.into_make_service()))
                } else {
                    tokio::spawn(axum_server::from_tcp(listener.into_std().unwrap())
                        .map(|inner| WriteTimeoutAcceptor::new(inner, idle)).handle(handle.clone())
                        .serve(proxy.into_make_service()))
                };
                let mut frontend = TestServer { handle: handle.clone(), task };
                let client = if is_tls { client.build().unwrap() } else { client.http2_prior_knowledge().build().unwrap() };
                let response = client.get(format!("{}://localhost:{}/v1/files/file-1/content", if is_tls { "https" } else { "http" }, addr.port()))
                    .send().await.unwrap();
                assert_eq!(response.version(), reqwest::Version::HTTP_2);
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                // Keep the response unread. No socket-write stall or shutdown can make this pass.
                tokio::time::timeout(Duration::from_secs(3), dropped.notified()).await
                    .expect("HTTP/2 transfer deadline retained upstream response");
                tokio::time::timeout(Duration::from_secs(1), async {
                    while handle.connection_count() != 0 { tokio::time::sleep(Duration::from_millis(10)).await; }
                }).await.expect("HTTP/2 transfer deadline retained downstream connection");
                drop(response);
                handle.shutdown();
                tokio::time::timeout(Duration::from_secs(1), &mut frontend.task).await.unwrap().unwrap().unwrap();
                drop(upstream);
            }
        })).await.unwrap();
    }

    struct TestServer {
        handle: axum_server::Handle,
        task: tokio::task::JoinHandle<io::Result<()>>,
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.handle.shutdown();
            self.task.abort();
        }
    }

    struct DropNotice(std::sync::Arc<tokio::sync::Notify>);

    impl Drop for DropNotice {
        fn drop(&mut self) {
            self.0.notify_one();
        }
    }
}
