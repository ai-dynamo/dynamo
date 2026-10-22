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

impl<S> ConnectionService<S> {
    pub(super) fn new(inner: S, deadline: ConnectionDeadline) -> Self {
        Self { inner, deadline }
    }
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
    pub(super) fn new(
        inner: I,
        timeout: Duration,
        connection_deadline: ConnectionDeadline,
    ) -> Self {
        Self {
            inner,
            timeout,
            stalled: None,
            is_write_pending: false,
            expired: false,
            connection_deadline,
            transfer_timer: None,
        }
    }

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
    use super::super::server::serve;
    use super::*;
    use axum::{
        Router,
        body::{Body, Bytes},
        routing::get,
    };
    use std::net::SocketAddr;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_rustls::TlsAcceptor;
    use tokio_util::sync::CancellationToken;
    use tokio_util::task::AbortOnDropHandle;

    #[tokio::test(start_paused = true)]
    async fn completed_transfers_remove_their_connection_deadline() {
        let (inner, _) = tokio::io::duplex(1);
        let connection = ConnectionDeadline::default();
        let first = connection.start(Duration::from_secs(1));
        let second = connection.start(Duration::from_secs(5));
        let mut writer = WriteTimeoutIo::new(inner, Duration::from_secs(30), connection.clone());
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
            let mut writer = WriteTimeoutIo::new(
                inner,
                Duration::from_millis(100),
                ConnectionDeadline::default(),
            );
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
    // Runs the same accept loop used by the frontend, without GPU/model infrastructure.
    // Serial: the HTTP/2 test below shortens the transfer deadline through the environment.
    #[tokio::test]
    #[serial_test::serial]
    async fn stalled_download_is_bounded_for_http_and_https() {
        tokio::time::timeout(Duration::from_secs(15), async {
            for tls in [false, true] {
                for shutdown in [false, true] {
                    let (upstream, released, _upstream) = spawn_upstream().await;
                    let cert =
                        rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
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
                    let cert =
                        rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
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
                        .add_root_certificate(
                            reqwest::Certificate::from_der(cert.cert.der()).unwrap(),
                        );
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
        let (_, router) = super::super::batch_proxy::router(
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

    // A client that connects and never sends a ClientHello is closed once the TLS
    // handshake timeout passes, instead of holding its connection task open.
    #[tokio::test(start_paused = true)]
    async fn silent_tls_client_is_closed_after_handshake_timeout() {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = serve(
            listener,
            Some(tls_acceptor(&cert)),
            Router::new(),
            Duration::from_secs(30),
            std::future::pending(),
        );
        let _server = AbortOnDropHandle::new(tokio::spawn(server));
        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let started = tokio::time::Instant::now();
        let mut buf = [0; 1];
        let read = tokio::time::timeout(Duration::from_secs(60), client.read(&mut buf))
            .await
            .expect("silent TLS client was not closed");
        assert_eq!(read.unwrap(), 0);
        assert!(started.elapsed() >= Duration::from_secs(10));
    }
}
