// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum::body::{Body, Bytes, HttpBody};
use axum::response::Response;
use axum_server::tls_rustls::RustlsConfig;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

pub mod disconnect;
pub mod error;
pub mod metrics;
pub mod request;

#[derive(Clone, Debug)]
pub struct TlsConfig {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub tls: Option<TlsConfig>,
    pub graceful_shutdown_timeout: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceStage {
    Ready = 0,
    Draining = 1,
    Stopping = 2,
}

impl ServiceStage {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Ready,
            1 => Self::Draining,
            _ => Self::Stopping,
        }
    }
}

impl std::fmt::Display for ServiceStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => f.write_str("ready"),
            Self::Draining => f.write_str("draining"),
            Self::Stopping => f.write_str("stopping"),
        }
    }
}

#[derive(Debug)]
pub struct ServiceObserver {
    stage: AtomicU8,
    inflight: AtomicU64,
    inflight_zero: Notify,
}

impl Default for ServiceObserver {
    fn default() -> Self {
        Self {
            stage: AtomicU8::new(ServiceStage::Ready as u8),
            inflight: AtomicU64::new(0),
            inflight_zero: Notify::new(),
        }
    }
}

impl ServiceObserver {
    pub fn stage(&self) -> ServiceStage {
        ServiceStage::from_u8(self.stage.load(Ordering::Acquire))
    }

    pub fn is_ready(&self) -> bool {
        self.stage() == ServiceStage::Ready
    }

    pub fn start_draining(&self) {
        tracing::info!(
            previous_stage = ?self.stage(),
            inflight_requests = self.inflight_count(),
            "frontend service entering draining stage"
        );
        self.stage
            .store(ServiceStage::Draining as u8, Ordering::Release);
    }

    pub fn start_stopping(&self) {
        tracing::info!(
            previous_stage = ?self.stage(),
            inflight_requests = self.inflight_count(),
            "frontend service entering stopping stage"
        );
        self.stage
            .store(ServiceStage::Stopping as u8, Ordering::Release);
    }

    pub fn acquire_inflight(self: &Arc<Self>) -> InflightPermit {
        self.inflight.fetch_add(1, Ordering::Relaxed);
        InflightPermit {
            observer: self.clone(),
        }
    }

    /// Admit a request only while the service is ready.
    pub fn try_acquire_inflight(self: &Arc<Self>) -> Option<InflightPermit> {
        if !self.is_ready() {
            return None;
        }
        let permit = self.acquire_inflight();
        if self.is_ready() { Some(permit) } else { None }
    }

    pub fn inflight_count(&self) -> u64 {
        self.inflight.load(Ordering::Acquire)
    }

    pub async fn wait_inflight_zero_or_timeout(&self, timeout: Duration) -> bool {
        tokio::time::timeout(timeout, async {
            loop {
                let notified = self.inflight_zero.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.inflight_count() == 0 {
                    break;
                }
                notified.as_mut().await;
            }
        })
        .await
        .is_ok()
    }
}

pub struct InflightPermit {
    observer: Arc<ServiceObserver>,
}

impl Drop for InflightPermit {
    fn drop(&mut self) {
        if self.observer.inflight.fetch_sub(1, Ordering::AcqRel) == 1
            && self.observer.stage() != ServiceStage::Ready
        {
            self.observer.inflight_zero.notify_waiters();
        }
    }
}

/// Hold an inflight permit until the complete HTTP body is sent or dropped.
pub fn track_inflight_response(response: Response, permit: InflightPermit) -> Response {
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(InflightBody {
            body,
            _permit: permit,
        }),
    )
}

/// Echo `x-request-id` from the request into the response for client correlation.
pub async fn echo_request_id_header(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let request_id = request.headers().get("x-request-id").cloned();
    let mut response = next.run(request).await;
    if let Some(request_id) = request_id {
        response.headers_mut().insert("x-request-id", request_id);
    }
    response
}

struct InflightBody {
    body: Body,
    _permit: InflightPermit,
}

impl HttpBody for InflightBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.get_mut().body).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }
}

/// Serves an Axum router without depending on Dynamo runtime state.
pub struct HttpServer {
    router: Router,
    config: HttpServerConfig,
    observer: Arc<ServiceObserver>,
    downstream_cancel_token: CancellationToken,
}

struct ShutdownOnDrop(axum_server::Handle);

impl Drop for ShutdownOnDrop {
    fn drop(&mut self) {
        self.0.shutdown();
    }
}

impl HttpServer {
    pub fn new(
        router: Router,
        config: HttpServerConfig,
        observer: Arc<ServiceObserver>,
        downstream_cancel_token: CancellationToken,
    ) -> Self {
        Self {
            router,
            config,
            observer,
            downstream_cancel_token,
        }
    }

    pub async fn run(self, cancel_token: CancellationToken) -> Result<()> {
        self.run_inner(cancel_token, None).await
    }

    pub async fn run_with_listener(
        self,
        cancel_token: CancellationToken,
        listener: tokio::net::TcpListener,
    ) -> Result<()> {
        self.run_inner(cancel_token, Some(listener)).await
    }

    async fn run_inner(
        self,
        cancel_token: CancellationToken,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<()> {
        let result = self.serve(cancel_token, listener).await;
        self.observer.start_stopping();
        self.downstream_cancel_token.cancel();
        result
    }

    async fn serve(
        &self,
        cancel_token: CancellationToken,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<()> {
        let address = format!("{}:{}", self.config.host, self.config.port);
        let protocol = if self.config.tls.is_some() {
            "HTTPS"
        } else {
            "HTTP"
        };
        tracing::info!(protocol, address, "Starting HTTP(S) service");

        if let Some(tls) = self.config.tls.clone() {
            if listener.is_some() {
                anyhow::bail!(
                    "Pre-bound listener is not supported in TLS mode; \
                     axum_server::bind_rustls owns its own bind. \
                     Use run() (which binds internally) when TLS is set."
                );
            }
            let addr: SocketAddr = address
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", address, e))?;

            if let Err(error) = rustls::crypto::aws_lc_rs::default_provider().install_default() {
                tracing::debug!(?error, "TLS crypto provider already installed");
            }
            let tls = RustlsConfig::from_pem_file(tls.cert_path, tls.key_path)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to create TLS config: {}", e))?;
            let handle = axum_server::Handle::new();
            let _shutdown_on_drop = ShutdownOnDrop(handle.clone());
            let server = axum_server::bind_rustls(addr, tls)
                .handle(handle.clone())
                .serve(self.router.clone().into_make_service());
            let mut server = tokio::spawn(server);

            tokio::select! {
                result = &mut server => {
                    result
                        .map_err(|e| anyhow::anyhow!("HTTPS server task failed: {}", e))?
                        .map_err(|e| anyhow::anyhow!("HTTPS server error: {}", e))?;
                }
                _ = cancel_token.cancelled() => {
                    self.observer.start_draining();
                    tracing::info!("HTTPS server shutdown requested");
                    let deadline = tokio::time::Instant::now()
                        + self.config.graceful_shutdown_timeout;
                    handle.graceful_shutdown(Some(self.config.graceful_shutdown_timeout));
                    let result = if !self.observer
                        .wait_inflight_zero_or_timeout(self.config.graceful_shutdown_timeout)
                        .await
                    {
                        tracing::warn!(
                            inflight_requests = self.observer.inflight_count(),
                            "Timed out waiting for inflight inference requests to drain"
                        );
                        handle.shutdown();
                        (&mut server).await
                    } else {
                        match tokio::time::timeout_at(deadline, &mut server).await {
                            Ok(result) => result,
                            Err(_) => {
                                tracing::warn!(
                                    inflight_requests = self.observer.inflight_count(),
                                    "Timed out waiting for HTTPS connections to drain"
                                );
                                handle.shutdown();
                                (&mut server).await
                            }
                        }
                    };
                    result
                        .map_err(|e| anyhow::anyhow!("HTTPS server task failed: {}", e))?
                        .map_err(|e| anyhow::anyhow!("HTTPS server error: {}", e))?;
                }
            }
        } else {
            let listener = match listener {
                Some(listener) => listener,
                None => {
                    let addr: SocketAddr = address
                        .parse()
                        .map_err(|e| anyhow::anyhow!("Invalid address '{}': {}", address, e))?;
                    tokio::net::TcpListener::bind(addr).await.map_err(|error| {
                        tracing::error!(protocol, address, %error, "Failed to bind server");
                        match error.kind() {
                            std::io::ErrorKind::AddrInUse => anyhow::anyhow!(
                                "Failed to start {} server: port {} already in use. Use --http-port to specify a different port.",
                                protocol,
                                self.config.port
                            ),
                            _ => anyhow::anyhow!(
                                "Failed to start {} server on {}: {}",
                                protocol,
                                address,
                                error
                            ),
                        }
                    })?
                }
            };

            let listener = listener.into_std()?;
            let handle = axum_server::Handle::new();
            let _shutdown_on_drop = ShutdownOnDrop(handle.clone());
            let server = axum_server::from_tcp(listener)
                .handle(handle.clone())
                .serve(self.router.clone().into_make_service());
            let mut server = tokio::spawn(server);
            tokio::select! {
                result = &mut server => {
                    result
                        .map_err(|e| anyhow::anyhow!("HTTP server task failed: {}", e))?
                        .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;
                }
                _ = cancel_token.cancelled() => {
                    self.observer.start_draining();
                    tracing::info!("HTTP server shutdown requested");
                    let deadline = tokio::time::Instant::now()
                        + self.config.graceful_shutdown_timeout;
                    handle.graceful_shutdown(Some(self.config.graceful_shutdown_timeout));
                    let result = if !self.observer
                        .wait_inflight_zero_or_timeout(self.config.graceful_shutdown_timeout)
                        .await
                    {
                        tracing::warn!(
                            inflight_requests = self.observer.inflight_count(),
                            "Timed out waiting for inflight inference requests to drain"
                        );
                        handle.shutdown();
                        (&mut server).await
                    } else {
                        match tokio::time::timeout_at(deadline, &mut server).await {
                            Ok(result) => result,
                            Err(_) => {
                                tracing::warn!(
                                    inflight_requests = self.observer.inflight_count(),
                                    "Timed out waiting for HTTP connections to drain"
                                );
                                handle.shutdown();
                                (&mut server).await
                            }
                        }
                    };
                    result
                        .map_err(|e| anyhow::anyhow!("HTTP server task failed: {}", e))?
                        .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use axum::extract::State;
    use axum::routing::get;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    #[derive(Clone)]
    struct StreamingState {
        observer: Arc<ServiceObserver>,
        release: CancellationToken,
    }

    async fn streaming_response(State(state): State<StreamingState>) -> Response {
        let permit = state.observer.acquire_inflight();
        let stream = async_stream::stream! {
            yield Ok::<_, io::Error>(Bytes::from_static(b"started"));
            state.release.cancelled().await;
        };
        track_inflight_response(Response::new(Body::from_stream(stream)), permit)
    }

    async fn connect_stream(address: SocketAddr) -> tokio::net::TcpStream {
        tokio::time::timeout(Duration::from_secs(1), async {
            let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
            stream
                .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let mut response = Vec::new();
            loop {
                let mut buffer = [0; 1024];
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0, "stream closed before the first body chunk");
                response.extend_from_slice(&buffer[..read]);
                if response
                    .windows(b"started".len())
                    .any(|part| part == b"started")
                {
                    return stream;
                }
            }
        })
        .await
        .expect("server did not start the streaming response")
    }

    #[tokio::test]
    async fn observer_waits_for_inflight_response() {
        let observer = Arc::new(ServiceObserver::default());
        let permit = observer.acquire_inflight();
        observer.start_draining();

        assert!(
            !observer
                .wait_inflight_zero_or_timeout(Duration::from_millis(1))
                .await
        );
        drop(permit);
        assert!(
            observer
                .wait_inflight_zero_or_timeout(Duration::from_secs(1))
                .await
        );
    }

    #[test]
    fn tracked_response_holds_permit_until_body_drop() {
        let observer = Arc::new(ServiceObserver::default());
        let permit = observer.try_acquire_inflight().unwrap();
        let response = track_inflight_response(Response::new(Body::from("hello")), permit);
        assert_eq!(observer.inflight_count(), 1);
        drop(response);
        assert_eq!(observer.inflight_count(), 0);
    }

    #[tokio::test]
    async fn startup_error_finalizes_without_cancelling_caller_token() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = occupied.local_addr().unwrap().port();
        let observer = Arc::new(ServiceObserver::default());
        let caller = CancellationToken::new();
        let downstream = CancellationToken::new();
        let server = HttpServer::new(
            Router::new(),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port,
                tls: None,
                graceful_shutdown_timeout: Duration::from_secs(1),
            },
            observer.clone(),
            downstream.clone(),
        );

        assert!(server.run(caller.clone()).await.is_err());
        assert_eq!(observer.stage(), ServiceStage::Stopping);
        assert!(downstream.is_cancelled());
        assert!(!caller.is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_drains_before_cancelling_downstream_work() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let observer = Arc::new(ServiceObserver::default());
        let shutdown = CancellationToken::new();
        let downstream = CancellationToken::new();
        let release = CancellationToken::new();
        let server = HttpServer::new(
            Router::new()
                .route("/", get(streaming_response))
                .with_state(StreamingState {
                    observer: observer.clone(),
                    release: release.clone(),
                }),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port: address.port(),
                tls: None,
                graceful_shutdown_timeout: Duration::from_secs(1),
            },
            observer.clone(),
            downstream.clone(),
        );
        let task = tokio::spawn(server.run_with_listener(shutdown.clone(), listener));
        let _client = connect_stream(address).await;
        assert_eq!(observer.inflight_count(), 1);

        shutdown.cancel();
        while observer.stage() == ServiceStage::Ready {
            tokio::task::yield_now().await;
        }
        assert_eq!(observer.stage(), ServiceStage::Draining);
        assert!(!downstream.is_cancelled());
        assert!(!task.is_finished());

        release.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("server did not stop")
            .unwrap()
            .unwrap();
        assert_eq!(observer.stage(), ServiceStage::Stopping);
        assert!(downstream.is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_waits_for_background_inflight_work() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let observer = Arc::new(ServiceObserver::default());
        let permit = observer.acquire_inflight();
        let shutdown = CancellationToken::new();
        let downstream = CancellationToken::new();
        let server = HttpServer::new(
            Router::new(),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port: listener.local_addr().unwrap().port(),
                tls: None,
                graceful_shutdown_timeout: Duration::from_secs(1),
            },
            observer.clone(),
            downstream.clone(),
        );
        let task = tokio::spawn(server.run_with_listener(shutdown.clone(), listener));

        shutdown.cancel();
        while observer.stage() == ServiceStage::Ready {
            tokio::task::yield_now().await;
        }
        assert_eq!(observer.stage(), ServiceStage::Draining);
        assert!(!downstream.is_cancelled());
        assert!(!task.is_finished());

        drop(permit);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("server did not stop after background work completed")
            .unwrap()
            .unwrap();
        assert_eq!(observer.stage(), ServiceStage::Stopping);
        assert!(downstream.is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_timeout_forces_open_response_closed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let grace = Duration::from_millis(20);
        let observer = Arc::new(ServiceObserver::default());
        let shutdown = CancellationToken::new();
        let downstream = CancellationToken::new();
        let server = HttpServer::new(
            Router::new()
                .route("/", get(streaming_response))
                .with_state(StreamingState {
                    observer: observer.clone(),
                    release: CancellationToken::new(),
                }),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port: address.port(),
                tls: None,
                graceful_shutdown_timeout: grace,
            },
            observer.clone(),
            downstream.clone(),
        );
        let task = tokio::spawn(server.run_with_listener(shutdown.clone(), listener));
        let mut client = connect_stream(address).await;
        assert_eq!(observer.inflight_count(), 1);

        let shutdown_started = tokio::time::Instant::now();
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("server ignored the graceful shutdown deadline")
            .unwrap()
            .unwrap();

        assert!(shutdown_started.elapsed() >= grace);
        assert_eq!(observer.stage(), ServiceStage::Stopping);
        assert_eq!(observer.inflight_count(), 0);
        assert!(downstream.is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let mut buffer = [0; 1024];
                match client.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        })
        .await
        .expect("forced shutdown left the client connection open");
    }

    #[tokio::test]
    async fn aborting_outer_task_releases_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let observer = Arc::new(ServiceObserver::default());
        let server = HttpServer::new(
            Router::new()
                .route("/", get(streaming_response))
                .with_state(StreamingState {
                    observer: observer.clone(),
                    release: CancellationToken::new(),
                }),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port: address.port(),
                tls: None,
                graceful_shutdown_timeout: Duration::from_secs(1),
            },
            observer,
            CancellationToken::new(),
        );
        let task = tokio::spawn(server.run_with_listener(CancellationToken::new(), listener));
        let mut client = connect_stream(address).await;

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let mut buffer = [0; 1024];
                match client.read(&mut buffer).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        })
        .await
        .expect("aborting HttpServer::run left the client connection open");
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match tokio::net::TcpListener::bind(address).await {
                    Ok(listener) => break listener,
                    Err(_) => tokio::time::sleep(Duration::from_millis(5)).await,
                }
            }
        })
        .await
        .expect("aborting HttpServer::run left the listener bound");
    }

    #[tokio::test]
    async fn prebound_listener_is_rejected_with_tls() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server = HttpServer::new(
            Router::new(),
            HttpServerConfig {
                host: "127.0.0.1".into(),
                port: listener.local_addr().unwrap().port(),
                tls: Some(TlsConfig {
                    cert_path: "unused".into(),
                    key_path: "unused".into(),
                }),
                graceful_shutdown_timeout: Duration::from_secs(1),
            },
            Arc::new(ServiceObserver::default()),
            CancellationToken::new(),
        );

        let error = server
            .run_with_listener(CancellationToken::new(), listener)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Pre-bound listener"));
    }
}
