// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future as _;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use anyhow::Context as _;
use parking_lot::{Mutex, RwLock};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
#[cfg(test)]
use tokio::sync::Notify;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;
use tonic::transport::server::Connected;

use super::super::protocol::{FILE_DESCRIPTOR_SET, KvEventRelayServer};
use super::super::transport_config::KvDcRelayTransportConfig;
use super::grpc::{KvEventRelayService, KvEventRelayServiceConfig, SubscriberLimits};
use super::load::{LoadUpdateHub, run_load_publisher};
use super::source::WanPublicationSource;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KvDcRelayTransportHealth {
    pub(crate) enabled: bool,
    pub(crate) serving: bool,
    pub(crate) bound_address: Option<SocketAddr>,
    pub(crate) last_error: Option<String>,
}

pub(crate) struct KvDcRelayTransport {
    cancel: CancellationToken,
    health: Arc<RwLock<KvDcRelayTransportHealth>>,
    task: Mutex<Option<JoinHandle<()>>>,
    #[cfg(test)]
    accepted: Arc<Notify>,
}

struct CancellableIo {
    stream: TcpStream,
    cancelled: Pin<Box<WaitForCancellationFutureOwned>>,
    #[cfg(test)]
    accepted: Option<Arc<Notify>>,
}

impl CancellableIo {
    fn new(stream: TcpStream, cancel: CancellationToken) -> Self {
        Self {
            stream,
            cancelled: Box::pin(cancel.cancelled_owned()),
            #[cfg(test)]
            accepted: None,
        }
    }

    fn poll_cancelled(&mut self, cx: &mut Context<'_>) -> io::Result<()> {
        if self.cancelled.as_mut().poll(cx).is_ready() {
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "KV Relay transport is shutting down",
            ))
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    fn notify_when_polled(mut self, accepted: Arc<Notify>) -> Self {
        self.accepted = Some(accepted);
        self
    }
}

impl AsyncRead for CancellableIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        #[cfg(test)]
        if let Some(accepted) = self.accepted.take() {
            accepted.notify_one();
        }
        if let Err(error) = self.poll_cancelled(cx) {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for CancellableIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        if let Err(error) = self.poll_cancelled(cx) {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), io::Error>> {
        if let Err(error) = self.poll_cancelled(cx) {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        if let Err(error) = self.poll_cancelled(cx) {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

impl Connected for CancellableIo {
    type ConnectInfo = <TcpStream as Connected>::ConnectInfo;

    fn connect_info(&self) -> Self::ConnectInfo {
        self.stream.connect_info()
    }
}

impl KvDcRelayTransport {
    pub(crate) async fn start(
        source: WanPublicationSource,
        config: KvDcRelayTransportConfig,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let reflection = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .context("building KV Relay reflection service")?;
        let listener = TcpListener::bind(config.bind)
            .await
            .with_context(|| format!("binding KV Relay gRPC listener at {}", config.bind))?;
        let bound_address = listener
            .local_addr()
            .context("reading bound KV Relay gRPC listener address")?;
        let cancel = source.lifecycle().child_token();
        let fatal_cancel = source.lifecycle().clone();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: false,
            bound_address: Some(bound_address),
            last_error: None,
        }));
        let load_window = Duration::from_millis(config.load_window_ms);
        let load_updates = LoadUpdateHub::new(&source, load_window, config.load_fanout_capacity);
        let limits = SubscriberLimits::new(
            config.max_catalog_subscribers,
            config.max_pool_streams_total,
            config.max_readiness_subscribers,
            config.max_load_subscribers,
        );
        let service = KvEventRelayServer::new(KvEventRelayService::new(
            source.clone(),
            cancel.clone(),
            KvEventRelayServiceConfig {
                pool_heartbeat_interval: Duration::from_millis(config.pool_heartbeat_interval_ms),
                readiness_heartbeat_interval: Duration::from_millis(
                    config.readiness_heartbeat_interval_ms,
                ),
                snapshot_progress_timeout: Duration::from_millis(
                    config.snapshot_progress_timeout_ms,
                ),
                load_updates: load_updates.clone(),
                limits,
                snapshot_encoding_permits: Arc::new(Semaphore::new(
                    config.publication_encoding_concurrency,
                )),
            },
        ))
        .accept_compressed(CompressionEncoding::Zstd)
        .send_compressed(CompressionEncoding::Zstd)
        .max_encoding_message_size(config.max_message_bytes)
        .max_decoding_message_size(config.max_message_bytes);
        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        let router = Server::builder()
            .http2_keepalive_interval(Some(Duration::from_millis(config.keepalive_interval_ms)))
            .http2_keepalive_timeout(Some(Duration::from_millis(config.keepalive_timeout_ms)))
            .add_service(service)
            .add_service(health_service)
            .add_service(reflection);

        health_reporter
            .set_serving::<KvEventRelayServer<KvEventRelayService>>()
            .await;
        health.write().serving = true;
        tracing::info!(
            bind = %bound_address,
            "Started KV DC Relay WAN plaintext gRPC transport"
        );

        let server_cancel = cancel.clone();
        let server_health = health.clone();
        #[cfg(test)]
        let accepted = Arc::new(Notify::new());
        #[cfg(test)]
        let accepted_for_server = accepted.clone();
        let server_task = tokio::spawn(async move {
            let connection_cancel = server_cancel.clone();
            let incoming = TcpListenerStream::new(listener).map(move |connection| {
                connection.map(|stream| {
                    let io = CancellableIo::new(stream, connection_cancel.clone());
                    #[cfg(test)]
                    let io = io.notify_when_polled(accepted_for_server.clone());
                    io
                })
            });
            let shutdown = async move {
                server_cancel.cancelled().await;
                health_reporter
                    .set_not_serving::<KvEventRelayServer<KvEventRelayService>>()
                    .await;
                server_health.write().serving = false;
            };
            router
                .serve_with_incoming_shutdown(incoming, shutdown)
                .await
                .context("KV Relay gRPC server failed")
        });
        let load_task = tokio::spawn(run_load_publisher(
            source,
            load_window,
            load_updates,
            cancel.clone(),
        ));
        let supervisor_cancel = cancel.clone();
        let supervisor_health = health.clone();
        let task = tokio::spawn(supervise_transport(
            server_task,
            load_task,
            supervisor_cancel,
            fatal_cancel,
            supervisor_health,
        ));
        Ok(Self {
            cancel,
            health,
            task: Mutex::new(Some(task)),
            #[cfg(test)]
            accepted,
        })
    }

    pub(crate) fn health(&self) -> KvDcRelayTransportHealth {
        self.health.read().clone()
    }

    #[cfg(test)]
    pub(crate) async fn wait_for_accepted_connection(&self) {
        self.accepted.notified().await;
    }

    pub(crate) async fn shutdown(&self) {
        self.cancel.cancel();
        let task = self.task.lock().take();
        if let Some(task) = task
            && let Err(error) = task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV Relay transport supervisor failed during shutdown");
        }
        self.health.write().serving = false;
    }
}

impl Drop for KvDcRelayTransport {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn supervise_transport(
    mut server: JoinHandle<anyhow::Result<()>>,
    mut load: JoinHandle<anyhow::Result<()>>,
    cancel: CancellationToken,
    fatal_cancel: CancellationToken,
    health: Arc<RwLock<KvDcRelayTransportHealth>>,
) {
    enum Exit {
        Cancelled,
        Server(Result<anyhow::Result<()>, tokio::task::JoinError>),
        Load(Result<anyhow::Result<()>, tokio::task::JoinError>),
    }
    let exit = tokio::select! {
        biased;
        _ = cancel.cancelled() => Exit::Cancelled,
        result = &mut server => Exit::Server(result),
        result = &mut load => Exit::Load(result),
    };
    let fatal = match &exit {
        Exit::Cancelled => None,
        Exit::Server(Ok(Ok(()))) => Some("KV Relay gRPC server stopped unexpectedly".to_string()),
        Exit::Server(Ok(Err(error))) => Some(error.to_string()),
        Exit::Server(Err(error)) => Some(format!("KV Relay gRPC server task failed: {error}")),
        Exit::Load(Ok(Ok(()))) => Some("KV Relay load publisher stopped unexpectedly".to_string()),
        Exit::Load(Ok(Err(error))) => Some(error.to_string()),
        Exit::Load(Err(error)) => Some(format!("KV Relay load publisher task failed: {error}")),
    };
    if let Some(reason) = fatal {
        tracing::error!(error = %reason, "KV DC Relay transport failed");
        health.write().last_error = Some(reason);
        fatal_cancel.cancel();
    }
    cancel.cancel();
    match exit {
        Exit::Cancelled => {
            let _ = tokio::join!(server, load);
        }
        Exit::Server(_) => {
            let _ = load.await;
        }
        Exit::Load(_) => {
            let _ = server.await;
        }
    }
    health.write().serving = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn supervised_load_task_failure_cancels_the_relay() {
        let cancel = CancellationToken::new();
        let fatal = CancellationToken::new();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: true,
            ..KvDcRelayTransportHealth::default()
        }));
        let server_cancel = cancel.clone();
        let server = tokio::spawn(async move {
            server_cancel.cancelled().await;
            Ok(())
        });
        let load = tokio::spawn(async { Ok(()) });
        supervise_transport(server, load, cancel, fatal.clone(), health.clone()).await;
        assert!(fatal.is_cancelled());
        assert!(!health.read().serving);
        assert!(
            health
                .read()
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("load publisher stopped unexpectedly"))
        );
    }

    #[tokio::test]
    async fn relay_root_cancellation_marks_wan_transport_not_serving() {
        let relay_cancel = CancellationToken::new();
        let transport_cancel = relay_cancel.child_token();
        let health = Arc::new(RwLock::new(KvDcRelayTransportHealth {
            enabled: true,
            serving: true,
            ..KvDcRelayTransportHealth::default()
        }));
        let server_cancel = transport_cancel.clone();
        let load_cancel = transport_cancel.clone();
        let server = tokio::spawn(async move {
            server_cancel.cancelled().await;
            Ok(())
        });
        let load = tokio::spawn(async move {
            load_cancel.cancelled().await;
            Ok(())
        });
        let supervisor = tokio::spawn(supervise_transport(
            server,
            load,
            transport_cancel,
            relay_cancel.clone(),
            health.clone(),
        ));

        relay_cancel.cancel();
        supervisor.await.unwrap();

        assert!(!health.read().serving);
        assert!(health.read().last_error.is_none());
    }
}
