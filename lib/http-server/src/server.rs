// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's HTTP(S) accept loop, with graceful shutdown bounded by a grace period.
//!
//! A connection task is polled on every wakeup, i.e. for every streamed response chunk, and
//! each poll also polls its shutdown signals. Every connection waits on its own child
//! [`CancellationToken`]s, whose lock and `Notify` no other connection touches. A shared
//! `tokio::sync::Notify` would lock its waiter list on every poll of a pending waiter.

use crate::lifecycle::ServiceObserver;
use anyhow::{Context, Result};
use dynamo_runtime::config::environment_names::llm as env_llm;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use std::{future::Future, pin::pin, time::Duration};

use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::write_timeout::{ConnectionDeadline, ConnectionService, WriteTimeoutIo};

/// The handshake timeout of axum-server's rustls acceptor, which served the frontend before.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Serves `router` on `listener` until `shutdown` resolves with a grace period.
///
/// Then it stops accepting connections and lets open ones finish their requests; any still
/// open after the grace period are closed, dropping their response bodies. Returning, or
/// being dropped, closes every connection that was not upgraded. An upgraded connection,
/// such as a WebSocket session, runs on its own task and stays open after `serve` returns.
///
/// Every connection is wrapped in [`WriteTimeoutIo`], and every request on it carries the
/// connection's [`ConnectionDeadline`].
pub async fn serve(
    listener: TcpListener,
    tls: Option<TlsAcceptor>,
    router: Router,
    write_idle_timeout: Duration,
    shutdown: impl Future<Output = Duration>,
) {
    // Build each route's handler once, as `Router::into_make_service` does, not per request.
    let router = router.with_state(());
    let graceful = CancellationToken::new();
    let hard = CancellationToken::new();
    let _close_connections = hard.clone().drop_guard();
    let connections = TaskTracker::new();

    let mut shutdown = pin!(shutdown);
    let grace_period = loop {
        let stream = tokio::select! {
            biased;
            grace_period = &mut shutdown => break grace_period,
            stream = accept(&listener) => stream,
        };
        let connection = serve_connection(
            stream,
            tls.clone(),
            router.clone(),
            write_idle_timeout,
            graceful.child_token(),
        );
        connections.spawn(hard.child_token().run_until_cancelled_owned(connection));
    };

    // Refuse new connections at once instead of leaving them in the backlog.
    drop(listener);
    graceful.cancel();
    connections.close();
    let _ = tokio::time::timeout(grace_period, connections.wait()).await;
}

async fn accept(listener: &TcpListener) -> TcpStream {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                // Streamed tokens are small writes. With Nagle's algorithm on, one can wait
                // for the client's delayed ACK of the previous write, 40 ms or more. Set on
                // the TCP socket, this also covers TLS connections.
                if let Err(error) = stream.set_nodelay(true) {
                    tracing::debug!(%error, "Failed to set TCP_NODELAY");
                }
                return stream;
            }
            // Errors such as EMFILE are transient: back off and retry, as axum-server did.
            Err(error) => {
                tracing::debug!(%error, "Failed to accept HTTP connection");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

async fn serve_connection(
    stream: TcpStream,
    tls: Option<TlsAcceptor>,
    router: Router,
    write_idle_timeout: Duration,
    graceful: CancellationToken,
) {
    let Some(tls) = tls else {
        return serve_io(stream, router, write_idle_timeout, graceful).await;
    };
    match tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, tls.accept(stream)).await {
        Ok(Ok(stream)) => serve_io(stream, router, write_idle_timeout, graceful).await,
        Ok(Err(error)) => tracing::debug!(%error, "TLS handshake failed"),
        Err(_) => tracing::debug!("TLS handshake timed out"),
    }
}

async fn serve_io<I>(
    io: I,
    router: Router,
    write_idle_timeout: Duration,
    graceful: CancellationToken,
) where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let deadline = ConnectionDeadline::default();
    let io = TokioIo::new(WriteTimeoutIo::new(
        io,
        write_idle_timeout,
        deadline.clone(),
    ));
    let service = TowerToHyperService::new(ConnectionService::new(router, deadline));
    let builder = Builder::new(TokioExecutor::new());
    let mut connection = pin!(builder.serve_connection_with_upgrades(io, service));
    if graceful
        .run_until_cancelled(connection.as_mut())
        .await
        .is_none()
    {
        connection.as_mut().graceful_shutdown();
        let _ = connection.await;
    }
}

pub struct HttpServerConfig {
    pub host: String,
    pub port: u16,
    pub enable_tls: bool,
    pub tls_cert_path: Option<PathBuf>,
    pub tls_key_path: Option<PathBuf>,
    pub tls_client_ca_cert_path: Option<PathBuf>,
}

pub struct HttpServer {
    router: Router,
    listener: TcpListener,
    tls: Option<TlsAcceptor>,
    protocol: &'static str,
}

impl HttpServer {
    /// Bind and prepare TLS before the caller starts background tasks.
    pub async fn prepare(
        router: Router,
        config: HttpServerConfig,
        listener: Option<TcpListener>,
    ) -> Result<Self> {
        let address = format!("{}:{}", config.host, config.port);
        let protocol = if config.enable_tls { "HTTPS" } else { "HTTP" };
        tracing::info!(protocol, address, "Starting HTTP(S) service");

        let tls = if config.enable_tls {
            let cert_path = config
                .tls_cert_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS certificate path not provided"))?;
            let key_path = config
                .tls_key_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("TLS private key path not provided"))?;
            let mut server_config = dynamo_runtime::tls_utils::server_tls_config(
                cert_path,
                key_path,
                config.tls_client_ca_cert_path.as_deref(),
            )
            .context("Failed to create TLS config")?;
            server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
            Some(tokio_rustls::TlsAcceptor::from(Arc::new(server_config)))
        } else {
            None
        };

        let listener = match listener {
            Some(l) => l,
            None => {
                let addr = listen_socket_addr(&config.host, config.port)?;
                bind_listener(addr).map_err(|e| {
                    tracing::error!(
                        protocol = %protocol,
                        address = %address,
                        error = %e,
                        "Failed to bind server to address"
                    );
                    match e.kind() {
                        std::io::ErrorKind::AddrInUse => anyhow::anyhow!(
                            "Failed to start {} server: port {} already in use. Use --http-port to specify a different port.",
                            protocol,
                            config.port
                        ),
                        _ => anyhow::anyhow!(
                            "Failed to start {} server on {}: {}",
                            protocol,
                            address,
                            e
                        ),
                    }
                })?
            }
        };
        tracing::info!(address = %listener.local_addr()?, "{protocol} server listening");

        Ok(Self {
            router,
            listener,
            tls,
            protocol,
        })
    }

    /// Serve while admitted response bodies drain before cancelling service state.
    pub async fn serve(
        self,
        state: Arc<ServiceObserver>,
        state_cancel: CancellationToken,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let observer = cancel_token.child_token();
        let protocol = self.protocol;
        // The server keeps accepting while inflight requests drain, then gives open
        // connections what is left of the shutdown budget.
        let shutdown = async {
            observer.cancelled().await;
            state.start_draining();
            tracing::info!("{protocol} server shutdown requested");
            let shutdown_timeout = Duration::from_secs(get_graceful_shutdown_timeout() as u64);
            let drain_started = tokio::time::Instant::now();
            if !state.wait_inflight_zero_or_timeout(shutdown_timeout).await {
                tracing::warn!(
                    inflight_requests = state.inflight_count(),
                    "Timed out waiting for inflight inference requests to drain"
                );
            }
            shutdown_timeout.saturating_sub(drain_started.elapsed())
        };
        serve(
            self.listener,
            self.tls,
            self.router,
            write_idle_timeout(),
            shutdown,
        )
        .await;
        state.start_stopping();
        state_cancel.cancel();
        Ok(())
    }
}

fn get_graceful_shutdown_timeout() -> usize {
    std::env::var(env_llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
}

fn write_idle_timeout() -> Duration {
    std::env::var(env_llm::DYN_HTTP_WRITE_IDLE_TIMEOUT_SECS)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .filter(|duration| tokio::time::Instant::now().checked_add(*duration).is_some())
        .unwrap_or(Duration::from_secs(120))
}

const DEFAULT_LISTEN_BACKLOG: u32 = 4096;

// `listen(2)` takes an `int`; anything above `i32::MAX` would go negative.
fn parse_listen_backlog(value: Result<String, std::env::VarError>) -> u32 {
    value
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|n| (1..=i32::MAX as u32).contains(n))
        .unwrap_or(DEFAULT_LISTEN_BACKLOG)
}

fn listen_backlog() -> u32 {
    parse_listen_backlog(std::env::var(env_llm::DYN_HTTP_LISTEN_BACKLOG))
}

/// Build the listen address from an IP literal host. IPv6 brackets are optional,
/// so `::` and `[::]` are equivalent.
pub fn listen_socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let literal = host.trim();
    let literal = literal
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(literal);
    let ip: IpAddr = literal
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid address '{host}:{port}': {e}"))?;
    Ok(SocketAddr::new(ip, port))
}

/// `tokio::net::TcpListener::bind` listens with a backlog of 128. A few thousand
/// clients connecting within seconds overflow that, and overflowed connections
/// are delayed or, depending on host TCP settings, fail.
fn bind_listener(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    let socket = if addr.is_ipv4() {
        tokio::net::TcpSocket::new_v4()?
    } else {
        tokio::net::TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    socket.bind(addr)?;
    socket.listen(listen_backlog())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accepted_connections_turn_nagle_off() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (_client, stream) = tokio::join!(TcpStream::connect(addr), accept(&listener));
        assert!(stream.nodelay().unwrap());
    }
    #[test]
    fn test_listen_backlog_env_var() {
        assert_eq!(
            parse_listen_backlog(Err(std::env::VarError::NotPresent)),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok("0".to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok("invalid".to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok((i32::MAX as u32 + 1).to_string())),
            DEFAULT_LISTEN_BACKLOG
        );
        assert_eq!(
            parse_listen_backlog(Ok(i32::MAX.to_string())),
            i32::MAX as u32
        );
        assert_eq!(parse_listen_backlog(Ok(" 8192 ".to_string())), 8192);
    }
    #[tokio::test]
    async fn test_bind_listener_accepts_connections() {
        let listener = bind_listener("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();
        assert_ne!(addr.port(), 0);
        let client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (_server_side, peer) = listener.accept().await.unwrap();
        assert_eq!(peer, client.local_addr().unwrap());
    }
    #[test]
    fn test_listen_socket_addr_accepts_ip_literals() {
        for (host, expected) in [
            ("0.0.0.0", "0.0.0.0:8000"),
            ("127.0.0.1", "127.0.0.1:8000"),
            ("::", "[::]:8000"),
            ("[::]", "[::]:8000"),
            (" [::1] ", "[::1]:8000"),
        ] {
            assert_eq!(
                listen_socket_addr(host, 8000).unwrap(),
                expected.parse::<SocketAddr>().unwrap(),
                "{host}"
            );
        }
        for host in ["localhost", "[::", "::1]", ""] {
            let error = listen_socket_addr(host, 8000).unwrap_err().to_string();
            assert!(
                error.starts_with(&format!("Invalid address '{host}:8000'")),
                "{host}: {error}"
            );
        }
    }
}
