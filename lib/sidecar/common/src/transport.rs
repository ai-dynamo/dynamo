// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write as _;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use dynamo_backend_common::DynamoError;
use futures::future::{BoxFuture, try_join_all};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::codegen::Service;
use tonic::codegen::http::Uri;
use tonic::transport::{Channel, Endpoint};

use crate::{GrpcEndpoint, GrpcTransportConfig, cannot_connect, invalid_argument};

pub const DEFAULT_MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const RETRY_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// The `:authority` of requests over a Unix socket, which has no host of its
/// own. gRPC sends `localhost`; tonic would otherwise send a placeholder.
const UNIX_SOCKET_ORIGIN: &str = "http://localhost";

/// Tonic's default plaintext TCP connector, except that it rejects a socket
/// connected to itself.
///
/// While the engine is not yet listening on a port inside the kernel's
/// ephemeral range (32768-60999 on Linux), a loopback connect can be given that
/// same port as its source port and complete a TCP simultaneous open with
/// itself. A normal close then leaves `127.0.0.1:<port>` in TIME_WAIT for 60s,
/// and the engine's gRPC bind on that port fails with EADDRINUSE.
#[derive(Clone)]
struct RejectSelfConnect(HttpConnector);

impl RejectSelfConnect {
    /// Matches the connector `Endpoint::connect` builds from default endpoint
    /// settings, except for the connect timeout: `connect_with_connector`
    /// already applies the endpoint's `connect_timeout` around this connector.
    ///
    /// TCP options set on the `Endpoint` (`tcp_keepalive` and `local_address`,
    /// plus the keepalive interval and retries in tonic 0.14) do not reach
    /// this connector. Nothing sets them today; set them here if that changes.
    fn new() -> Self {
        let mut http = HttpConnector::new();
        http.enforce_http(false);
        http.set_nodelay(true);
        Self(http)
    }
}

impl Service<Uri> for RejectSelfConnect {
    type Response = TokioIo<TcpStream>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let connecting = self.0.call(uri);
        Box::pin(async move {
            let stream = connecting.await?;
            reject_self_connect(stream.inner())?;
            Ok(stream)
        })
    }
}

/// Fails if `stream` is connected to itself, after arranging for its drop to
/// reset the connection so the port is free for the server immediately.
fn reject_self_connect(stream: &TcpStream) -> io::Result<()> {
    let local = stream.local_addr()?;
    if local != stream.peer_addr()? {
        return Ok(());
    }
    stream.set_zero_linger()?;
    Err(io::Error::new(
        io::ErrorKind::ConnectionRefused,
        format!("TCP connection to {local} connected to itself; nothing is listening yet"),
    ))
}

/// Connected channels distributed in round-robin order.
pub struct GrpcChannelPool {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl GrpcChannelPool {
    /// `bootstrap`: true for synchronous constructors that run before logging
    /// initialization and need retry warnings on stderr. Deferred launcher
    /// discovery and `LLMEngine::start` pass false to use tracing.
    ///
    /// A `unix://` endpoint is retried like a TCP one, because its socket may
    /// not exist until the engine has started, and each of its channels
    /// reconnects through the same path after the engine drops it. Tonic dials
    /// the socket itself (a self-connect is a TCP-only hazard); connecting to
    /// one never waits, so the attempt timeout does not apply to it.
    pub async fn connect(
        peer: &str,
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        bootstrap: bool,
    ) -> Result<Self, DynamoError> {
        let endpoint_label = endpoint.to_string();
        let mut tonic_endpoint =
            Endpoint::from_shared(endpoint_label.clone()).map_err(|error| {
                invalid_argument(format!("invalid {peer} endpoint after validation: {error}"))
            })?;
        let unix_socket = endpoint.is_unix_socket();
        if unix_socket {
            tonic_endpoint = tonic_endpoint.origin(Uri::from_static(UNIX_SOCKET_ORIGIN));
        }
        let channels = connect_pool_with(
            peer,
            &endpoint_label,
            transport,
            bootstrap,
            |_, attempt_timeout| {
                let endpoint = tonic_endpoint.clone().connect_timeout(attempt_timeout);
                async move {
                    if unix_socket {
                        endpoint.connect().await
                    } else {
                        endpoint
                            .connect_with_connector(RejectSelfConnect::new())
                            .await
                    }
                }
            },
        )
        .await?;
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn next_channel(&self) -> Channel {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        self.channels[index].clone()
    }
}

pub(crate) async fn connect_pool_with<C, F, E>(
    peer: &str,
    endpoint: &str,
    transport: GrpcTransportConfig,
    bootstrap: bool,
    connect: impl Fn(usize, Duration) -> F,
) -> Result<Vec<C>, DynamoError>
where
    F: std::future::Future<Output = Result<C, E>>,
    E: std::error::Error + 'static,
{
    let deadline = checked_instant_add(
        Instant::now(),
        transport.startup_deadline,
        "gRPC startup deadline",
    )?;
    let first = connect_until_ready(
        peer,
        endpoint,
        1,
        transport,
        deadline,
        bootstrap,
        |timeout| connect(1, timeout),
    )
    .await?;
    let mut channels = vec![first];
    let remaining = try_join_all((2..=transport.connections.get()).map(|slot| {
        let connect = &connect;
        connect_until_ready(
            peer,
            endpoint,
            slot,
            transport,
            deadline,
            bootstrap,
            move |timeout| connect(slot, timeout),
        )
    }))
    .await?;
    channels.extend(remaining);
    Ok(channels)
}

async fn connect_until_ready<C, F, E>(
    peer: &str,
    endpoint_label: &str,
    pool_slot: usize,
    transport: GrpcTransportConfig,
    deadline: Instant,
    bootstrap: bool,
    connect: impl Fn(Duration) -> F,
) -> Result<C, DynamoError>
where
    F: std::future::Future<Output = Result<C, E>>,
    E: std::error::Error + 'static,
{
    let started = Instant::now();
    let mut attempt = 0_u64;
    let mut last_error = None;
    let mut last_logged_at = None;
    let mut last_logged_error = None;
    let mut suppressed_attempts = 0_u64;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(startup_timeout(
                peer,
                endpoint_label,
                pool_slot,
                attempt,
                started.elapsed(),
                transport,
                last_error.as_deref(),
            ));
        }

        attempt += 1;
        match timeout_at(
            deadline,
            connect(transport.connect_attempt_timeout.min(remaining)),
        )
        .await
        {
            Ok(Ok(channel)) => return Ok(channel),
            Ok(Err(error)) => {
                let detailed_error = format_error_chain(&error);
                let now = Instant::now();
                let error_changed = last_logged_error.as_deref() != Some(detailed_error.as_str());
                let log_interval_elapsed = last_logged_at
                    .is_none_or(|last| now.duration_since(last) >= RETRY_LOG_INTERVAL);
                if error_changed || log_interval_elapsed {
                    // Synchronous constructors may precede logging setup;
                    // deferred launcher discovery already has a subscriber.
                    if bootstrap {
                        eprintln!(
                            "{peer} gRPC connection attempt failed; retrying (endpoint={endpoint_label}, pool_slot={pool_slot}, attempt={attempt}, elapsed={:?}, remaining={:?}, retry_interval={:?}, suppressed_attempts={suppressed_attempts}, error={detailed_error})",
                            started.elapsed(),
                            deadline.saturating_duration_since(now),
                            transport.retry_interval,
                        );
                    } else {
                        tracing::warn!(
                            peer,
                            endpoint = %endpoint_label,
                            pool_slot,
                            attempt,
                            elapsed = ?started.elapsed(),
                            remaining = ?deadline.saturating_duration_since(now),
                            retry_interval = ?transport.retry_interval,
                            suppressed_attempts,
                            error = ?error,
                            "sidecar gRPC connection attempt failed; retrying"
                        );
                    }
                    last_logged_at = Some(now);
                    last_logged_error = Some(detailed_error.clone());
                    suppressed_attempts = 0;
                } else {
                    suppressed_attempts += 1;
                }
                last_error = Some(detailed_error);
            }
            Err(_) => {
                return Err(startup_timeout(
                    peer,
                    endpoint_label,
                    pool_slot,
                    attempt,
                    started.elapsed(),
                    transport,
                    last_error.as_deref(),
                ));
            }
        }

        if Instant::now() >= deadline {
            return Err(startup_timeout(
                peer,
                endpoint_label,
                pool_slot,
                attempt,
                started.elapsed(),
                transport,
                last_error.as_deref(),
            ));
        }
        let retry_at = checked_instant_add(
            Instant::now(),
            transport.retry_interval,
            "gRPC retry interval",
        )?;
        sleep_until(retry_at.min(deadline)).await;
    }
}

/// The instant by which startup work -- connecting, waiting for the engine to
/// answer, discovering the model -- must be done. Every sidecar bounds its whole
/// startup path by this one deadline rather than by per-RPC constants, so
/// `--grpc-startup-deadline-secs` actually governs how long a slow-loading engine is
/// given.
pub fn startup_deadline(duration: Duration) -> Result<Instant, DynamoError> {
    checked_instant_add(Instant::now(), duration, "gRPC startup deadline")
}

fn checked_instant_add(
    instant: Instant,
    duration: Duration,
    setting: &str,
) -> Result<Instant, DynamoError> {
    instant.checked_add(duration).ok_or_else(|| {
        invalid_argument(format!(
            "{setting} {duration:?} exceeds the supported monotonic clock range"
        ))
    })
}

fn startup_timeout(
    peer: &str,
    endpoint: &str,
    pool_slot: usize,
    attempts: u64,
    elapsed: Duration,
    transport: GrpcTransportConfig,
    last_error: Option<&str>,
) -> DynamoError {
    let cause = last_error.unwrap_or("the connection attempt exceeded the startup deadline");
    cannot_connect(format!(
        "failed to establish the {peer} gRPC connection pool to {endpoint} for pool slot {pool_slot} within {:?} after {attempts} attempts over {elapsed:?}: {cause}",
        transport.startup_deadline,
    ))
}

/// Flattens an error's `source()` chain into one string ("outer: middle:
/// inner"). `Display` alone on a `tonic::transport::Error` gives only the
/// constant "transport error" -- this is what callers should log instead.
pub fn format_error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let _ = write!(message, ": {cause}");
        source = cause.source();
    }
    message
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use tokio::net::{TcpListener, TcpSocket};

    use super::{GrpcChannelPool, reject_self_connect};
    use crate::{GrpcEndpoint, GrpcTransportConfig};

    #[tokio::test]
    async fn self_connect_is_rejected_without_leaving_time_wait() {
        // Connecting a socket to its own bound address is a deterministic TCP
        // simultaneous open: the state an unlucky ephemeral source port produces.
        let socket = TcpSocket::new_v4().expect("socket");
        socket.bind("127.0.0.1:0".parse().unwrap()).expect("bind");
        let address = socket.local_addr().expect("address");
        let stream = socket.connect(address).await.expect("self-connect");

        let error = reject_self_connect(&stream).expect_err("self-connect must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::ConnectionRefused);
        drop(stream);

        // A TIME_WAIT left by the client socket would fail this bind.
        let server = TcpSocket::new_v4().expect("socket");
        server.set_reuseaddr(false).expect("SO_REUSEADDR");
        server
            .bind(address)
            .expect("the engine port must be free after a rejected self-connect");
    }

    #[tokio::test]
    async fn startup_deadline_caps_connection_retries() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        drop(listener);

        let transport = GrpcTransportConfig {
            connections: NonZeroUsize::new(2).unwrap(),
            connect_attempt_timeout: Duration::from_millis(50),
            retry_interval: Duration::from_millis(10),
            startup_deadline: Duration::from_millis(100),
        };
        let endpoint = GrpcEndpoint::parse(&address.to_string(), "--test-endpoint").unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(300),
            GrpcChannelPool::connect("test", &endpoint, transport, false),
        )
        .await
        .expect("connection retries must respect the startup deadline");

        let error = match result {
            Ok(_) => panic!("the endpoint is closed"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("within 100ms"),
            "unexpected error: {error}"
        );
    }
}
