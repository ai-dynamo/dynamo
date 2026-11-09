// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's HTTP(S) accept loop, with graceful shutdown bounded by a grace period.
//!
//! A connection task is polled on every wakeup, i.e. for every streamed response chunk, and
//! each poll also polls its shutdown signals. Every connection waits on its own child
//! [`CancellationToken`]s, whose lock and `Notify` no other connection touches. A shared
//! `tokio::sync::Notify` would lock its waiter list on every poll of a pending waiter.

use std::pin::pin;
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;

use super::shutdown::ShutdownController;
use super::write_timeout::{ConnectionDeadline, ConnectionService, WriteTimeoutIo};

/// The handshake timeout of axum-server's rustls acceptor, which served the frontend before.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Serves `router` on `listener`, driven by the shared [`ShutdownController`].
///
/// The accept loop runs until the controller stops accepting (Draining), then drops the
/// listener so new connections are refused at once. Each connection runs on the
/// controller's tracker: it retires with the protocol's own mechanism (HTTP/1.1 keep-alive
/// off, HTTP/2 GOAWAY) when the controller stops accepting, finishes its in-flight
/// exchanges, and is dropped if still open at the controller's hard deadline. An upgraded
/// connection, such as a WebSocket session, stays on the tracker after `serve` returns.
///
/// Every connection is wrapped in [`WriteTimeoutIo`], and every request on it carries the
/// connection's [`ConnectionDeadline`].
pub(super) async fn serve(
    listener: TcpListener,
    tls: Option<TlsAcceptor>,
    router: Router,
    write_idle_timeout: Duration,
    controller: &ShutdownController,
) {
    // Build each route's handler once, as `Router::into_make_service` does, not per request.
    let router = router.with_state(());
    let stop_accepting = controller.stop_accepting_token();

    loop {
        let stream = tokio::select! {
            biased;
            _ = stop_accepting.cancelled() => break,
            stream = accept(&listener) => stream,
        };
        let connection = serve_connection(
            stream,
            tls.clone(),
            router.clone(),
            write_idle_timeout,
            controller.stop_accepting_token(),
            controller.remaining(),
        );
        // The hard deadline drops the connection future; the tracker lets the
        // controller observe when every connection has ended (early exit).
        controller.spawn_connection(connection);
    }

    // Refuse new connections at once instead of leaving them in the backlog. The
    // controller then drives graceful retirement, abort and hard close across
    // every connection it tracks.
    drop(listener);
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
    remaining_budget: Option<Duration>,
) {
    let Some(tls) = tls else {
        return serve_io(stream, router, write_idle_timeout, graceful).await;
    };
    // Bound the handshake by the shorter of the fixed timeout and whatever is left of
    // the shutdown budget, so a client stalling mid-handshake cannot outlast the drain.
    let handshake_timeout = remaining_budget
        .map(|remaining| remaining.min(TLS_HANDSHAKE_TIMEOUT))
        .unwrap_or(TLS_HANDSHAKE_TIMEOUT);
    match tokio::time::timeout(handshake_timeout, tls.accept(stream)).await {
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
}
