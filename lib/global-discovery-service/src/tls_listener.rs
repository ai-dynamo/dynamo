// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A TLS listener for the discovery service.
//!
//! A background task accepts TCP connections and runs each TLS handshake on
//! its own task, so a client that stalls its handshake holds up nobody else.
//! Finished handshakes are handed to the HTTP server. Dropping the listener
//! stops the task and every handshake still running.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::serve::Listener;
use rustls::ServerConfig;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;

/// How long a client has to finish its TLS handshake.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long to wait before accepting again after an accept error, such as
/// running out of file descriptors.
const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(50);

type Accepted = (TlsStream<TcpStream>, SocketAddr);

pub(crate) struct TlsListener {
    local_addr: SocketAddr,
    accepted: mpsc::Receiver<Accepted>,
    acceptor_task: JoinHandle<()>,
}

impl TlsListener {
    pub(crate) fn new(listener: TcpListener, config: ServerConfig) -> std::io::Result<Self> {
        let local_addr = listener.local_addr()?;
        let (sender, accepted) = mpsc::channel(64);
        let acceptor = TlsAcceptor::from(Arc::new(config));
        let acceptor_task = tokio::spawn(accept_loop(listener, acceptor, sender));
        Ok(Self {
            local_addr,
            accepted,
            acceptor_task,
        })
    }
}

impl Drop for TlsListener {
    fn drop(&mut self) {
        self.acceptor_task.abort();
    }
}

impl Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> Accepted {
        match self.accepted.recv().await {
            Some(accepted) => accepted,
            // The accept loop only ends when the listener is dropped, and
            // the server does not call `accept` after that.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        Ok(self.local_addr)
    }
}

async fn accept_loop(
    listener: TcpListener,
    acceptor: TlsAcceptor,
    accepted: mpsc::Sender<Accepted>,
) {
    // Dropped with this task, which aborts the handshakes still running.
    let mut handshakes = JoinSet::new();
    loop {
        while handshakes.try_join_next().is_some() {}
        let (stream, peer) = match listener.accept().await {
            Ok(connection) => connection,
            Err(error) => {
                tracing::debug!(%error, "failed to accept a connection");
                tokio::time::sleep(ACCEPT_ERROR_BACKOFF).await;
                continue;
            }
        };
        let acceptor = acceptor.clone();
        let accepted = accepted.clone();
        handshakes.spawn(async move {
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, acceptor.accept(stream)).await {
                Ok(Ok(stream)) => {
                    let _ = accepted.send((stream, peer)).await;
                }
                Ok(Err(error)) => tracing::debug!(%peer, %error, "TLS handshake failed"),
                Err(_) => tracing::debug!(%peer, "TLS handshake timed out"),
            }
        });
    }
}
