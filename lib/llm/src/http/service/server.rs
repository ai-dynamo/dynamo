// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's HTTP(S) accept loop, with graceful shutdown bounded by a grace period.
//!
//! A connection task is polled on every wakeup, i.e. for every streamed response chunk, and
//! each poll also polls its shutdown signals. Those signals are per-connection oneshot
//! channels, so a poll touches no state shared with other connections. A shared
//! `tokio::sync::Notify` would lock its waiter list on every poll of a pending waiter.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use parking_lot::Mutex;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, oneshot};
use tokio_rustls::TlsAcceptor;
use tokio_util::task::TaskTracker;

use super::write_timeout::{ConnectionDeadline, ConnectionService, WriteTimeoutIo};

/// The handshake timeout of axum-server's rustls acceptor, which served the frontend before.
const TLS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Shuts down a running [`serve`] loop.
#[derive(Clone, Default)]
pub(super) struct Handle(Arc<HandleInner>);

#[derive(Default)]
struct HandleInner {
    graceful: ShutdownSignal,
    hard: ShutdownSignal,
    grace_period: Mutex<Option<Duration>>,
    connections: TaskTracker,
    address: Mutex<Option<SocketAddr>>,
    listening: Notify,
}

impl Handle {
    /// Stops accepting connections and lets open connections finish their requests.
    /// Connections still open after `grace_period` are closed; `None` waits indefinitely.
    pub(super) fn graceful_shutdown(&self, grace_period: Option<Duration>) {
        *self.0.grace_period.lock() = grace_period;
        self.0.graceful.trigger();
    }

    /// Closes the listener and every connection at once.
    #[cfg(test)]
    pub(super) fn shutdown(&self) {
        self.0.hard.trigger();
    }

    /// Open connections, including those still in the TLS handshake.
    #[cfg(test)]
    pub(super) fn connection_count(&self) -> usize {
        self.0.connections.len()
    }

    /// The listener's address, once [`serve`] has started.
    pub(super) async fn listening(&self) -> Option<SocketAddr> {
        // `notify_waiters` wakes a `Notified` from the moment it is created.
        let listening = self.0.listening.notified();
        if let Some(address) = *self.0.address.lock() {
            return Some(address);
        }
        listening.await;
        *self.0.address.lock()
    }
}

/// Serves `router` on `listener` until `handle` shuts it down.
///
/// Every connection is wrapped in [`WriteTimeoutIo`], and every request on it carries the
/// connection's [`ConnectionDeadline`].
pub(super) async fn serve(
    listener: TcpListener,
    tls: Option<TlsAcceptor>,
    router: Router,
    write_idle_timeout: Duration,
    handle: Handle,
) -> io::Result<()> {
    let inner = &handle.0;
    *inner.address.lock() = listener.local_addr().ok();
    inner.listening.notify_waiters();

    let accept_loop = async {
        let mut graceful = pin!(inner.graceful.subscribe());
        loop {
            let stream = tokio::select! {
                biased;
                () = &mut graceful => return,
                stream = accept(&listener) => stream,
            };
            inner.connections.spawn(serve_connection(
                stream,
                tls.clone(),
                router.clone(),
                write_idle_timeout,
                inner.graceful.subscribe(),
                inner.hard.subscribe(),
            ));
        }
    };
    tokio::select! {
        biased;
        () = inner.hard.subscribe() => return Ok(()),
        () = accept_loop => {}
    }

    // Refuse new connections at once instead of leaving them in the backlog.
    drop(listener);
    inner.connections.close();
    let grace_period = *inner.grace_period.lock();
    let grace_period_elapsed = async {
        match grace_period {
            Some(grace_period) => tokio::time::sleep(grace_period).await,
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        biased;
        () = inner.hard.subscribe() => {}
        () = inner.connections.wait() => {}
        () = grace_period_elapsed => inner.hard.trigger(),
    }
    Ok(())
}

async fn accept(listener: &TcpListener) -> TcpStream {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => return stream,
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
    graceful: impl Future<Output = ()>,
    hard: impl Future<Output = ()>,
) {
    let Some(tls) = tls else {
        return serve_io(stream, router, write_idle_timeout, graceful, hard).await;
    };
    let mut hard = pin!(hard);
    let handshake = tokio::time::timeout(TLS_HANDSHAKE_TIMEOUT, tls.accept(stream));
    let stream = tokio::select! {
        biased;
        () = &mut hard => return,
        stream = handshake => match stream {
            Ok(Ok(stream)) => stream,
            Ok(Err(error)) => {
                tracing::debug!(%error, "TLS handshake failed");
                return;
            }
            Err(_) => {
                tracing::debug!("TLS handshake timed out");
                return;
            }
        },
    };
    serve_io(stream, router, write_idle_timeout, graceful, hard).await
}

async fn serve_io<I>(
    io: I,
    router: Router,
    write_idle_timeout: Duration,
    graceful: impl Future<Output = ()>,
    hard: impl Future<Output = ()>,
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
    let mut hard = pin!(hard);
    tokio::select! {
        biased;
        // Dropping the connection also drops its response bodies.
        () = &mut hard => return,
        () = graceful => connection.as_mut().graceful_shutdown(),
        _ = &mut connection => return,
    }
    tokio::select! {
        biased;
        () = hard => {}
        _ = connection => {}
    }
}

/// A shutdown notification that each subscriber receives on its own oneshot channel.
#[derive(Clone)]
struct ShutdownSignal(Arc<Mutex<Option<Subscribers>>>);

#[derive(Default)]
struct Subscribers {
    next_key: u64,
    senders: HashMap<u64, oneshot::Sender<()>>,
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Some(Subscribers::default()))))
    }
}

impl ShutdownSignal {
    /// Returns a future that completes once [`Self::trigger`] has been called.
    ///
    /// Dropping the future unsubscribes, so finished connections don't leave senders behind.
    fn subscribe(&self) -> impl Future<Output = ()> + Send + 'static {
        let (tx, rx) = oneshot::channel();
        let mut subscribers = self.0.lock();
        let (key, late_tx) = match subscribers.as_mut() {
            Some(subscribers) => {
                let key = subscribers.next_key;
                subscribers.next_key += 1;
                subscribers.senders.insert(key, tx);
                (Some(key), None)
            }
            None => (None, Some(tx)),
        };
        drop(subscribers);
        // After a trigger, the sender is dropped here, so a late subscriber completes at once.
        drop(late_tx);

        let unsubscribe = Unsubscribe {
            signal: self.clone(),
            key,
        };
        async move {
            let _unsubscribe = unsubscribe;
            let _ = rx.await;
        }
    }

    fn trigger(&self) {
        // Drop the senders after releasing the lock: dropping one wakes its task.
        let subscribers = self.0.lock().take();
        drop(subscribers);
    }
}

/// Removes a subscriber's sender when its future is dropped.
struct Unsubscribe {
    signal: ShutdownSignal,
    key: Option<u64>,
}

impl Drop for Unsubscribe {
    fn drop(&mut self) {
        let Some(key) = self.key else { return };
        let sender = self
            .signal
            .0
            .lock()
            .as_mut()
            .and_then(|subscribers| subscribers.senders.remove(&key));
        // Drop the sender after releasing the lock, as in `ShutdownSignal::trigger`.
        drop(sender);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_signal_notifies_live_and_late_subscribers() {
        let signal = ShutdownSignal::default();
        let live = signal.subscribe();
        // Finished subscribers are removed right away.
        for _ in 0..1000 {
            drop(signal.subscribe());
        }
        assert_eq!(signal.0.lock().as_ref().unwrap().senders.len(), 1);

        signal.trigger();
        live.await;
        // A connection accepted after shutdown sees it right away.
        signal.subscribe().await;
    }
}
