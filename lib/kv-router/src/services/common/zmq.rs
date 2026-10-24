// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::future::poll_fn;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc as StdArc, OnceLock};
use std::task::{Context, Poll, ready};

use anyhow::{Result, anyhow};
#[cfg(feature = "standalone-indexer")]
use std::sync::Arc;
use tokio::io::unix::AsyncFd;
#[cfg(feature = "standalone-indexer")]
use tokio::sync::Mutex;

pub(crate) type MultipartMessage = Vec<Vec<u8>>;
#[cfg(feature = "standalone-indexer")]
pub(crate) type SharedSocket = Arc<Mutex<ZmqSocket>>;

const ZMQ_RCVTIMEOUT_MS: i32 = 100;
const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_HEARTBEAT_IVL_MS: i32 = 5000;
const ZMQ_HEARTBEAT_TIMEOUT_MS: i32 = 15000;
const ZMQ_HEARTBEAT_TTL_MS: i32 = 15000;
const ZMQ_LINGER_MS: i32 = 0;

/// I/O threads for each shared context; the same variable the runtime event plane reads.
const IO_THREADS_ENV: &str = "DYN_ZMQ_IO_THREADS";
const DEFAULT_IO_THREADS: i32 = 4;
/// Sockets per shared context, under libzmq's default `ZMQ_MAX_SOCKETS` of 1023, which the
/// `zmq` crate cannot raise.
const SOCKETS_PER_CONTEXT: usize = 1000;

/// Process-wide ZMQ contexts. A context owns an I/O thread pool and a reaper thread, so a
/// context per socket costs two threads per KV-event source; one SUB and one replay DEALER
/// per worker rank made thousands of threads at large worker counts. Sockets share a
/// context until it holds `capacity` of them, then the pool adds another.
struct ContextPool {
    contexts: std::sync::Mutex<Vec<PooledContext>>,
    capacity: usize,
    io_threads: i32,
}

struct PooledContext {
    context: zmq::Context,
    sockets: StdArc<AtomicUsize>,
}

/// A socket's slot in a pooled context, released when the socket closes.
struct ContextLease(StdArc<AtomicUsize>);

impl Drop for ContextLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl ContextPool {
    fn new(capacity: usize, io_threads: i32) -> Self {
        Self {
            contexts: std::sync::Mutex::new(Vec::new()),
            capacity,
            io_threads,
        }
    }

    fn global() -> &'static Self {
        static POOL: OnceLock<ContextPool> = OnceLock::new();
        POOL.get_or_init(|| Self::new(SOCKETS_PER_CONTEXT, io_threads_from_env()))
    }

    /// Only this method adds sockets, under the lock, so no context exceeds `capacity`.
    fn lease(&self) -> Result<(zmq::Context, ContextLease)> {
        let mut contexts = self.contexts.lock().unwrap();
        let index = match contexts
            .iter()
            .position(|pooled| pooled.sockets.load(Ordering::Acquire) < self.capacity)
        {
            Some(index) => index,
            None => {
                let context = zmq::Context::new();
                // Takes effect only before the context's first socket.
                context.set_io_threads(self.io_threads)?;
                contexts.push(PooledContext {
                    context,
                    sockets: StdArc::default(),
                });
                contexts.len() - 1
            }
        };
        let pooled = &contexts[index];
        pooled.sockets.fetch_add(1, Ordering::AcqRel);
        Ok((
            pooled.context.clone(),
            ContextLease(StdArc::clone(&pooled.sockets)),
        ))
    }
}

fn io_threads_from_env() -> i32 {
    let Some(value) = std::env::var_os(IO_THREADS_ENV) else {
        return DEFAULT_IO_THREADS;
    };
    match value.to_str().and_then(|value| value.parse::<i32>().ok()) {
        Some(threads) if threads > 0 => threads,
        _ => {
            tracing::warn!(
                value = ?value,
                default = DEFAULT_IO_THREADS,
                "{IO_THREADS_ENV} must be a positive integer; using the default"
            );
            DEFAULT_IO_THREADS
        }
    }
}

struct SocketWrapper {
    // Field order matters: the socket closes before its context slot is released.
    socket: zmq::Socket,
    fd: RawFd,
    _lease: ContextLease,
}

impl SocketWrapper {
    fn new(socket: zmq::Socket, lease: ContextLease) -> Result<Self> {
        Ok(Self {
            fd: socket.get_fd()?,
            socket,
            _lease: lease,
        })
    }
}

impl AsRawFd for SocketWrapper {
    fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

pub(crate) struct ZmqSocket(AsyncFd<SocketWrapper>);

impl ZmqSocket {
    fn new(socket: zmq::Socket, lease: ContextLease) -> Result<Self> {
        let wrapper = SocketWrapper::new(socket, lease)?;
        // SAFETY: `fd` is the socket's `ZMQ_FD`, owned by the `zmq::Socket` that the wrapper
        // owns, so it stays open until the wrapper is dropped, and `as_raw_fd` always returns it.
        let fd = unsafe { AsyncFd::register(wrapper) }.map_err(std::io::Error::from)?;
        Ok(Self(fd))
    }

    fn socket(&self) -> &zmq::Socket {
        &self.0.get_ref().socket
    }

    pub(crate) fn connect(&self, endpoint: &str) -> Result<()> {
        set_ipv6_for(self.socket(), endpoint)?;
        self.socket().connect(endpoint)?;
        Ok(())
    }

    #[cfg(any(feature = "standalone-slot-tracker", feature = "standalone-selection"))]
    pub(crate) fn disconnect(&self, endpoint: &str) -> Result<()> {
        self.socket().disconnect(endpoint)?;
        Ok(())
    }

    fn poll_socket_event(
        &mut self,
        cx: &mut Context<'_>,
        event: zmq::PollEvents,
    ) -> Poll<Result<()>> {
        if self.socket().get_events()?.contains(event) {
            Poll::Ready(Ok(()))
        } else {
            self.clear_read_ready(cx)?;
            Poll::Pending
        }
    }

    fn clear_read_ready(&mut self, cx: &mut Context<'_>) -> Result<()> {
        if let Poll::Ready(mut guard) = self.0.poll_read_ready(cx)? {
            guard.clear_ready();
            cx.waker().wake_by_ref();
        }
        Ok(())
    }

    fn poll_recv_multipart(&mut self, cx: &mut Context<'_>) -> Poll<Result<MultipartMessage>> {
        ready!(self.poll_socket_event(cx, zmq::POLLIN))?;

        let mut frames = Vec::new();
        loop {
            let mut msg = zmq::Message::new();
            match self.socket().recv(&mut msg, zmq::DONTWAIT) {
                Ok(_) => {
                    let more = msg.get_more();
                    frames.push(msg.to_vec());
                    if !more {
                        return Poll::Ready(Ok(frames));
                    }
                }
                Err(zmq::Error::EAGAIN) if frames.is_empty() => {
                    self.clear_read_ready(cx)?;
                    return Poll::Pending;
                }
                Err(zmq::Error::EAGAIN) => {
                    return Poll::Ready(Err(anyhow!(
                        "multipart receive interrupted after {} frames",
                        frames.len()
                    )));
                }
                Err(error) => return Poll::Ready(Err(error.into())),
            }
        }
    }

    fn poll_send_multipart(
        &mut self,
        cx: &mut Context<'_>,
        buffer: &mut VecDeque<zmq::Message>,
    ) -> Poll<Result<()>> {
        while !buffer.is_empty() {
            ready!(self.poll_socket_event(cx, zmq::POLLOUT))?;

            while let Some(frame) = buffer.pop_front() {
                let mut flags = zmq::DONTWAIT;
                if !buffer.is_empty() {
                    flags |= zmq::SNDMORE;
                }

                match self.socket().send(&*frame, flags) {
                    Ok(_) => {}
                    Err(zmq::Error::EAGAIN) => {
                        buffer.push_front(frame);
                        self.clear_read_ready(cx)?;
                        return Poll::Pending;
                    }
                    Err(error) => return Poll::Ready(Err(error.into())),
                }
            }
        }

        Poll::Ready(Ok(()))
    }

    pub(crate) async fn recv_multipart(&mut self) -> Result<MultipartMessage> {
        poll_fn(|cx| self.poll_recv_multipart(cx)).await
    }

    pub(crate) async fn send_multipart(&mut self, frames: MultipartMessage) -> Result<()> {
        let mut buffer = frames
            .into_iter()
            .map(zmq::Message::from)
            .collect::<VecDeque<_>>();
        poll_fn(|cx| self.poll_send_multipart(cx, &mut buffer)).await
    }
}

/// libzmq resolves TCP hosts as IPv4 unless `ZMQ_IPV6` is set, and snapshots the option per
/// bind or connect. [`validate_endpoint`] rejects unbracketed IPv6 literals.
fn set_ipv6_for(socket: &zmq::Socket, endpoint: &str) -> Result<()> {
    socket.set_ipv6(endpoint.starts_with("tcp://["))?;
    Ok(())
}

fn configure_common_socket(socket: &zmq::Socket) -> Result<()> {
    socket.set_linger(ZMQ_LINGER_MS)?;
    socket.set_reconnect_ivl(ZMQ_RECONNECT_IVL_MS)?;
    socket.set_reconnect_ivl_max(ZMQ_RECONNECT_IVL_MAX_MS)?;
    socket.set_tcp_keepalive(ZMQ_TCP_KEEPALIVE)?;
    socket.set_heartbeat_ivl(ZMQ_HEARTBEAT_IVL_MS)?;
    socket.set_heartbeat_timeout(ZMQ_HEARTBEAT_TIMEOUT_MS)?;
    socket.set_heartbeat_ttl(ZMQ_HEARTBEAT_TTL_MS)?;
    Ok(())
}

fn configure_receive_socket(socket: &zmq::Socket) -> Result<()> {
    configure_common_socket(socket)?;
    socket.set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)?;
    Ok(())
}

#[cfg(feature = "standalone-indexer")]
fn configure_bidirectional_socket(socket: &zmq::Socket) -> Result<()> {
    configure_receive_socket(socket)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

#[cfg(any(
    feature = "standalone-slot-tracker",
    feature = "standalone-selection",
    test
))]
fn configure_send_socket(socket: &zmq::Socket) -> Result<()> {
    configure_common_socket(socket)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

fn build_socket<F>(socket_type: zmq::SocketType, configure: F) -> Result<ZmqSocket>
where
    F: FnOnce(&zmq::Socket) -> Result<()>,
{
    build_socket_in(ContextPool::global(), socket_type, configure)
}

fn build_socket_in<F>(
    pool: &ContextPool,
    socket_type: zmq::SocketType,
    configure: F,
) -> Result<ZmqSocket>
where
    F: FnOnce(&zmq::Socket) -> Result<()>,
{
    let (context, lease) = pool.lease()?;
    let socket = context.socket(socket_type)?;
    configure(&socket)?;
    ZmqSocket::new(socket, lease)
}

#[cfg(any(test, feature = "standalone-indexer"))]
pub(crate) fn create_sub_socket(topic: &[u8]) -> Result<ZmqSocket> {
    create_sub_socket_topics(&[topic])
}

pub(crate) fn create_sub_socket_topics(topics: &[&[u8]]) -> Result<ZmqSocket> {
    build_socket(zmq::SUB, |socket| {
        configure_receive_socket(socket)?;
        for topic in topics {
            socket.set_subscribe(topic)?;
        }
        Ok(())
    })
}

#[cfg(feature = "standalone-indexer")]
pub(crate) fn connect_sub_socket(endpoint: &str) -> Result<SharedSocket> {
    let socket = create_sub_socket(b"")?;
    socket.connect(endpoint)?;
    Ok(Arc::new(Mutex::new(socket)))
}

#[cfg(feature = "standalone-indexer")]
pub(crate) fn connect_dealer_socket(endpoint: &str) -> Result<SharedSocket> {
    Ok(Arc::new(Mutex::new(build_socket(zmq::DEALER, |socket| {
        configure_bidirectional_socket(socket)?;
        set_ipv6_for(socket, endpoint)?;
        socket.connect(endpoint)?;
        Ok(())
    })?)))
}

#[cfg(any(
    feature = "standalone-slot-tracker",
    feature = "standalone-selection",
    test
))]
pub(crate) fn create_bound_pub_socket(endpoint: &str) -> Result<ZmqSocket> {
    build_socket(zmq::PUB, |socket| {
        configure_send_socket(socket)?;
        set_ipv6_for(socket, endpoint)?;
        socket.bind(endpoint)?;
        Ok(())
    })
}

#[cfg(feature = "standalone-indexer")]
pub(crate) async fn recv_multipart(socket: &SharedSocket) -> Result<MultipartMessage> {
    let mut socket = socket.lock().await;
    socket.recv_multipart().await
}

#[cfg(feature = "standalone-indexer")]
pub(crate) async fn send_multipart(socket: &SharedSocket, frames: MultipartMessage) -> Result<()> {
    let mut socket = socket.lock().await;
    socket.send_multipart(frames).await
}

pub(crate) fn validate_endpoint(endpoint: &str) -> Result<()> {
    let (scheme, address) = endpoint
        .split_once("://")
        .ok_or_else(|| anyhow!("invalid ZMQ endpoint `{endpoint}`: missing scheme"))?;

    if address.is_empty() {
        return Err(anyhow!(
            "invalid ZMQ endpoint `{endpoint}`: missing address"
        ));
    }

    match scheme {
        "tcp" => {
            let (host, port) = address
                .rsplit_once(':')
                .ok_or_else(|| anyhow!("invalid ZMQ endpoint `{endpoint}`: missing TCP port"))?;
            if host.is_empty() {
                return Err(anyhow!(
                    "invalid ZMQ endpoint `{endpoint}`: missing TCP host"
                ));
            }
            if host.starts_with('[') {
                if !host.ends_with(']') {
                    return Err(anyhow!(
                        "invalid ZMQ endpoint `{endpoint}`: missing closing `]`"
                    ));
                }
            } else if host.contains(':') {
                return Err(anyhow!(
                    "invalid ZMQ endpoint `{endpoint}`: missing TCP port"
                ));
            }
            port.parse::<u16>().map_err(|error| {
                anyhow!("invalid ZMQ endpoint `{endpoint}`: invalid TCP port: {error}")
            })?;
            Ok(())
        }
        "ipc" | "inproc" => Ok(()),
        other => Err(anyhow!(
            "invalid ZMQ endpoint `{endpoint}`: unsupported scheme `{other}`"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn pool_adds_a_context_when_full_and_reuses_freed_slots() {
        let pool = ContextPool::new(2, 1);
        let sub = || build_socket_in(&pool, zmq::SUB, |_| Ok(())).unwrap();
        let contexts = || pool.contexts.lock().unwrap().len();
        let first = sub();
        let _second = sub();
        assert_eq!(contexts(), 1);
        let _third = sub();
        assert_eq!(contexts(), 2, "a full context gets a sibling");
        drop(first);
        let _reused = sub();
        assert_eq!(contexts(), 2, "a closed socket frees its slot");
    }

    #[tokio::test]
    async fn bracketed_ipv6_endpoints_connect() {
        if let Err(error) = std::net::TcpListener::bind("[::1]:0") {
            eprintln!("Skipping IPv6 ZMQ test: {error}");
            return;
        }
        let mut publisher = create_bound_pub_socket("tcp://[::1]:*").unwrap();
        let endpoint = publisher.socket().get_last_endpoint().unwrap().unwrap();
        validate_endpoint(&endpoint).unwrap();
        let mut subscriber = create_sub_socket(b"").unwrap();
        subscriber.connect(&endpoint).unwrap();

        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                publisher
                    .send_multipart(vec![b"ipv6".to_vec()])
                    .await
                    .unwrap();
                if let Ok(frames) =
                    tokio::time::timeout(Duration::from_millis(50), subscriber.recv_multipart())
                        .await
                {
                    assert_eq!(frames.unwrap(), vec![b"ipv6".to_vec()]);
                    break;
                }
            }
        })
        .await
        .expect("IPv6 SUB socket should receive the message");
    }
}
