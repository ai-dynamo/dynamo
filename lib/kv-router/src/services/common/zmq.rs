// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::future::poll_fn;
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::{Mutex, OnceLock};
use std::task::{Context, Poll, ready};

use anyhow::{Result, anyhow};
use tokio::io::unix::AsyncFd;

pub(crate) type MultipartMessage = Vec<Vec<u8>>;

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

/// Process-wide ZMQ contexts. A context owns an I/O thread pool and a reaper thread, so a
/// context per socket cost two threads per socket: thousands at large worker counts.
///
/// libzmq caps a context at `ZMQ_MAX_SOCKETS` (1023, which the `zmq` crate cannot raise),
/// and a closed socket keeps its slot until the context's reaper thread has freed it. So the
/// pool lets libzmq decide: a socket goes to the first context that accepts it, and the pool
/// adds a context only when every existing one refuses with `EMFILE`.
struct ContextPool {
    contexts: Mutex<Vec<zmq::Context>>,
    io_threads: i32,
}

impl ContextPool {
    fn new(io_threads: i32) -> Self {
        Self {
            contexts: Mutex::new(Vec::new()),
            io_threads,
        }
    }

    fn global() -> &'static Self {
        static POOL: OnceLock<ContextPool> = OnceLock::new();
        POOL.get_or_init(|| Self::new(io_threads_from_env()))
    }

    fn socket(&self, socket_type: zmq::SocketType) -> Result<zmq::Socket> {
        let mut contexts = self.contexts.lock().unwrap();
        for context in contexts.iter() {
            match context.socket(socket_type) {
                Err(zmq::Error::EMFILE) => continue,
                result => return Ok(result?),
            }
        }
        let context = zmq::Context::new();
        // Takes effect only before the context's first socket.
        context.set_io_threads(self.io_threads)?;
        let socket = context.socket(socket_type)?;
        contexts.push(context);
        Ok(socket)
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
    socket: zmq::Socket,
    fd: RawFd,
}

impl SocketWrapper {
    fn new(socket: zmq::Socket) -> Result<Self> {
        Ok(Self {
            fd: socket.get_fd()?,
            socket,
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
    fn new(socket: zmq::Socket) -> Result<Self> {
        let wrapper = SocketWrapper::new(socket)?;
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
    let socket = ContextPool::global().socket(socket_type)?;
    configure(&socket)?;
    ZmqSocket::new(socket)
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
pub(crate) fn connect_sub_socket(endpoint: &str) -> Result<ZmqSocket> {
    let socket = create_sub_socket(b"")?;
    socket.connect(endpoint)?;
    Ok(socket)
}

#[cfg(feature = "standalone-indexer")]
pub(crate) fn connect_dealer_socket(endpoint: &str) -> Result<ZmqSocket> {
    build_socket(zmq::DEALER, |socket| {
        configure_bidirectional_socket(socket)?;
        set_ipv6_for(socket, endpoint)?;
        socket.connect(endpoint)?;
        Ok(())
    })
}

#[cfg(any(
    feature = "standalone-slot-tracker",
    feature = "standalone-selection",
    test
))]
pub(crate) fn create_bound_pub_socket(endpoint: &str) -> Result<ZmqSocket> {
    // Not pooled: the socket holds the only reference to its context, so dropping it
    // terminates the context, which waits until libzmq has closed the listener. The port is
    // free again as soon as the publisher's owner has shut down.
    let socket = zmq::Context::new().socket(zmq::PUB)?;
    configure_send_socket(&socket)?;
    set_ipv6_for(&socket, endpoint)?;
    socket.bind(endpoint)?;
    ZmqSocket::new(socket)
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

    /// Soft `RLIMIT_NOFILE` from `/proc/self/limits`; 0 when it can't be read.
    fn open_file_limit() -> u64 {
        std::fs::read_to_string("/proc/self/limits")
            .ok()
            .and_then(|limits| {
                let line = limits
                    .lines()
                    .find(|line| line.starts_with("Max open files"))?;
                line.split_whitespace().nth(3)?.parse().ok()
            })
            .unwrap_or(0)
    }

    #[test]
    fn pool_adds_a_context_when_libzmq_refuses_a_socket() {
        // libzmq holds an eventfd per socket and refuses the 1024th socket in a context.
        if open_file_limit() < 4096 {
            eprintln!("Skipping ZMQ context pool test: needs more than 1024 open files");
            return;
        }
        let pool = ContextPool::new(1);
        let _sockets: Vec<_> = (0..1024).map(|_| pool.socket(zmq::SUB).unwrap()).collect();
        assert_eq!(pool.contexts.lock().unwrap().len(), 2);
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
