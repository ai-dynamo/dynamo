// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::future::poll_fn;
use std::os::unix::io::{AsRawFd, RawFd};
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
        Ok(Self(AsyncFd::new(SocketWrapper::new(socket)?)?))
    }

    fn socket(&self) -> &zmq::Socket {
        &self.0.get_ref().socket
    }

    pub(crate) fn connect(&self, endpoint: &str) -> Result<()> {
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
    let context = zmq::Context::new();
    let socket = context.socket(socket_type)?;
    configure(&socket)?;
    ZmqSocket::new(socket)
}

pub(crate) fn create_sub_socket(topic: &[u8]) -> Result<ZmqSocket> {
    build_socket(zmq::SUB, |socket| {
        configure_receive_socket(socket)?;
        socket.set_subscribe(topic)?;
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
