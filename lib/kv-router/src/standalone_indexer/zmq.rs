// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

pub(super) type MultipartMessage = Vec<Vec<u8>>;
pub(super) type SharedSocket = Arc<Mutex<zmq::Socket>>;

const ZMQ_RCVTIMEOUT_MS: i32 = 100;
const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_HEARTBEAT_IVL_MS: i32 = 5000;
const ZMQ_HEARTBEAT_TIMEOUT_MS: i32 = 15000;
const ZMQ_HEARTBEAT_TTL_MS: i32 = 15000;
const ZMQ_LINGER_MS: i32 = 0;

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

fn configure_bidirectional_socket(socket: &zmq::Socket) -> Result<()> {
    configure_receive_socket(socket)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

#[cfg(test)]
fn configure_send_socket(socket: &zmq::Socket) -> Result<()> {
    configure_common_socket(socket)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

async fn spawn_socket<F>(socket_type: zmq::SocketType, configure: F) -> Result<SharedSocket>
where
    F: FnOnce(&zmq::Socket) -> Result<()> + Send + 'static,
{
    tokio::task::spawn_blocking(move || -> Result<SharedSocket> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(socket_type)?;
        configure(&socket)?;
        Ok(Arc::new(Mutex::new(socket)))
    })
    .await
    .context("failed to join ZMQ socket task")?
}

pub(super) async fn connect_sub_socket(endpoint: &str) -> Result<SharedSocket> {
    let endpoint = endpoint.to_string();
    spawn_socket(zmq::SUB, move |socket| {
        configure_receive_socket(socket)?;
        socket.set_subscribe(b"")?;
        socket.connect(&endpoint)?;
        Ok(())
    })
    .await
}

pub(super) async fn connect_dealer_socket(endpoint: &str) -> Result<SharedSocket> {
    let endpoint = endpoint.to_string();
    spawn_socket(zmq::DEALER, move |socket| {
        configure_bidirectional_socket(socket)?;
        socket.connect(&endpoint)?;
        Ok(())
    })
    .await
}

#[cfg(test)]
pub(super) async fn bind_pub_socket(endpoint: &str) -> Result<SharedSocket> {
    let endpoint = endpoint.to_string();
    spawn_socket(zmq::PUB, move |socket| {
        configure_send_socket(socket)?;
        socket.bind(&endpoint)?;
        Ok(())
    })
    .await
}

pub(super) async fn recv_multipart(socket: &SharedSocket) -> Result<Option<MultipartMessage>> {
    let socket = Arc::clone(socket);
    tokio::task::spawn_blocking(move || -> Result<Option<MultipartMessage>> {
        let socket = socket.lock().expect("ZMQ socket mutex poisoned");
        match socket.recv_multipart(0) {
            Ok(frames) => Ok(Some(frames)),
            Err(zmq::Error::EAGAIN) => Ok(None),
            Err(error) => Err(error.into()),
        }
    })
    .await
    .context("failed to join ZMQ recv task")?
}

pub(super) async fn send_multipart(socket: &SharedSocket, frames: MultipartMessage) -> Result<()> {
    let socket = Arc::clone(socket);
    tokio::task::spawn_blocking(move || -> Result<()> {
        let socket = socket.lock().expect("ZMQ socket mutex poisoned");
        socket.send_multipart(frames, 0)?;
        Ok(())
    })
    .await
    .context("failed to join ZMQ send task")?
}
