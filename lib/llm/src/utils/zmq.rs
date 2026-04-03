// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_runtime::transports::zmq::{
    MultipartMessage, SharedRawZmqSocket, spawn_configured_raw_socket,
};

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

fn configure_send_socket(socket: &zmq::Socket) -> Result<()> {
    configure_common_socket(socket)?;
    socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
    Ok(())
}

pub(crate) async fn connect_sub_socket(
    endpoint: &str,
    topic: Option<&str>,
) -> Result<SharedRawZmqSocket> {
    let endpoint = endpoint.to_string();
    let topic = topic.unwrap_or("").to_string();
    spawn_configured_raw_socket(zmq::SUB, move |socket| {
        configure_receive_socket(socket)?;
        socket.set_subscribe(topic.as_bytes())?;
        socket.connect(&endpoint)?;
        Ok(())
    })
    .await
}

pub(crate) async fn bind_pub_socket(endpoint: &str) -> Result<SharedRawZmqSocket> {
    let endpoint = endpoint.to_string();
    spawn_configured_raw_socket(zmq::PUB, move |socket| {
        configure_send_socket(socket)?;
        socket.bind(&endpoint)?;
        Ok(())
    })
    .await
}

pub(crate) async fn bind_router_socket(endpoint: &str) -> Result<SharedRawZmqSocket> {
    let endpoint = endpoint.to_string();
    spawn_configured_raw_socket(zmq::ROUTER, move |socket| {
        configure_bidirectional_socket(socket)?;
        socket.bind(&endpoint)?;
        Ok(())
    })
    .await
}

pub(crate) async fn send_multipart(
    socket: &SharedRawZmqSocket,
    frames: MultipartMessage,
) -> Result<()> {
    let socket = Arc::clone(socket);
    tokio::task::spawn_blocking(move || -> Result<()> {
        let socket = socket.lock().expect("ZMQ socket mutex poisoned");
        socket.send_multipart(frames, 0)?;
        Ok(())
    })
    .await
    .context("failed to join ZMQ send task")?
}
