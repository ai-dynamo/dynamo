// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use futures::{SinkExt, StreamExt};
use tmq::{
    Context, Multipart, SocketBuilder,
    publish::{Publish, publish},
    router::{Router, router},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub(crate) type MultipartMessage = Vec<Vec<u8>>;
pub(crate) type SharedPubSocket = Arc<Mutex<Publish>>;
pub(crate) type SubSocket = Subscribe;

const ZMQ_RCVTIMEOUT_MS: i32 = 100;
const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_LINGER_MS: i32 = 0;
const INITIAL_SETUP_BACKOFF_MS: u64 = 10;
const MAX_SETUP_BACKOFF_MS: u64 = 5000;
const MAX_SETUP_BACKOFF_EXPONENT: u32 = 8;

fn calculate_setup_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_SETUP_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_SETUP_BACKOFF_EXPONENT)),
        MAX_SETUP_BACKOFF_MS,
    )
}

fn configure_common_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder
        .set_linger(ZMQ_LINGER_MS)
        .set_reconnect_ivl(ZMQ_RECONNECT_IVL_MS)
        .set_reconnect_ivl_max(ZMQ_RECONNECT_IVL_MAX_MS)
        .set_tcp_keepalive(ZMQ_TCP_KEEPALIVE)
}

fn configure_receive_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
}

fn configure_bidirectional_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_receive_builder(builder).set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

fn configure_send_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

pub(crate) async fn connect_sub_socket(endpoint: &str, topic: Option<&str>) -> Result<SubSocket> {
    let ctx = Context::new();
    let socket = configure_receive_builder(subscribe(&ctx))
        .connect(endpoint)?
        .subscribe(topic.unwrap_or("").as_bytes())?;
    Ok(socket)
}

pub(crate) async fn connect_sub_socket_with_retry(
    endpoint: &str,
    topic: Option<&str>,
    cancellation_token: &CancellationToken,
    log_prefix: &str,
) -> Option<SubSocket> {
    let mut consecutive_errors = 0u32;

    loop {
        if cancellation_token.is_cancelled() {
            tracing::debug!("{log_prefix}: cancelled before connecting to {endpoint}");
            return None;
        }

        match connect_sub_socket(endpoint, topic).await {
            Ok(socket) => return Some(socket),
            Err(error) => {
                consecutive_errors += 1;
                let backoff_ms = calculate_setup_backoff_ms(consecutive_errors);
                tracing::warn!(
                    error = %error,
                    consecutive_errors = consecutive_errors,
                    backoff_ms = backoff_ms,
                    "{log_prefix}: failed to connect ZMQ SUB during setup, retrying"
                );
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => return None,
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
            }
        }
    }
}

pub(crate) async fn bind_pub_socket(endpoint: &str) -> Result<SharedPubSocket> {
    let ctx = Context::new();
    let socket = configure_send_builder(publish(&ctx)).bind(endpoint)?;
    Ok(Arc::new(Mutex::new(socket)))
}

pub(crate) async fn bind_router_socket(endpoint: &str) -> Result<Router> {
    let ctx = Context::new();
    let socket = configure_bidirectional_builder(router(&ctx)).bind(endpoint)?;
    Ok(socket)
}

pub(crate) fn multipart_message(multipart: Multipart) -> MultipartMessage {
    multipart.into_iter().map(|frame| frame.to_vec()).collect()
}

pub(crate) async fn recv_multipart<S>(socket: &mut S) -> Result<MultipartMessage>
where
    S: futures::Stream<Item = std::result::Result<Multipart, tmq::TmqError>> + Unpin,
{
    match socket.next().await {
        Some(Ok(multipart)) => Ok(multipart_message(multipart)),
        Some(Err(error)) => Err(error.into()),
        None => Err(anyhow!("ZMQ stream ended")),
    }
}

pub(crate) async fn send_multipart<S>(
    socket: &Arc<Mutex<S>>,
    frames: MultipartMessage,
) -> Result<()>
where
    S: futures::Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.lock().await.send(Multipart::from(frames)).await?;
    Ok(())
}

pub(crate) async fn send_multipart_direct<S>(socket: &mut S, frames: MultipartMessage) -> Result<()>
where
    S: futures::Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.send(Multipart::from(frames)).await?;
    Ok(())
}
