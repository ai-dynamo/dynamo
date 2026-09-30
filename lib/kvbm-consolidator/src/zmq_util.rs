// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal ZMQ helpers for the consolidator crate.
//!
//! Mirrors `lib/llm/src/utils/zmq.rs` without pulling in the whole `dynamo-llm` crate.

use std::sync::Arc;

use anyhow::Result;
use futures::SinkExt;
use tmq::{
    Context, Multipart, SocketBuilder,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::Mutex;

pub type MultipartMessage = Vec<Vec<u8>>;
pub type SharedPubSocket = Arc<Mutex<Publish>>;
pub type SubSocket = Subscribe;

const ZMQ_RCVTIMEOUT_MS: i32 = 100;
const ZMQ_SNDTIMEOUT_MS: i32 = 0;
const ZMQ_RECONNECT_IVL_MS: i32 = 100;
const ZMQ_RECONNECT_IVL_MAX_MS: i32 = 5000;
const ZMQ_TCP_KEEPALIVE: i32 = 1;
const ZMQ_LINGER_MS: i32 = 0;

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

pub async fn connect_sub_socket(endpoint: &str, topic: Option<&str>) -> Result<SubSocket> {
    let ctx = Context::new();
    let socket = configure_common_builder(subscribe(&ctx))
        .set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
        .connect(endpoint)?
        .subscribe(topic.unwrap_or("").as_bytes())?;
    Ok(socket)
}

pub async fn bind_pub_socket(endpoint: &str) -> Result<SharedPubSocket> {
    let ctx = Context::new();
    let socket = configure_common_builder(publish(&ctx))
        .set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
        .bind(endpoint)?;
    Ok(Arc::new(Mutex::new(socket)))
}

pub fn multipart_message(multipart: Multipart) -> MultipartMessage {
    multipart.into_iter().map(|frame| frame.to_vec()).collect()
}

pub async fn send_multipart<S>(socket: &Arc<Mutex<S>>, frames: MultipartMessage) -> Result<()>
where
    S: futures::Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.lock().await.send(Multipart::from(frames)).await?;
    Ok(())
}
