// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::transports::zmq::ipv6_option_for;
use futures::SinkExt;
use tmq::{
    Context, Multipart, SocketBuilder,
    publish::{Publish, publish},
    pull::{Pull, pull},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::Mutex;

pub(crate) type MultipartMessage = Vec<Vec<u8>>;
#[cfg_attr(not(feature = "block-manager"), allow(dead_code))]
pub(crate) type SharedPubSocket = Arc<Mutex<Publish>>;
pub(crate) type SubSocket = Subscribe;
pub(crate) type PullSocket = Pull;

const ZMQ_RCVTIMEOUT_MS: i32 = 100;
#[cfg_attr(not(feature = "block-manager"), allow(dead_code))]
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

fn configure_receive_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
}

#[cfg_attr(not(feature = "block-manager"), allow(dead_code))]
fn configure_send_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_common_builder(builder).set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

pub(crate) async fn connect_sub_socket(endpoint: &str, topic: Option<&str>) -> Result<SubSocket> {
    let ctx = Context::new();
    let socket = configure_receive_builder(subscribe(&ctx))
        .set_ipv6(ipv6_option_for(endpoint)?)
        .connect(endpoint)?
        .subscribe(topic.unwrap_or("").as_bytes())?;
    Ok(socket)
}

#[cfg_attr(not(feature = "block-manager"), allow(dead_code))]
pub(crate) async fn bind_pub_socket(endpoint: &str) -> Result<SharedPubSocket> {
    let ctx = Context::new();
    let socket = configure_send_builder(publish(&ctx))
        .set_ipv6(ipv6_option_for(endpoint)?)
        .bind(endpoint)?;
    Ok(Arc::new(Mutex::new(socket)))
}

pub(crate) async fn bind_pull_socket(endpoint: &str) -> Result<PullSocket> {
    let ctx = Context::new();
    let socket = configure_receive_builder(pull(&ctx))
        .set_ipv6(ipv6_option_for(endpoint)?)
        .bind(endpoint)?;
    Ok(socket)
}

#[cfg(test)]
pub(crate) async fn connect_push_socket(endpoint: &str) -> Result<tmq::push::Push> {
    let ctx = Context::new();
    let socket = configure_send_builder(tmq::push::push(&ctx))
        .set_ipv6(ipv6_option_for(endpoint)?)
        .connect(endpoint)?;
    Ok(socket)
}

pub(crate) fn multipart_message(multipart: Multipart) -> MultipartMessage {
    multipart.into_iter().map(|frame| frame.to_vec()).collect()
}

#[cfg_attr(not(feature = "block-manager"), allow(dead_code))]
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

#[cfg(test)]
pub(crate) async fn send_multipart_direct<S>(socket: &mut S, frames: MultipartMessage) -> Result<()>
where
    S: futures::Sink<Multipart, Error = tmq::TmqError> + Unpin,
{
    socket.send(Multipart::from(frames)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tmq::AsZmqSocket;

    #[tokio::test]
    async fn bracketed_ipv6_endpoints_connect() {
        if let Err(error) = std::net::TcpListener::bind("[::1]:0") {
            eprintln!("Skipping IPv6 ZMQ test: {error}");
            return;
        }
        let mut pull = bind_pull_socket("tcp://[::1]:*").await.unwrap();
        let endpoint = pull.get_socket().get_last_endpoint().unwrap().unwrap();
        let mut push = connect_push_socket(&endpoint).await.unwrap();

        send_multipart_direct(&mut push, vec![b"ipv6".to_vec()])
            .await
            .unwrap();
        let received = tokio::time::timeout(std::time::Duration::from_secs(5), pull.next())
            .await
            .expect("IPv6 PULL socket should receive the message")
            .unwrap()
            .unwrap();
        assert_eq!(multipart_message(received), vec![b"ipv6".to_vec()]);
    }

    #[tokio::test]
    async fn unbracketed_ipv6_endpoints_are_rejected() {
        let error = connect_sub_socket("tcp://::1:5555", None)
            .await
            .err()
            .expect("unbracketed IPv6 endpoint should be rejected");
        assert!(
            error
                .to_string()
                .contains("IPv6 addresses must be bracketed"),
            "{error}"
        );
    }
}
