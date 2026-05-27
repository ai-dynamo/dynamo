// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Transport Module
//!
//! Brief overview of the request-response transport:
//!
//! The request plane (TCP, NATS, etc.) carries a two-part message whose header is a
//! `RequestControlMessage` — embedding the [`ConnectionInfo`] that tells the worker where to call home
//! — and whose data half is the serialized request body if request streaming is not needed.
//! All subsequent streaming bytes (responses, and request-stream) flow over the TCP socket established afterwards.
//!
//! For simplicity, if request streaming is needed, the `RequestControlMessage` should not contain
//! the request body. Instead, all requests of the stream should be sent over the TCP socket.
//!
//! The TCP transport is the implementation that produces and consumes [`ConnectionInfo`] and
//! carries the streaming response (and, optionally, request-stream) bytes between two peers
//! on separate sockets from the initial request.
//!
//! # Roles
//!
//! The TCP transport has two sides:
//!
//! - Request sender: The upstream that **initiates the transfer** runs [`server::TcpStreamServer`],
//!   registers what it expects to receive, and listens. It publishes its address + a per-stream
//!   subject UUID via [`TcpStreamConnectionInfo`], which is serialized into a [`ConnectionInfo`].
//! - Request receiver: The downstream that **acknowledges the transfer** runs [`client::TcpClient`],
//!   reads the connection info out of the request, dials the listener, and identifies itself with
//!   a `CallHomeHandshake` to the request sender.
//!
//! Although TCP is bidirectional, we keep separate sockets for the request stream and the response
//! stream to match Dynamo's design principles. To establish both, the request receiver must receive
//! two [`ConnectionInfo`] objects and run two handshakes — each [`StreamType`] is its own TCP
//! connection with its own subject UUID.
//!
//! # Server-Client Interaction
//!
//! See the test cases below for detailed examples. Note that the response stream expects the client
//! to send a [`ResponseStreamPrologue`] in order to properly establish the stream.
//!
//! # Stream Types
//!
//! [`StreamType::Response`] — worker pushes engine output back to the upstream. Server side is
//! `process_response_stream` (delivers a [`StreamReceiver`] to the awaiting registrant once the
//! client has sent its [`ResponseStreamPrologue`]). Client side is
//! [`client::TcpClient::create_response_stream`] (returns a [`StreamSender`]; spawns reader/writer
//! tasks plus a connection monitor that waits for the server's FIN).
//!
//! [`StreamType::Request`] — upstream pushes the request body (or a stream of follow-up frames)
//! into a downstream worker. Server side is `process_request_stream` (delivers a [`StreamSender`]
//! immediately; there is no prologue today). Client side is
//! [`client::TcpClient::create_request_stream`] (returns a [`StreamReceiver`]; spawns a single
//! task that handles both directions on the socket).
//!
//! # Registration and Lifecycle
//!
//! [`ResponseService::register`] takes [`StreamOptions`] with `enable_request_stream` /
//! `enable_response_stream` flags and returns [`PendingConnections`] holding zero, one, or two
//! [`RegisteredStream`]s. Each [`RegisteredStream`] carries a [`ConnectionInfo`] and a oneshot
//! that resolves to the [`StreamSender`] / [`StreamReceiver`] once the downstream dials in and the
//! handshake completes. Once registered, the pending entry remains until the downstream successfully
//! establishes the stream. Two mechanisms ensure the pending entry is removed when the downstream
//! cannot be reached:
//!
//! 1. The returned [`RegisteredStream`] is RAII — dropping it without `into_parts()` removes the
//!    pending entry from the server's subject tables. This is typically used by the request sender
//!    up until the `RequestControlMessage` is sent and the stream is established.
//! 2. The server tracks `subject UUID → oneshot` in `tx_subjects` / `rx_subjects`.
//!    [`server::TcpStreamServer::associate_instance`] links one or both
//!    subjects to a discovery instance so [`server::TcpStreamServer::cancel_instance_streams`] can
//!    drop both halves' oneshots together when a worker disappears. Tombstones (`TOMBSTONE_TTL`)
//!    are the safety net that closes the cancel-vs-register race.
//!
//! # CallHome Handshake
//!
//! The first message a [`client::TcpClient`] sends on a freshly-opened socket is a
//! `CallHomeHandshake` header-only frame carrying `{ subject, stream_type }`. The server pops
//! the matching entry out of `tx_subjects` (for [`StreamType::Request`]) or `rx_subjects` (for
//! [`StreamType::Response`]) and resolves the registrant's oneshot. After that the socket carries
//! framed [`TwoPartCodec`] messages: data frames in the natural direction, control frames
//! (and, for response streams, the prologue) interleaved.
//!
//! # Control / Shutdown Protocol
//!
//! [`ControlMessage`] frames are header-only frames interleaved with data on either socket:
//!
//! - [`ControlMessage::Sentinel`] — per-direction clean end-of-stream; the producing side emits
//!   it before closing the socket. Used on both the request and response sockets.
//! - [`ControlMessage::Stop`] — sender asks the receiver to cancel; the receiving side calls
//!   `context.stop()`.
//! - [`ControlMessage::Kill`] — hard cancel; `context.kill()` and break out.
//!
//! `Stop` and `Kill` only flow from upstream to downstream (i.e. frontend → worker). This is an
//! expected asymmetry: the upstream can cancel the downstream operation for various reasons,
//! but the downstream cannot cancel the upstream operation. A downstream that cannot consume
//! the stream simply drops its socket, which the upstream surfaces as a write error and
//! interprets as a hint for recovery or failure propagation.
//!
//! The cancellation direction is fixed, but the two streams carry **data** in opposite
//! directions, so the practical handling differs per stream:
//!
//! ## Response stream (downstream → upstream) — bidirectional
//!
//! - Upstream writes: `Stop` / `Kill` (any time, to cancel).
//! - Downstream writes: data frames, then `Sentinel` on clean close (skipped on kill/stop).
//!
//! ## Request stream (upstream → downstream) — unidirectional after the handshake
//!
//! - Upstream writes: data frames, then exactly one closing frame — `Sentinel` (clean drain)
//!   / `Stop` (`context.stopped()`) / `Kill` (`context.killed()`).
//! - Downstream writes: nothing. Its TCP write half is closed right after the CallHome handshake.

pub mod client;
pub mod server;

pub mod test_utils;

use super::ControlMessage;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::{
    ConnectionInfo, PendingConnections, RegisteredStream, ResponseService, ResponseStreamPrologue,
    StreamOptions, StreamReceiver, StreamSender, StreamType, codec::TwoPartCodec,
};

const TCP_TRANSPORT: &str = "tcp_server";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpStreamConnectionInfo {
    pub address: String,
    pub subject: String,
    pub context: String,
    pub stream_type: StreamType,
}

impl From<TcpStreamConnectionInfo> for ConnectionInfo {
    fn from(info: TcpStreamConnectionInfo) -> Self {
        // Need to consider the below. If failure should be fatal, keep the below with .expect()
        // But if there is a default value, we can use:
        // unwrap_or_else(|e| {
        //     eprintln!("Failed to serialize TcpStreamConnectionInfo: {:?}", e);
        //     "{}".to_string() // Provide a fallback empty JSON string or default value
        ConnectionInfo {
            transport: TCP_TRANSPORT.to_string(),
            info: serde_json::to_string(&info)
                .expect("Failed to serialize TcpStreamConnectionInfo"),
        }
    }
}

impl TryFrom<ConnectionInfo> for TcpStreamConnectionInfo {
    type Error = anyhow::Error;

    fn try_from(info: ConnectionInfo) -> Result<Self, Self::Error> {
        if info.transport != TCP_TRANSPORT {
            return Err(anyhow::anyhow!(
                "Invalid transport; TcpClient requires the transport to be `tcp_server`; however {} was passed",
                info.transport
            ));
        }

        serde_json::from_str(&info.info)
            .map_err(|e| anyhow::anyhow!("Failed parse ConnectionInfo: {:?}", e))
    }
}

/// First message sent over a CallHome stream which will map the newly created socket to a specific
/// response data stream which was registered with the same subject.
///
/// This is a transport specific message as part of forming/completing a CallHome TcpStream.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallHomeHandshake {
    subject: String,
    stream_type: StreamType,
}

#[cfg(test)]
mod tests {
    use crate::engine::AsyncEngineContextProvider;

    use super::*;
    use crate::pipeline::Context;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestMessage {
        foo: String,
    }

    /// Round-trip a request-stream connection: register on the server with
    /// `enable_request_stream(true)`, dial in via `TcpClient::create_request_stream`,
    /// send a frame from the upstream `StreamSender`, and assert it arrives on the
    /// downstream `StreamReceiver` returned by the client.
    #[tokio::test]
    async fn test_tcp_stream_request_stream_client_server() {
        // [server] start the server and register the request stream
        let options = server::ServerOptions::default();
        let server = server::TcpStreamServer::new(options).await.unwrap();

        let context_upstream = Context::new(());

        let options = StreamOptions::builder()
            .context(context_upstream.context())
            .enable_request_stream(true)
            .enable_response_stream(false)
            .build()
            .unwrap();

        let pending_connection = server.register(options).await;

        let connection_info = pending_connection
            .send_stream
            .as_ref()
            .unwrap()
            .connection_info
            .clone();

        // [client] Assume to receive the connection info from the server via the request plane,
        // create the request stream and dial in to the server
        let context_downstream = Context::with_id((), context_upstream.id().to_string());

        let mut recv_stream = client::TcpClient::create_request_stream(
            context_downstream.context(),
            connection_info,
            None,
        )
        .await
        .unwrap();

        // [server] After client dials in, the server can pick up its `StreamSender` half.
        let (_conn_info, stream_provider) = pending_connection.send_stream.unwrap().into_parts();
        let send_stream = stream_provider.await.unwrap().unwrap();

        let msg = TestMessage {
            foo: "request-frame".to_string(),
        };
        let payload = serde_json::to_vec(&msg).unwrap();

        send_stream.send(payload.into()).await.unwrap();

        // [client] The client can now receive the response from the server
        let data = recv_stream.rx.recv().await.unwrap();
        let recv_msg = serde_json::from_slice::<TestMessage>(&data).unwrap();
        assert_eq!(msg.foo, recv_msg.foo);

        // Dropping the upstream `StreamSender` should cleanly close the request
        // stream — the downstream receiver should observe `None`.
        drop(send_stream);
        assert!(recv_stream.rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_tcp_stream_client_server() {
        // [server] start the server and register the response stream
        let options = server::ServerOptions::builder().port(9124).build().unwrap();
        let server = server::TcpStreamServer::new(options).await.unwrap();

        let context_rank0 = Context::new(());

        let options = StreamOptions::builder()
            .context(context_rank0.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending_connection = server.register(options).await;

        let connection_info = pending_connection
            .recv_stream
            .as_ref()
            .unwrap()
            .connection_info
            .clone();

        // [client] set up the other rank and create the response stream
        let context_rank1 = Context::with_id((), context_rank0.id().to_string());

        let mut send_stream = client::TcpClient::create_response_stream(
            context_rank1.context(),
            connection_info,
            None,
        )
        .await
        .unwrap();

        // the client can now setup it's end of the stream and if it errors, it can send a message
        // to the server to stop the stream
        //
        // this step must be done before the next step on the server can complete, i.e.
        // the server's stream is now blocked on receiving the prologue message
        //
        // let's improve this and use an enum like Ok/Err; currently, None means good-to-go, and
        // Some(String) means an error happened on this downstream node and we need to alert the
        // upstream node that an error occurred
        send_stream.send_prologue(None).await.unwrap();

        // [server] After client sends the prologue, the server can pick up its `StreamReceiver` half.
        let (_conn_info, stream_provider) = pending_connection.recv_stream.unwrap().into_parts();
        let recv_stream = stream_provider.await.unwrap();

        // [client] The client can now send the response message to the server
        let msg = TestMessage {
            foo: "bar".to_string(),
        };

        let payload = serde_json::to_vec(&msg).unwrap();

        send_stream.send(payload.into()).await.unwrap();

        // [server] The server can now receive the response message from the client

        let data = recv_stream.unwrap().rx.recv().await.unwrap();

        let recv_msg = serde_json::from_slice::<TestMessage>(&data).unwrap();

        assert_eq!(msg.foo, recv_msg.foo);

        drop(send_stream);

        // let data = recv_stream.rx.recv().await;

        // assert!(data.is_none());
    }
}
