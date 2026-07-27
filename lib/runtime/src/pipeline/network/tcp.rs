// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Transport Module
//!
//! TODO: this design-and-implementation overview should eventually move into
//! the architecture docs (`docs/design-docs/architecture.md`); kept here for
//! now until there's a home for transport-level design notes.
//!
//! Brief overview of the request-response transport:
//!
//! The request plane (TCP, NATS, etc.) carries a two-part message whose header is a
//! `RequestControlMessage` — embedding the [`ConnectionInfo`] that tells the worker where to call home
//! — and whose data half is the serialized request body if request streaming is not needed.
//! Subsequent request-stream bytes use a dedicated TCP socket. Responses use
//! persistent multiplexed TCP connections shared by logical response streams.
//!
//! For simplicity, if request streaming is needed, the `RequestControlMessage` should not contain
//! the request body. Instead, all requests of the stream should be sent over the TCP socket.
//!
//! The TCP transport is the implementation that produces and consumes [`ConnectionInfo`] and
//! carries response and optional request-stream bytes between two peers on
//! sockets separate from the initial request.
//!
//! # Roles
//!
//! The TCP transport has two sides:
//!
//! - Request sender: The upstream that **initiates the transfer** runs [`server::TcpStreamServer`],
//!   registers what it expects to receive, and listens. It publishes dedicated
//!   request-stream information or versioned response-mux information through [`ConnectionInfo`].
//! - Request receiver: The downstream that **acknowledges the transfer** runs [`client::TcpClient`],
//!   reads the connection info out of the request, dials the listener, and identifies itself with
//!   a `CallHomeHandshake` to the request sender.
//!
//! Request streams retain one dedicated socket per logical stream. Response
//! streams are mandatory mux-v1 streams: each worker/frontend pair warms four
//! persistent connections and identifies logical responses by UUID.
//!
//! # Server-Client Interaction
//!
//! See the test cases below for detailed examples. A response `Prologue` activates
//! the logical stream before any `Data` frames are sent.
//!
//! # Stream Types
//!
//! [`StreamType::Response`] — worker pushes engine output through the response
//! mux pool. A `Prologue` activates the pending stream, `Data` carries output,
//! and `End` completes only that logical stream.
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
//! 2. The server tracks dedicated request subjects and pending response-mux UUIDs.
//!    [`server::TcpStreamServer::associate_instance`] links one or both
//!    subjects to a discovery instance so [`server::TcpStreamServer::cancel_instance_streams`] can
//!    drop both halves' oneshots together when a worker disappears. Tombstones (`TOMBSTONE_TTL`)
//!    are the safety net that closes the cancel-vs-register race.
//!
//! # Handshakes
//!
//! A dedicated request socket starts with a `CallHomeHandshake` carrying its
//! subject and [`StreamType::Request`]. A response-mux connection instead uses
//! [`mux::MuxCodec`] from the first byte: `ConnectionHello` carries the protocol
//! version and frontend UUID, and the frontend returns an empty
//! `ConnectionReady`. Incompatible response versions are rejected without a
//! legacy fallback.
//!
//! # Control / Shutdown Protocol
//!
//! Dedicated request sockets use header-only [`ControlMessage`] frames:
//!
//! - [`ControlMessage::Sentinel`] — per-direction clean end-of-stream; the producing side emits
//!   it before closing the request socket.
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
//! ## Response mux (downstream → upstream) — bidirectional
//!
//! - Upstream writes: mux `Stop`, `Kill`, `WindowUpdate`, and `Reset` frames.
//! - Downstream writes: mux `Prologue`, `Data`, `End`, and `Reset` frames.
//!
//! ## Request stream (upstream → downstream) — unidirectional after the handshake
//!
//! - Upstream writes: data frames, then exactly one closing frame — `Sentinel` (clean drain)
//!   / `Stop` (`context.stopped()`) / `Kill` (`context.killed()`).
//! - Downstream writes: nothing. Its TCP write half is closed right after the CallHome handshake.

pub mod client;
pub mod mux;
pub mod server;

pub mod test_utils;

use super::ControlMessage;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::{
    ConnectionInfo, PendingConnections, RegisteredStream, ResponseService, StreamOptions,
    StreamReceiver, StreamSender, StreamType, codec::TwoPartCodec,
};

const TCP_TRANSPORT: &str = "tcp_server";
pub const TCP_RESPONSE_MUX_TRANSPORT: &str = "tcp_response_mux_v1";

/// Read Linux's kernel-maintained count of data-bearing TCP segments for one
/// socket. This is used only by opt-in benchmark diagnostics; keeping the
/// query here avoids coupling the transport to packet-capture privileges.
#[cfg(target_os = "linux")]
pub(crate) fn tcp_data_segments_out(stream: &tokio::net::TcpStream) -> Option<u64> {
    use std::os::fd::AsRawFd;

    tcp_data_segments_out_fd(stream.as_raw_fd())
}

#[cfg(target_os = "linux")]
pub(crate) fn tcp_data_segments_out_fd(fd: std::os::fd::RawFd) -> Option<u64> {
    #[repr(C)]
    // Layout through tcpi_data_segs_out from Linux uapi/linux/tcp.h::tcp_info.
    struct LinuxTcpInfoThroughDataSegments {
        _header: [u8; 8],
        _metrics: [u32; 24],
        _rates_and_bytes: [u64; 4],
        _segments_out: u32,
        _segments_in: u32,
        _notsent_bytes: u32,
        _min_rtt: u32,
        _data_segments_in: u32,
        data_segments_out: u32,
    }

    let mut info = std::mem::MaybeUninit::<LinuxTcpInfoThroughDataSegments>::zeroed();
    let mut len = std::mem::size_of::<LinuxTcpInfoThroughDataSegments>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_INFO,
            info.as_mut_ptr().cast(),
            &mut len,
        )
    };
    if result != 0 || len < std::mem::size_of::<LinuxTcpInfoThroughDataSegments>() as _ {
        return None;
    }
    Some(unsafe { info.assume_init() }.data_segments_out as u64)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn tcp_data_segments_out(_stream: &tokio::net::TcpStream) -> Option<u64> {
    None
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn tcp_data_segments_out_fd(_fd: std::os::fd::RawFd) -> Option<u64> {
    None
}

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

/// Connection information for one logical response stream carried by the
/// frontend's persistent response-mux listener.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseMuxConnectionInfo {
    pub address: String,
    pub frontend_server_id: uuid::Uuid,
    pub stream_id: uuid::Uuid,
    pub context: String,
}

impl From<ResponseMuxConnectionInfo> for ConnectionInfo {
    fn from(info: ResponseMuxConnectionInfo) -> Self {
        Self {
            transport: TCP_RESPONSE_MUX_TRANSPORT.to_string(),
            info: serde_json::to_string(&info)
                .expect("Failed to serialize ResponseMuxConnectionInfo"),
        }
    }
}

impl TryFrom<ConnectionInfo> for ResponseMuxConnectionInfo {
    type Error = anyhow::Error;

    fn try_from(info: ConnectionInfo) -> Result<Self, Self::Error> {
        if info.transport != TCP_RESPONSE_MUX_TRANSPORT {
            return Err(anyhow::anyhow!(
                "Invalid transport; response mux requires `{TCP_RESPONSE_MUX_TRANSPORT}`; got {}",
                info.transport
            ));
        }
        serde_json::from_str(&info.info)
            .map_err(|e| anyhow::anyhow!("Failed to parse response mux connection info: {e}"))
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
        let context_downstream = Context::with_id_and_metadata(
            (),
            context_upstream.id().to_string(),
            Default::default(),
        );

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
        let data = recv_stream.recv().await.unwrap();
        let recv_msg = serde_json::from_slice::<TestMessage>(&data).unwrap();
        assert_eq!(msg.foo, recv_msg.foo);

        // Dropping the upstream `StreamSender` should cleanly close the request
        // stream — the downstream receiver should observe `None`.
        drop(send_stream);
        assert!(recv_stream.recv().await.is_none());
    }
}
