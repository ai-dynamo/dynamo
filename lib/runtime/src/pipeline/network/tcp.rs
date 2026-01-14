// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP Transport Module
//!
//! The TCP Transport module consists of two main components: Client and Server. The Client is
//! the downstream node that is responsible for connecting back to the upstream node (Server).
//!
//! Both Client and Server are given a Stream object that they can specialize for their specific
//! needs, i.e. if they are SingleIn/ManyIn or SingleOut/ManyOut.
//!
//! The Request object will carry the Transport Type and Connection details, i.e. how the receiver
//! of a Request is able to communicate back to the source of the Request.
//!
//! There are two types of TcpStream:
//! - CallHome stream - the address for the listening socket is forward via some mechanism which then
//!   connects back to the source of the CallHome stream. To match the socket with an awaiting data
//!   stream, the CallHomeHandshake is used.

pub mod client;
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

    #[tokio::test]
    async fn test_tcp_stream_client_server() {
        println!("Test Started");
        let options = server::ServerOptions::builder().port(9124).build().unwrap();
        println!("Test Started");
        let server = server::TcpStreamServer::new(options).await.unwrap();
        println!("Server created");

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

        // set up the other rank
        let context_rank1 = Context::with_id((), context_rank0.id().to_string());

        // connect to the server socket
        let mut send_stream =
            client::TcpClient::create_response_stream(context_rank1.context(), connection_info)
                .await
                .unwrap();
        println!("Client connected");

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

        // [server] next - now pending connections should be connected
        let recv_stream = pending_connection
            .recv_stream
            .unwrap()
            .stream_provider
            .await
            .unwrap();

        println!("Server paired");

        let msg = TestMessage {
            foo: "bar".to_string(),
        };

        let payload = serde_json::to_vec(&msg).unwrap();

        send_stream.send(payload.into()).await.unwrap();

        println!("Client sent message");

        let data = recv_stream.unwrap().rx.recv().await.unwrap();

        println!("Server received message");

        let recv_msg = serde_json::from_slice::<TestMessage>(&data).unwrap();

        assert_eq!(msg.foo, recv_msg.foo);
        println!("message match");

        drop(send_stream);

        // let data = recv_stream.rx.recv().await;

        // assert!(data.is_none());
    }
}
