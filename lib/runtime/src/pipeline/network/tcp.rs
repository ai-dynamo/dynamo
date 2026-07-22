// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TCP callback transport for streaming frontend-to-worker requests.
//!
//! Worker responses use the fixed-lane QUIC transport. TCP remains here only
//! for the optional bidirectional request callback: the frontend registers a
//! subject, the worker dials the advertised listener, and the frontend pushes
//! request frames down that socket.

pub mod client;
pub mod server;
pub mod test_utils;

use serde::{Deserialize, Serialize};

use super::{ConnectionInfo, StreamType};

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
        ConnectionInfo {
            transport: TCP_TRANSPORT.to_string(),
            info: serde_json::to_string(&info).expect("TcpStreamConnectionInfo must serialize"),
        }
    }
}

impl TryFrom<ConnectionInfo> for TcpStreamConnectionInfo {
    type Error = anyhow::Error;

    fn try_from(info: ConnectionInfo) -> Result<Self, Self::Error> {
        if info.transport != TCP_TRANSPORT {
            anyhow::bail!(
                "invalid transport for TCP request callback: {}",
                info.transport
            );
        }
        Ok(serde_json::from_str(&info.info)?)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CallHomeHandshake {
    subject: String,
    stream_type: StreamType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::AsyncEngineContextProvider,
        pipeline::{Context, network::DEFAULT_SEND_BUFFER_COUNT},
    };
    use bytes::Bytes;

    #[tokio::test]
    async fn bidirectional_request_callback_round_trip() {
        let server = server::TcpStreamServer::new(server::ServerOptions::default())
            .await
            .unwrap();
        let upstream = Context::new(());
        let registered = server.register_request(upstream.context(), DEFAULT_SEND_BUFFER_COUNT);
        let connection_info = registered.connection_info.clone();
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut receiver =
            client::TcpClient::create_request_stream(downstream.context(), connection_info, None)
                .await
                .unwrap();
        let (_, provider) = registered.into_parts();
        let sender = provider.await.unwrap().unwrap();
        sender
            .send(Bytes::from_static(b"request-frame"))
            .await
            .unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"request-frame")
        );
        drop(sender);
        assert!(receiver.rx.recv().await.is_none());
    }
}
