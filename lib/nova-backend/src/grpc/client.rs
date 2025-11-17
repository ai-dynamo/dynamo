// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC client implementation using tonic with protobuf
//!
//! This module handles client connections using generated gRPC code.

use bytes::Bytes;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tracing::{debug, error};

use super::server::proto::{FramedData, nova_streaming_client::NovaStreamingClient};
use crate::MessageType;
use crate::tcp::TcpFrameCodec;

/// Handle to a gRPC connection
#[derive(Clone)]
pub struct ConnectionHandle {
    tx: mpsc::Sender<FramedData>,
}

impl ConnectionHandle {
    /// Try to send a framed message (non-blocking)
    pub fn try_send(
        &self,
        msg_type: MessageType,
        header: Bytes,
        payload: Bytes,
    ) -> Result<(), mpsc::error::SendError<(MessageType, Bytes, Bytes)>> {
        // Build preamble
        let preamble = match TcpFrameCodec::build_preamble(
            msg_type,
            header.len() as u32,
            payload.len() as u32,
        ) {
            Ok(p) => p,
            Err(_e) => {
                return Err(mpsc::error::SendError((msg_type, header, payload)));
            }
        };

        // Construct FramedData with 3 separate fields
        let framed_data = FramedData {
            preamble: preamble.to_vec(),
            header: header.to_vec(),
            payload: payload.to_vec(),
        };

        self.tx
            .try_send(framed_data)
            .map_err(|_| mpsc::error::SendError((msg_type, header, payload)))
    }

    /// Send a framed message (async, waits if channel is full)
    pub async fn send(
        &self,
        msg_type: MessageType,
        header: Bytes,
        payload: Bytes,
    ) -> Result<(), mpsc::error::SendError<(MessageType, Bytes, Bytes)>> {
        // Build preamble
        let preamble = match TcpFrameCodec::build_preamble(
            msg_type,
            header.len() as u32,
            payload.len() as u32,
        ) {
            Ok(p) => p,
            Err(_e) => {
                return Err(mpsc::error::SendError((msg_type, header, payload)));
            }
        };

        // Construct FramedData with 3 separate fields
        let framed_data = FramedData {
            preamble: preamble.to_vec(),
            header: header.to_vec(),
            payload: payload.to_vec(),
        };

        self.tx
            .send(framed_data)
            .await
            .map_err(|_| mpsc::error::SendError((msg_type, header, payload)))
    }
}

/// Establish a gRPC connection to a peer
pub async fn establish_connection(
    url: String,
    instance_id: dynamo_identity::InstanceId,
    cancel_token: CancellationToken,
) -> anyhow::Result<ConnectionHandle> {
    // Parse URL to get endpoint
    // Format: "grpc://host:port"
    let url = url
        .trim_start_matches("grpc://")
        .trim_start_matches("http://");
    let endpoint_url = format!("http://{}", url);

    debug!("Connecting to {} at {}", instance_id, endpoint_url);

    // Create tonic channel
    let channel = Channel::from_shared(endpoint_url)?
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(5))
        .connect()
        .await?;

    debug!("Connected to {} at {}", instance_id, url);

    // Create gRPC client
    let mut client = NovaStreamingClient::new(channel);

    // Create channel for sending frames
    let (tx, rx) = mpsc::channel::<FramedData>(256);

    // Convert receiver to stream
    let request_stream = ReceiverStream::new(rx);

    // Start the bidirectional stream
    let response_stream = client.stream(request_stream).await?.into_inner();

    // Spawn task to handle responses (though we don't expect any)
    tokio::spawn(async move {
        let mut stream = response_stream;
        while let Some(result) = futures::StreamExt::next(&mut stream).await {
            match result {
                Ok(_frame) => {
                    // Unexpected response from server
                    debug!("Received unexpected response from server");
                }
                Err(e) => {
                    if !cancel_token.is_cancelled() {
                        error!("Stream error: {}", e);
                    }
                    break;
                }
            }
        }
        debug!("Response stream closed");
    });

    Ok(ConnectionHandle { tx })
}
