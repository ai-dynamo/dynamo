// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC server implementation using tonic with protobuf
//!
//! This module implements a gRPC server using generated code from nova.proto.
//! Messages are wrapped in protobuf but contain our custom TCP-framed data.

use std::net::SocketAddr;
use std::pin::Pin;

use futures::Stream;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};

use crate::MessageType;
use crate::tcp::TcpFrameCodec;
use crate::transport::TransportAdapter;

// Include generated proto code
pub mod proto {
    tonic::include_proto!("nova.streaming.v1");
}

use proto::{
    FramedData,
    nova_streaming_server::{NovaStreaming, NovaStreamingServer},
};

/// gRPC server implementation
pub struct GrpcServer {
    bind_addr: SocketAddr,
    service: NovaStreamingImpl,
    cancel_token: CancellationToken,
}

impl GrpcServer {
    /// Create a new gRPC server with a pre-bound listener
    pub fn with_listener(
        listener: TcpListener,
        channels: TransportAdapter,
        cancel_token: CancellationToken,
    ) -> Self {
        // Get the bind address from the listener
        let bind_addr = listener
            .local_addr()
            .unwrap_or_else(|_| "0.0.0.0:0".parse().unwrap());

        // tonic::transport::Server will bind its own listener, so drop this one
        drop(listener);

        let service = NovaStreamingImpl { channels };

        Self {
            bind_addr,
            service,
            cancel_token,
        }
    }

    /// Run the gRPC server
    pub async fn run(self) -> anyhow::Result<()> {
        info!("Starting gRPC server on {}", self.bind_addr);

        // Create the tonic server with our service
        // Note: add_service() returns a Router in tonic 0.13
        let router = Server::builder().add_service(NovaStreamingServer::new(self.service));

        router
            .serve_with_shutdown(self.bind_addr, self.cancel_token.cancelled())
            .await?;

        info!("gRPC server shut down");
        Ok(())
    }
}

/// Implementation of the NovaStreaming service
#[derive(Clone)]
struct NovaStreamingImpl {
    channels: TransportAdapter,
}

#[tonic::async_trait]
impl NovaStreaming for NovaStreamingImpl {
    type StreamStream = Pin<Box<dyn Stream<Item = Result<FramedData, Status>> + Send>>;

    async fn stream(
        &self,
        request: Request<Streaming<FramedData>>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let mut in_stream = request.into_inner();
        let channels = self.channels.clone();

        debug!("New gRPC streaming connection established");

        // Spawn task to handle incoming messages
        tokio::spawn(async move {
            while let Some(result) = futures::StreamExt::next(&mut in_stream).await {
                match result {
                    Ok(framed_data) => {
                        // Extract the 3 separate blobs from protobuf
                        let preamble = framed_data.preamble;
                        let header = framed_data.header;
                        let payload = framed_data.payload;

                        // Parse message type from preamble
                        match TcpFrameCodec::parse_message_type_from_preamble(&preamble) {
                            Ok(message_type) => {
                                debug!(
                                    "Received message: type={:?}, header_len={}, payload_len={}",
                                    message_type,
                                    header.len(),
                                    payload.len()
                                );

                                // Route to appropriate channel
                                let channel = match message_type {
                                    MessageType::Message => &channels.message_stream,
                                    MessageType::Response => &channels.response_stream,
                                    MessageType::Ack | MessageType::Event => &channels.event_stream,
                                };

                                // Convert Vec<u8> to Bytes
                                let header = bytes::Bytes::from(header);
                                let payload = bytes::Bytes::from(payload);

                                if let Err(e) = channel.try_send((header, payload)) {
                                    error!("Failed to route message to channel: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Failed to parse message type from preamble: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Stream error: {}", e);
                        break;
                    }
                }
            }
            debug!("gRPC streaming connection closed");
        });

        // Return empty response stream (we don't use server->client direction)
        let output_stream = futures::stream::empty();
        Ok(Response::new(Box::pin(output_stream)))
    }
}
