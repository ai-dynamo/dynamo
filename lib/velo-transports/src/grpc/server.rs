// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC server implementation for the VeloStreaming service.
//!
//! This module implements the tonic server-side handler for bidirectional
//! streaming. Inbound `FramedData` messages are parsed using `TcpFrameCodec`
//! and routed to the appropriate adapter channel based on message type.

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, warn};

use crate::MessageType;
use crate::tcp::TcpFrameCodec;
use crate::transport::{ShutdownState, TransportAdapter};

use super::proto;
use super::proto::velo_streaming_server::VeloStreaming;

/// Tonic service implementation for bidirectional gRPC streaming.
///
/// Each inbound `Stream` RPC opens a bidirectional channel. Inbound frames
/// are decoded, routed through the [`TransportAdapter`], and drain-aware
/// rejection is handled by sending `ShuttingDown` frames back on the
/// response stream.
pub struct VeloStreamingService {
    adapter: TransportAdapter,
    shutdown_state: ShutdownState,
}

impl VeloStreamingService {
    /// Create a new service instance with the given adapter and shutdown state.
    pub fn new(adapter: TransportAdapter, shutdown_state: ShutdownState) -> Self {
        Self {
            adapter,
            shutdown_state,
        }
    }
}

#[tonic::async_trait]
impl VeloStreaming for VeloStreamingService {
    type StreamStream = ReceiverStream<Result<proto::FramedData, Status>>;

    async fn stream(
        &self,
        request: Request<Streaming<proto::FramedData>>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let mut inbound = request.into_inner();
        let adapter = self.adapter.clone();
        let shutdown_state = self.shutdown_state.clone();

        // Response channel for sending frames back to the client (e.g. ShuttingDown).
        let (response_tx, response_rx) = mpsc::channel::<Result<proto::FramedData, Status>>(256);

        tokio::spawn(async move {
            while let Ok(Some(framed_data)) = inbound.message().await {
                let msg_type =
                    match TcpFrameCodec::parse_message_type_from_preamble(&framed_data.preamble) {
                        Ok(mt) => mt,
                        Err(e) => {
                            warn!("gRPC server: invalid preamble: {}", e);
                            continue;
                        }
                    };

                // During drain: reject new Message frames with ShuttingDown,
                // but always pass through Response/Ack/Event frames.
                if shutdown_state.is_draining() && msg_type == MessageType::Message {
                    debug!("gRPC server: rejecting Message during drain (sending ShuttingDown)");
                    let preamble = match TcpFrameCodec::build_preamble(
                        MessageType::ShuttingDown,
                        framed_data.header.len() as u32,
                        0,
                    ) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("gRPC server: failed to build ShuttingDown preamble: {}", e);
                            continue;
                        }
                    };
                    let reject = proto::FramedData {
                        preamble: preamble.to_vec(),
                        header: framed_data.header,
                        payload: Vec::new(),
                    };
                    if response_tx.send(Ok(reject)).await.is_err() {
                        break;
                    }
                    continue;
                }

                // Route to the appropriate adapter channel.
                let sender = match msg_type {
                    MessageType::Message => &adapter.message_stream,
                    MessageType::Response => &adapter.response_stream,
                    MessageType::Ack | MessageType::Event => &adapter.event_stream,
                    MessageType::ShuttingDown => &adapter.response_stream,
                };

                if let Err(e) = sender
                    .send_async((
                        Bytes::from(framed_data.header),
                        Bytes::from(framed_data.payload),
                    ))
                    .await
                {
                    warn!("gRPC server: failed to route {:?} frame: {}", msg_type, e);
                    break;
                }
            }

            debug!("gRPC server: inbound stream ended");
        });

        Ok(Response::new(ReceiverStream::new(response_rx)))
    }
}
