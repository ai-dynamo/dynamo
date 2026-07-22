// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use futures::{SinkExt, StreamExt};
use prometheus::IntCounter;
use tokio::net::TcpStream;
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{CallHomeHandshake, TcpStreamConnectionInfo};
use crate::{
    engine::AsyncEngineContext,
    pipeline::network::{
        ConnectionInfo, ControlMessage, StreamReceiver, StreamType, TwoPartCodec,
        codec::{TwoPartMessage, TwoPartMessageType},
    },
};

pub struct TcpClient;

impl TcpClient {
    async fn connect(address: &str) -> std::io::Result<TcpStream> {
        let backoff = std::time::Duration::from_millis(200);
        loop {
            match TcpStream::connect(address).await {
                Ok(socket) => {
                    socket.set_nodelay(true)?;
                    return Ok(socket);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AddrNotAvailable => {
                    tracing::warn!(%error, "TCP request callback connect retry");
                    tokio::time::sleep(backoff).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub async fn create_request_stream(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation_counter: Option<IntCounter>,
    ) -> Result<StreamReceiver> {
        let info = TcpStreamConnectionInfo::try_from(info)
            .context("tcp request callback connection info")?;
        if info.stream_type != StreamType::Request {
            return Err(anyhow!("TCP callback connection is not a request stream"));
        }
        if info.context != context.id() {
            return Err(anyhow!(
                "TCP callback context mismatch: expected {}, got {}",
                context.id(),
                info.context
            ));
        }

        let stream = Self::connect(&info.address).await?;
        let (read_half, write_half) = tokio::io::split(stream);
        let reader = FramedRead::new(read_half, TwoPartCodec::default());
        let mut writer = FramedWrite::new(write_half, TwoPartCodec::default());
        let handshake = CallHomeHandshake {
            subject: info.subject,
            stream_type: StreamType::Request,
        };
        writer
            .send(TwoPartMessage::from_header(
                serde_json::to_vec(&handshake)?.into(),
            ))
            .await?;
        drop(writer);

        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(read_request_stream(
            reader,
            bytes_tx,
            context,
            cancellation_counter,
        ));
        Ok(StreamReceiver { rx: bytes_rx })
    }
}

async fn read_request_stream(
    mut reader: FramedRead<tokio::io::ReadHalf<TcpStream>, TwoPartCodec>,
    bytes_tx: tokio::sync::mpsc::Sender<bytes::Bytes>,
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
) {
    let mut cancelled = false;
    loop {
        tokio::select! {
            biased;
            _ = context.killed() => break,
            _ = context.stopped() => break,
            _ = bytes_tx.closed() => break,
            message = reader.next() => match message {
                Some(Ok(message)) => match message.into_message_type() {
                    TwoPartMessageType::DataOnly(data) => {
                        if bytes_tx.send(data).await.is_err() { break; }
                    }
                    TwoPartMessageType::HeaderOnly(header) => {
                        match serde_json::from_slice::<ControlMessage>(&header) {
                            Ok(ControlMessage::Sentinel) => break,
                            Ok(ControlMessage::Stop) => {
                                cancelled = true;
                                context.stop();
                                break;
                            }
                            Ok(ControlMessage::Kill) => {
                                cancelled = true;
                                context.kill();
                                break;
                            }
                            Err(error) => {
                                tracing::warn!(%error, "invalid TCP request callback control");
                                cancelled = true;
                                context.kill();
                                break;
                            }
                        }
                    }
                    _ => {
                        cancelled = true;
                        context.kill();
                        break;
                    }
                },
                Some(Err(error)) => {
                    tracing::warn!(%error, "TCP request callback read failed");
                    cancelled = true;
                    context.kill();
                    break;
                }
                None => {
                    cancelled = true;
                    context.kill();
                    break;
                }
            }
        }
    }
    if cancelled && let Some(counter) = cancellation_counter {
        counter.inc();
    }
}
