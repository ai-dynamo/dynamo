// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::io::{AsyncReadExt, ReadHalf, WriteHalf};
use tokio::{
    io::AsyncWriteExt,
    net::TcpStream,
    time::{self, Duration, Instant},
};
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{CallHomeHandshake, ControlMessage, TcpStreamConnectionInfo};
use crate::engine::AsyncEngineContext;
use crate::pipeline::network::{
    ConnectionInfo, ResponseStreamPrologue, StreamSender,
    codec::{TwoPartCodec, TwoPartMessage},
    tcp::StreamType,
};
use anyhow::{Context, Result, anyhow as error}; // Import SinkExt to use the `send` method

#[allow(dead_code)]
pub struct TcpClient {
    worker_id: String,
}

impl Default for TcpClient {
    fn default() -> Self {
        TcpClient {
            worker_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl TcpClient {
    pub fn new(worker_id: String) -> Self {
        TcpClient { worker_id }
    }

    async fn connect(address: &str) -> std::io::Result<TcpStream> {
        // try to connect to the address; retry with linear backoff if AddrNotAvailable
        let backoff = std::time::Duration::from_millis(200);
        loop {
            match TcpStream::connect(address).await {
                Ok(socket) => {
                    socket.set_nodelay(true)?;
                    return Ok(socket);
                }
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::AddrNotAvailable {
                        tracing::warn!("retry warning: failed to connect: {:?}", e);
                        tokio::time::sleep(backoff).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    pub async fn create_response_stream(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
    ) -> Result<StreamSender> {
        let info =
            TcpStreamConnectionInfo::try_from(info).context("tcp-stream-connection-info-error")?;
        tracing::trace!("Creating response stream for {:?}", info);

        if info.stream_type != StreamType::Response {
            return Err(error!(
                "Invalid stream type; TcpClient requires the stream type to be `response`; however {:?} was passed",
                info.stream_type
            ));
        }

        if info.context != context.id() {
            return Err(error!(
                "Invalid context; TcpClient requires the context to be {:?}; however {:?} was passed",
                context.id(),
                info.context
            ));
        }

        let stream = TcpClient::connect(&info.address).await?;
        let peer_port = stream.peer_addr().ok().map(|addr| addr.port());
        let (read_half, write_half) = tokio::io::split(stream);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let mut framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

        // this is a oneshot channel that will be used to signal when the stream is closed
        // when the stream sender is dropped, the bytes_rx will be closed and the forwarder task will exit
        // the forwarder task will capture the alive_rx half of the oneshot channel; this will close the alive channel
        // so the holder of the alive_tx half will be notified that the stream is closed; the alive_tx channel will be
        // captured by the monitor task
        let (alive_tx, alive_rx) = tokio::sync::oneshot::channel::<()>();

        let reader_task = tokio::spawn(handle_reader(framed_reader, context.clone(), alive_tx));

        // transport specific handshake message
        let handshake = CallHomeHandshake {
            subject: info.subject.clone(),
            stream_type: StreamType::Response,
        };

        let handshake_bytes = match serde_json::to_vec(&handshake) {
            Ok(hb) => hb,
            Err(err) => {
                return Err(error!(
                    "create_response_stream: Error converting CallHomeHandshake to JSON array: {err:#}"
                ));
            }
        };
        let msg = TwoPartMessage::from_header(handshake_bytes.into());

        // issue the the first tcp handshake message
        framed_writer
            .send(msg)
            .await
            .map_err(|e| error!("failed to send handshake: {:?}", e))?;

        // set up the channel to send bytes to the transport layer
        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel(64);

        // forwards the bytes send from this stream to the transport layer; hold the alive_rx half of the oneshot channel

        let writer_task = tokio::spawn(handle_writer(framed_writer, bytes_rx, alive_rx, context));

        let subject = info.subject.clone();
        tokio::spawn(async move {
            // await both tasks
            let (reader, writer) = tokio::join!(reader_task, writer_task);

            match (reader, writer) {
                (Ok(reader), Ok(writer)) => {
                    let reader = reader.into_inner();

                    let writer = match writer {
                        Ok(writer) => writer.into_inner(),
                        Err(e) => {
                            tracing::error!("failed to join writer task: {:?}", e);
                            return Err(e);
                        }
                    };

                    let mut stream = reader.unsplit(writer);

                    // await the tcp server to shutdown the socket connection
                    // set a timeout for the server shutdown
                    let mut buf = vec![0u8; 1024];
                    let deadline = Instant::now() + Duration::from_secs(10);
                    loop {
                        let n = time::timeout_at(deadline, stream.read(&mut buf))
                            .await
                            .inspect_err(|_| {
                                tracing::debug!("server did not close socket within the deadline");
                            })?
                            .inspect_err(|e| {
                                tracing::debug!("failed to read from stream: {:?}", e);
                            })?;
                        if n == 0 {
                            // Server has closed (FIN)
                            break;
                        }
                    }

                    Ok(())
                }
                (Err(reader_err), Ok(_)) => {
                    tracing::error!(
                        "reader task failed to join (peer_port: {peer_port:?}, subject: {subject}): {reader_err:?}"
                    );
                    anyhow::bail!(
                        "reader task failed to join (peer_port: {peer_port:?}, subject: {subject}): {reader_err:?}"
                    );
                }
                (Ok(_), Err(writer_err)) => {
                    tracing::error!(
                        "writer task failed to join (peer_port: {peer_port:?}, subject: {subject}): {writer_err:?}"
                    );
                    anyhow::bail!(
                        "writer task failed to join (peer_port: {peer_port:?}, subject: {subject}): {writer_err:?}"
                    );
                }
                (Err(reader_err), Err(writer_err)) => {
                    tracing::error!(
                        "both reader and writer tasks failed to join (peer_port: {peer_port:?}, subject: {subject}) - reader: {reader_err:?}, writer: {writer_err:?}"
                    );
                    anyhow::bail!(
                        "both reader and writer tasks failed to join (peer_port: {peer_port:?}, subject: {subject}) - reader: {reader_err:?}, writer: {writer_err:?}"
                    );
                }
            }
        });

        // set up the prologue for the stream
        // this might have transport specific metadata in the future
        let prologue = Some(ResponseStreamPrologue { error: None });

        // create the stream sender
        let stream_sender = StreamSender {
            tx: bytes_tx,
            prologue,
        };

        Ok(stream_sender)
    }
}

async fn handle_reader(
    framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    context: Arc<dyn AsyncEngineContext>,
    alive_tx: tokio::sync::oneshot::Sender<()>,
) -> FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec> {
    let mut framed_reader = framed_reader;
    let mut alive_tx = alive_tx;
    loop {
        tokio::select! {
            msg = framed_reader.next() => {
                match msg {
                    Some(Ok(two_part_msg)) => {
                        match two_part_msg.optional_parts() {
                           (Some(bytes), None) => {
                                let msg = match serde_json::from_slice::<ControlMessage>(bytes) {
                                    Ok(msg) => msg,
                                    Err(_) => {
                                        // TODO(#171) - address fatal errors
                                        panic!("fatal error - invalid control message detected");
                                    }
                                };

                                match msg {
                                    ControlMessage::Stop => {
                                        context.stop();
                                    }
                                    ControlMessage::Kill => {
                                        context.kill();
                                    }
                                    ControlMessage::Sentinel => {
                                        // TODO(#171) - address fatal errors
                                        panic!("received a sentinel message; this should never happen");
                                    }
                                }
                           }
                           _ => {
                                panic!("received a non-control message; this should never happen");
                           }
                        }
                    }
                    Some(Err(e)) => {
                        // TODO(#171) - address fatal errors
                        // in this case the binary representation of the message is invalid
                        panic!("fatal error - failed to decode message from stream; invalid line protocol: {e:?}");
                    }
                    None => {
                        tracing::debug!("tcp stream closed by server");
                        break;
                    }
                }
            }
            _ = alive_tx.closed() => {
                break;
            }
        }
    }
    framed_reader
}

async fn handle_writer(
    mut framed_writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
    mut bytes_rx: tokio::sync::mpsc::Receiver<TwoPartMessage>,
    alive_rx: tokio::sync::oneshot::Receiver<()>,
    context: Arc<dyn AsyncEngineContext>,
) -> Result<FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>> {
    // Only send sentinel for normal channel closure
    let mut send_sentinel = true;

    loop {
        let msg = tokio::select! {
            biased;

            _ = context.killed() => {
                tracing::trace!("context kill signal received; shutting down");
                send_sentinel = false;
                break;
            }

            _ = context.stopped() => {
                tracing::trace!("context stop signal received; shutting down");
                send_sentinel = false;
                break;
            }

            msg = bytes_rx.recv() => {
                match msg {
                    Some(msg) => msg,
                    None => {
                        tracing::trace!("response channel closed; shutting down");
                        break;
                    }
                }
            }
        };

        if let Err(e) = framed_writer.send(msg).await {
            tracing::trace!(
                "failed to send message to network; possible disconnect: {:?}",
                e
            );
            send_sentinel = false;
            break;
        }
    }

    // Send sentinel only on normal closure
    if send_sentinel {
        let message = serde_json::to_vec(&ControlMessage::Sentinel)?;
        let msg = TwoPartMessage::from_header(message.into());
        framed_writer.send(msg).await?;
    }

    drop(alive_rx);
    Ok(framed_writer)
}
