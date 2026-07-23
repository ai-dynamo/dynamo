// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::{io::AsyncWriteExt, net::TcpStream};
use tokio_util::codec::{FramedRead, FramedWrite};

use prometheus::IntCounter;

use super::{CallHomeHandshake, ControlMessage, TcpStreamConnectionInfo};
use crate::engine::AsyncEngineContext;
use crate::pipeline::network::{
    ConnectionInfo, StreamReceiver, StreamRxItem,
    codec::{TwoPartCodec, TwoPartMessage, TwoPartMessageType},
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

    /// Dial the upstream TCP server with `StreamType::Request`, then return a
    /// [`StreamReceiver`] that yields the data frames the upstream pushes down.
    ///
    /// The request stream is unidirectional after the handshake: the write half
    /// is dropped as soon as the `CallHomeHandshake` is sent, so the downstream
    /// never writes anything back (no `Sentinel` ack). The spawned reader task
    /// forwards `TwoPartMessage::DataOnly` payloads into the channel and
    /// translates `ControlMessage::Stop` / `Kill` into context cancellation;
    /// `ControlMessage::Sentinel` terminates the task cleanly. A TCP close
    /// before any `Sentinel` is treated as a truncated input (cancellation +
    /// `context.kill()`), and dropping the returned `StreamReceiver` also stops
    /// the task.
    pub async fn create_request_stream(
        context: Arc<dyn AsyncEngineContext>,
        info: ConnectionInfo,
        cancellation_counter: Option<IntCounter>,
    ) -> Result<StreamReceiver> {
        let info =
            TcpStreamConnectionInfo::try_from(info).context("tcp-stream-connection-info-error")?;
        tracing::trace!("Creating request stream for {:?}", info);

        if info.stream_type != StreamType::Request {
            return Err(error!(
                "Invalid stream type; TcpClient::create_request_stream requires the stream type to be `request`; however {:?} was passed",
                info.stream_type
            ));
        }

        if info.context != context.id() {
            return Err(error!(
                "Invalid context; TcpClient::create_request_stream requires the context to be {:?}; however {:?} was passed",
                context.id(),
                info.context
            ));
        }

        let stream = TcpClient::connect(&info.address).await?;
        let (read_half, write_half) = tokio::io::split(stream);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let mut framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

        let handshake = CallHomeHandshake {
            subject: info.subject.clone(),
            stream_type: StreamType::Request,
        };
        let handshake_bytes = serde_json::to_vec(&handshake).map_err(|err| {
            error!(
                "create_request_stream: Error converting CallHomeHandshake to JSON array: {err:#}"
            )
        })?;
        framed_writer
            .send(TwoPartMessage::from_header(handshake_bytes.into()))
            .await
            .map_err(|e| error!("failed to send request-stream handshake: {:?}", e))?;

        // Request stream is unidirectional after the handshake: the downstream
        // never writes again, so close the write half immediately.
        drop(framed_writer);

        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel::<StreamRxItem>(64);

        tokio::spawn(handle_request_reader(
            framed_reader,
            bytes_tx,
            context,
            cancellation_counter,
        ));

        Ok(StreamReceiver::dedicated(bytes_rx))
    }
}

async fn handle_request_reader(
    mut framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    bytes_tx: tokio::sync::mpsc::Sender<StreamRxItem>,
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
) {
    let cancellation_seen = {
        // Keep cancellation and receiver-close notifications alive across frames instead of
        // rebuilding their watch/Notify state for every request-stream message.
        let killed = context.killed();
        let stopped = context.stopped();
        // The function explicitly drops `bytes_tx` after the loop, so let the persistent
        // `closed()` future borrow a stream-lifetime clone instead.
        let bytes_closed_tx = bytes_tx.clone();
        let bytes_closed = bytes_closed_tx.closed();
        tokio::pin!(killed, stopped, bytes_closed);

        // Only mark cancellation on fatal errors or explicit upstream cancellation.
        let mut cancellation_seen = false;
        loop {
            tokio::select! {
                biased;

                _ = &mut killed => {
                    tracing::trace!("context kill signal received on request stream; shutting down");
                    break;
                }

                _ = &mut stopped => {
                    tracing::trace!("context stop signal received on request stream; shutting down");
                    break;
                }

                // Downstream consumer dropped the StreamReceiver. Exit promptly
                // instead of staying parked on `framed_reader.next()` until the
                // socket closes — the data has nowhere to go. This is the consumer's
                // own choice, so it is not a cancellation (no kill, no count).
                _ = &mut bytes_closed => {
                    tracing::debug!("downstream consumer dropped; exiting request-stream reader");
                    break;
                }

                msg = framed_reader.next() => {
                    match msg {
                        Some(Ok(two_part_msg)) => match two_part_msg.into_message_type() {
                            TwoPartMessageType::HeaderOnly(header) => {
                                let ctrl = match serde_json::from_slice::<ControlMessage>(&header) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        tracing::warn!(
                                            err = ?e,
                                            "invalid control message, closing connection"
                                        );
                                        cancellation_seen = true;
                                        context.kill();
                                        break;
                                    }
                                };
                                match ctrl {
                                    ControlMessage::Stop => {
                                        cancellation_seen = true;
                                        context.stop();
                                        break;
                                    }
                                    ControlMessage::Kill => {
                                        cancellation_seen = true;
                                        context.kill();
                                        break;
                                    }
                                    ControlMessage::Sentinel => {
                                        tracing::trace!("upstream signaled end of request stream");
                                        break;
                                    }
                                }
                            }
                            TwoPartMessageType::DataOnly(data) => {
                                if bytes_tx.send(StreamRxItem::dedicated(data)).await.is_err() {
                                    tracing::debug!("downstream consumer dropped; exiting request-stream reader");
                                    break;
                                }
                            }
                            _ => {
                                tracing::warn!("fatal error - unexpected message shape on request stream");
                                cancellation_seen = true;
                                context.kill();
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            tracing::warn!("fatal error - failed to decode message on request stream: {e:?}");
                            cancellation_seen = true;
                            context.kill();
                            break;
                        }
                        None => {
                            // Socket closed before a Sentinel/Stop/Kill: the request
                            // input is truncated. Kill the context so the consumer
                            // sees an aborted stream rather than a clean end, and
                            // count it as a cancellation.
                            tracing::warn!("request stream closed by upstream before sentinel; treating as truncated");
                            cancellation_seen = true;
                            context.kill();
                            break;
                        }
                    }
                }
            }
        }

        // Leaving this scope drops the pinned `closed()` future and its sender clone
        // before the original sender below signals end-of-stream.
        cancellation_seen
    };

    if cancellation_seen && let Some(counter) = &cancellation_counter {
        counter.inc();
    }

    // Dropping bytes_tx closes the receiver side, signaling end-of-stream to the
    // engine consumer.
    drop(bytes_tx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::Controller;
    use crate::pipeline::network::tcp::test_utils::create_tcp_pair;
    use bytes::Bytes;
    use std::sync::Arc;
    use tokio::io::AsyncWriteExt;
    use tokio::sync::mpsc;

    fn control_message(msg: &ControlMessage) -> TwoPartMessage {
        TwoPartMessage::from_header(serde_json::to_vec(msg).unwrap().into())
    }

    // ==================== handle_request_reader tests ====================

    struct RequestReaderHarness {
        framed_server: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        bytes_tx: mpsc::Sender<StreamRxItem>,
        bytes_rx: mpsc::Receiver<StreamRxItem>,
        controller: Arc<Controller>,
    }

    async fn request_reader_harness() -> RequestReaderHarness {
        let (client, server) = create_tcp_pair().await;
        let (read_half, _write_half) = tokio::io::split(client);
        let (_server_read, server_write) = tokio::io::split(server);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_server = FramedWrite::new(server_write, TwoPartCodec::default());
        let (bytes_tx, bytes_rx) = mpsc::channel::<StreamRxItem>(64);
        let controller = Arc::new(Controller::default());

        RequestReaderHarness {
            framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx,
            controller,
        }
    }

    /// Receiving Stop calls context.stop(), increments the counter, and exits.
    #[tokio::test]
    async fn test_handle_request_reader_stop_control_message() {
        let RequestReaderHarness {
            mut framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx: _bytes_rx,
            controller,
        } = request_reader_harness().await;

        let counter = IntCounter::new("tcp_request_reader_stop_test", "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(
                framed_reader,
                bytes_tx,
                controller_clone,
                Some(counter_clone),
            )
            .await
        });

        framed_server
            .send(control_message(&ControlMessage::Stop))
            .await
            .unwrap();

        handle.await.unwrap();

        assert!(controller.is_stopped(), "Stop should call context.stop()");
        assert!(!controller.is_killed(), "Stop should not kill the context");
        assert_eq!(counter.get(), 1, "cancellation counter should increment");
    }

    /// Receiving Kill calls context.kill(), increments the counter, and exits.
    #[tokio::test]
    async fn test_handle_request_reader_kill_control_message() {
        let RequestReaderHarness {
            mut framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx: _bytes_rx,
            controller,
        } = request_reader_harness().await;

        let counter = IntCounter::new("tcp_request_reader_kill_test", "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(
                framed_reader,
                bytes_tx,
                controller_clone,
                Some(counter_clone),
            )
            .await
        });

        framed_server
            .send(control_message(&ControlMessage::Kill))
            .await
            .unwrap();

        handle.await.unwrap();

        assert!(controller.is_killed(), "Kill should call context.kill()");
        assert_eq!(counter.get(), 1, "cancellation counter should increment");
    }

    /// Receiving Sentinel exits cleanly without touching the context or counter.
    #[tokio::test]
    async fn test_handle_request_reader_sentinel_control_message() {
        let RequestReaderHarness {
            mut framed_server,
            framed_reader,
            bytes_tx,
            mut bytes_rx,
            controller,
        } = request_reader_harness().await;

        let counter = IntCounter::new("tcp_request_reader_sentinel_test", "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(
                framed_reader,
                bytes_tx,
                controller_clone,
                Some(counter_clone),
            )
            .await
        });

        framed_server
            .send(control_message(&ControlMessage::Sentinel))
            .await
            .unwrap();

        handle.await.unwrap();

        assert!(
            !controller.is_stopped(),
            "Sentinel must not stop the context"
        );
        assert!(
            !controller.is_killed(),
            "Sentinel must not kill the context"
        );
        assert_eq!(counter.get(), 0, "Sentinel must not increment counter");
        assert!(
            bytes_rx.recv().await.is_none(),
            "bytes_tx should be dropped on exit"
        );
    }

    /// DataOnly frames are forwarded to bytes_tx; the loop continues until a
    /// terminator arrives (here, Sentinel).
    #[tokio::test]
    async fn test_handle_request_reader_forwards_data() {
        let RequestReaderHarness {
            mut framed_server,
            framed_reader,
            bytes_tx,
            mut bytes_rx,
            controller,
        } = request_reader_harness().await;

        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(framed_reader, bytes_tx, controller_clone, None).await
        });

        framed_server
            .send(TwoPartMessage::from_data(Bytes::from_static(b"hello")))
            .await
            .unwrap();
        framed_server
            .send(TwoPartMessage::from_data(Bytes::from_static(b"world")))
            .await
            .unwrap();

        assert_eq!(bytes_rx.recv().await.unwrap().as_ref(), b"hello");
        assert_eq!(bytes_rx.recv().await.unwrap().as_ref(), b"world");

        framed_server
            .send(control_message(&ControlMessage::Sentinel))
            .await
            .unwrap();

        handle.await.unwrap();
        assert!(
            bytes_rx.recv().await.is_none(),
            "channel should close after Sentinel"
        );
    }

    /// External context.kill() exits the reader without touching the wire.
    #[tokio::test]
    async fn test_handle_request_reader_exits_on_context_killed() {
        let RequestReaderHarness {
            framed_server: _framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx: _bytes_rx,
            controller,
        } = request_reader_harness().await;

        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(framed_reader, bytes_tx, controller_clone, None).await
        });

        controller.kill();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        assert!(
            result.is_ok(),
            "handler should exit promptly on context.kill()"
        );
    }

    /// External context.stop() exits the reader without touching the wire.
    #[tokio::test]
    async fn test_handle_request_reader_exits_on_context_stopped() {
        let RequestReaderHarness {
            framed_server: _framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx: _bytes_rx,
            controller,
        } = request_reader_harness().await;

        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(framed_reader, bytes_tx, controller_clone, None).await
        });

        controller.stop();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        assert!(
            result.is_ok(),
            "handler should exit promptly on context.stop()"
        );
    }

    /// Socket EOF exits the reader and drops bytes_tx.
    /// EOF before a closing Sentinel is a truncated request input: the handler
    /// kills the context and counts a cancellation so the consumer sees an
    /// aborted stream rather than a clean end.
    #[tokio::test]
    async fn test_handle_request_reader_exits_on_stream_closed() {
        let RequestReaderHarness {
            mut framed_server,
            framed_reader,
            bytes_tx,
            mut bytes_rx,
            controller,
        } = request_reader_harness().await;

        let counter =
            IntCounter::new("tcp_request_reader_eof_truncation_test", "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(
                framed_reader,
                bytes_tx,
                controller_clone,
                Some(counter_clone),
            )
            .await
        });

        framed_server.close().await.unwrap();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        assert!(result.is_ok(), "handler should exit on EOF");
        assert!(
            controller.is_killed(),
            "EOF before sentinel should kill the context (truncated input)"
        );
        assert_eq!(
            counter.get(),
            1,
            "EOF before sentinel should count as a cancellation"
        );
        assert!(
            bytes_rx.recv().await.is_none(),
            "bytes_tx should be dropped"
        );
    }

    /// Dropping the returned StreamReceiver makes the reader exit promptly via
    /// the `bytes_tx.closed()` arm, even while parked on the socket with no
    /// incoming frame. This is the consumer's own choice, so it is not counted
    /// as a cancellation and the context is left untouched.
    #[tokio::test]
    async fn test_handle_request_reader_exits_when_receiver_dropped() {
        let RequestReaderHarness {
            framed_server,
            framed_reader,
            bytes_tx,
            bytes_rx,
            controller,
        } = request_reader_harness().await;

        // Keep the socket open so the only exit path is the receiver drop.
        let _framed_server = framed_server;

        let counter =
            IntCounter::new("tcp_request_reader_receiver_drop_test", "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let handle = tokio::spawn(async move {
            handle_request_reader(
                framed_reader,
                bytes_tx,
                controller_clone,
                Some(counter_clone),
            )
            .await
        });

        // Drop the consumer; the reader is parked on `framed_reader.next()`.
        drop(bytes_rx);

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        assert!(
            result.is_ok(),
            "handler should exit promptly when the receiver is dropped"
        );
        assert!(
            !controller.is_killed() && !controller.is_stopped(),
            "consumer drop is not a cancellation"
        );
        assert_eq!(
            counter.get(),
            0,
            "consumer drop must not count as cancellation"
        );
    }
}
