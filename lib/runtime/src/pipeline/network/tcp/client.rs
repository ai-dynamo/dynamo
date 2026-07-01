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

use prometheus::IntCounter;
use tracing::Instrument;

use super::{CallHomeHandshake, ControlMessage, TcpStreamConnectionInfo};
use crate::engine::AsyncEngineContext;
use crate::pipeline::network::{
    ConnectionInfo, ResponseStreamPrologue, StreamReceiver, StreamSender,
    codec::{TwoPartCodec, TwoPartMessage, TwoPartMessageType},
    tcp::StreamType,
};
use anyhow::{Context, Result, anyhow as error}; // Import SinkExt to use the `send` method

#[allow(dead_code)]
pub struct TcpClient {
    worker_id: String,
}

fn record_upstream_cancellation(context: &dyn AsyncEngineContext, signal: &'static str) {
    tracing::info!(
        target: "request_span",
        {
            request_id = context.id(),
            { "cancellation.signal" } = signal,
            { "cancellation.source" } = "upstream"
        },
        "request cancellation received"
    );
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
        cancellation_counter: Option<IntCounter>,
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

        let reader_span = tracing::Span::current();
        let reader_task = tokio::spawn(
            handle_reader(
                framed_reader,
                context.clone(),
                alive_tx,
                cancellation_counter,
            )
            .instrument(reader_span),
        );

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
        let writer_context = context.clone();
        let writer_task = tokio::spawn(handle_writer(
            framed_writer,
            bytes_rx,
            alive_rx,
            writer_context,
        ));

        let subject = info.subject.clone();
        let monitor_context = context;
        // Spawn the connection monitor; errors are already logged inside
        // wait_for_connection_tasks, so the Result is intentionally dropped.
        tokio::spawn(async move {
            let _ = wait_for_connection_tasks(
                reader_task,
                writer_task,
                monitor_context,
                peer_port,
                subject,
            )
            .await;
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

    /// Symmetric to [`Self::create_response_stream`] for the request-stream half:
    /// dial the upstream TCP server with `StreamType::Request`, then return a
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

        let (bytes_tx, bytes_rx) = tokio::sync::mpsc::channel::<bytes::Bytes>(64);

        let reader_span = tracing::Span::current();
        tokio::spawn(
            handle_request_reader(framed_reader, bytes_tx, context, cancellation_counter)
                .instrument(reader_span),
        );

        Ok(StreamReceiver { rx: bytes_rx })
    }
}

async fn handle_request_reader(
    mut framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    bytes_tx: tokio::sync::mpsc::Sender<bytes::Bytes>,
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
) {
    // Only mark cancellation on fatal errors or explicit upstream cancellation.
    let mut cancellation_seen = false;
    loop {
        tokio::select! {
            biased;

            _ = context.killed() => {
                tracing::trace!("context kill signal received on request stream; shutting down");
                break;
            }

            _ = context.stopped() => {
                tracing::trace!("context stop signal received on request stream; shutting down");
                break;
            }

            // Downstream consumer dropped the StreamReceiver. Exit promptly
            // instead of staying parked on `framed_reader.next()` until the
            // socket closes — the data has nowhere to go. This is the consumer's
            // own choice, so it is not a cancellation (no kill, no count).
            _ = bytes_tx.closed() => {
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
                                    record_upstream_cancellation(context.as_ref(), "stop");
                                    context.stop();
                                    break;
                                }
                                ControlMessage::Kill => {
                                    cancellation_seen = true;
                                    record_upstream_cancellation(context.as_ref(), "kill");
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
                            if bytes_tx.send(data).await.is_err() {
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

    if cancellation_seen && let Some(counter) = &cancellation_counter {
        counter.inc();
    }

    // Dropping bytes_tx closes the receiver side, signaling end-of-stream to the
    // engine consumer.
    drop(bytes_tx);
}

async fn wait_for_connection_tasks(
    reader_task: tokio::task::JoinHandle<FramedRead<ReadHalf<TcpStream>, TwoPartCodec>>,
    writer_task: tokio::task::JoinHandle<Result<FramedWrite<WriteHalf<TcpStream>, TwoPartCodec>>>,
    context: Arc<dyn AsyncEngineContext>,
    peer_port: Option<u16>,
    subject: String,
) -> Result<()> {
    // Await the reader first and abort the writer on reader Err — the
    // writer parks on `bytes_rx.recv()` and won't wake on its own.
    let reader = match reader_task.await {
        Ok(reader) => reader,
        Err(reader_err) => {
            writer_task.abort();
            let _ = writer_task.await;
            tracing::error!(
                subject = %subject,
                peer_port = ?peer_port,
                err = ?reader_err,
                "reader task failed to join"
            );
            return Err(reader_err.into());
        }
    };

    let writer = match writer_task.await {
        Ok(writer) => writer,
        Err(writer_err) => {
            tracing::error!(
                subject = %subject,
                peer_port = ?peer_port,
                err = ?writer_err,
                "writer task failed to join"
            );
            return Err(writer_err.into());
        }
    };

    let reader = reader.into_inner();
    let writer = match writer {
        Ok(writer) => writer.into_inner(),
        Err(e) => {
            tracing::error!(
                subject = %subject,
                peer_port = ?peer_port,
                err = ?e,
                "writer task returned error"
            );
            return Err(e);
        }
    };

    let stream = reader.unsplit(writer);
    wait_for_server_shutdown(stream, context).await
}

async fn wait_for_server_shutdown(
    mut stream: TcpStream,
    context: Arc<dyn AsyncEngineContext>,
) -> Result<()> {
    // `handle_writer` skips the closing sentinel on both `killed` and
    // `stopped`, so the server has nothing to react to in either case;
    // sitting in the read loop until the 10 s deadline would be dead time.
    if context.is_killed() || context.is_stopped() {
        tracing::debug!("stream context killed or stopped; skipping server FIN wait");
        return Ok(());
    }

    // Await the tcp server to shutdown the socket connection, bounded by a
    // timeout so normal sentinel shutdown cannot hang indefinitely.
    let mut buf = [0u8; 1024];
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let n = time::timeout_at(deadline, stream.read(&mut buf))
            .await
            .inspect_err(|_| {
                tracing::debug!("server did not close socket within the deadline");
            })?
            .inspect_err(|e| {
                tracing::debug!(err = ?e, "failed to read from stream");
            })?;
        if n == 0 {
            // Server has closed (FIN)
            break;
        }
    }

    Ok(())
}

async fn handle_reader(
    framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    context: Arc<dyn AsyncEngineContext>,
    alive_tx: tokio::sync::oneshot::Sender<()>,
    cancellation_counter: Option<IntCounter>,
) -> FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec> {
    let mut framed_reader = framed_reader;
    let mut alive_tx = alive_tx;
    // Set on every cancellation arm; counted once after the loop.
    let mut cancellation_seen = false;
    loop {
        tokio::select! {
            msg = framed_reader.next() => {
                match msg {
                    Some(Ok(two_part_msg)) => {
                        match two_part_msg.optional_parts() {
                           (Some(bytes), None) => {
                                let msg = match serde_json::from_slice::<ControlMessage>(bytes) {
                                    Ok(msg) => msg,
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

                                // Stop/Kill intentionally do not `break`: the
                                // reader keeps running so a later Kill can
                                // upgrade an earlier Stop (and vice versa).
                                // The loop still exits promptly via the
                                // `alive_tx.closed()` arm once `handle_writer`
                                // reacts to `context.stop()` / `context.kill()`.
                                match msg {
                                    ControlMessage::Stop => {
                                        cancellation_seen = true;
                                        record_upstream_cancellation(context.as_ref(), "stop");
                                        context.stop();
                                    }
                                    ControlMessage::Kill => {
                                        cancellation_seen = true;
                                        record_upstream_cancellation(context.as_ref(), "kill");
                                        context.kill();
                                    }
                                    ControlMessage::Sentinel => {
                                        tracing::warn!(
                                            "unexpected sentinel on client reader, closing connection"
                                        );
                                        cancellation_seen = true;
                                        context.kill();
                                        break;
                                    }
                                }
                           }
                           _ => {
                                tracing::warn!(
                                    "unexpected non-control message on client reader, closing connection"
                                );
                                cancellation_seen = true;
                                context.kill();
                                break;
                           }
                        }
                    }
                    Some(Err(e)) => {
                        // Kill the engine context so the producer stops
                        // generating responses that can no longer be delivered.
                        tracing::warn!(err = ?e, "tcp stream read error, closing connection");
                        cancellation_seen = true;
                        context.kill();
                        break;
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
    if cancellation_seen && let Some(counter) = &cancellation_counter {
        counter.inc();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::Controller;
    use crate::pipeline::network::tcp::test_utils::create_tcp_pair;
    use bytes::Bytes;
    use futures::StreamExt;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio::sync::{mpsc, oneshot};
    use tokio_util::codec::FramedRead;
    use tracing::field::{Field, Visit};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context as TraceContext, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::util::SubscriberInitExt;

    type CapturedCancellationEvent = (HashMap<String, String>, Option<String>);

    #[derive(Default)]
    struct CancellationEventCapture(Mutex<Vec<CapturedCancellationEvent>>);

    struct EventFieldVisitor<'a>(&'a mut HashMap<String, String>);

    impl Visit for EventFieldVisitor<'_> {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    struct CancellationEventLayer(Arc<CancellationEventCapture>);

    impl<S> Layer<S> for CancellationEventLayer
    where
        S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_event(&self, event: &tracing::Event<'_>, ctx: TraceContext<'_, S>) {
            if event.metadata().target() != "request_span" {
                return;
            }
            let mut fields = HashMap::new();
            event.record(&mut EventFieldVisitor(&mut fields));
            if fields
                .get("message")
                .is_none_or(|message| message.trim_matches('"') != "request cancellation received")
            {
                return;
            }
            let parent = ctx.event_span(event).map(|span| span.name().to_string());
            self.0.0.lock().unwrap().push((fields, parent));
        }
    }

    #[tokio::test]
    async fn upstream_cancellation_event_is_parented_to_worker_request_span() {
        let captured = Arc::new(CancellationEventCapture::default());
        let _subscriber = tracing_subscriber::registry()
            .with(CancellationEventLayer(captured.clone()))
            .set_default();
        let controller = Arc::new(Controller::new("request-123".to_string()));
        let span = tracing::info_span!(target: "request_span", "handle_payload");

        async {
            record_upstream_cancellation(controller.as_ref(), "stop");
        }
        .instrument(span)
        .await;

        let events = captured.0.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].0.get("cancellation.signal").map(String::as_str),
            Some("stop")
        );
        assert_eq!(
            events[0].0.get("cancellation.source").map(String::as_str),
            Some("upstream")
        );
        assert_eq!(events[0].1.as_deref(), Some("handle_payload"));
    }

    struct WriterHarness {
        server: tokio::net::TcpStream,
        framed_writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        bytes_tx: mpsc::Sender<TwoPartMessage>,
        bytes_rx: mpsc::Receiver<TwoPartMessage>,
        alive_tx: oneshot::Sender<()>,
        alive_rx: oneshot::Receiver<()>,
        controller: Arc<Controller>,
    }

    /// Creates a reusable writer harness with paired TCP streams and test channels.
    async fn writer_harness() -> WriterHarness {
        let (client, server) = create_tcp_pair().await;
        let (_, write_half) = tokio::io::split(client);
        let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

        let (bytes_tx, bytes_rx) = mpsc::channel(64);
        let (alive_tx, alive_rx) = oneshot::channel::<()>();
        let controller = Arc::new(Controller::default());

        WriterHarness {
            server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_tx,
            alive_rx,
            controller,
        }
    }

    async fn recv_msg(reader: &mut FramedRead<TcpStream, TwoPartCodec>) -> TwoPartMessage {
        reader
            .next()
            .await
            .expect("expected message")
            .expect("failed to decode message")
    }

    fn assert_data_only_message(msg: TwoPartMessage, expected: &[u8]) {
        let (header, data) = msg.optional_parts();
        assert!(header.is_none(), "data-only message should not have header");
        assert_eq!(
            data.expect("data payload missing").as_ref(),
            expected,
            "data payload should match"
        );
    }

    fn assert_header_only_message(msg: TwoPartMessage, expected: &[u8]) {
        let (header, data) = msg.optional_parts();
        assert!(data.is_none(), "header-only message should not carry data");
        assert_eq!(
            header.expect("header missing").as_ref(),
            expected,
            "header payload should match"
        );
    }

    fn assert_header_and_data_message(
        msg: TwoPartMessage,
        expected_header: &[u8],
        expected_data: &[u8],
    ) {
        let (header, data) = msg.optional_parts();
        assert_eq!(
            header.expect("header missing").as_ref(),
            expected_header,
            "header payload should match"
        );
        assert_eq!(
            data.expect("data missing").as_ref(),
            expected_data,
            "data payload should match"
        );
    }

    fn assert_sentinel_message(msg: TwoPartMessage) {
        let (header, data) = msg.optional_parts();
        assert!(data.is_none(), "sentinel should not include a data section");
        let expected_sentinel = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
        assert_eq!(
            header.expect("sentinel header missing").as_ref(),
            expected_sentinel.as_slice(),
            "sentinel header should match serialized ControlMessage::Sentinel"
        );
    }

    /// Test that handle_writer forwards messages from the channel to the framed writer
    #[tokio::test]
    async fn test_handle_writer_forwards_messages() {
        let WriterHarness {
            server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Send test messages
        let test_msg = TwoPartMessage::from_data(Bytes::from("test data"));
        bytes_tx.send(test_msg).await.unwrap();

        // Close the sender to trigger normal termination
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // Decode from server side to verify data and sentinel were sent
        let mut reader = FramedRead::new(server, TwoPartCodec::default());

        let msg = recv_msg(&mut reader).await;
        assert_data_only_message(msg, b"test data");

        let sentinel = recv_msg(&mut reader).await;
        assert_sentinel_message(sentinel);
    }

    /// Test that handle_writer sends sentinel on normal channel closure
    #[tokio::test]
    async fn test_handle_writer_sends_sentinel_on_normal_closure() {
        let WriterHarness {
            mut server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Close the sender immediately to trigger normal termination
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // Read from server side to verify sentinel was sent
        let mut buffer = vec![0u8; 1024];
        let n = server.read(&mut buffer).await.unwrap();

        // Buffer should contain the sentinel message
        assert!(n > 0, "Expected sentinel to be written to the TCP stream");

        // Verify it contains the sentinel message by checking for the JSON
        let sentinel_json = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
        assert!(
            buffer[..n]
                .windows(sentinel_json.len())
                .any(|w| w == sentinel_json.as_slice()),
            "Buffer should contain sentinel message. Buffer: {:?}",
            String::from_utf8_lossy(&buffer[..n])
        );
    }

    /// A normally completed response stream sends Sentinel, receives the
    /// server's FIN, and leaves the cancellation metric unchanged.
    #[tokio::test]
    async fn test_normal_response_stream_completion_does_not_count_cancellation() {
        let (client, server) = create_tcp_pair().await;
        let (read_half, write_half) = tokio::io::split(client);
        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());
        let (bytes_tx, bytes_rx) = mpsc::channel(64);
        let (alive_tx, alive_rx) = oneshot::channel::<()>();
        let controller = Arc::new(Controller::default());
        let cancellation_counter = IntCounter::new(
            "tcp_client_normal_completion_cancellations_test",
            "test cancellation counter",
        )
        .unwrap();

        let reader_context = controller.clone();
        let counter_clone = cancellation_counter.clone();
        let reader_task = tokio::spawn(async move {
            handle_reader(framed_reader, reader_context, alive_tx, Some(counter_clone)).await
        });
        let writer_context = controller.clone();
        let writer_task = tokio::spawn(async move {
            handle_writer(framed_writer, bytes_rx, alive_rx, writer_context).await
        });

        drop(bytes_tx);

        let mut server_reader = FramedRead::new(server, TwoPartCodec::default());
        let sentinel = recv_msg(&mut server_reader).await;
        assert_sentinel_message(sentinel);
        drop(server_reader);

        writer_task.await.unwrap().unwrap();
        reader_task.await.unwrap();

        assert!(
            !controller.is_stopped() && !controller.is_killed(),
            "normal response completion must not cancel the context"
        );
        assert_eq!(
            cancellation_counter.get(),
            0,
            "normal response completion must not increment the cancellation counter"
        );
    }

    /// Test that handle_writer does NOT send sentinel when context is killed
    #[tokio::test]
    async fn test_handle_writer_no_sentinel_on_context_killed() {
        let WriterHarness {
            mut server,
            framed_writer,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Kill the context
        controller.kill();

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // Drop the writer to close the connection, then try to read. Otherwise,
        // the test will hang on `server.read()`
        drop(result);

        // Read from server side - should get no sentinel
        let mut buffer = vec![0u8; 1024];
        let n = server.read(&mut buffer).await.unwrap();

        // Buffer should be empty (no sentinel sent)
        let sentinel_json = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
        assert!(
            n == 0
                || !buffer[..n]
                    .windows(sentinel_json.len())
                    .any(|w| w == sentinel_json.as_slice()),
            "Buffer should NOT contain sentinel message when context is killed"
        );
    }

    /// Test that handle_writer does NOT send sentinel when context is stopped
    #[tokio::test]
    async fn test_handle_writer_no_sentinel_on_context_stopped() {
        let WriterHarness {
            mut server,
            framed_writer,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Stop the context
        controller.stop();

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // Drop the writer to close the connection, then try to read. Otherwise,
        // the test will hang on `server.read()`
        drop(result);

        // Read from server side - should get no sentinel
        let mut buffer = vec![0u8; 1024];
        let n = server.read(&mut buffer).await.unwrap();

        // Buffer should be empty (no sentinel sent)
        let sentinel_json = serde_json::to_vec(&ControlMessage::Sentinel).unwrap();
        assert!(
            n == 0
                || !buffer[..n]
                    .windows(sentinel_json.len())
                    .any(|w| w == sentinel_json.as_slice()),
            "Buffer should NOT contain sentinel message when context is stopped"
        );
    }

    /// Test that handle_writer handles multiple messages correctly
    #[tokio::test]
    async fn test_handle_writer_multiple_messages() {
        let WriterHarness {
            server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Send multiple messages
        for i in 0..5 {
            let test_msg = TwoPartMessage::from_data(Bytes::from(format!("message {}", i)));
            bytes_tx.send(test_msg).await.unwrap();
        }

        // Close the sender to trigger normal termination
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // Decode from server side to verify all messages plus sentinel
        let mut reader = FramedRead::new(server, TwoPartCodec::default());
        for i in 0..5 {
            let msg = recv_msg(&mut reader).await;
            assert_data_only_message(msg, format!("message {}", i).as_bytes());
        }

        let sentinel = recv_msg(&mut reader).await;
        assert_sentinel_message(sentinel);
    }

    /// Test that alive_rx is dropped after handle_writer completes
    #[tokio::test]
    async fn test_handle_writer_drops_alive_rx() {
        let WriterHarness {
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_tx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Close the sender to trigger normal termination
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        // alive_tx should now be closed because alive_rx was dropped
        assert!(alive_tx.is_closed());
    }

    /// Test handle_writer with header-only messages (control messages)
    #[tokio::test]
    async fn test_handle_writer_header_only_messages() {
        let WriterHarness {
            server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Send a header-only message
        let header_msg = TwoPartMessage::from_header(Bytes::from("header content"));
        bytes_tx.send(header_msg).await.unwrap();

        // Close the sender
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        let mut reader = FramedRead::new(server, TwoPartCodec::default());

        let header_msg = recv_msg(&mut reader).await;
        assert_header_only_message(header_msg, b"header content");

        let sentinel = recv_msg(&mut reader).await;
        assert_sentinel_message(sentinel);
    }

    /// Test handle_writer with mixed header and data messages
    #[tokio::test]
    async fn test_handle_writer_mixed_messages() {
        let WriterHarness {
            server,
            framed_writer,
            bytes_tx,
            bytes_rx,
            alive_rx,
            controller,
            ..
        } = writer_harness().await;

        // Send mixed messages
        bytes_tx
            .send(TwoPartMessage::from_header(Bytes::from("header1")))
            .await
            .unwrap();
        bytes_tx
            .send(TwoPartMessage::from_data(Bytes::from("data1")))
            .await
            .unwrap();
        bytes_tx
            .send(TwoPartMessage::from_parts(
                Bytes::from("header2"),
                Bytes::from("data2"),
            ))
            .await
            .unwrap();

        // Close the sender
        drop(bytes_tx);

        let result = handle_writer(framed_writer, bytes_rx, alive_rx, controller).await;

        assert!(result.is_ok());

        let mut reader = FramedRead::new(server, TwoPartCodec::default());

        let first = recv_msg(&mut reader).await;
        assert_header_only_message(first, b"header1");

        let second = recv_msg(&mut reader).await;
        assert_data_only_message(second, b"data1");

        let third = recv_msg(&mut reader).await;
        assert_header_and_data_message(third, b"header2", b"data2");

        let sentinel = recv_msg(&mut reader).await;
        assert_sentinel_message(sentinel);
    }

    /// Killed or stopped contexts skip the server FIN deadline.
    #[tokio::test]
    async fn test_wait_for_server_shutdown_skips_terminal_context() {
        for action in [Controller::kill as fn(&Controller), Controller::stop] {
            let (client, _server) = create_tcp_pair().await;
            let controller = Arc::new(Controller::default());
            action(&controller);

            let context: Arc<dyn AsyncEngineContext> = controller;
            let result = tokio::time::timeout(
                std::time::Duration::from_millis(50),
                wait_for_server_shutdown(client, context),
            )
            .await;

            assert!(result.is_ok(), "terminal context should not wait for FIN");
            assert!(
                result.unwrap().is_ok(),
                "terminal context shutdown should succeed"
            );
        }
    }

    /// Read error in the connection monitor kills the context and skips the FIN wait.
    #[tokio::test]
    async fn test_connection_monitor_skips_fin_wait_after_read_error_kills_context() {
        let (client, mut server) = create_tcp_pair().await;
        let (read_half, write_half) = tokio::io::split(client);
        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());
        let (_bytes_tx, bytes_rx) = mpsc::channel(64);
        let (alive_tx, alive_rx) = oneshot::channel::<()>();
        let controller = Arc::new(Controller::default());

        let reader_context = controller.clone();
        let reader_task = tokio::spawn(async move {
            handle_reader(framed_reader, reader_context, alive_tx, None).await
        });
        let writer_context = controller.clone();
        let writer_task = tokio::spawn(async move {
            handle_writer(framed_writer, bytes_rx, alive_rx, writer_context).await
        });

        // Bypass the codec and write a complete but invalid TwoPartCodec
        // header. This drives the client reader into Some(Err(_)) without
        // closing the server side of the socket.
        server.write_all(&[0xFF; 24]).await.unwrap();

        let monitor_context: Arc<dyn AsyncEngineContext> = controller.clone();
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            wait_for_connection_tasks(
                reader_task,
                writer_task,
                monitor_context,
                None,
                "test-subject".to_string(),
            ),
        )
        .await;

        assert!(
            result.is_ok(),
            "connection monitor should not wait for the FIN deadline after read error"
        );
        assert!(result.unwrap().is_ok(), "connection monitor should succeed");
        assert!(
            controller.is_killed(),
            "read error should kill the stream context"
        );
    }

    /// Reader-side panic must abort the writer and return promptly rather than
    /// hanging on `tokio::join!`. Locks in the fix added with this function's
    /// sequential-await + writer-abort behavior.
    ///
    /// Setup: spawn a reader task that panics immediately (so
    /// `reader_task.await` yields `Err(JoinError::panic)`), and a writer task
    /// that parks indefinitely waiting for application bytes (so without the
    /// abort, `tokio::join!` on the previous implementation would never wake).
    /// Expect: `wait_for_connection_tasks` returns Err within the timeout.
    #[tokio::test]
    async fn test_connection_monitor_aborts_writer_when_reader_panics() {
        // Reader task that panics immediately. The explicit JoinHandle type
        // pins the inferred return type to the one wait_for_connection_tasks
        // expects; `panic!` is type `!`, which coerces to that type.
        let reader_task: tokio::task::JoinHandle<
            FramedRead<ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        > = tokio::spawn(async {
            panic!("simulated reader panic to trigger JoinError");
        });

        // Writer task that would block indefinitely waiting on application
        // bytes. Under the pre-fix `tokio::join!` implementation, this would
        // prevent the function from returning when the reader panicked.
        // After the fix, the abort drives this task to completion promptly.
        let writer_task: tokio::task::JoinHandle<
            Result<FramedWrite<WriteHalf<tokio::net::TcpStream>, TwoPartCodec>>,
        > = tokio::spawn(async {
            std::future::pending::<()>().await;
            unreachable!()
        });

        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();

        // 250 ms is generous — the abort + JoinHandle resolution should fire
        // sub-millisecond. We are checking for "doesn't hang", not "fast".
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            wait_for_connection_tasks(
                reader_task,
                writer_task,
                context,
                None,
                "test-reader-panic".to_string(),
            ),
        )
        .await;

        // Outer timeout must not fire: the abort path must surface the reader
        // JoinError before the writer would have produced any bytes.
        assert!(
            result.is_ok(),
            "wait_for_connection_tasks must return after reader panic, \
             not hang waiting on the writer"
        );

        // The inner result must be Err — the reader's JoinError propagates.
        assert!(
            result.unwrap().is_err(),
            "reader panic should propagate as Err from wait_for_connection_tasks"
        );
    }

    // ==================== handle_reader tests ====================

    struct ReaderHarness {
        framed_server: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        alive_tx: oneshot::Sender<()>,
        alive_rx: oneshot::Receiver<()>,
        controller: Arc<Controller>,
    }

    /// Creates a reusable reader harness with paired TCP streams and test channels.
    async fn reader_harness() -> ReaderHarness {
        let (client, server) = create_tcp_pair().await;
        let (read_half, _write_half) = tokio::io::split(client);
        let (_server_read, server_write) = tokio::io::split(server);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_server = FramedWrite::new(server_write, TwoPartCodec::default());
        let (alive_tx, alive_rx) = oneshot::channel::<()>();
        let controller = Arc::new(Controller::default());

        ReaderHarness {
            framed_server,
            framed_reader,
            alive_tx,
            alive_rx,
            controller,
        }
    }

    fn control_message(msg: &ControlMessage) -> TwoPartMessage {
        let msg_bytes = serde_json::to_vec(msg).unwrap();
        TwoPartMessage::from_header(Bytes::from(msg_bytes))
    }

    /// Test that handle_reader handles Stop control message by calling context.stop()
    #[tokio::test]
    async fn test_handle_reader_stop_control_message() {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;

        // Spawn the reader task
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(framed_reader, controller_clone, alive_tx, None).await
        });

        // Send Stop control message from server
        framed_server
            .send(control_message(&ControlMessage::Stop))
            .await
            .unwrap();

        // Close the framed server to signal EOF to the client
        framed_server.close().await.unwrap();

        // Wait for reader to finish
        let _ = reader_handle.await.unwrap();

        // Verify that stop was called on the controller
        assert!(
            controller.is_stopped(),
            "Controller should be stopped after receiving Stop message"
        );
    }

    /// Test that handle_reader handles Kill control message by calling context.kill()
    #[tokio::test]
    async fn test_handle_reader_kill_control_message() {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;

        // Spawn the reader task
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(framed_reader, controller_clone, alive_tx, None).await
        });

        // Send Kill control message from server
        framed_server
            .send(control_message(&ControlMessage::Kill))
            .await
            .unwrap();

        // Close the framed server to signal EOF to the client
        framed_server.close().await.unwrap();

        // Wait for reader to finish
        let _ = reader_handle.await.unwrap();

        // Verify that kill was called on the controller
        assert!(
            controller.is_killed(),
            "Controller should be killed after receiving Kill message"
        );
    }

    /// Test that handle_reader exits when alive channel is closed
    #[tokio::test]
    async fn test_handle_reader_exits_on_alive_channel_closed() {
        let ReaderHarness {
            framed_reader,
            alive_tx,
            alive_rx,
            controller,
            ..
        } = reader_harness().await;

        // Spawn the reader task
        let reader_handle =
            tokio::spawn(
                async move { handle_reader(framed_reader, controller, alive_tx, None).await },
            );

        // Drop the alive_rx to close the channel (simulating writer finishing)
        drop(alive_rx);

        // Reader should exit due to alive channel closure
        let result = reader_handle.await;

        assert!(
            result.is_ok(),
            "handle_reader should exit when alive channel is closed"
        );
    }

    /// Response-stream EOF is the normal server FIN after a closing Sentinel.
    /// It must not be counted as a cancellation by itself.
    #[tokio::test]
    async fn test_handle_reader_eof_does_not_count_cancellation() {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;
        let cancellation_counter = IntCounter::new(
            "tcp_client_reader_clean_eof_cancellations_test",
            "test cancellation counter",
        )
        .unwrap();

        // Spawn the reader task
        let counter_clone = cancellation_counter.clone();
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(
                framed_reader,
                controller_clone,
                alive_tx,
                Some(counter_clone),
            )
            .await
        });

        // Close the framed server to signal EOF to the client
        framed_server.close().await.unwrap();

        // Reader should exit due to stream closure
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), reader_handle).await;

        assert!(
            result.is_ok(),
            "handle_reader should exit when stream is closed"
        );
        assert!(
            !controller.is_stopped() && !controller.is_killed(),
            "response-stream EOF must not cancel the context"
        );
        assert_eq!(
            cancellation_counter.get(),
            0,
            "response-stream EOF must not increment the cancellation counter"
        );
    }

    /// Test that handle_reader handles multiple control messages in sequence
    #[tokio::test]
    async fn test_handle_reader_multiple_control_messages() {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;

        // Spawn the reader task
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(framed_reader, controller_clone, alive_tx, None).await
        });

        // Send multiple Stop messages (first one will stop, subsequent ones are no-ops)
        framed_server
            .send(control_message(&ControlMessage::Stop))
            .await
            .unwrap();
        framed_server
            .send(control_message(&ControlMessage::Stop))
            .await
            .unwrap();

        // Close the framed server to signal EOF to the client
        framed_server.close().await.unwrap();

        // Wait for reader to finish
        let _ = reader_handle.await.unwrap();

        // Verify that stop was called
        assert!(
            controller.is_stopped(),
            "Controller should be stopped after receiving Stop messages"
        );
    }

    /// Test handle_reader with Stop followed by Kill
    #[tokio::test]
    async fn test_handle_reader_stop_then_kill() {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;

        // Spawn the reader task
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(framed_reader, controller_clone, alive_tx, None).await
        });

        // Send Stop first, then Kill
        framed_server
            .send(control_message(&ControlMessage::Stop))
            .await
            .unwrap();
        framed_server
            .send(control_message(&ControlMessage::Kill))
            .await
            .unwrap();

        // Close the framed server to signal EOF to the client
        framed_server.close().await.unwrap();

        // Wait for reader to finish
        let _ = reader_handle.await.unwrap();

        // Verify that kill was called (which sets killed state)
        assert!(
            controller.is_killed(),
            "Controller should be killed after receiving Kill message"
        );
    }

    /// Read errors kill the context and are counted as cancellations.
    #[tokio::test]
    async fn test_handle_reader_increments_cancellation_counter_on_read_error() {
        let ReaderHarness {
            framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;
        let cancellation_counter = IntCounter::new(
            "tcp_client_reader_read_error_cancellations_test",
            "test cancellation counter",
        )
        .unwrap();

        let counter_clone = cancellation_counter.clone();
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(
                framed_reader,
                controller_clone,
                alive_tx,
                Some(counter_clone),
            )
            .await
        });

        let mut raw_writer = framed_server.into_inner();
        raw_writer.write_all(&[0u8; 8]).await.unwrap();
        raw_writer.shutdown().await.unwrap();

        let _ = reader_handle.await.unwrap();

        assert!(
            controller.is_killed(),
            "Controller should be killed after TCP stream read error"
        );
        assert_eq!(
            cancellation_counter.get(),
            1,
            "read-error close should increment cancellation metric once"
        );
    }

    /// Drives `handle_reader` against a single message and returns the
    /// controller + cancellation counter for assertions.
    async fn run_reader_with(
        msg: TwoPartMessage,
        counter_name: &str,
    ) -> (Arc<Controller>, IntCounter) {
        let ReaderHarness {
            mut framed_server,
            framed_reader,
            alive_tx,
            alive_rx: _alive_rx,
            controller,
        } = reader_harness().await;
        let counter = IntCounter::new(counter_name, "test counter").unwrap();

        let counter_clone = counter.clone();
        let controller_clone = controller.clone();
        let reader_handle = tokio::spawn(async move {
            handle_reader(
                framed_reader,
                controller_clone,
                alive_tx,
                Some(counter_clone),
            )
            .await
        });

        framed_server.send(msg).await.unwrap();
        let _ = reader_handle.await.unwrap();

        (controller, counter)
    }

    /// Each protocol-violating message variant must kill only this stream
    /// (controller killed, cancellation counted once) and never panic the
    /// worker. Covers the three non-read-error panic arms in `handle_reader`:
    /// undecodable control bytes, server-sent Sentinel, and non-control
    /// (data-only) messages.
    #[tokio::test]
    async fn test_handle_reader_kills_on_protocol_violations() {
        let cases: Vec<(&str, TwoPartMessage)> = vec![
            (
                "invalid control bytes",
                TwoPartMessage::from_header(Bytes::from_static(b"not a valid control message")),
            ),
            (
                "sentinel from server",
                control_message(&ControlMessage::Sentinel),
            ),
            (
                "non-control (data-only)",
                TwoPartMessage::from_data(Bytes::from_static(b"unexpected payload")),
            ),
        ];

        for (i, (label, msg)) in cases.into_iter().enumerate() {
            let counter_name = format!("tcp_client_reader_protocol_violation_test_{i}");
            let (controller, counter) = run_reader_with(msg, &counter_name).await;
            assert!(
                controller.is_killed(),
                "{label}: should kill stream context"
            );
            assert_eq!(counter.get(), 1, "{label}: should be counted once");
        }
    }

    // ==================== handle_request_reader tests ====================

    struct RequestReaderHarness {
        framed_server: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        bytes_tx: mpsc::Sender<Bytes>,
        bytes_rx: mpsc::Receiver<Bytes>,
        controller: Arc<Controller>,
    }

    async fn request_reader_harness() -> RequestReaderHarness {
        let (client, server) = create_tcp_pair().await;
        let (read_half, _write_half) = tokio::io::split(client);
        let (_server_read, server_write) = tokio::io::split(server);

        let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
        let framed_server = FramedWrite::new(server_write, TwoPartCodec::default());
        let (bytes_tx, bytes_rx) = mpsc::channel::<Bytes>(64);
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
