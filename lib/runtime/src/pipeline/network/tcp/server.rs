// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use core::panic;
use socket2::{Domain, SockAddr, Socket, Type};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr, TcpListener},
    os::fd::{AsFd, FromRawFd},
    sync::Arc,
};
use tokio::sync::Mutex;

use bytes::Bytes;
use derive_builder::Builder;
use futures::{SinkExt, StreamExt};
use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};

use serde::{Deserialize, Serialize};
use tokio::{
    io::AsyncWriteExt,
    sync::{mpsc, oneshot},
    time,
};
use tokio_util::codec::{FramedRead, FramedWrite};

use super::{
    CallHomeHandshake, ControlMessage, PendingConnections, RegisteredStream, StreamOptions,
    StreamReceiver, StreamSender, TcpStreamConnectionInfo, TwoPartCodec,
};
use crate::engine::AsyncEngineContext;
use crate::pipeline::{
    PipelineError,
    network::{
        ResponseService, ResponseStreamPrologue,
        codec::{TwoPartMessage, TwoPartMessageType},
        tcp::StreamType,
    },
};
use anyhow::{Context, Result, anyhow as error};

// Trait for IP address resolution - allows dependency injection for testing
pub trait IpResolver {
    fn local_ip(&self) -> Result<std::net::IpAddr, Error>;
    fn local_ipv6(&self) -> Result<std::net::IpAddr, Error>;
}

// Default implementation using the real local_ip_address crate
pub struct DefaultIpResolver;

impl IpResolver for DefaultIpResolver {
    fn local_ip(&self) -> Result<std::net::IpAddr, Error> {
        local_ip()
    }

    fn local_ipv6(&self) -> Result<std::net::IpAddr, Error> {
        local_ipv6()
    }
}

#[allow(dead_code)]
type ResponseType = TwoPartMessage;

#[derive(Debug, Serialize, Deserialize, Clone, Builder, Default)]
pub struct ServerOptions {
    #[builder(default = "0")]
    pub port: u16,

    #[builder(default)]
    pub interface: Option<String>,
}

impl ServerOptions {
    pub fn builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }
}

/// A [`TcpStreamServer`] is a TCP service that listens on a port for incoming response connections.
/// A Response connection is a connection that is established by a client with the intention of sending
/// specific data back to the server.
pub struct TcpStreamServer {
    local_ip: String,
    local_port: u16,
    state: Arc<Mutex<State>>,
}

// pub struct TcpStreamReceiver {
//     address: TcpStreamConnectionInfo,
//     state: Arc<Mutex<State>>,
//     rx: mpsc::Receiver<ResponseType>,
// }

#[allow(dead_code)]
struct RequestedSendConnection {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamSender, String>>,
}

struct RequestedRecvConnection {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamReceiver, String>>,
}

// /// When registering a new TcpStream on the server, the registration method will return a [`Connections`] object.
// /// This [`Connections`] object will have two [`oneshot::Receiver`] objects, one for the [`TcpStreamSender`] and one for the [`TcpStreamReceiver`].
// /// The [`Connections`] object can be awaited to get the [`TcpStreamSender`] and [`TcpStreamReceiver`] objects; these objects will
// /// be made available when the matching Client has connected to the server.
// pub struct Connections {
//     pub address: TcpStreamConnectionInfo,

//     /// The [`oneshot::Receiver`] for the [`TcpStreamSender`]. Awaiting this object will return the [`TcpStreamSender`] object once
//     /// the client has connected to the server.
//     pub sender: Option<oneshot::Receiver<StreamSender>>,

//     /// The [`oneshot::Receiver`] for the [`TcpStreamReceiver`]. Awaiting this object will return the [`TcpStreamReceiver`] object once
//     /// the client has connected to the server.
//     pub receiver: Option<oneshot::Receiver<StreamReceiver>>,
// }

#[derive(Default)]
struct State {
    tx_subjects: HashMap<String, RequestedSendConnection>,
    rx_subjects: HashMap<String, RequestedRecvConnection>,
    handle: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl TcpStreamServer {
    pub fn options_builder() -> ServerOptionsBuilder {
        ServerOptionsBuilder::default()
    }

    pub async fn new(options: ServerOptions) -> Result<Arc<Self>, PipelineError> {
        Self::new_with_resolver(options, DefaultIpResolver).await
    }

    pub async fn new_with_resolver<R: IpResolver>(
        options: ServerOptions,
        resolver: R,
    ) -> Result<Arc<Self>, PipelineError> {
        let local_ip = match options.interface {
            Some(interface) => {
                let interfaces: HashMap<String, std::net::IpAddr> =
                    list_afinet_netifas()?.into_iter().collect();

                interfaces
                    .get(&interface)
                    .ok_or(PipelineError::Generic(format!(
                        "Interface not found: {}",
                        interface
                    )))?
                    .to_string()
            }
            None => {
                let resolved_ip = resolver.local_ip().or_else(|err| match err {
                    Error::LocalIpAddressNotFound => resolver.local_ipv6(),
                    _ => Err(err),
                });

                match resolved_ip {
                    Ok(addr) => addr,
                    Err(Error::LocalIpAddressNotFound) => IpAddr::from([127, 0, 0, 1]),
                    Err(err) => return Err(err.into()),
                }
                .to_string()
            }
        };

        let state = Arc::new(Mutex::new(State::default()));

        let local_port = Self::start(local_ip.clone(), options.port, state.clone())
            .await
            .map_err(|e| {
                PipelineError::Generic(format!("Failed to start TcpStreamServer: {}", e))
            })?;

        tracing::debug!("tcp transport service on {local_ip}:{local_port}");

        Ok(Arc::new(Self {
            local_ip,
            local_port,
            state,
        }))
    }

    #[allow(clippy::await_holding_lock)]
    async fn start(local_ip: String, local_port: u16, state: Arc<Mutex<State>>) -> Result<u16> {
        let addr = format!("{}:{}", local_ip, local_port);
        let state_clone = state.clone();
        let mut guard = state.lock().await;
        if guard.handle.is_some() {
            panic!("TcpStreamServer already started");
        }
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<u16>>();
        let handle = tokio::spawn(tcp_listener(addr, state_clone, ready_tx));
        guard.handle = Some(handle);
        drop(guard);
        let local_port = ready_rx.await??;
        Ok(local_port)
    }
}

// todo - possible rename ResponseService to ResponseServer
#[async_trait::async_trait]
impl ResponseService for TcpStreamServer {
    /// Register a new subject and sender with the response subscriber
    /// Produces an RAII object that will deregister the subject when dropped
    ///
    /// we need to register both data in and data out entries
    /// there might be forward pipeline that want to consume the data out stream
    /// and there might be a response stream that wants to consume the data in stream
    /// on registration, we need to specific if we want data-in, data-out or both
    /// this will map to the type of service that is runniing, i.e. Single or Many In //
    /// Single or Many Out
    ///
    /// todo(ryan) - return a connection object that can be awaited. when successfully connected,
    /// can ask for the sender and receiver
    ///
    /// OR
    ///
    /// we make it into register sender and register receiver, both would return a connection object
    /// and when a connection is established, we'd get the respective sender or receiver
    ///
    /// the registration probably needs to be done in one-go, so we should use a builder object for
    /// requesting a receiver and optional sender
    async fn register(&self, options: StreamOptions) -> PendingConnections {
        // oneshot channels to pass back the sender and receiver objects

        let address = format!("{}:{}", self.local_ip, self.local_port);
        tracing::debug!("Registering new TcpStream on {}", address);

        let send_stream = if options.enable_request_stream {
            let sender_subject = uuid::Uuid::new_v4().to_string();

            let (pending_sender_tx, pending_sender_rx) = oneshot::channel();

            let connection_info = RequestedSendConnection {
                context: options.context.clone(),
                connection: pending_sender_tx,
            };

            let mut state = self.state.lock().await;
            state
                .tx_subjects
                .insert(sender_subject.clone(), connection_info);

            let registered_stream = RegisteredStream {
                connection_info: TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: sender_subject.clone(),
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Request,
                }
                .into(),
                stream_provider: pending_sender_rx,
            };

            Some(registered_stream)
        } else {
            None
        };

        let recv_stream = if options.enable_response_stream {
            let (pending_recver_tx, pending_recver_rx) = oneshot::channel();
            let receiver_subject = uuid::Uuid::new_v4().to_string();

            let connection_info = RequestedRecvConnection {
                context: options.context.clone(),
                connection: pending_recver_tx,
            };

            let mut state = self.state.lock().await;
            state
                .rx_subjects
                .insert(receiver_subject.clone(), connection_info);

            let registered_stream = RegisteredStream {
                connection_info: TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: receiver_subject.clone(),
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Response,
                }
                .into(),
                stream_provider: pending_recver_rx,
            };

            Some(registered_stream)
        } else {
            None
        };

        PendingConnections {
            send_stream,
            recv_stream,
        }
    }
}

// this method listens on a tcp port for incoming connections
// new connections are expected to send a protocol specific handshake
// for us to determine the subject they are interested in, in this case,
// we expect the first message to be [`FirstMessage`] from which we find
// the sender, then we spawn a task to forward all bytes from the tcp stream
// to the sender
async fn tcp_listener(
    addr: String,
    state: Arc<Mutex<State>>,
    read_tx: tokio::sync::oneshot::Sender<Result<u16>>,
) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start TcpListender on {}: {}", addr, e));

    let listener = match listener {
        Ok(listener) => {
            let addr = listener
                .local_addr()
                .map_err(|e| anyhow::anyhow!("Failed get SocketAddr: {:?}", e))
                .unwrap();

            read_tx
                .send(Ok(addr.port()))
                .expect("Failed to send ready signal");

            listener
        }
        Err(e) => {
            read_tx.send(Err(e)).expect("Failed to send ready signal");
            return Err(anyhow::anyhow!("Failed to start TcpListender on {}", addr));
        }
    };

    loop {
        // todo - add instrumentation
        // todo - add counter for all accepted connections
        // todo - add gauge for all inflight connections
        // todo - add counter for incoming bytes
        // todo - add counter for outgoing bytes
        let (stream, _addr) = match listener.accept().await {
            Ok((stream, _addr)) => (stream, _addr),
            Err(e) => {
                // the client should retry, so we don't need to abort
                tracing::warn!("failed to accept tcp connection: {}", e);
                eprintln!("failed to accept tcp connection: {}", e);
                continue;
            }
        };

        match stream.set_nodelay(true) {
            Ok(_) => (),
            Err(e) => {
                tracing::warn!("failed to set tcp stream to nodelay: {}", e);
            }
        }

        match stream.set_linger(Some(std::time::Duration::from_secs(0))) {
            Ok(_) => (),
            Err(e) => {
                tracing::warn!("failed to set tcp stream to linger: {}", e);
            }
        }

        tokio::spawn(handle_connection(stream, state.clone()));
    }
}

// #[instrument(level = "trace"), skip(state)]
// todo - clone before spawn and trace process_stream
async fn handle_connection(stream: tokio::net::TcpStream, state: Arc<Mutex<State>>) {
    let result = process_stream(stream, state).await;
    match result {
        Ok(_) => tracing::trace!("successfully processed tcp connection"),
        Err(e) => {
            tracing::warn!("failed to handle tcp connection: {}", e);
            #[cfg(debug_assertions)]
            eprintln!("failed to handle tcp connection: {}", e);
        }
    }
}

/// This method is responsible for the internal tcp stream handshake
/// The handshake will specialize the stream as a request/sender or response/receiver stream
async fn process_stream(stream: tokio::net::TcpStream, state: Arc<Mutex<State>>) -> Result<()> {
    // split the socket in to a reader and writer
    let (read_half, write_half) = tokio::io::split(stream);

    // attach the codec to the reader and writer to get framed readers and writers
    let mut framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
    let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());

    // the internal tcp [`CallHomeHandshake`] connects the socket to the requester
    // here we await this first message as a raw bytes two part message
    let first_message = framed_reader
        .next()
        .await
        .ok_or(error!("Connection closed without a ControlMessage"))??;

    // we await on the raw bytes which should come in as a header only message
    // todo - improve error handling - check for no data
    let handshake: CallHomeHandshake = match first_message.header() {
        Some(header) => serde_json::from_slice(header).map_err(|e| {
            error!("Failed to deserialize the first message as a valid `CallHomeHandshake`: {e}",)
        })?,
        None => {
            return Err(error!("Expected ControlMessage, got DataMessage"));
        }
    };

    // branch here to handle sender stream or receiver stream
    match handshake.stream_type {
        StreamType::Request => process_request_stream().await,
        StreamType::Response => {
            process_response_stream(handshake.subject, state, framed_reader, framed_writer).await
        }
    }
}

async fn process_request_stream() -> Result<()> {
    Ok(())
}

async fn process_response_stream(
    subject: String,
    state: Arc<Mutex<State>>,
    mut reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
) -> Result<()> {
    let response_stream = state
        .lock()
        .await
        .rx_subjects
        .remove(&subject)
        .ok_or(error!("Subject not found: {}; upstream publisher specified a subject unknown to the downsteam subscriber", subject))?;

    // unwrap response_stream
    let RequestedRecvConnection {
        context,
        connection,
    } = response_stream;

    // the [`Prologue`]
    // there must be a second control message it indicate the other segment's generate method was successful
    let prologue = reader
        .next()
        .await
        .ok_or(error!("Connection closed without a ControlMessge"))??;

    // deserialize prologue
    let prologue = match prologue.into_message_type() {
        TwoPartMessageType::HeaderOnly(header) => {
            let prologue: ResponseStreamPrologue = serde_json::from_slice(&header)
                .map_err(|e| error!("Failed to deserialize ControlMessage: {}", e))?;
            prologue
        }
        _ => {
            panic!("Expected HeaderOnly ControlMessage; internally logic error")
        }
    };

    // await the control message of GTG or Error, if error, then connection.send(Err(String)), which should fail the
    // generate call chain
    //
    // note: this second control message might be delayed, but the expensive part of setting up the connection
    // is both complete and ready for data flow; awaiting here is not a performance hit or problem and it allows
    // us to trace the initial setup time vs the time to prologue
    if let Some(error) = &prologue.error {
        let _ = connection.send(Err(error.clone()));
        return Err(error!("Received error prologue: {}", error));
    }

    // we need to know the buffer size from the registration options; add this to the RequestRecvConnection object
    let (response_tx, response_rx) = mpsc::channel(64);

    if connection
        .send(Ok(crate::pipeline::network::StreamReceiver {
            rx: response_rx,
        }))
        .is_err()
    {
        return Err(error!(
            "The requester of the stream has been dropped before the connection was established"
        ));
    }

    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(1);

    // sender task
    // issues control messages to the sender and when finished shuts down the socket
    // this should be the last task to finish and must
    let send_task = tokio::spawn(network_send_handler(writer, control_rx));

    // forward task
    let recv_task = tokio::spawn(network_receive_handler(
        reader,
        response_tx,
        control_tx,
        context.clone(),
    ));

    // check the results of each of the tasks
    let (monitor_result, forward_result) = tokio::join!(send_task, recv_task);

    monitor_result?;
    forward_result?;

    Ok(())
}

async fn network_receive_handler(
    mut framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
    response_tx: mpsc::Sender<Bytes>,
    control_tx: mpsc::Sender<ControlMessage>,
    context: Arc<dyn AsyncEngineContext>,
) {
    // loop over reading the tcp stream and checking if the writer is closed
    let mut can_stop = true;
    loop {
        tokio::select! {
            biased;

            _ = response_tx.closed() => {
                tracing::trace!("response channel closed before the client finished writing data");
                control_tx.send(ControlMessage::Kill).await.expect("the control channel should not be closed");
                break;
            }

            _ = context.killed() => {
                tracing::trace!("context kill signal received; shutting down");
                control_tx.send(ControlMessage::Kill).await.expect("the control channel should not be closed");
                break;
            }

            _ = context.stopped(), if can_stop => {
                tracing::trace!("context stop signal received; shutting down");
                can_stop = false;
                control_tx.send(ControlMessage::Stop).await.expect("the control channel should not be closed");
            }

            msg = framed_reader.next() => {
                match msg {
                    Some(Ok(msg)) => {
                        let (header, data) = msg.into_parts();

                        // received a control message
                        if !header.is_empty() {
                            match process_control_message(header) {
                                Ok(ControlAction::Continue) => {}
                                Ok(ControlAction::Shutdown) => {
                                    assert!(data.is_empty(), "received sentinel message with data; this should never happen");
                                    tracing::trace!("received sentinel message; shutting down");
                                    break;
                                }
                                Err(e) => {
                                    // TODO(#171) - address fatal errors
                                    panic!("{:?}", e);
                                }
                            }
                        }

                        if !data.is_empty()
                            && let Err(err) = response_tx.send(data).await {
                                tracing::debug!("forwarding body/data message to response channel failed: {}", err);
                                control_tx.send(ControlMessage::Kill).await.expect("the control channel should not be closed");
                                break;
                            };
                    }
                    Some(Err(_)) => {
                        // TODO(#171) - address fatal errors
                        panic!("invalid message issued over socket; this should never happen");
                    }
                    None => {
                        // this is allowed but we try to avoid it
                        // the logic is that the client will tell us when its is done and the server
                        // will close the connection naturally when the sentinel message is received
                        // the client closing early represents a transport error outside the control of the
                        // transport library
                        tracing::trace!("tcp stream was closed by client");
                        break;
                    }
                }
            }

        }
    }
}

async fn network_send_handler(
    socket_tx: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
    control_rx: mpsc::Receiver<ControlMessage>,
) {
    let mut socket_tx: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec> = socket_tx;
    let mut control_rx = control_rx;

    while let Some(control_msg) = control_rx.recv().await {
        assert_ne!(
            control_msg,
            ControlMessage::Sentinel,
            "received sentinel message; this should never happen"
        );
        let bytes = serde_json::to_vec(&control_msg).expect("failed to serialize control message");
        let message = TwoPartMessage::from_header(bytes.into());
        match socket_tx.send(message).await {
            Ok(_) => tracing::debug!("issued control message {control_msg:?} to sender"),
            Err(_) => {
                tracing::debug!("failed to send control message {control_msg:?} to sender")
            }
        }
    }

    let mut inner = socket_tx.into_inner();
    if let Err(e) = inner.flush().await {
        tracing::debug!("failed to flush socket: {}", e);
    }
    if let Err(e) = inner.shutdown().await {
        tracing::debug!("failed to shutdown socket: {}", e);
    }
}

enum ControlAction {
    Continue,
    Shutdown,
}

fn process_control_message(message: Bytes) -> Result<ControlAction> {
    match serde_json::from_slice::<ControlMessage>(&message)? {
        ControlMessage::Sentinel => {
            // the client issued a sentinel message
            // it has finished writing data and is now awaiting the server to close the connection
            tracing::trace!("sentinel received; shutting down");
            Ok(ControlAction::Shutdown)
        }
        ControlMessage::Kill | ControlMessage::Stop => {
            // TODO(#171) - address fatal errors
            anyhow::bail!(
                "fatal error - unexpected control message received - this should never happen"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AsyncEngineContextProvider;
    use crate::pipeline::Context;
    use crate::pipeline::context::Controller;
    use crate::pipeline::network::ControlMessage;
    use crate::pipeline::network::codec::TwoPartMessage;
    use crate::pipeline::network::tcp::test_utils::create_tcp_pair;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // ==================== Test Setup Helpers ====================

    /// Test fixture for network_send_handler tests
    struct SendHandlerTestSetup {
        server: tokio::net::TcpStream,
        framed_writer: FramedWrite<tokio::io::WriteHalf<tokio::net::TcpStream>, TwoPartCodec>,
        control_tx: mpsc::Sender<ControlMessage>,
        control_rx: mpsc::Receiver<ControlMessage>,
    }

    impl SendHandlerTestSetup {
        async fn new() -> Self {
            let (client, server) = create_tcp_pair().await;
            let (_, write_half) = tokio::io::split(client);
            let framed_writer = FramedWrite::new(write_half, TwoPartCodec::default());
            let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(1);

            Self {
                server,
                framed_writer,
                control_tx,
                control_rx,
            }
        }

        /// Spawn the handler and return components needed for testing
        fn spawn_handler(
            self,
        ) -> (
            tokio::task::JoinHandle<()>,
            tokio::net::TcpStream,
            mpsc::Sender<ControlMessage>,
        ) {
            let handle = tokio::spawn(network_send_handler(self.framed_writer, self.control_rx));
            (handle, self.server, self.control_tx)
        }
    }

    /// Test fixture for network_receive_handler tests
    struct ReceiveHandlerTestSetup {
        server_write: tokio::io::WriteHalf<tokio::net::TcpStream>,
        framed_reader: FramedRead<tokio::io::ReadHalf<tokio::net::TcpStream>, TwoPartCodec>,
        response_tx: mpsc::Sender<Bytes>,
        response_rx: mpsc::Receiver<Bytes>,
        control_tx: mpsc::Sender<ControlMessage>,
        control_rx: mpsc::Receiver<ControlMessage>,
        controller: Arc<Controller>,
    }

    impl ReceiveHandlerTestSetup {
        async fn new() -> Self {
            Self::with_response_channel_size(10).await
        }

        async fn with_response_channel_size(size: usize) -> Self {
            let (client, server) = create_tcp_pair().await;
            let (read_half, _write_half) = tokio::io::split(client);
            let (_server_read, server_write) = tokio::io::split(server);

            let framed_reader = FramedRead::new(read_half, TwoPartCodec::default());
            let (response_tx, response_rx) = mpsc::channel::<Bytes>(size);
            let (control_tx, control_rx) = mpsc::channel::<ControlMessage>(10);
            let controller = Arc::new(Controller::default());

            Self {
                server_write,
                framed_reader,
                response_tx,
                response_rx,
                control_tx,
                control_rx,
                controller,
            }
        }

        /// Spawn the handler and return components needed for testing
        fn spawn_handler(
            self,
        ) -> (
            tokio::task::JoinHandle<()>,
            tokio::io::WriteHalf<tokio::net::TcpStream>,
            mpsc::Receiver<Bytes>,
            mpsc::Receiver<ControlMessage>,
            Arc<Controller>,
        ) {
            let handle = tokio::spawn(network_receive_handler(
                self.framed_reader,
                self.response_tx,
                self.control_tx,
                self.controller.clone(),
            ));
            (
                handle,
                self.server_write,
                self.response_rx,
                self.control_rx,
                self.controller,
            )
        }
    }

    /// Helper to encode a control message to bytes using the TwoPartCodec
    fn encode_control_message(msg: &ControlMessage) -> Bytes {
        let msg_bytes = serde_json::to_vec(msg).unwrap();
        let two_part_msg = TwoPartMessage::from_header(Bytes::from(msg_bytes));
        TwoPartCodec::default().encode_message(two_part_msg).unwrap()
    }

    /// Helper to encode a data message to bytes using the TwoPartCodec
    fn encode_data_message(data: &[u8]) -> Bytes {
        let two_part_msg = TwoPartMessage::from_data(Bytes::from(data.to_vec()));
        TwoPartCodec::default().encode_message(two_part_msg).unwrap()
    }

    // ==================== Mock Resolvers ====================

    // Mock resolver that always fails to simulate the fallback scenario
    struct FailingIpResolver;

    impl IpResolver for FailingIpResolver {
        fn local_ip(&self) -> Result<std::net::IpAddr, Error> {
            Err(Error::LocalIpAddressNotFound)
        }

        fn local_ipv6(&self) -> Result<std::net::IpAddr, Error> {
            Err(Error::LocalIpAddressNotFound)
        }
    }

    #[tokio::test]
    async fn test_tcp_stream_server_default_behavior() {
        // Test that TcpStreamServer::new works with default options
        // This verifies normal operation when IP detection succeeds
        let options = ServerOptions::default();
        let result = TcpStreamServer::new(options).await;

        assert!(
            result.is_ok(),
            "TcpStreamServer::new should succeed with default options"
        );

        let server = result.unwrap();

        // Verify the server can be used by registering a stream
        let context = Context::new(());
        let stream_options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending_connection = server.register(stream_options).await;

        // Verify connection info is available and valid
        let connection_info = pending_connection
            .recv_stream
            .as_ref()
            .unwrap()
            .connection_info
            .clone();

        let tcp_info: TcpStreamConnectionInfo = connection_info.try_into().unwrap();
        let socket_addr = tcp_info.address.parse::<std::net::SocketAddr>().unwrap();

        // Should have a valid port assigned
        assert!(
            socket_addr.port() > 0,
            "Server should be assigned a valid port number"
        );

        println!(
            "Server created successfully with address: {}",
            tcp_info.address
        );
    }

    #[tokio::test]
    async fn test_tcp_stream_server_fallback_to_loopback() {
        // Test fallback behavior using a mock resolver that always fails
        // This guarantees the fallback logic is triggered

        let options = ServerOptions::builder().port(0).build().unwrap();

        // Use the failing resolver to force the fallback
        let result = TcpStreamServer::new_with_resolver(options, FailingIpResolver).await;
        assert!(
            result.is_ok(),
            "Server creation should succeed with fallback even when IP detection fails"
        );

        let server = result.unwrap();

        // Get the actual bound address by registering a stream
        let context = Context::new(());
        let stream_options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending_connection = server.register(stream_options).await;
        let connection_info = pending_connection
            .recv_stream
            .as_ref()
            .unwrap()
            .connection_info
            .clone();

        let tcp_info: TcpStreamConnectionInfo = connection_info.try_into().unwrap();
        let socket_addr = tcp_info.address.parse::<std::net::SocketAddr>().unwrap();

        // With the failing resolver, fallback should ALWAYS be used
        let ip = socket_addr.ip();
        assert!(
            ip.is_loopback(),
            "Should use loopback when IP detection fails"
        );

        // Verify it's specifically 127.0.0.1 (the fallback value from the patch)
        assert_eq!(
            ip,
            std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            "Fallback should use exactly 127.0.0.1, got: {}",
            ip
        );

        println!("SUCCESS: Fallback to 127.0.0.1 was confirmed: {}", ip);

        // The server should work with the fallback IP
        assert!(socket_addr.port() > 0, "Server should have a valid port");
    }

    /// Test that network_send_handler sends control messages over the socket
    #[tokio::test]
    async fn test_network_send_handler_sends_control_messages() {
        let setup = SendHandlerTestSetup::new().await;
        let (handle, mut server, control_tx) = setup.spawn_handler();

        // Send a Kill message
        control_tx.send(ControlMessage::Kill).await.unwrap();
        drop(control_tx);

        // Wait for handler to complete
        handle.await.unwrap();

        // Read from server side to verify message was sent
        let mut buffer = vec![0u8; 1024];
        let n = server.read(&mut buffer).await.unwrap();

        // Buffer should contain the Kill message
        assert!(n > 0, "Expected control message to be written to the TCP stream");

        let kill_json = serde_json::to_vec(&ControlMessage::Kill).unwrap();
        assert!(
            buffer[..n]
                .windows(kill_json.len())
                .any(|w| w == kill_json.as_slice()),
            "Buffer should contain Kill message. Buffer: {:?}",
            String::from_utf8_lossy(&buffer[..n])
        );
    }

    /// Test that network_send_handler sends Stop message correctly
    #[tokio::test]
    async fn test_network_send_handler_sends_stop_message() {
        let setup = SendHandlerTestSetup::new().await;
        let (handle, mut server, control_tx) = setup.spawn_handler();

        // Send a Stop message
        control_tx.send(ControlMessage::Stop).await.unwrap();
        drop(control_tx);

        // Wait for handler to complete
        handle.await.unwrap();

        // Read from server side
        let mut buffer = vec![0u8; 1024];
        let n = server.read(&mut buffer).await.unwrap();

        let stop_json = serde_json::to_vec(&ControlMessage::Stop).unwrap();
        assert!(
            buffer[..n]
                .windows(stop_json.len())
                .any(|w| w == stop_json.as_slice()),
            "Buffer should contain Stop message"
        );
    }

    /// Test that network_send_handler shuts down cleanly when channel closes
    #[tokio::test]
    async fn test_network_send_handler_shuts_down_on_channel_close() {
        let setup = SendHandlerTestSetup::new().await;
        let (handle, _server, control_tx) = setup.spawn_handler();

        // Immediately close the channel without sending anything
        drop(control_tx);

        // Handler should complete without panicking
        let result = handle.await;
        assert!(result.is_ok(), "Handler should complete successfully when channel closes");
    }

    /// Test that network_send_handler panics on Sentinel (which should never be sent)
    #[tokio::test]
    #[should_panic(expected = "received sentinel message")]
    async fn test_network_send_handler_panics_on_sentinel() {
        let setup = SendHandlerTestSetup::new().await;

        // Send a Sentinel message (which should cause a panic)
        setup.control_tx.send(ControlMessage::Sentinel).await.unwrap();

        // This should panic
        network_send_handler(setup.framed_writer, setup.control_rx).await;
    }

    // ==================== network_receive_handler tests ====================

    /// Test that network_receive_handler forwards data messages to response_tx
    #[tokio::test]
    async fn test_network_receive_handler_forwards_data() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, mut response_rx, mut control_rx, _controller) =
            setup.spawn_handler();

        // Send data message from server
        let test_data = b"hello world";
        let encoded = encode_data_message(test_data);
        server_write.write_all(&encoded).await.unwrap();
        server_write.flush().await.unwrap();

        // Send sentinel to gracefully close
        let sentinel = encode_control_message(&ControlMessage::Sentinel);
        server_write.write_all(&sentinel).await.unwrap();
        server_write.flush().await.unwrap();
        server_write.shutdown().await.unwrap();

        // Wait for handler to complete
        handle.await.unwrap();

        // Verify data was forwarded
        let received = response_rx.recv().await.unwrap();
        assert_eq!(&received[..], test_data);

        // No control messages should have been sent
        assert!(control_rx.try_recv().is_err());
    }

    /// Test that network_receive_handler handles sentinel and shuts down
    #[tokio::test]
    async fn test_network_receive_handler_sentinel_shutdown() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, _response_rx, _control_rx, _controller) =
            setup.spawn_handler();

        // Send sentinel message
        let sentinel = encode_control_message(&ControlMessage::Sentinel);
        server_write.write_all(&sentinel).await.unwrap();
        server_write.flush().await.unwrap();
        server_write.shutdown().await.unwrap();

        // Handler should complete without error
        let result = handle.await;
        assert!(result.is_ok(), "Handler should complete successfully on sentinel");
    }

    /// Test that network_receive_handler sends Kill when response_tx is closed
    #[tokio::test]
    async fn test_network_receive_handler_response_channel_closed() {
        let mut setup = ReceiveHandlerTestSetup::new().await;

        // Drop the response receiver to close the channel before spawning
        drop(setup.response_rx);

        // Spawn the receive handler (response_rx already dropped)
        let handle = tokio::spawn(network_receive_handler(
            setup.framed_reader,
            setup.response_tx,
            setup.control_tx,
            setup.controller,
        ));

        // Handler should complete
        handle.await.unwrap();

        // Should have sent Kill message
        let control_msg = setup.control_rx.recv().await.unwrap();
        assert!(
            matches!(control_msg, ControlMessage::Kill),
            "Expected Kill message when response channel is closed"
        );
    }

    /// Test that network_receive_handler sends Kill when context is killed
    #[tokio::test]
    async fn test_network_receive_handler_context_killed() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, _server_write, _response_rx, mut control_rx, controller) =
            setup.spawn_handler();

        // Kill the context
        controller.kill();

        // Handler should complete
        handle.await.unwrap();

        // Should have sent Kill message
        let control_msg = control_rx.recv().await.unwrap();
        assert!(
            matches!(control_msg, ControlMessage::Kill),
            "Expected Kill message when context is killed"
        );
    }

    /// Test that network_receive_handler sends Stop when context is stopped
    #[tokio::test]
    async fn test_network_receive_handler_context_stopped() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, _response_rx, mut control_rx, controller) =
            setup.spawn_handler();

        // Stop the context
        controller.stop();

        // recv() blocks until message arrives
        let control_msg = control_rx.recv().await.expect("Control channel closed unexpectedly");
        assert!(
            matches!(control_msg, ControlMessage::Stop),
            "Expected Stop message when context is stopped"
        );

        // Send sentinel to allow graceful shutdown
        let sentinel = encode_control_message(&ControlMessage::Sentinel);
        server_write.write_all(&sentinel).await.unwrap();
        server_write.flush().await.unwrap();
        server_write.shutdown().await.unwrap();

        // Handler should complete
        handle.await.unwrap();
    }

    /// Test that network_receive_handler handles TCP stream closed by client
    #[tokio::test]
    async fn test_network_receive_handler_tcp_stream_closed() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, _response_rx, _control_rx, _controller) =
            setup.spawn_handler();

        // Close the server side (simulates client closing connection)
        server_write.shutdown().await.unwrap();
        drop(server_write);

        // Handler should complete without panicking
        let result = handle.await;
        assert!(result.is_ok(), "Handler should complete when TCP stream is closed");
    }

    /// Test that network_receive_handler forwards multiple data messages
    #[tokio::test]
    async fn test_network_receive_handler_multiple_data_messages() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, mut response_rx, _control_rx, _controller) =
            setup.spawn_handler();

        // Send multiple data messages
        let messages = [b"message 1".as_slice(), b"message 2", b"message 3"];
        for msg in &messages {
            let encoded = encode_data_message(msg);
            server_write.write_all(&encoded).await.unwrap();
        }
        server_write.flush().await.unwrap();

        // Send sentinel to gracefully close
        let sentinel = encode_control_message(&ControlMessage::Sentinel);
        server_write.write_all(&sentinel).await.unwrap();
        server_write.flush().await.unwrap();
        server_write.shutdown().await.unwrap();

        // Wait for handler to complete
        handle.await.unwrap();

        // Verify all messages were forwarded in order
        for expected in &messages {
            let received = response_rx.recv().await.unwrap();
            assert_eq!(&received[..], *expected);
        }
    }

    /// Test that network_receive_handler sends Kill when data forwarding fails
    #[tokio::test]
    async fn test_network_receive_handler_data_forward_failure() {
        let mut setup = ReceiveHandlerTestSetup::with_response_channel_size(1).await;

        // Drop the response receiver to cause send failures
        drop(setup.response_rx);

        // Spawn the receive handler
        let handle = tokio::spawn(network_receive_handler(
            setup.framed_reader,
            setup.response_tx,
            setup.control_tx,
            setup.controller,
        ));

        // Send data message - this should fail to forward
        let test_data = b"test data";
        let encoded = encode_data_message(test_data);
        setup.server_write.write_all(&encoded).await.unwrap();
        setup.server_write.flush().await.unwrap();
        setup.server_write.shutdown().await.unwrap();

        // Wait for handler to complete
        handle.await.unwrap();

        // Should have sent Kill message due to forward failure
        let control_msg = setup.control_rx.recv().await.unwrap();
        assert!(
            matches!(control_msg, ControlMessage::Kill),
            "Expected Kill message when data forwarding fails"
        );
    }

    /// Test that network_receive_handler only sends Stop once even if context.stopped() is called multiple times
    #[tokio::test]
    async fn test_network_receive_handler_stop_sent_only_once() {
        let setup = ReceiveHandlerTestSetup::new().await;
        let (handle, mut server_write, _response_rx, mut control_rx, controller) =
            setup.spawn_handler();

        // Stop the context
        controller.stop();

        // recv() blocks until message arrives
        let control_msg = control_rx.recv().await.expect("Control channel closed unexpectedly");
        assert!(matches!(control_msg, ControlMessage::Stop));

        // Send sentinel to allow graceful shutdown
        let sentinel = encode_control_message(&ControlMessage::Sentinel);
        server_write.write_all(&sentinel).await.unwrap();
        server_write.flush().await.unwrap();
        server_write.shutdown().await.unwrap();

        // Handler should complete
        handle.await.unwrap();

        // No additional control messages should be present (Stop should only be sent once)
        assert!(
            control_rx.try_recv().is_err(),
            "Stop should only be sent once"
        );
    }
}
