// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use core::panic;
use socket2::{Domain, SockAddr, Socket, Type};
use std::{
    collections::HashMap,
    net::{SocketAddr, TcpListener},
    os::fd::{AsFd, FromRawFd},
    sync::Arc,
};
use tokio::sync::Mutex;

use bytes::Bytes;
use derive_builder::Builder;
use futures::{SinkExt, StreamExt};
use local_ip_address::{list_afinet_netifas, local_ip};
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
    network::{
        codec::{TwoPartMessage, TwoPartMessageType},
        tcp::StreamType,
        ResponseService, ResponseStreamPrologue,
    },
    PipelineError,
};
use crate::{error, ErrorContext, Result};

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
            None => local_ip().unwrap().to_string(),
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
                error!(
                    "Failed to deserialize the first message as a valid `CallHomeHandshake`: {e}",
                )
            })?,
            None => {
                return Err(error!("Expected ControlMessage, got DataMessage"));
            }
        };

        // branch here to handle sender stream or receiver stream
        match handshake.stream_type {
            StreamType::Request => process_request_stream().await,
            StreamType::Response => {
                process_response_stream(handshake.subject, state, framed_reader, framed_writer)
                    .await
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
            .lock().await
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
            return Err(error!("The requester of the stream has been dropped before the connection was established"));
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

                            if !data.is_empty() {
                                if let Err(err) = response_tx.send(data).await {
                                    tracing::debug!("forwarding body/data message to response channel failed: {}", err);
                                    control_tx.send(ControlMessage::Kill).await.expect("the control channel should not be closed");
                                    break;
                                };
                            }
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
        let mut socket_tx = socket_tx;
        let mut control_rx = control_rx;

        while let Some(control_msg) = control_rx.recv().await {
            assert_ne!(
                control_msg,
                ControlMessage::Sentinel,
                "received sentinel message; this should never happen"
            );
            let bytes =
                serde_json::to_vec(&control_msg).expect("failed to serialize control message");
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
