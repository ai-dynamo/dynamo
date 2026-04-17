// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use core::panic;
use socket2::{Domain, SockAddr, Socket, Type};
use std::{
    collections::{HashMap, HashSet},
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
    /// Maps subject UUID -> (endpoint_name, instance_id) for reverse lookup
    /// during cleanup.  The endpoint name is included so that cancellation is
    /// scoped to the specific endpoint that was removed, not to every endpoint
    /// registered on the same runtime (which share a connection_id / instance_id).
    subject_instance: HashMap<String, (String, u64)>,
    /// Maps (endpoint_name, instance_id) -> set of subject UUIDs for batch
    /// cancellation when the discovery plane reports the instance is gone.
    instance_subjects: HashMap<(String, u64), HashSet<String>>,
    /// Tombstone set keyed by (endpoint_name, instance_id). Closes the race
    /// window where `cancel_instance_streams()` fires before `associate_instance()`
    /// for the same request. Cleared by `clear_instance_tombstone()` when the
    /// instance reappears in the discovery set.
    removed_instances: HashSet<(String, u64)>,
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
                    // Only fall back to loopback when no routable IP exists at all;
                    // propagate other resolver errors (I/O, platform) so
                    // misconfigured hosts fail fast instead of silently binding
                    // to 127.0.0.1.
                    Err(Error::LocalIpAddressNotFound) => {
                        tracing::warn!(
                            "No routable local IP address found; falling back to 127.0.0.1"
                        );
                        IpAddr::from([127, 0, 0, 1])
                    }
                    Err(err) => {
                        return Err(PipelineError::Generic(format!(
                            "Failed to resolve local IP address: {err}"
                        )));
                    }
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

    /// Associate a registered response-stream subject with a backend instance.
    ///
    /// Called by the egress router after `register()` so that when the discovery
    /// plane reports the instance as removed, we can cancel all pending subjects
    /// for that instance in one shot.
    ///
    /// If the instance has already been removed (tombstoned by a prior
    /// `cancel_instance_streams` call), the subject is immediately cancelled
    /// instead of being tracked. This closes the race window where the
    /// discovery watcher fires before the request path calls this method.
    /// Returns `true` if the subject was tracked normally, `false` if the
    /// instance was tombstoned and the subject was immediately cancelled.
    /// When `false` is returned the caller should skip `send_request` and
    /// return a migratable `Disconnected` error directly.
    ///
    /// The `endpoint` parameter scopes the key to a specific endpoint name
    /// (e.g. `"generate"`), preventing cross-endpoint cancellation when
    /// multiple endpoints on the same runtime share a `connection_id`.
    pub async fn associate_instance(
        &self,
        subject: &str,
        endpoint: &str,
        instance_id: u64,
    ) -> bool {
        let key = (endpoint.to_string(), instance_id);
        let mut state = self.state.lock().await;
        if state.removed_instances.contains(&key) {
            // Instance was already removed -- cancel immediately.
            tracing::warn!(
                subject,
                endpoint,
                instance_id,
                "Cancelling subject immediately: instance already removed (tombstoned)"
            );
            state.rx_subjects.remove(subject);
            return false;
        }
        state
            .subject_instance
            .insert(subject.to_string(), key.clone());
        state
            .instance_subjects
            .entry(key)
            .or_default()
            .insert(subject.to_string());
        true
    }

    /// Cancel a single pending response-stream registration.
    ///
    /// Removes the subject from `rx_subjects` (dropping the `oneshot::Sender`,
    /// which causes the waiting `oneshot::Receiver` to resolve with
    /// `RecvError`) and cleans up the instance-tracking maps.
    pub async fn cancel_recv_stream(&self, subject: &str) {
        let mut state = self.state.lock().await;
        state.rx_subjects.remove(subject);
        if let Some(key) = state.subject_instance.remove(subject) {
            if let Some(subjects) = state.instance_subjects.get_mut(&key) {
                subjects.remove(subject);
                if subjects.is_empty() {
                    state.instance_subjects.remove(&key);
                }
            }
        }
    }

    /// Cancel **all** pending response-stream registrations for a given instance.
    ///
    /// Called when the discovery plane reports the instance as removed (etcd lease
    /// expired or explicit unregister). Dropping each `oneshot::Sender` causes
    /// the corresponding `oneshot::Receiver.await` in `AddressedPushRouter::generate()`
    /// to resolve with `RecvError`, which is converted to a migratable
    /// `DynamoError(Disconnected)` so the migration layer can retry on another worker.
    ///
    /// The instance is also tombstoned so that any concurrent
    /// `associate_instance()` call for the same instance (race with the
    /// request path) will immediately cancel the late-arriving subject.
    ///
    /// The `endpoint` parameter scopes the tombstone to a specific endpoint,
    /// preventing sibling endpoints on the same runtime from being affected.
    ///
    /// Returns the number of streams cancelled.
    pub async fn cancel_instance_streams(&self, endpoint: &str, instance_id: u64) -> usize {
        let key = (endpoint.to_string(), instance_id);
        let mut state = self.state.lock().await;
        // Tombstone the (endpoint, instance) pair so late associate_instance() calls cancel.
        state.removed_instances.insert(key.clone());
        let subjects = match state.instance_subjects.remove(&key) {
            Some(subjects) => subjects,
            None => return 0,
        };
        let count = subjects.len();
        for subject in &subjects {
            state.rx_subjects.remove(subject);
            state.subject_instance.remove(subject);
        }
        count
    }

    /// Remove an (endpoint, instance) pair from the tombstone set.
    ///
    /// Called by the discovery watcher when an instance reappears in the
    /// discovery set (re-registered after restart). Without this, the
    /// tombstone would cause every future subject for that endpoint+instance
    /// to be immediately cancelled even though the instance is alive again.
    pub async fn clear_instance_tombstone(&self, endpoint: &str, instance_id: u64) {
        let key = (endpoint.to_string(), instance_id);
        let mut state = self.state.lock().await;
        state.removed_instances.remove(&key);
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
        tracing::debug!("Registering new TcpStream on {address}");

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

            let cleanup_subject = sender_subject.clone();
            let cleanup_state = self.state.clone();
            let registered_stream = RegisteredStream::new(
                TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: sender_subject,
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Request,
                }
                .into(),
                pending_sender_rx,
            )
            .with_cleanup(move || {
                // Synchronous removal -- fire-and-forget via a spawned task
                // because Drop cannot be async.
                tokio::spawn(async move {
                    let mut state = cleanup_state.lock().await;
                    state.tx_subjects.remove(&cleanup_subject);
                });
            });

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

            let cleanup_subject = receiver_subject.clone();
            let cleanup_state = self.state.clone();
            let registered_stream = RegisteredStream::new(
                TcpStreamConnectionInfo {
                    address: address.clone(),
                    subject: receiver_subject,
                    context: options.context.id().to_string(),
                    stream_type: StreamType::Response,
                }
                .into(),
                pending_recver_rx,
            )
            .with_cleanup(move || {
                // Synchronous removal -- fire-and-forget via a spawned task
                // because Drop cannot be async.
                tokio::spawn(async move {
                    let mut state = cleanup_state.lock().await;
                    state.rx_subjects.remove(&cleanup_subject);
                    if let Some(key) = state.subject_instance.remove(&cleanup_subject) {
                        if let Some(subjects) = state.instance_subjects.get_mut(&key) {
                            subjects.remove(&cleanup_subject);
                            if subjects.is_empty() {
                                state.instance_subjects.remove(&key);
                            }
                        }
                    }
                });
            });

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
                tracing::warn!("failed to accept tcp connection: {e}");
                eprintln!("failed to accept tcp connection: {}", e);
                continue;
            }
        };

        match stream.set_nodelay(true) {
            Ok(_) => (),
            Err(e) => {
                tracing::warn!("failed to set tcp stream to nodelay: {e}");
            }
        }

        match stream.set_linger(Some(std::time::Duration::from_secs(0))) {
            Ok(_) => (),
            Err(e) => {
                tracing::warn!("failed to set tcp stream to linger: {e}");
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
                tracing::warn!("failed to handle tcp connection: {e}");
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
        let response_stream = {
            let mut guard = state.lock().await;
            let conn = guard
                .rx_subjects
                .remove(&subject)
                .ok_or(error!("Subject not found: {}; upstream publisher specified a subject unknown to the downsteam subscriber", subject))?;
            // Clean up instance-tracking maps on the normal (worker-connected) path.
            if let Some(key) = guard.subject_instance.remove(&subject) {
                if let Some(subjects) = guard.instance_subjects.get_mut(&key) {
                    subjects.remove(&subject);
                    if subjects.is_empty() {
                        guard.instance_subjects.remove(&key);
                    }
                }
            }
            conn
        };

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
                                    tracing::debug!("forwarding body/data message to response channel failed: {err}");
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
            tracing::debug!("failed to flush socket: {e}");
        }
        if let Err(e) = inner.shutdown().await {
            tracing::debug!("failed to shutdown socket: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AsyncEngineContextProvider;
    use crate::pipeline::Context;

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

    /// Create a test server using the failing IP resolver (falls back to loopback).
    async fn test_server() -> Arc<TcpStreamServer> {
        TcpStreamServer::new_with_resolver(
            ServerOptions::builder().port(0).build().unwrap(),
            FailingIpResolver,
        )
        .await
        .unwrap()
    }

    /// Helper: register a response stream and extract its subject string.
    async fn register_and_get_subject(
        server: &TcpStreamServer,
    ) -> (
        String,
        tokio::sync::oneshot::Receiver<Result<super::StreamReceiver, String>>,
    ) {
        let context = Context::new(());
        let options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending = server.register(options).await;
        let recv_stream = pending.recv_stream.unwrap();
        let (conn_info, provider) = recv_stream.into_parts();
        let tcp_info: TcpStreamConnectionInfo = conn_info.try_into().unwrap();
        (tcp_info.subject, provider)
    }

    #[tokio::test]
    async fn test_cancel_instance_streams_unblocks_receiver() {
        let server = test_server().await;

        let (subject, provider) = register_and_get_subject(&server).await;

        // Associate the subject with endpoint "generate", instance 42
        assert!(server.associate_instance(&subject, "generate", 42).await);

        // Cancel all streams for endpoint "generate", instance 42
        let cancelled = server.cancel_instance_streams("generate", 42).await;
        assert_eq!(cancelled, 1);

        // The oneshot receiver should now resolve with an error (sender dropped)
        let result = provider.await;
        assert!(result.is_err(), "Expected RecvError after cancellation");
    }

    #[tokio::test]
    async fn test_cancel_instance_streams_multiple_subjects() {
        let server = test_server().await;

        let (subj1, prov1) = register_and_get_subject(&server).await;
        let (subj2, prov2) = register_and_get_subject(&server).await;
        let (subj3, prov3) = register_and_get_subject(&server).await;

        // Associate first two with instance 10, third with instance 20
        assert!(server.associate_instance(&subj1, "generate", 10).await);
        assert!(server.associate_instance(&subj2, "generate", 10).await);
        assert!(server.associate_instance(&subj3, "generate", 20).await);

        // Cancel instance 10 -- should cancel 2 subjects
        let cancelled = server.cancel_instance_streams("generate", 10).await;
        assert_eq!(cancelled, 2);

        assert!(prov1.await.is_err());
        assert!(prov2.await.is_err());

        // Instance 20 should be unaffected -- cancel it separately
        let cancelled = server.cancel_instance_streams("generate", 20).await;
        assert_eq!(cancelled, 1);
        assert!(prov3.await.is_err());
    }

    #[tokio::test]
    async fn test_cancel_instance_streams_nonexistent_instance() {
        let server = test_server().await;

        // Cancelling a nonexistent instance should return 0 and not panic
        let cancelled = server.cancel_instance_streams("generate", 999).await;
        assert_eq!(cancelled, 0);
    }

    #[tokio::test]
    async fn test_cancel_recv_stream_cleans_up_instance_tracking() {
        let server = test_server().await;

        let (subject, _provider) = register_and_get_subject(&server).await;
        assert!(server.associate_instance(&subject, "generate", 42).await);

        // Cancel the individual subject
        server.cancel_recv_stream(&subject).await;

        // Instance should have no remaining subjects
        let cancelled = server.cancel_instance_streams("generate", 42).await;
        assert_eq!(
            cancelled, 0,
            "Instance tracking should have been cleaned up"
        );
    }

    #[tokio::test]
    async fn test_registered_stream_drop_runs_cleanup() {
        let server = test_server().await;

        // Register a response stream but DON'T call into_parts -- just drop it
        let context = Context::new(());
        let options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending = server.register(options).await;
        let recv_stream = pending.recv_stream.unwrap();

        // Get the subject before dropping
        let tcp_info: TcpStreamConnectionInfo =
            recv_stream.connection_info.clone().try_into().unwrap();
        let subject = tcp_info.subject.clone();

        // Verify it's in rx_subjects
        {
            let state = server.state.lock().await;
            assert!(state.rx_subjects.contains_key(&subject));
        }

        // Drop the RegisteredStream -- RAII cleanup should fire
        drop(recv_stream);

        // Give the spawned cleanup task a moment to run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Verify it's been removed from rx_subjects
        {
            let state = server.state.lock().await;
            assert!(
                !state.rx_subjects.contains_key(&subject),
                "RAII cleanup should have removed the rx_subjects entry"
            );
        }
    }

    #[tokio::test]
    async fn test_registered_stream_into_parts_disarms_cleanup() {
        let server = test_server().await;

        let context = Context::new(());
        let options = StreamOptions::builder()
            .context(context.context())
            .enable_request_stream(false)
            .enable_response_stream(true)
            .build()
            .unwrap();

        let pending = server.register(options).await;
        let recv_stream = pending.recv_stream.unwrap();

        let tcp_info: TcpStreamConnectionInfo =
            recv_stream.connection_info.clone().try_into().unwrap();
        let subject = tcp_info.subject.clone();

        // Call into_parts to disarm the cleanup
        let (_conn_info, _provider) = recv_stream.into_parts();

        // Give any potential cleanup a moment to run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // The entry should still be in rx_subjects (cleanup was disarmed)
        {
            let state = server.state.lock().await;
            assert!(
                state.rx_subjects.contains_key(&subject),
                "into_parts() should disarm the RAII cleanup"
            );
        }
    }

    #[tokio::test]
    async fn test_associate_after_cancel_is_immediately_cancelled() {
        // Simulates the race: cancel_instance_streams fires before associate_instance.
        let server = test_server().await;

        // Cancel instance 42 for endpoint "generate" BEFORE any subject is registered (tombstone).
        let cancelled = server.cancel_instance_streams("generate", 42).await;
        assert_eq!(cancelled, 0);

        // Now register a subject and try to associate it with the tombstoned instance.
        let (subject, provider) = register_and_get_subject(&server).await;
        let associated = server.associate_instance(&subject, "generate", 42).await;

        // associate_instance should return false when the instance is tombstoned.
        assert!(
            !associated,
            "associate_instance on a tombstoned instance should return false"
        );

        // The provider should resolve with an error because associate_instance
        // found the tombstone and immediately cancelled the subject.
        let result = provider.await;
        assert!(
            result.is_err(),
            "Late associate_instance on a tombstoned instance should immediately cancel"
        );
    }

    #[tokio::test]
    async fn test_clear_tombstone_allows_new_associations() {
        let server = test_server().await;

        // Tombstone instance 42 for endpoint "generate".
        server.cancel_instance_streams("generate", 42).await;

        // Clear the tombstone (simulates instance coming back in discovery).
        server.clear_instance_tombstone("generate", 42).await;

        // Now associate should work normally (subject NOT cancelled).
        let (subject, _provider) = register_and_get_subject(&server).await;
        assert!(server.associate_instance(&subject, "generate", 42).await);

        // Subject should be tracked, not cancelled.
        let cancelled = server.cancel_instance_streams("generate", 42).await;
        assert_eq!(
            cancelled, 1,
            "After clearing tombstone, subjects should be tracked normally"
        );
    }

    #[tokio::test]
    async fn test_cancel_does_not_affect_sibling_endpoint() {
        // Bug 1 regression test: cancelling "generate" must not cancel "prefill"
        // subjects that share the same instance_id (same backend runtime).
        let server = test_server().await;

        let (gen_subj, gen_prov) = register_and_get_subject(&server).await;
        let (pre_subj, pre_prov) = register_and_get_subject(&server).await;

        // Same instance_id 42, but different endpoints.
        assert!(server.associate_instance(&gen_subj, "generate", 42).await);
        assert!(server.associate_instance(&pre_subj, "prefill", 42).await);

        // Cancel only the "generate" endpoint's subjects.
        let cancelled = server.cancel_instance_streams("generate", 42).await;
        assert_eq!(
            cancelled, 1,
            "Only the generate subject should be cancelled"
        );

        // generate provider must be cancelled.
        assert!(gen_prov.await.is_err());

        // prefill provider must still be pending (not cancelled).
        // We verify by doing a try-style check -- the oneshot should not be resolved yet.
        // We just check it can still be cancelled by its own endpoint.
        let still_pending = server.cancel_instance_streams("prefill", 42).await;
        assert_eq!(still_pending, 1, "prefill subject should still be tracked");
        assert!(pre_prov.await.is_err());
    }

    #[tokio::test]
    async fn test_tombstone_is_endpoint_scoped() {
        // Tombstoning "generate" must not prevent new associations on "prefill"
        // for the same instance_id.
        let server = test_server().await;

        // Tombstone "generate" for instance 42.
        server.cancel_instance_streams("generate", 42).await;

        // A new subject for "generate" should be rejected.
        let (gen_subj, gen_prov) = register_and_get_subject(&server).await;
        assert!(
            !server.associate_instance(&gen_subj, "generate", 42).await,
            "generate should be tombstoned"
        );
        assert!(gen_prov.await.is_err());

        // A new subject for "prefill" with the same instance_id should be accepted.
        let (pre_subj, _pre_prov) = register_and_get_subject(&server).await;
        assert!(
            server.associate_instance(&pre_subj, "prefill", 42).await,
            "prefill tombstone is independent; subject should be tracked"
        );
        let count = server.cancel_instance_streams("prefill", 42).await;
        assert_eq!(count, 1, "prefill subject should be tracked normally");
    }
}
