// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ Transport
//!
//! This module provides a ZMQ transport for the [crate::DistributedRuntime].
//!
//! Currently, the [Server] consists of a [async_zmq::Router] and the [Client] leverages
//! a [async_zmq::Dealer].
//!
//! The distributed service pattern we will use is based on the Harmony pattern described in
//! [Chapter 8: A Framework for Distributed Computing](https://zguide.zeromq.org/docs/chapter8/#True-Peer-Connectivity-Harmony-Pattern).
//!
//! This is similar to the TCP implementation; however, the TCP implementation used a direct
//! connection between the client and server per stream. The ZMQ transport will enable the
//! equivalent of a connection pool per upstream service at the cost of needing an extra internal
//! routing step per service endpoint.

use anyhow::{Result, anyhow};
use async_zmq::{Context, Dealer, Router, Sink, SinkExt, StreamExt};
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, os::fd::FromRawFd, sync::Arc, time::Duration, vec::IntoIter};
use tokio::{
    sync::{Mutex, mpsc},
    task::{JoinError, JoinHandle},
};
use tokio_util::sync::CancellationToken;

// Core message types
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ControlMessage {
    Cancel { request_id: String },
    CancelAck { request_id: String },
    Error { request_id: String, error: String },
    Complete { request_id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum MessageType {
    Data(Vec<u8>),
    Control(ControlMessage),
}

enum StreamAction {
    SendEager(usize),
    SendDelayed(usize),
    Close,
}

// Router state management
struct RouterState {
    active_streams: HashMap<String, mpsc::Sender<Bytes>>,
    control_channels: HashMap<String, mpsc::Sender<ControlMessage>>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            active_streams: HashMap::new(),
            control_channels: HashMap::new(),
        }
    }

    fn register_stream(
        &mut self,
        request_id: String,
        data_tx: mpsc::Sender<Bytes>,
        control_tx: mpsc::Sender<ControlMessage>,
    ) {
        self.active_streams.insert(request_id.clone(), data_tx);
        self.control_channels.insert(request_id, control_tx);
    }

    fn remove_stream(&mut self, request_id: &str) {
        self.active_streams.remove(request_id);
        self.control_channels.remove(request_id);
    }
}

// Server implementation
#[derive(Clone, Dissolve)]
pub struct Server {
    state: Arc<Mutex<RouterState>>,
    cancel_token: CancellationToken,
    fd: i32,
}

impl Server {
    /// Create a new [Server] which is a [async_zmq::Router] with the given [async_zmq::Context] and address to bind
    /// the ZMQ [async_zmq::Router] socket.
    ///
    /// If the event loop processing the router fails with an error, the signal is propagated through the [CancellationToken]
    /// by issuing a [CancellationToken::cancel].
    ///
    /// The [Server] is how you interact with the running instance.
    ///
    /// The [ServerExecutionHandle] is the handle for background task executing the [Server].
    pub async fn new(
        context: &Context,
        address: &str,
        cancel_token: CancellationToken,
    ) -> Result<(Self, ServerExecutionHandle)> {
        let router = async_zmq::router(address)?.with_context(context).bind()?;
        let fd = router.as_raw_socket().get_fd()?;
        let state = Arc::new(Mutex::new(RouterState::new()));

        // can cancel the router's event loop
        let child = cancel_token.child_token();
        let primary_task = tokio::spawn(Self::run(router, state.clone(), child.child_token()));

        // this task captures the primary cancellation token, so if an error occurs, we can cancel the router's event loop
        // but we also propagate the error to the caller's cancellation token
        let watch_task = tokio::spawn(async move {
            let result = primary_task.await.inspect_err(|e| {
                tracing::error!("zmq server/router task failed: {}", e);
                cancel_token.cancel();
            })?;
            result.inspect_err(|e| {
                tracing::error!("zmq server/router task failed: {}", e);
                cancel_token.cancel();
            })
        });

        let handle = ServerExecutionHandle {
            task: watch_task,
            cancel_token: child.clone(),
        };

        Ok((
            Self {
                state,
                cancel_token: child,
                fd,
            },
            handle,
        ))
    }

    // pub async fn register_stream(&)

    async fn run(
        router: Router<IntoIter<Vec<u8>>, Vec<u8>>,
        state: Arc<Mutex<RouterState>>,
        token: CancellationToken,
    ) -> Result<()> {
        let mut router = router;

        // todo - move this into the Server impl to discover the os port being used
        // let fd = router.as_raw_socket().get_fd()?;
        // let sock = unsafe { socket2::Socket::from_raw_fd(fd) };
        // let addr = sock.local_addr()?;
        // let port = addr.as_socket().map(|s| s.port());

        // if let Some(port) = port {
        //     tracing::info!("Server listening on port {}", port);
        // }

        loop {
            let frames = tokio::select! {
                biased;

                frames = router.next() => {
                    match frames {
                        Some(Ok(frames)) => {
                            frames
                        },
                        Some(Err(e)) => {
                            tracing::warn!("Error receiving message: {}", e);
                            continue;
                        }
                        None => break,
                    }
                }

                _ = token.cancelled() => {
                    tracing::info!("Server shutting down");
                    break;
                }
            };

            // we should have at least 3 frames
            // 0: identity
            // 1: request_id
            // 2: message type

            // if the contract is broken, we should exit
            if frames.len() != 3 {
                anyhow::bail!(
                    "Fatal Error -- Broken contract -- Expected 3 frames, got {}",
                    frames.len()
                );
            }

            let request_id = String::from_utf8_lossy(&frames[1]).to_string();
            let message = frames[2].to_vec();
            let message_size = message.len();

            if let Some(tx) = state.lock().await.active_streams.get(&request_id) {
                // first we try to send the data eagerly without blocking
                let action = match tx.try_send(message.into()) {
                    Ok(_) => {
                        tracing::trace!(
                            request_id,
                            "response data sent eagerly to stream: {} bytes",
                            message_size
                        );
                        StreamAction::SendEager(message_size)
                    }
                    Err(e) => match e {
                        mpsc::error::TrySendError::Closed(_) => {
                            tracing::info!(request_id, "response stream was closed");
                            StreamAction::Close
                        }
                        mpsc::error::TrySendError::Full(data) => {
                            tracing::warn!(
                                request_id,
                                "response stream is full; backpressure alert"
                            );
                            // todo - add timeout - we are blocking all other streams
                            if (tx.send(data).await).is_err() {
                                StreamAction::Close
                            } else {
                                StreamAction::SendDelayed(message_size)
                            }
                        }
                    },
                };

                match action {
                    StreamAction::SendEager(_size) => {
                        // increment bytes_received
                        // increment messages_received
                        // increment eager_messages_received
                    }
                    StreamAction::SendDelayed(_size) => {
                        // increment bytes_received
                        // increment messages_received
                        // increment delayed_messages_received
                    }
                    StreamAction::Close => {
                        state.lock().await.active_streams.remove(&request_id);
                    }
                }
            } else {
                // increment bytes_dropped
                // increment messages_dropped
                tracing::trace!(request_id, "no active stream for request_id");
            }
        }

        Ok(())
    }
}

/// The [ServerExecutionHandle] is the handle for background task executing the [Server].
///
/// You can use this to check if the server is finished or cancelled.
///
/// You can also join on the task to wait for it to finish.
pub struct ServerExecutionHandle {
    task: JoinHandle<Result<()>>,
    cancel_token: CancellationToken,
}

impl ServerExecutionHandle {
    /// Check if the task awaiting on the [Server]s background event loop has finished.
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Check if the server's event loop has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Cancel the server's event loop.
    ///
    /// This will signal the server to stop processing requests and exit.
    ///
    /// This will not wait for the server to finish, it will exit immediately.
    ///
    /// This will not propagate to the [CancellationToken] used to start the [Server]
    /// unless an error happens during the shutdown process.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Join on the task awaiting on the [Server]s background event loop.
    ///
    /// This will return the result of the [Server]s background event loop.
    pub async fn join(self) -> Result<()> {
        self.task.await?
    }
}

// Client implementation
pub struct Client {
    dealer: Dealer<IntoIter<Vec<u8>>, Vec<u8>>,
}

impl Client {
    fn new(context: &Context, address: &str) -> Result<Self> {
        let dealer = async_zmq::dealer(address)?
            .with_context(context)
            .connect()?;

        Ok(Self { dealer })
    }

    fn dealer(&mut self) -> &mut Dealer<IntoIter<Vec<u8>>, Vec<u8>> {
        &mut self.dealer
    }

    // async fn send_data(&self, data: Vec<u8>) -> Result<()> {
    //     let msg_type = MessageType::Data(data);
    //     let type_bytes = serde_json::to_vec(&msg_type)?;

    //     self.dealer
    //         .send_multipart(&[type_bytes, self.request_id.as_bytes().to_vec()])
    //         .await
    //         .map_err(|e| anyhow!("Failed to send data: {}", e))
    // }

    // async fn send_control(&self, msg: ControlMessage) -> Result<()> {
    //     let msg_type = MessageType::Control(msg);
    //     let type_bytes = serde_json::to_vec(&msg_type)?;

    //     self.dealer
    //         .send_multipart(&[type_bytes])
    //         .await
    //         .map_err(|e| anyhow!("Failed to send control message: {}", e))
    // }

    // async fn receive(&self) -> Result<MessageType> {
    //     let frames = self
    //         .dealer
    //         .recv_multipart()
    //         .await
    //         .map_err(|e| anyhow!("Failed to receive message: {}", e))?;

    //     if frames.is_empty() {
    //         return Err(anyhow!("Received empty message"));
    //     }

    //     serde_json::from_slice(&frames[0])
    //         .map_err(|e| anyhow!("Failed to deserialize message: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_basic_communication() -> Result<()> {
        let context = Context::new();
        let address = "tcp://127.0.0.1:1337";
        let token = CancellationToken::new();

        // Start server
        let (server, handle) = Server::new(&context, address, token.clone()).await?;
        let state = server.state.clone();

        let id = "test-request".to_string();
        let (tx, mut rx) = tokio::sync::mpsc::channel(512);
        state.lock().await.active_streams.insert(id.clone(), tx);

        // Create client
        let mut client = Client::new(&context, address)?;

        client
            .dealer()
            .send(vec![id.as_bytes().to_vec(), id.as_bytes().to_vec()].into())
            .await?;

        let receive_result = rx.recv().await;

        let received = receive_result.unwrap();

        // convert to string
        let received_str = String::from_utf8_lossy(&received).to_string();
        assert_eq!(received_str, "test-request");

        client.dealer().close().await?;

        handle.cancel();
        handle.join().await?;

        println!("done");

        Ok(())
    }

    // #[tokio::test]
    // async fn test_multiple_streams() -> Result<()> {
    //     // Similar to above but with multiple clients/streams
    //     Ok(())
    // }

    // #[tokio::test]
    // async fn test_error_handling() -> Result<()> {
    //     // Test various error conditions
    //     Ok(())
    // }
}
