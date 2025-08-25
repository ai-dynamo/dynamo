// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tmq::AsZmqSocket;

use super::*;
use utils::*;

use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tmq::{
    Context, Message, Multipart,
    publish::{Publish, publish},
    pull::{Pull, pull},
    push::{Push, push},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::{Mutex, oneshot};
use tokio_util::sync::CancellationToken;

use futures_util::{SinkExt, StreamExt};

struct PendingMessage {
    remaining_workers: usize,
    completion_indicator: oneshot::Sender<()>,
}

pub struct LeaderSockets {
    pub pub_socket: Publish,
    pub pub_url: String,
    pub ack_socket: Pull,
    pub ack_url: String,
}

pub fn new_leader_sockets(url: &str) -> Result<LeaderSockets> {
    let url = format!("{}:0", url);

    let context = Context::new();
    let pub_socket = publish(&context).bind(url.as_str())?;
    let pub_url = pub_socket
        .get_socket()
        .get_last_endpoint()
        .unwrap()
        .unwrap();

    let ack_socket = pull(&context).bind(url.as_str())?;
    let ack_url = ack_socket
        .get_socket()
        .get_last_endpoint()
        .unwrap()
        .unwrap();

    Ok(LeaderSockets {
        pub_socket,
        pub_url,
        ack_socket,
        ack_url,
    })
}

/// The ActiveMessageLeader is responsible for sending commands to all workers.
/// On the leader side, we use two sockets:
/// 1. A publish socket to send messages to all workers.
/// 2. A pull socket to receive ACKs from workers.
pub struct ZmqActiveMessageLeader {
    // Our socket to broadcast messages.
    pub_socket: Arc<Mutex<Publish>>,
    // Message ID counter. Used for ACKs
    message_id: Arc<Mutex<usize>>,
    // Map of currently pending messages (messages that haven't been ACKed by all workers).
    pending_messages: Arc<Mutex<HashMap<usize, PendingMessage>>>,
    // Number of workers we're waiting for.
    num_workers: Arc<usize>,
}

impl ZmqActiveMessageLeader {
    pub async fn new(
        leader_sockets: LeaderSockets,
        num_workers: usize,
        timeout: Duration,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let pub_socket = Arc::new(Mutex::new(leader_sockets.pub_socket));
        let pull_socket = leader_sockets.ack_socket;

        tracing::info!(
            "ZmqActiveMessageLeader: Bound to pub: {} and pull: {}",
            leader_sockets.pub_url,
            leader_sockets.ack_url
        );

        let pending_messages = Arc::new(Mutex::new(HashMap::new()));

        let pending_messages_clone = pending_messages.clone();
        CriticalTaskExecutionHandle::new(
            |cancel_token| Self::pull_worker(pull_socket, pending_messages_clone, cancel_token),
            cancel_token,
            "ZmqActiveMessageLeader: Pull worker",
        )?
        .detach();

        let self_ = Self {
            pub_socket,
            message_id: Arc::new(Mutex::new(0)),
            pending_messages,
            num_workers: Arc::new(num_workers),
        };

        // Ping our workers.
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!("Timed out waiting for workers."));
            }

            // Try to send a ping to all workers.
            tracing::info!("ZmqActiveMessageLeader: Pinging workers...");
            let ping_receiver = self_.broadcast(ZMQ_PING_MESSAGE, vec![]).await?;

            tokio::select! {
                // If we receive an ACK from every worker, we're done.
                _ = ping_receiver => {
                    tracing::info!("ZmqActiveMessageLeader: Worker ping successful. Startup complete.");
                    break;
                }
                // Wait for 1 second before pinging again.
                _ = tokio::time::sleep(Duration::from_millis(1000)) => {
                    tracing::info!("ZmqActiveMessageLeader: Ping timed out. Retrying...");
                    continue;
                }
            }
        }

        Ok(self_)
    }

    /// Broadcast a message to all workers.
    /// Returns a receiver that will be notified when all workers have ACKed the message.
    pub async fn broadcast(
        &self,
        function: &str,
        data: Vec<Vec<u8>>,
    ) -> Result<oneshot::Receiver<()>> {
        // Generate a unique id.
        let id = {
            let mut id = self.message_id.lock().await;
            *id += 1;
            *id
        };

        let (completion_indicator, completion_receiver) = oneshot::channel();

        let pending_message = PendingMessage {
            // We start with the number of workers we're waiting for.
            remaining_workers: *self.num_workers,
            completion_indicator,
        };

        // Add the message to the pending messages map.
        self.pending_messages
            .lock()
            .await
            .insert(id, pending_message);

        // id, function, data
        let mut message: VecDeque<Message> = VecDeque::with_capacity(data.len() + 2);
        message.push_back(id.to_be_bytes().as_slice().into());
        message.push_back(function.into());
        for data in data {
            message.push_back(data.into());
        }

        tracing::debug!(
            "ZmqActiveMessageLeader: Broadcasting message with id: {}",
            id
        );
        self.pub_socket
            .lock()
            .await
            .send(Multipart(message))
            .await?;

        Ok(completion_receiver)
    }

    /// Pull worker is responsible for receiving ACKs from workers.
    async fn pull_worker(
        mut pull_socket: Pull,
        pending_messages: Arc<Mutex<HashMap<usize, PendingMessage>>>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(Ok(message)) = pull_socket.next() => {
                    // The leader should only ever receive ACKs.
                    // ACKs have no data.
                    if message.len() != 1 {
                        tracing::error!(
                            "Received message with unexpected length: {:?}",
                            message.len()
                        );
                        continue;
                    }

                    // TODO: This looks ugly.
                    let arr: [u8; std::mem::size_of::<usize>()] = (*message[0]).try_into()?;
                    let id = usize::from_be_bytes(arr);

                    let mut pending_messages = pending_messages.lock().await;
                    // TODO: Should we error if we can't find the pending message?
                    // if let std::collections::hash_map::Entry::Occupied(mut entry) =
                    //     pending_messages.entry(id)
                    // {
                    //     entry.get_mut().remaining_workers -= 1;
                    //     tracing::debug!(
                    //         "ZmqActiveMessageLeader: Received ACK for message with id: {}. There are {} remaining workers.",
                    //         id,
                    //         entry.get().remaining_workers
                    //     );
                    //     // If all workers have ACKed, notify the completion indicator.
                    //     if entry.get().remaining_workers == 0 {
                    //         let e = entry.remove();
                    //         tracing::debug!(
                    //             "ZmqActiveMessageLeader: Message with id: {} completed.",
                    //             id
                    //         );
                    //         // It's possible that the receiver has already been dropped,
                    //         // so ignore any send error here.
                    //         let _ = e.completion_indicator.send(());
                    //     }
                    // }

                    match pending_messages.entry(id) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            let pending_message = entry.get_mut();
                            debug_assert!(pending_message.remaining_workers > 0);
                            pending_message.remaining_workers -= 1;
                            tracing::debug!(
                                "ZmqActiveMessageLeader: Received ACK for message with id: {}. There are {} remaining workers.",
                                id,
                                pending_message.remaining_workers
                            );
                            if pending_message.remaining_workers == 0 {
                                let e = entry.remove();
                                tracing::debug!("ZmqActiveMessageLeader: Message with id: {} completed.", id);
                                let _ = e.completion_indicator.send(());
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(_) => {
                            tracing::error!("Received ACK for unknown message with id: {}", id);
                        }
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::info!("ZmqActiveMessageLeader: Pull worker cancelled.");
                    break;
                }
            }
        }
        tracing::info!("ZmqActiveMessageLeader: Pull worker exiting.");
        Ok(())
    }
}

/// A message handle is used to track a message.
/// It contains a way to ACK the message, as well as the data.
pub struct MessageHandle {
    message_id: usize,
    function: String,
    pub data: Vec<Vec<u8>>,
    push_handle: Arc<Mutex<Push>>,
    acked: bool,
}

impl MessageHandle {
    pub fn new(message: Multipart, push_handle: Arc<Mutex<Push>>) -> Result<Self> {
        // We always need at least the message id and the function name.
        if message.len() < 2 {
            return Err(anyhow::anyhow!(
                "Received message with unexpected length: {:?}",
                message.len()
            ));
        }
        let arr: [u8; std::mem::size_of::<usize>()] = (*message[0]).try_into()?;
        let id = usize::from_be_bytes(arr);
        let function = message[1]
            .as_str()
            .ok_or(anyhow::anyhow!("Unable to parse function name."))?
            .to_string();

        // Skip the message id and function name: Everything else is data.
        let data = message.into_iter().skip(2).map(|m| (*m).to_vec()).collect();

        Ok(Self {
            message_id: id,
            function,
            data,
            push_handle,
            acked: false,
        })
    }

    /// ACK the message, which notifies the leader.
    pub async fn ack(&mut self) -> Result<()> {
        // We can only ACK once.
        if self.acked {
            return Err(anyhow::anyhow!("Message was already acked!"));
        }

        self.acked = true;

        let id = self.message_id;
        let mut message = VecDeque::with_capacity(1);
        message.push_back(id.to_be_bytes().as_slice().into());
        let message = Multipart(message);
        self.push_handle.lock().await.send(message).await?;
        tracing::debug!("ZmqActiveMessageWorker: ACKed message with id: {}", id);
        Ok(())
    }
}

/// We must always ACK a message.
/// Panic if we don't.
impl Drop for MessageHandle {
    fn drop(&mut self) {
        if !self.acked {
            panic!("Message was not acked!");
        }
    }
}

/// A handler is responsible for handling a message.
/// We have to use this instead of AsyncFn because AsyncFn isn't dyn compatible.
#[async_trait]
pub trait Handler: Send + Sync {
    async fn handle(&self, message: MessageHandle) -> Result<()>;
}

/// A super simple handler that responds to a ping.
/// This is used in the startup sequence to check worker liveness.
struct Ping;

#[async_trait]
impl Handler for Ping {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if !message.data.is_empty() {
            return Err(anyhow::anyhow!("Ping message should not have data."));
        }
        message.ack().await?;
        Ok(())
    }
}

type MessageHandlers = HashMap<String, Arc<dyn Handler>>;

/// The ActiveMessageWorker receives commands from the leader, and ACKs them.
pub struct ZmqActiveMessageWorker {}

impl ZmqActiveMessageWorker {
    pub fn new(
        sub_url: &str,
        push_url: &str,
        mut message_handlers: MessageHandlers,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        let context = Context::new();

        let sub_socket = subscribe(&context)
            .connect(sub_url)?
            .subscribe("".as_bytes())?;
        let push_socket = Arc::new(Mutex::new(push(&context).connect(push_url)?));

        tracing::info!(
            "ZmqActiveMessageWorker: Bound to sub: {} and push: {}",
            sub_url,
            push_url
        );

        // Add our ping handler.
        message_handlers.insert(ZMQ_PING_MESSAGE.to_string(), Arc::new(Ping));
        let message_handlers = Arc::new(message_handlers);

        CriticalTaskExecutionHandle::new(
            |cancel_token| {
                Self::sub_worker(sub_socket, push_socket, message_handlers, cancel_token)
            },
            cancel_token,
            "ZmqActiveMessageWorker: Sub worker",
        )?
        .detach();

        Ok(Self {})
    }

    async fn sub_worker(
        mut sub_socket: Subscribe,
        push_socket: Arc<Mutex<Push>>,
        message_handlers: Arc<MessageHandlers>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(Ok(message)) = sub_socket.next() => {
                    if message.len() < 2 {
                        tracing::error!(
                            "Received message with unexpected length: {:?}",
                            message.len()
                        );
                        continue;
                    }

                    // Try to parse our message.
                    let message_handle = MessageHandle::new(message, push_socket.clone())?;

                    // Check if the function name is registered.
                    // TODO: We may want to make this dynamic, and expose a function
                    // to dynamically add/remove handlers.
                    if let Some(handler) = message_handlers.get(&message_handle.function) {
                        tracing::debug!(
                            "ZmqActiveMessageWorker: Handling message with id: {} for function: {}",
                            message_handle.message_id,
                            message_handle.function
                        );
                        let handler_clone = handler.clone();
                        let handle_text = format!("ZmqActiveMessageWorker: Handler for function: {}", message_handle.function);
                        CriticalTaskExecutionHandle::new(
                            move |_| async move { handler_clone.handle(message_handle).await },
                            cancel_token.clone(),
                            handle_text.as_str(),
                        )?
                        .detach();
                    } else {
                        tracing::error!("No handler found for function: {}", message_handle.function);
                    }
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }
}
