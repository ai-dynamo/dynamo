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

use bincode;
use futures_util::{SinkExt, StreamExt};
use std::cmp::min;

struct PendingMessage {
    remaining_workers: usize,
    completion_indicator: Option<oneshot::Sender<()>>,
    // If true, collect one payload (bytes) from each worker reply.
    want_payload: bool,
    // Collected raw payloads (one per worker), if want_payload == true
    payloads: Option<Vec<Vec<u8>>>,
}

pub struct LeaderSockets {
    pub pub_socket: Publish,
    pub pub_url: String,
    pub ack_socket: Pull,
    pub ack_url: String,
}

pub fn new_leader_sockets(pub_url: &str, ack_url: &str) -> Result<LeaderSockets> {
    let context = Context::new();
    let pub_socket = publish(&context).bind(pub_url)?;
    let pub_url = pub_socket
        .get_socket()
        .get_last_endpoint()
        .unwrap()
        .unwrap();

    let ack_socket = pull(&context).bind(ack_url)?;
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
    /// Handshake-first constructor: collects WorkerMetaData, broadcasts LeaderMetadata,
    /// waits for allocation ACKs, then runs the final ping loop.
    pub async fn new_with_handshake<F>(
        leader_sockets: LeaderSockets,
        num_workers: usize,
        overall_timeout: Duration,
        cancel_token: CancellationToken,
        make_leader_meta: F,
    ) -> Result<Self>
    where
        F: Fn(&[WorkerMetadata]) -> LeaderMetadata + Send + Sync + 'static,
    {
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
            |ct| Self::pull_worker(pull_socket, pending_messages_clone, ct),
            cancel_token.clone(),
            "ZmqActiveMessageLeader: Pull worker",
        )?
        .detach();

        let this = Self {
            pub_socket,
            message_id: Arc::new(Mutex::new(0)),
            pending_messages,
            num_workers: Arc::new(num_workers),
        };

        let deadline = Instant::now() + overall_timeout;

        // 1) Collect KvbmWorkerData from ALL workers in a single round.
        // Keep rebroadcasting until we get exactly `num_workers` replies to the SAME broadcast.
        let workers_payloads: Vec<Vec<u8>> = loop {
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "Handshake timed out (device-config collection)."
                ));
            }
            let remain = deadline.saturating_duration_since(Instant::now());
            let round_to = min(Duration::from_secs(2), remain);

            tracing::info!("Handshake: requesting worker device configs...");
            match this
                .broadcast_collect(
                    ZMQ_WORKER_METADATA_MESSAGE,
                    &[],
                    /* want_payload */ true,
                    round_to,
                )
                .await
            {
                Ok(payloads) if payloads.len() == num_workers => {
                    tracing::info!(
                        "Handshake: received {} worker metadata replies in this round.",
                        payloads.len()
                    );
                    break payloads;
                }
                Ok(payloads) => {
                    tracing::warn!(
                        "Handshake: got {} / {} worker metadata replies; rebroadcasting...",
                        payloads.len(),
                        num_workers
                    );
                    continue;
                }
                Err(e) => {
                    tracing::debug!(
                        "Handshake: worker metadata round timed out/failed: {e}; retrying..."
                    );
                    continue;
                }
            }
        };

        let mut workers: Vec<WorkerMetadata> = Vec::with_capacity(workers_payloads.len());

        for payload in workers_payloads {
            let worker: WorkerMetadata =
                bincode::serde::decode_from_slice(&payload, bincode::config::standard())?.0;
            workers.push(worker);
        }

        // 2) Compute & broadcast LeaderMetadata; wait for ALL acks in the SAME round.
        let leader_meta = make_leader_meta(&workers);
        let leader_meta_bytes =
            bincode::serde::encode_to_vec(&leader_meta, bincode::config::standard())?;

        loop {
            if Instant::now() >= deadline {
                return Err(anyhow::anyhow!(
                    "Handshake timed out (allocation-config broadcast)."
                ));
            }
            let remain = deadline.saturating_duration_since(Instant::now());
            let round_to = min(Duration::from_secs(2), remain);

            tracing::info!("Handshake: broadcasting allocation config to workers...");
            match this
                .broadcast_collect(
                    ZMQ_LEADER_METADATA_MESSAGE,
                    std::slice::from_ref(&leader_meta_bytes),
                    /* want_payload */ false,
                    round_to,
                )
                .await
            {
                Ok(_) => {
                    // Success: all workers acked in this round.
                    tracing::info!("Handshake: all workers acked allocation config.");
                    break;
                }
                Err(e) => {
                    tracing::warn!(
                        "Handshake: allocation-config round incomplete: {e}; rebroadcasting..."
                    );
                    continue;
                }
            }
        }

        // 3) Final readiness ping loop (workers only ACK after allocation ready)
        let ping_deadline = deadline;
        loop {
            if Instant::now() >= ping_deadline {
                return Err(anyhow::anyhow!(
                    "Timed out waiting for ping readiness after handshake."
                ));
            }
            tracing::debug!("Handshake: final readiness ping...");
            let ping = this.broadcast(ZMQ_PING_MESSAGE, vec![]).await?;
            tokio::select! {
                _ = ping => break,
                _ = tokio::time::sleep(Duration::from_millis(500)) => continue,
                _ = cancel_token.cancelled() => return Err(anyhow::anyhow!("Startup canceled")),
            }
        }

        Ok(this)
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
            completion_indicator: Some(completion_indicator),
            want_payload: false,
            payloads: None,
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

    /// Generic broadcast that can collect one reply payload from each worker.
    /// - `function`: handler name on workers
    /// - `data_frames`: optional extra frames after [id, function]
    /// - `want_payload`: if true, expects replies shaped as [id, function, payload]
    ///   Returns payloads (empty if want_payload == false).
    pub async fn broadcast_collect(
        &self,
        function: &str,
        data_frames: &[Vec<u8>],
        want_payload: bool,
        timeout: Duration,
    ) -> Result<Vec<Vec<u8>>> {
        // Generate a unique id.
        let id = {
            let mut id = self.message_id.lock().await;
            *id += 1;
            *id
        };

        let (completion_indicator, completion_receiver) = oneshot::channel();
        let pending_message = PendingMessage {
            remaining_workers: *self.num_workers,
            completion_indicator: Some(completion_indicator),
            want_payload,
            payloads: want_payload.then(|| Vec::with_capacity(*self.num_workers)),
        };
        self.pending_messages
            .lock()
            .await
            .insert(id, pending_message);

        // Build message: [id, function, ...data]
        let mut message: VecDeque<Message> = VecDeque::with_capacity(2 + data_frames.len());
        message.push_back(id.to_be_bytes().as_slice().into());
        message.push_back(function.into());
        for df in data_frames {
            message.push_back(df.clone().into());
        }
        self.pub_socket
            .lock()
            .await
            .send(Multipart(message))
            .await?;

        // Await all replies or timeout.
        tokio::select! {
            _ = completion_receiver => { /* done */ }
            _ = tokio::time::sleep(timeout) => {
                let mut map = self.pending_messages.lock().await;
                map.remove(&id);
                return Err(anyhow::anyhow!("Timed out waiting for '{}' responses", function));
            }
        }

        // Extract payloads (if any).
        let mut map = self.pending_messages.lock().await;
        let entry = map
            .remove(&id)
            .ok_or_else(|| anyhow::anyhow!("pending entry missing"))?;
        Ok(entry.payloads.unwrap_or_default())
    }

    async fn pull_worker(
        mut pull_socket: Pull,
        pending_messages: Arc<Mutex<HashMap<usize, PendingMessage>>>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(Ok(message)) = pull_socket.next() => {
                if message.is_empty() {
                    tracing::error!("Leader PULL: empty message");
                    continue;
                }
                let arr: [u8; std::mem::size_of::<usize>()] = (*message[0]).try_into()?;
                let id = usize::from_be_bytes(arr);

                let mut map = pending_messages.lock().await;

                if let Some(pm) = map.get_mut(&id) {
                    // payload reply or pure ACK?
                    if message.len() == 1 {
                        if pm.remaining_workers > 0 { pm.remaining_workers -= 1; }
                    } else {
                        if pm.want_payload && message.len() >= 3
                            && let Some(bufs) = pm.payloads.as_mut() {
                                bufs.push((*message[2]).to_vec());
                            }
                        if pm.remaining_workers > 0 { pm.remaining_workers -= 1; }
                    }

                    tracing::debug!(
                        "Leader PULL: got {} for id {} (remaining={})",
                        if message.len()==1 { "ACK" } else { "REPLY" }, id, pm.remaining_workers
                    );

                    // IMPORTANT: do NOT remove here; just notify completion.
                    if pm.remaining_workers == 0
                        && let Some(tx) = pm.completion_indicator.take() {
                            let _ = tx.send(());
                        }
                } else {
                    // Late reply for a round we've already collected/removed.
                    tracing::debug!("Leader PULL: late/unknown id {}", id);
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
    pub message_id: usize,
    function: String,
    pub data: Vec<Vec<u8>>,
    pub push_handle: Arc<Mutex<Push>>,
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

    /// Reply to the leader with arbitrary payload frames and mark as acked.
    /// Frames shape: [id, function, payload_0, payload_1, ...]
    pub async fn reply(
        &mut self,
        function: &str,
        payload_frames: &[Vec<u8>],
    ) -> anyhow::Result<()> {
        let mut frames: std::collections::VecDeque<tmq::Message> =
            std::collections::VecDeque::with_capacity(2 + payload_frames.len());
        frames.push_back(self.message_id.to_be_bytes().as_slice().into());
        frames.push_back(function.into());
        for p in payload_frames {
            frames.push_back(p.clone().into());
        }
        self.push_handle
            .lock()
            .await
            .send(tmq::Multipart(frames))
            .await?;
        // Mark as acked so Drop won't panic; leader treats the reply as the "ack".
        self.acked = true;
        Ok(())
    }

    /// Mark this message as handled locally without sending an ACK/reply.
    /// Use when intentionally ignoring a message (e.g. ping before readiness).
    pub fn mark_handled(&mut self) {
        self.acked = true;
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

type MessageHandlers = HashMap<String, Arc<dyn Handler>>;

/// The ActiveMessageWorker receives commands from the leader, and ACKs them.
pub struct ZmqActiveMessageWorker {}

impl ZmqActiveMessageWorker {
    pub fn new(
        sub_url: &str,
        push_url: &str,
        message_handlers: MessageHandlers,
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
