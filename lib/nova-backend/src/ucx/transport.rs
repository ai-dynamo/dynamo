// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-performance UCX transport with lazy endpoint creation

use anyhow::Result;
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::transport::{HealthCheckError, TransportError, TransportErrorHandler};
use crate::{MessageType, PeerInfo, Transport, TransportAdapter, TransportKey, WorkerAddress};

use super::runtime::{GetWorkerAddressRequest, SendTask, spawn_ucx_runtime};

/// High-performance UCX transport with lock-free concurrent access
///
/// This transport uses DashMap for lock-free concurrent access to peer information.
/// All UCX operations (which use Rc) are marshalled to a dedicated LocalSet thread.
pub struct UcxTransport {
    // Identity (immutable, no wrapper needed)
    key: TransportKey,
    local_address: Arc<std::sync::Mutex<WorkerAddress>>,

    // Shared mutable state with DashMap (lock-free)
    peers: Arc<DashMap<dynamo_identity::InstanceId, Bytes>>, // decoded UCX worker addresses

    // Channels to send tasks to LocalSet runtime
    msg_tx: flume::Sender<SendTask>,
    resp_tx: flume::Sender<SendTask>,
    event_tx: flume::Sender<SendTask>,

    // Receiver sides (stored until start() is called)
    msg_rx: Arc<std::sync::Mutex<Option<flume::Receiver<SendTask>>>>,
    resp_rx: Arc<std::sync::Mutex<Option<flume::Receiver<SendTask>>>>,
    event_rx: Arc<std::sync::Mutex<Option<flume::Receiver<SendTask>>>>,

    // Channel to request worker address from runtime (set during start)
    worker_addr_tx: Arc<std::sync::Mutex<Option<mpsc::UnboundedSender<GetWorkerAddressRequest>>>>,

    // Channel to send control messages to endpoint manager (set during start)
    endpoint_control_tx: Arc<
        std::sync::Mutex<Option<mpsc::UnboundedSender<super::runtime::EndpointControlMessage>>>,
    >,

    // Shutdown coordination
    cancel_token: CancellationToken,

    // Thread handle (kept for cleanup)
    runtime_thread: Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
}

impl UcxTransport {
    /// Create a new UCX transport
    ///
    /// Note: The transport is not started until `start()` is called.
    fn new(key: TransportKey) -> Self {
        // Create channels for sending tasks to runtime
        let (msg_tx, msg_rx) = flume::unbounded();
        let (resp_tx, resp_rx) = flume::unbounded();
        let (event_tx, event_rx) = flume::unbounded();

        Self {
            key,
            local_address: Arc::new(std::sync::Mutex::new(
                WorkerAddress::builder().build().unwrap(),
            )),
            peers: Arc::new(DashMap::new()),
            msg_tx,
            resp_tx,
            event_tx,
            msg_rx: Arc::new(std::sync::Mutex::new(Some(msg_rx))),
            resp_rx: Arc::new(std::sync::Mutex::new(Some(resp_rx))),
            event_rx: Arc::new(std::sync::Mutex::new(Some(event_rx))),
            worker_addr_tx: Arc::new(std::sync::Mutex::new(None)),
            endpoint_control_tx: Arc::new(std::sync::Mutex::new(None)),
            cancel_token: CancellationToken::new(),
            runtime_thread: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Slow path for send_message when fast path fails
    #[cold]
    fn send_slow_path(tx: flume::Sender<SendTask>, task: SendTask) {
        // Spawn task to handle backpressure
        tokio::spawn(async move {
            if let Err(e) = tx.send_async(task).await {
                error!("Failed to send to UCX runtime: {}", e);
            }
        });
    }
}

impl Transport for UcxTransport {
    fn key(&self) -> TransportKey {
        self.key.clone()
    }

    fn address(&self) -> WorkerAddress {
        self.local_address.lock().unwrap().clone()
    }

    fn register(&self, peer_info: PeerInfo) -> Result<(), TransportError> {
        // Get endpoint from peer's address
        let encoded_address = peer_info
            .worker_address()
            .get_entry(&self.key)
            .map_err(|_| TransportError::NoEndpoint)?
            .ok_or(TransportError::NoEndpoint)?;

        // // endpoint is already base64-encoded (from start()), decode it once
        // let base64_blob = String::from_utf8(endpoint.to_vec())
        //     .map_err(|_| TransportError::InvalidEndpoint)?;

        // Decode base64 once during registration
        let ucx_bytes =
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &encoded_address)
                .map_err(|_| TransportError::InvalidEndpoint)?;

        let decoded_address = Bytes::from(ucx_bytes);

        // Store decoded UCX worker address
        self.peers
            .insert(peer_info.instance_id(), decoded_address.clone());

        debug!(
            "Registered peer {} with UCX blob ({} bytes decoded from {} bytes base64)",
            peer_info.instance_id(),
            decoded_address.len(),
            encoded_address.len()
        );

        Ok(())
    }

    #[inline]
    fn send_message(
        &self,
        instance_id: dynamo_identity::InstanceId,
        header: Vec<u8>,
        payload: Vec<u8>,
        message_type: MessageType,
        on_error: std::sync::Arc<dyn TransportErrorHandler>,
    ) {
        // Convert to Bytes (caller's thread)
        let header = Bytes::from(header);
        let payload = Bytes::from(payload);

        // Look up peer UCX blob (fast DashMap read)
        let peer = match self.peers.get(&instance_id) {
            Some(p) => p.clone(),
            None => {
                on_error.on_error(
                    header,
                    payload,
                    format!("peer not registered: {}", instance_id),
                );
                return;
            }
        };

        let task = SendTask {
            target_instance: instance_id,
            target_blob: peer,
            header,
            payload,
            on_error,
        };

        // Select channel based on MessageType
        let tx = match message_type {
            MessageType::Message => &self.msg_tx,
            MessageType::Response => &self.resp_tx,
            MessageType::Ack | MessageType::Event => &self.event_tx,
        };

        // Fast path: non-blocking send
        if tx.try_send(task.clone()).is_ok() {
            return; // Fast path success
        }

        // Slow path: spawn task to handle backpressure
        Self::send_slow_path(tx.clone(), task);
    }

    fn start(
        &self,
        _instance_id: dynamo_identity::InstanceId,
        channels: TransportAdapter,
        _rt: tokio::runtime::Handle,
    ) -> futures::future::BoxFuture<'_, anyhow::Result<()>> {
        let msg_rx = self.msg_rx.lock().unwrap().take();
        let resp_rx = self.resp_rx.lock().unwrap().take();
        let event_rx = self.event_rx.lock().unwrap().take();

        let cancel_token = self.cancel_token.clone();
        let runtime_thread = self.runtime_thread.clone();
        let worker_addr_tx_arc = self.worker_addr_tx.clone();
        let endpoint_control_tx_arc = self.endpoint_control_tx.clone();
        let local_address_arc = self.local_address.clone();
        let key = self.key.clone();

        Box::pin(async move {
            // Take receiver sides from storage (they can only be used once)
            let msg_rx = msg_rx.ok_or_else(|| anyhow::anyhow!("Transport already started"))?;
            let resp_rx = resp_rx.ok_or_else(|| anyhow::anyhow!("Transport already started"))?;
            let event_rx = event_rx.ok_or_else(|| anyhow::anyhow!("Transport already started"))?;

            // Spawn UCX runtime (requires dedicated thread for LocalSet)
            let (worker_addr_tx, endpoint_control_tx, thread) =
                spawn_ucx_runtime(channels, msg_rx, resp_rx, event_rx, cancel_token)?;

            // Store thread handle and channels
            *runtime_thread.lock().unwrap() = Some(thread);
            *worker_addr_tx_arc.lock().unwrap() = Some(worker_addr_tx.clone());
            *endpoint_control_tx_arc.lock().unwrap() = Some(endpoint_control_tx);

            // Get worker address from runtime
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            worker_addr_tx.send(GetWorkerAddressRequest { reply: reply_tx })?;

            // Wait for worker address - spawn a thread to avoid blocking the runtime
            let ucx_blob = std::thread::spawn(move || reply_rx.blocking_recv())
                .join()
                .map_err(|_| anyhow::anyhow!("Worker address wait thread panicked"))?
                .map_err(|_| anyhow::anyhow!("UCX runtime died before sending worker address"))??;

            // Build local address with base64 encoded UCX blob
            let base64_blob = base64::engine::general_purpose::STANDARD.encode(&ucx_blob);
            let mut addr_builder = WorkerAddress::builder();
            addr_builder.add_entry(key, base64_blob.as_bytes().to_vec())?;
            let local_address = addr_builder.build()?;

            *local_address_arc.lock().unwrap() = local_address;

            info!(
                "UCX transport started with worker address ({} bytes)",
                ucx_blob.len()
            );

            Ok(())
        })
    }

    fn shutdown(&self) {
        info!("Shutting down UCX transport");
        self.cancel_token.cancel();

        // Wait for runtime thread to exit
        if let Some(handle) = self.runtime_thread.lock().unwrap().take()
            && let Err(e) = handle.join()
        {
            error!("UCX runtime thread panicked: {:?}", e);
        }

        // Clear peers
        self.peers.clear();
    }

    fn check_health(
        &self,
        instance_id: dynamo_identity::InstanceId,
        timeout: Duration,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<(), HealthCheckError>> + Send + '_>,
    > {
        // First verify peer is registered
        if !self.peers.contains_key(&instance_id) {
            return Box::pin(async move { Err(HealthCheckError::PeerNotRegistered) });
        }

        // Send health check control message to endpoint manager (before async block)
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let send_result = self
            .endpoint_control_tx
            .lock()
            .unwrap()
            .as_ref()
            .ok_or(HealthCheckError::TransportNotStarted)
            .and_then(|tx| {
                tx.send(super::runtime::EndpointControlMessage::CheckHealth {
                    instance_id,
                    reply: reply_tx,
                })
                .map_err(|_| HealthCheckError::TransportNotStarted)
            });

        if let Err(e) = send_result {
            return Box::pin(async move { Err(e) });
        }

        Box::pin(async move {
            // Wait for reply with timeout
            match tokio::time::timeout(timeout, reply_rx).await {
                Ok(Ok(status)) => {
                    use super::runtime::EndpointHealthStatus;
                    match status {
                        EndpointHealthStatus::Healthy => Ok(()),
                        EndpointHealthStatus::Unhealthy => Err(HealthCheckError::ConnectionFailed),
                        EndpointHealthStatus::NotConnected => Err(HealthCheckError::NeverConnected),
                    }
                }
                Ok(Err(_)) => Err(HealthCheckError::ConnectionFailed),
                Err(_) => Err(HealthCheckError::Timeout),
            }
        })
    }
}

/// Builder for UcxTransport
pub struct UcxTransportBuilder {
    key: Option<TransportKey>,
}

impl UcxTransportBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self { key: None }
    }

    /// Set the transport key
    pub fn key(mut self, key: TransportKey) -> Self {
        self.key = Some(key);
        self
    }

    /// Build the UcxTransport
    pub fn build(self) -> Result<UcxTransport> {
        let key = self.key.unwrap_or_else(|| TransportKey::from("ucx"));

        Ok(UcxTransport::new(key))
    }
}

impl Default for UcxTransportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let result = UcxTransportBuilder::new().build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_key() {
        let result = UcxTransportBuilder::new()
            .key(TransportKey::from("my-ucx"))
            .build();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().key().as_str(), "my-ucx");
    }
}
