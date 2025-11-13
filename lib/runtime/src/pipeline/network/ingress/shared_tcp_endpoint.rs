// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared TCP Server with Endpoint Multiplexing
//!
//! Provides a shared TCP server that can handle multiple endpoints on a single port
//! by adding endpoint routing to the TCP wire protocol.

use crate::SystemHealth;
use crate::pipeline::network::PushWorkHandler;
use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio_util::bytes::BytesMut;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Default maximum message size for TCP server (32 MB)
const DEFAULT_MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// Get maximum message size from environment or use default
fn get_max_message_size() -> usize {
    std::env::var("DYN_TCP_MAX_MESSAGE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_MESSAGE_SIZE)
}

/// Shared TCP server that handles multiple endpoints on a single port
pub struct SharedTcpServer {
    handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
    bind_addr: SocketAddr,
    cancellation_token: CancellationToken,
}

struct EndpointHandler {
    service_handler: Arc<dyn PushWorkHandler>,
    instance_id: u64,
    namespace: String,
    component_name: String,
    endpoint_name: String,
    system_health: Arc<Mutex<SystemHealth>>,
    inflight: Arc<AtomicU64>,
    notify: Arc<Notify>,
}

impl SharedTcpServer {
    pub fn new(bind_addr: SocketAddr, cancellation_token: CancellationToken) -> Arc<Self> {
        Arc::new(Self {
            handlers: Arc::new(DashMap::new()),
            bind_addr,
            cancellation_token,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn register_endpoint(
        &self,
        endpoint_path: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        endpoint_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        let handler = Arc::new(EndpointHandler {
            service_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name: endpoint_name.clone(),
            system_health,
            inflight: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        });

        self.handlers.insert(endpoint_path, handler);

        tracing::info!(
            "Registered endpoint '{}' with shared TCP server on {}",
            endpoint_name,
            self.bind_addr
        );

        Ok(())
    }

    pub async fn unregister_endpoint(&self, endpoint_path: &str, endpoint_name: &str) {
        self.handlers.remove(endpoint_path);
        tracing::info!(
            "Unregistered endpoint '{}' from shared TCP server",
            endpoint_name
        );
    }

    pub async fn start(self: Arc<Self>) -> Result<()> {
        tracing::info!("Starting shared TCP server on {}", self.bind_addr);

        let listener = TcpListener::bind(&self.bind_addr).await?;
        let cancellation_token = self.cancellation_token.clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            tracing::trace!("Accepted TCP connection from {}", peer_addr);

                            let handlers = self.handlers.clone();
                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(stream, handlers).await {
                                    tracing::debug!("TCP connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!("Failed to accept TCP connection: {}", e);
                        }
                    }
                }
                _ = cancellation_token.cancelled() => {
                    tracing::info!("SharedTcpServer received cancellation signal, shutting down");
                    return Ok(());
                }
            }
        }
    }

    async fn handle_connection(
        stream: TcpStream,
        handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
    ) -> Result<()> {
        use crate::pipeline::network::codec::{TcpRequestMessage, TcpResponseMessage};

        // Split stream into read and write halves for concurrent operations
        let (read_half, write_half) = tokio::io::split(stream);

        // Channel for sending responses to the write task (zero-copy Bytes)
        let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();

        // Spawn write task
        let write_task = tokio::spawn(Self::write_loop(write_half, response_rx));

        // Run read task in current context
        let read_result = Self::read_loop(read_half, handlers, response_tx).await;

        // Write task will end when response_tx is dropped
        write_task.await??;

        read_result
    }

    async fn read_loop(
        mut read_half: tokio::io::ReadHalf<TcpStream>,
        handlers: Arc<DashMap<String, Arc<EndpointHandler>>>,
        response_tx: tokio::sync::mpsc::UnboundedSender<Bytes>,
    ) -> Result<()> {
        use crate::pipeline::network::codec::{TcpRequestMessage, TcpResponseMessage};

        loop {
            // Read endpoint path length (2 bytes)
            let mut path_len_buf = [0u8; 2];
            match read_half.read_exact(&mut path_len_buf).await {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }

            let path_len = u16::from_be_bytes(path_len_buf) as usize;

            // Read endpoint path
            let mut path_buf = vec![0u8; path_len];
            read_half.read_exact(&mut path_buf).await?;

            // Read payload length (4 bytes)
            let mut len_buf = [0u8; 4];
            read_half.read_exact(&mut len_buf).await?;
            let payload_len = u32::from_be_bytes(len_buf) as usize;

            // Sanity check - enforce maximum message size
            let max_message_size = get_max_message_size();
            if payload_len > max_message_size {
                tracing::warn!(
                    "Request too large: {} bytes (max: {} bytes), closing connection",
                    payload_len,
                    max_message_size
                );
                // Send error response
                let error_response =
                    TcpResponseMessage::new(Bytes::from_static(b"Request too large"));
                if let Ok(encoded) = error_response.encode() {
                    let _ = response_tx.send(encoded);
                }
                break;
            }

            // Read request payload
            let mut payload_buf = vec![0u8; payload_len];
            read_half.read_exact(&mut payload_buf).await?;

            // Reconstruct the full message buffer for decoding using BytesMut
            let mut full_msg = BytesMut::with_capacity(2 + path_len + 4 + payload_len);
            full_msg.extend_from_slice(&path_len_buf);
            full_msg.extend_from_slice(&path_buf);
            full_msg.extend_from_slice(&len_buf);
            full_msg.extend_from_slice(&payload_buf);

            // Decode using codec (zero-copy conversion)
            let full_msg_bytes = full_msg.freeze();
            let request_msg = match TcpRequestMessage::decode(&full_msg_bytes) {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!("Failed to decode TCP request: {}", e);
                    // Send error response
                    let error_response =
                        TcpResponseMessage::new(Bytes::from(format!("Decode error: {}", e)));
                    if let Ok(encoded) = error_response.encode() {
                        let _ = response_tx.send(encoded);
                    }
                    continue;
                }
            };

            let endpoint_path = request_msg.endpoint_path;
            let payload = request_msg.payload;

            // Look up handler (lock-free read with DashMap)
            let handler = handlers.get(&endpoint_path).map(|h| h.clone());

            let handler = match handler {
                Some(h) => h,
                None => {
                    tracing::warn!("No handler found for endpoint: {}", endpoint_path);
                    // Send error response using codec
                    let error_response = TcpResponseMessage::new(Bytes::from(format!(
                        "Unknown endpoint: {}",
                        endpoint_path
                    )));
                    if let Ok(encoded) = error_response.encode() {
                        let _ = response_tx.send(encoded);
                    }
                    continue;
                }
            };

            handler.inflight.fetch_add(1, Ordering::SeqCst);

            // Send acknowledgment immediately using codec (non-blocking, zero-copy)
            let ack_response = TcpResponseMessage::empty();
            if let Ok(encoded_ack) = ack_response.encode() {
                // Send to write task without blocking reads
                if response_tx.send(encoded_ack).is_err() {
                    tracing::debug!("Write task closed, ending read loop");
                    break;
                }
            }

            // Process request asynchronously
            let service_handler = handler.service_handler.clone();
            let inflight = handler.inflight.clone();
            let notify = handler.notify.clone();
            let instance_id = handler.instance_id;
            let namespace = handler.namespace.clone();
            let component_name = handler.component_name.clone();
            let endpoint_name = handler.endpoint_name.clone();

            tokio::spawn(async move {
                tracing::trace!(instance_id, "handling TCP request");

                let result = service_handler
                    .handle_payload(payload)
                    .instrument(tracing::info_span!(
                        "handle_payload",
                        component = component_name.as_str(),
                        endpoint = endpoint_name.as_str(),
                        namespace = namespace.as_str(),
                        instance_id = instance_id,
                    ))
                    .await;

                match result {
                    Ok(_) => {
                        tracing::trace!(instance_id, "TCP request handled successfully");
                    }
                    Err(e) => {
                        tracing::warn!("Failed to handle TCP request: {}", e);
                    }
                }

                inflight.fetch_sub(1, Ordering::SeqCst);
                notify.notify_one();
            });
        }

        Ok(())
    }

    async fn write_loop(
        mut write_half: tokio::io::WriteHalf<TcpStream>,
        mut response_rx: tokio::sync::mpsc::UnboundedReceiver<Bytes>,
    ) -> Result<()> {
        while let Some(response) = response_rx.recv().await {
            write_half.write_all(&response).await?;
            write_half.flush().await?;
        }
        Ok(())
    }
}

// Implement RequestPlaneServer trait for SharedTcpServer
#[async_trait::async_trait]
impl super::unified_server::RequestPlaneServer for SharedTcpServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        // For TCP, we use endpoint_name as both the endpoint_path (routing key) and endpoint_name
        self.register_endpoint(
            endpoint_name.clone(),
            service_handler,
            instance_id,
            namespace,
            component_name,
            endpoint_name,
            system_health,
        )
        .await
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        self.unregister_endpoint(endpoint_name, endpoint_name).await;
        Ok(())
    }

    fn address(&self) -> String {
        format!("tcp://{}:{}", self.bind_addr.ip(), self.bind_addr.port())
    }

    fn transport_name(&self) -> &'static str {
        "tcp"
    }

    fn is_healthy(&self) -> bool {
        // Server is healthy if it has been created
        // TODO: Add more sophisticated health checks (e.g., check if listener is active)
        true
    }
}
