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
use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

/// Shared TCP server that handles multiple endpoints on a single port
pub struct SharedTcpServer {
    handlers: Arc<tokio::sync::RwLock<HashMap<String, Arc<EndpointHandler>>>>,
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
            handlers: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            bind_addr,
            cancellation_token,
        })
    }

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

        self.handlers.write().await.insert(endpoint_path, handler);

        tracing::info!(
            "Registered endpoint '{}' with shared TCP server on {}",
            endpoint_name,
            self.bind_addr
        );

        Ok(())
    }

    pub async fn unregister_endpoint(&self, endpoint_path: &str, endpoint_name: &str) {
        self.handlers.write().await.remove(endpoint_path);
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
        mut stream: TcpStream,
        handlers: Arc<tokio::sync::RwLock<HashMap<String, Arc<EndpointHandler>>>>,
    ) -> Result<()> {
        loop {
            // Read endpoint path length
            let mut path_len_buf = [0u8; 2];
            match stream.read_exact(&mut path_len_buf).await {
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
            stream.read_exact(&mut path_buf).await?;
            let endpoint_path = String::from_utf8(path_buf)?;

            // Read payload length
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf).await?;
            let len = u32::from_be_bytes(len_buf) as usize;

            // Sanity check
            if len > 16 * 1024 * 1024 {
                anyhow::bail!("Request too large: {} bytes", len);
            }

            // Read request payload
            let mut payload = vec![0u8; len];
            stream.read_exact(&mut payload).await?;

            // Look up handler
            let handler = {
                let handlers_read = handlers.read().await;
                handlers_read.get(&endpoint_path).cloned()
            };

            let handler = match handler {
                Some(h) => h,
                None => {
                    tracing::warn!("No handler found for endpoint: {}", endpoint_path);
                    // Send error response
                    let error_msg = format!("Unknown endpoint: {}", endpoint_path);
                    let error_bytes = error_msg.as_bytes();
                    let error_len = error_bytes.len() as u32;
                    stream.write_all(&error_len.to_be_bytes()).await?;
                    stream.write_all(error_bytes).await?;
                    stream.flush().await?;
                    continue;
                }
            };

            handler.inflight.fetch_add(1, Ordering::SeqCst);

            // Send acknowledgment immediately
            let ack = b"";
            let ack_len = ack.len() as u32;
            stream.write_all(&ack_len.to_be_bytes()).await?;
            stream.write_all(ack).await?;
            stream.flush().await?;

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
                    .handle_payload(Bytes::from(payload))
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
}
