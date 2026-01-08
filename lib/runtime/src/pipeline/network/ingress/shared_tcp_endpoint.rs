// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use parking_lot::{Mutex, RwLock};
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
    /// The address to bind to (may have port 0 for OS-assigned port)
    bind_addr: SocketAddr,
    /// The actual bound address (populated after bind_and_start, contains actual port)
    actual_addr: RwLock<Option<SocketAddr>>,
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
            // address we requested to bind to.
            bind_addr,
            // actual address after free port assignment (if DYN_TCP_RPC_PORT is not specified)
            actual_addr: RwLock::new(None),
            cancellation_token,
        })
    }

    /// Bind the server and start accepting connections.
    ///
    /// This method binds to the configured address first, then starts the accept loop.
    /// If the configured port is 0, the OS will assign a free port.
    /// The actual bound address is stored and can be retrieved via `actual_address()`.
    ///
    /// Returns the actual bound address (useful when port 0 was specified).
    pub async fn bind_and_start(self: Arc<Self>) -> Result<SocketAddr> {
        tracing::info!("Binding TCP server to {}", self.bind_addr);

        let listener = TcpListener::bind(&self.bind_addr).await?;
        let actual_addr = listener.local_addr()?;

        tracing::info!(
            requested = %self.bind_addr,
            actual = %actual_addr,
            "TCP server bound successfully"
        );

        // Store the actual bound address
        *self.actual_addr.write() = Some(actual_addr);

        // Start accepting connections in a background task
        let server = self.clone();
        tokio::spawn(async move {
            server.accept_loop(listener).await;
        });

        Ok(actual_addr)
    }

    /// Get the actual bound address (after bind_and_start has been called).
    ///
    /// Returns None if the server hasn't been started yet.
    pub fn actual_address(&self) -> Option<SocketAddr> {
        *self.actual_addr.read()
    }

    /// Internal accept loop - runs after binding
    async fn accept_loop(self: Arc<Self>, listener: TcpListener) {
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
                                    tracing::error!("TCP connection error: {}", e);
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
                    return;
                }
            }
        }
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
            system_health: system_health.clone(),
            inflight: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        });

        // Insert handler FIRST to ensure it's ready to receive requests
        self.handlers.insert(endpoint_path, handler);

        // THEN set health status to Ready (after handler is registered and ready)
        system_health
            .lock()
            .set_endpoint_health_status(&endpoint_name, crate::HealthStatus::Ready);

        tracing::info!(
            "Registered endpoint '{}' with shared TCP server on {}",
            endpoint_name,
            self.actual_address().unwrap_or(self.bind_addr)
        );

        Ok(())
    }

    pub async fn unregister_endpoint(&self, endpoint_path: &str, endpoint_name: &str) {
        if let Some((_, handler)) = self.handlers.remove(endpoint_path) {
            handler
                .system_health
                .lock()
                .set_endpoint_health_status(endpoint_name, crate::HealthStatus::NotReady);
            tracing::info!(
                endpoint_name = %endpoint_name,
                endpoint_path = %endpoint_path,
                "Unregistered TCP endpoint handler"
            );

            let inflight_count = handler.inflight.load(Ordering::SeqCst);
            if inflight_count > 0 {
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    inflight_count = inflight_count,
                    "Waiting for inflight TCP requests to complete"
                );
                while handler.inflight.load(Ordering::SeqCst) > 0 {
                    handler.notify.notified().await;
                }
                tracing::info!(
                    endpoint_name = %endpoint_name,
                    "All inflight TCP requests completed"
                );
            }
        }
    }

    /// Start the server (legacy method - prefer bind_and_start for new code).
    ///
    /// This method is kept for backwards compatibility. It binds and starts
    /// the server but doesn't return the actual bound address.
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let cancel_token = self.cancellation_token.clone();
        self.bind_and_start().await?;
        // Wait for cancellation (the accept loop runs in background)
        cancel_token.cancelled().await;
        Ok(())
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
        // Include instance_id in the routing key to avoid collisions when multiple workers
        // share the same TCP server (e.g., --num-workers > 1 in tests)
        let endpoint_path = format!("{instance_id:x}/{endpoint_name}");
        self.register_endpoint(
            endpoint_path,
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
        // With multiple workers per process, each registers with a unique key
        // "{instance_id}/{endpoint_name}". Find and remove all matching entries.
        let suffix = format!("/{endpoint_name}");
        let keys_to_remove: Vec<String> = self
            .handlers
            .iter()
            .filter(|entry| entry.key().ends_with(&suffix))
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_remove {
            self.unregister_endpoint(&key, endpoint_name).await;
        }
        Ok(())
    }

    fn address(&self) -> String {
        // Return actual bound address if available (after bind_and_start),
        // otherwise fall back to configured bind address
        let addr = self.actual_address().unwrap_or(self.bind_addr);
        format!("tcp://{}:{}", addr.ip(), addr.port())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::error::PipelineError;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::time::Instant;

    /// Mock handler that simulates slow request processing for testing
    struct SlowMockHandler {
        /// Tracks if a request is currently being processed
        request_in_flight: Arc<AtomicBool>,
        /// Notifies when request processing starts
        request_started: Arc<Notify>,
        /// Notifies when request processing completes
        request_completed: Arc<Notify>,
        /// Duration to simulate request processing
        processing_duration: Duration,
    }

    impl SlowMockHandler {
        fn new(processing_duration: Duration) -> Self {
            Self {
                request_in_flight: Arc::new(AtomicBool::new(false)),
                request_started: Arc::new(Notify::new()),
                request_completed: Arc::new(Notify::new()),
                processing_duration,
            }
        }
    }

    #[async_trait]
    impl PushWorkHandler for SlowMockHandler {
        async fn handle_payload(&self, _payload: Bytes) -> Result<(), PipelineError> {
            self.request_in_flight.store(true, Ordering::SeqCst);
            self.request_started.notify_one();

            tracing::debug!(
                "SlowMockHandler: Request started, sleeping for {:?}",
                self.processing_duration
            );

            // Simulate slow request processing
            tokio::time::sleep(self.processing_duration).await;

            tracing::debug!("SlowMockHandler: Request completed");

            self.request_in_flight.store(false, Ordering::SeqCst);
            self.request_completed.notify_one();
            Ok(())
        }

        fn add_metrics(
            &self,
            _endpoint: &crate::component::Endpoint,
            _metrics_labels: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_graceful_shutdown_waits_for_inflight_tcp_requests() {
        // Initialize tracing for test debugging
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        let cancellation_token = CancellationToken::new();
        let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

        // Create SharedTcpServer
        let server = SharedTcpServer::new(bind_addr, cancellation_token.clone());

        // Create a handler that takes 1s to process requests
        let handler = Arc::new(SlowMockHandler::new(Duration::from_secs(1)));
        let request_started = handler.request_started.clone();
        let request_completed = handler.request_completed.clone();
        let request_in_flight = handler.request_in_flight.clone();

        // Register endpoint
        let endpoint_path = "test_endpoint".to_string();
        let system_health = Arc::new(Mutex::new(SystemHealth::new(
            crate::HealthStatus::Ready,
            vec![],
            "/health".to_string(),
            "/live".to_string(),
        )));

        server
            .register_endpoint(
                endpoint_path.clone(),
                handler.clone() as Arc<dyn PushWorkHandler>,
                1,
                "test_namespace".to_string(),
                "test_component".to_string(),
                "test_endpoint".to_string(),
                system_health,
            )
            .await
            .expect("Failed to register endpoint");

        tracing::debug!("Endpoint registered");

        // Get the endpoint handler to simulate request processing
        let endpoint_handler = server
            .handlers
            .get(&endpoint_path)
            .expect("Handler should be registered")
            .clone();

        // Spawn a task that simulates an inflight request
        let request_task = tokio::spawn({
            let handler = handler.clone();
            async move {
                let payload = Bytes::from("test payload");
                handler.handle_payload(payload).await
            }
        });

        // Increment inflight counter manually to simulate the request being tracked
        endpoint_handler.inflight.fetch_add(1, Ordering::SeqCst);

        // Wait for request to start processing
        tokio::select! {
            _ = request_started.notified() => {
                tracing::debug!("Request processing started");
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                panic!("Timeout waiting for request to start");
            }
        }

        // Verify request is in flight
        assert!(
            request_in_flight.load(Ordering::SeqCst),
            "Request should be in flight"
        );

        // Now unregister the endpoint while request is inflight
        let unregister_start = Instant::now();
        tracing::debug!("Starting unregister_endpoint with inflight request");

        // Spawn unregister in a separate task so we can monitor its behavior
        let unregister_task = tokio::spawn({
            let server = server.clone();
            let endpoint_path = endpoint_path.clone();
            async move {
                server
                    .unregister_endpoint(&endpoint_path, "test_endpoint")
                    .await;
                Instant::now()
            }
        });

        // Give unregister a moment to remove handler and start waiting
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify that unregister_endpoint hasn't returned yet (it should be waiting)
        assert!(
            !unregister_task.is_finished(),
            "unregister_endpoint should still be waiting for inflight request"
        );

        tracing::debug!("Verified unregister is waiting, now waiting for request to complete");

        // Wait for the request to complete
        tokio::select! {
            _ = request_completed.notified() => {
                tracing::debug!("Request completed");
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                panic!("Timeout waiting for request to complete");
            }
        }

        // Decrement inflight counter and notify (simulating what the real code does)
        endpoint_handler.inflight.fetch_sub(1, Ordering::SeqCst);
        endpoint_handler.notify.notify_one();

        // Now wait for unregister to complete
        let unregister_end = tokio::time::timeout(Duration::from_secs(2), unregister_task)
            .await
            .expect("unregister_endpoint should complete after inflight request finishes")
            .expect("unregister task should not panic");

        let unregister_duration = unregister_end - unregister_start;

        tracing::debug!("unregister_endpoint completed in {:?}", unregister_duration);

        // Verify unregister_endpoint waited for the inflight request
        assert!(
            unregister_duration >= Duration::from_secs(1),
            "unregister_endpoint should have waited ~1s for inflight request, but only took {:?}",
            unregister_duration
        );

        // Verify request completed successfully
        assert!(
            !request_in_flight.load(Ordering::SeqCst),
            "Request should have completed"
        );

        // Wait for request task to finish
        request_task
            .await
            .expect("Request task should complete")
            .expect("Request should succeed");

        tracing::info!("Test passed: unregister_endpoint properly waited for inflight TCP request");
    }
}
