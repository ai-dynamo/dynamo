// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-performance TCP listener for ActiveMessage transport
//!
//! This module provides a TCP server that accepts incoming connections,
//! decodes framed messages using zero-copy techniques, and routes them
//! to the appropriate transport streams.

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::StreamExt;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener as TokioTcpListener, TcpStream};
use tokio::runtime::{Handle, Runtime};
use tokio_util::codec::Framed;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::{MessageType, TransportAdapter, TransportErrorHandler};

use super::framing::TcpFrameCodec;

/// Runtime configuration for the TCP listener
pub enum RuntimeConfig {
    /// Use an existing tokio runtime handle
    Handle(Handle),
    /// Use a provided tokio runtime
    Runtime(Arc<Runtime>),
    /// Create a single-threaded runtime pinned to the specified CPU core
    CpuPin(usize),
}

/// High-performance TCP listener for ActiveMessage transport
///
/// This listener accepts incoming TCP connections and routes decoded frames
/// to the appropriate transport streams with zero-copy performance.
pub struct TcpListener {
    bind_addr: SocketAddr,
    adapter: TransportAdapter,
    error_handler: Arc<dyn TransportErrorHandler>,
    cancel_token: CancellationToken,
    runtime_config: RuntimeConfig,
    listener: Option<std::net::TcpListener>,
}

impl TcpListener {
    /// Create a new builder for TcpListener
    pub fn builder() -> TcpListenerBuilder {
        TcpListenerBuilder::new()
    }

    /// Start the listener and serve incoming connections
    ///
    /// This method blocks or spawns based on the runtime configuration:
    /// - For Handle/Runtime: spawns tasks and returns immediately
    /// - For CpuPin: creates a pinned runtime and blocks until cancellation
    pub async fn serve(mut self) -> Result<()> {
        // Extract runtime config to avoid borrow issues
        let runtime_config = std::mem::replace(
            &mut self.runtime_config,
            RuntimeConfig::Handle(Handle::current()),
        );

        match runtime_config {
            RuntimeConfig::Handle(handle) => {
                handle.spawn(self.run_server());
                Ok(())
            }
            RuntimeConfig::Runtime(rt) => {
                rt.spawn(self.run_server());
                Ok(())
            }
            RuntimeConfig::CpuPin(cpu_id) => {
                let rt = Self::create_pinned_runtime(cpu_id)
                    .context("Failed to create CPU-pinned runtime")?;
                rt.block_on(self.run_server())
            }
        }
    }

    /// Create a single-threaded runtime pinned to a specific CPU core
    #[cfg(target_os = "linux")]
    fn create_pinned_runtime(cpu_id: usize) -> Result<Runtime> {
        use nix::sched::{CpuSet, sched_setaffinity};
        use nix::unistd::Pid;

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .thread_name("tcp-listener-pinned")
            .on_thread_start(move || {
                let mut cpu_set = CpuSet::new();
                if cpu_set.set(cpu_id).is_ok() {
                    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpu_set) {
                        error!("Failed to pin thread to CPU {}: {}", cpu_id, e);
                    } else {
                        debug!("Successfully pinned TCP listener to CPU {}", cpu_id);
                    }
                }
            })
            .build()
            .context("Failed to build tokio runtime")
    }

    /// Create a single-threaded runtime without CPU pinning (non-Linux platforms)
    #[cfg(not(target_os = "linux"))]
    fn create_pinned_runtime(cpu_id: usize) -> Result<Runtime> {
        warn!(
            "CPU pinning requested (CPU {}) but not supported on this platform",
            cpu_id
        );
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .thread_name("tcp-listener")
            .build()
            .context("Failed to build tokio runtime")
    }

    /// Main server loop that accepts connections
    async fn run_server(self) -> Result<()> {
        // Use pre-bound listener if provided, otherwise bind to the address
        let listener = if let Some(std_listener) = self.listener {
            // Set non-blocking for tokio conversion
            std_listener
                .set_nonblocking(true)
                .context("Failed to set listener to non-blocking")?;

            TokioTcpListener::from_std(std_listener)
                .context("Failed to convert std TcpListener to tokio TcpListener")?
        } else {
            TokioTcpListener::bind(self.bind_addr)
                .await
                .context(format!("Failed to bind TCP listener to {}", self.bind_addr))?
        };

        let local_addr = listener
            .local_addr()
            .context("Failed to get local address")?;
        info!("TCP listener bound to {}", local_addr);

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            debug!("Accepted TCP connection from {}", peer_addr);

                            let adapter = self.adapter.clone();
                            let error_handler = self.error_handler.clone();
                            let cancel_token = self.cancel_token.clone();

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(
                                    stream,
                                    peer_addr,
                                    adapter,
                                    error_handler,
                                    cancel_token,
                                )
                                .await
                                {
                                    warn!("Error handling connection from {}: {}", peer_addr, e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept TCP connection: {}", e);
                        }
                    }
                }
                _ = self.cancel_token.cancelled() => {
                    info!("TCP listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a single TCP connection
    async fn handle_connection(
        stream: TcpStream,
        peer_addr: SocketAddr,
        adapter: TransportAdapter,
        error_handler: Arc<dyn TransportErrorHandler>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        debug!("Configuring connection from {}", peer_addr);

        // Configure socket for high performance
        if let Err(e) = stream.set_nodelay(true) {
            warn!("Failed to set TCP_NODELAY on {}: {}", peer_addr, e);
        }

        if let Err(e) = stream.set_linger(Some(Duration::from_secs(1))) {
            warn!("Failed to set linger on {}: {}", peer_addr, e);
        }

        // Set keep-alive to detect dead connections
        let keepalive = socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));

        let sock_ref = socket2::SockRef::from(&stream);
        if let Err(e) = sock_ref.set_tcp_keepalive(&keepalive) {
            warn!("Failed to set TCP keepalive on {}: {}", peer_addr, e);
        }

        // Set large receive buffer for high throughput
        if let Err(e) = sock_ref.set_recv_buffer_size(1_048_576) {
            warn!("Failed to set receive buffer size on {}: {}", peer_addr, e);
        }

        // Create framed stream with zero-copy codec
        let mut framed = Framed::new(stream, TcpFrameCodec::new());

        debug!("Connection from {} ready for frames", peer_addr);

        loop {
            tokio::select! {
                frame_result = framed.next() => {
                    match frame_result {
                        Some(Ok((msg_type, header, payload))) => {
                            // Route frame to appropriate stream based on type
                            if let Err(e) = Self::route_frame(
                                msg_type,
                                header,
                                payload,
                                &adapter,
                                &error_handler,
                            )
                            .await
                            {
                                warn!(
                                    "Failed to route {:?} frame from {}: {}",
                                    msg_type, peer_addr, e
                                );
                            }
                        }
                        Some(Err(e)) => {
                            error!("Frame decode error from {}: {}", peer_addr, e);
                            break;
                        }
                        None => {
                            // Connection closed gracefully (FIN received)
                            debug!("Connection from {} closed gracefully", peer_addr);
                            break;
                        }
                    }
                }
                _ = cancel_token.cancelled() => {
                    // if the cancel_token is triggered, we should continue to accept responses for outstanding requests
                    // however, we should stop accepting new requests
                    //
                    // todo:
                    // - drop the adapter.message_stream
                    // - if the a new message comes in, reply with an error
                    // - we should stay alive until all outstanding requests have finished
                    //   - under that condition, we should get a second stage of cancellation
                    //   - we will need a stage_two_cancel_token instead of the current stage_one_cancel_token
                    // - on stage two shutdown, we issue a SHUT_WR which will issue a FIN, then we must continue to pull all
                    //   outstanding responses from framed until it's closed
                    // - once all open recv sockets have closed, then we can issue stage_three_cancel_token.cancel() which
                    //   will tell the senders they can exit because all outstanding requests have finished and no more
                    //   incoming messages can be received.
                    //
                    // - this will close all the adapter channels and begin the cascading shutdown of the transport
                    // - we will need an inflight counter for receives - which the response manager can track from it's free
                    //   list - once all awaiting responses have returned to the arena we are done.

                    debug!("Connection handler for {} cancelled", peer_addr);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Route a decoded frame to the appropriate stream
    ///
    /// This function performs zero-copy routing by transferring ownership of
    /// the Bytes to the flume channel. On error, it invokes the error callback
    /// with the original data (requiring a clone).
    async fn route_frame(
        msg_type: MessageType,
        header: Bytes,
        payload: Bytes,
        adapter: &TransportAdapter,
        error_handler: &Arc<dyn TransportErrorHandler>,
    ) -> Result<()> {
        let sender = match msg_type {
            MessageType::Message => &adapter.message_stream,
            MessageType::Response => &adapter.response_stream,
            MessageType::Ack | MessageType::Event => &adapter.event_stream,
        };

        // Try to send with ownership transfer (zero-copy)
        match sender.send_async((header, payload)).await {
            Ok(_) => Ok(()),
            Err(e) => {
                // Send failed - invoke error callback with the data
                error_handler.on_error(
                    e.0.0, // header
                    e.0.1, // payload
                    format!("Failed to route {:?}", msg_type),
                );
                Err(anyhow::anyhow!("Failed to send to stream"))
            }
        }
    }
}

/// Builder for TcpListener
pub struct TcpListenerBuilder {
    bind_addr: Option<SocketAddr>,
    adapter: Option<TransportAdapter>,
    error_handler: Option<Arc<dyn TransportErrorHandler>>,
    cancel_token: Option<CancellationToken>,
    runtime_config: Option<RuntimeConfig>,
    listener: Option<std::net::TcpListener>,
}

impl TcpListenerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            bind_addr: None,
            adapter: None,
            error_handler: None,
            cancel_token: None,
            runtime_config: None,
            listener: None,
        }
    }

    /// Set the bind address
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }

    /// Set the transport adapter
    pub fn adapter(mut self, adapter: TransportAdapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Set the error handler
    pub fn error_handler(mut self, handler: Arc<dyn TransportErrorHandler>) -> Self {
        self.error_handler = Some(handler);
        self
    }

    /// Set the cancellation token
    pub fn cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    /// Use an existing tokio runtime handle
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.runtime_config = Some(RuntimeConfig::Handle(handle));
        self
    }

    /// Use a provided tokio runtime
    pub fn with_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime_config = Some(RuntimeConfig::Runtime(runtime));
        self
    }

    /// Create a single-threaded runtime pinned to a specific CPU core
    pub fn with_cpu_pin(mut self, cpu_id: usize) -> Self {
        self.runtime_config = Some(RuntimeConfig::CpuPin(cpu_id));
        self
    }

    /// Use a pre-bound TcpListener
    ///
    /// This is useful for tests where you want to bind to port 0 and avoid port races.
    /// When provided, the bind_addr should still be set (for logging/debugging purposes).
    pub fn listener(mut self, listener: Option<std::net::TcpListener>) -> Self {
        self.listener = listener;
        self
    }

    /// Build the TcpListener
    pub fn build(self) -> Result<TcpListener> {
        let bind_addr = self
            .bind_addr
            .ok_or_else(|| anyhow::anyhow!("bind_addr is required"))?;
        let adapter = self
            .adapter
            .ok_or_else(|| anyhow::anyhow!("adapter is required"))?;
        let error_handler = self
            .error_handler
            .ok_or_else(|| anyhow::anyhow!("error_handler is required"))?;
        let cancel_token = self.cancel_token.unwrap_or_default();
        let runtime_config = self
            .runtime_config
            .unwrap_or_else(|| RuntimeConfig::Handle(Handle::current()));

        Ok(TcpListener {
            bind_addr,
            adapter,
            error_handler,
            cancel_token,
            runtime_config,
            listener: self.listener,
        })
    }
}

impl Default for TcpListenerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::make_channels;
    use std::net::{IpAddr, Ipv4Addr};

    struct TestErrorHandler;

    impl TransportErrorHandler for TestErrorHandler {
        fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
            eprintln!("Test error handler: {}", error);
        }
    }

    #[test]
    fn test_builder_requires_fields() {
        let result = TcpListener::builder().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_builder_with_all_fields() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (adapter, _streams) = make_channels();
        let error_handler = Arc::new(TestErrorHandler);

        let result = TcpListener::builder()
            .bind_addr(bind_addr)
            .adapter(adapter)
            .error_handler(error_handler)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_cpu_pin() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (adapter, _streams) = make_channels();
        let error_handler = Arc::new(TestErrorHandler);

        let result = TcpListener::builder()
            .bind_addr(bind_addr)
            .adapter(adapter)
            .error_handler(error_handler)
            .with_cpu_pin(0)
            .build();

        assert!(result.is_ok());
    }
}
