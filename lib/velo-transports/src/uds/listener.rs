// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! UDS listener for ActiveMessage transport
//!
//! Mirrors `tcp/listener.rs` but uses `UnixListener`/`UnixStream`.
//! Reuses `TcpFrameCodec` for framing. Supports drain-aware frame handling
//! via `ShutdownState`.

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UnixListener as TokioUnixListener;
use tokio::net::UnixStream;
use tokio_util::codec::Framed;
use tracing::{debug, error, info, warn};

use crate::{MessageType, ShutdownState, TransportAdapter, TransportErrorHandler};

use crate::tcp::TcpFrameCodec;

/// UDS listener for ActiveMessage transport
///
/// Accepts incoming Unix domain socket connections and routes decoded frames
/// to the appropriate transport streams. Supports graceful drain: during drain,
/// new `Message` frames are rejected with a `ShuttingDown` response while
/// `Response`/`Event`/`Ack` frames continue to flow.
pub struct UdsListener {
    socket_path: PathBuf,
    adapter: TransportAdapter,
    error_handler: Arc<dyn TransportErrorHandler>,
    shutdown_state: ShutdownState,
}

/// UDS listener that has been bound to a socket path, ready to accept connections.
///
/// Created by [`UdsListener::bind`]. Holding this value proves the OS-level bind
/// succeeded, so callers can detect failures before spawning a task.
pub struct BoundUdsListener {
    socket_path: PathBuf,
    adapter: TransportAdapter,
    error_handler: Arc<dyn TransportErrorHandler>,
    shutdown_state: ShutdownState,
    listener: TokioUnixListener,
}

impl UdsListener {
    /// Create a new builder for UdsListener
    pub fn builder() -> UdsListenerBuilder {
        UdsListenerBuilder::new()
    }

    /// Bind to the socket path and return a [`BoundUdsListener`] ready to serve.
    ///
    /// `TokioUnixListener::bind` is synchronous, so this method is also
    /// synchronous. Callers that need to propagate bind failures before spawning
    /// a task should call `bind()` first, then spawn `bound.serve()`.
    pub fn bind(self) -> Result<BoundUdsListener> {
        let listener = TokioUnixListener::bind(&self.socket_path)
            .with_context(|| format!("Failed to bind UDS listener to {:?}", self.socket_path))?;
        info!("UDS listener bound to {:?}", self.socket_path);
        Ok(BoundUdsListener {
            socket_path: self.socket_path,
            adapter: self.adapter,
            error_handler: self.error_handler,
            shutdown_state: self.shutdown_state,
            listener,
        })
    }

    /// Convenience shim: bind and serve in one call.
    pub async fn serve(self) -> Result<()> {
        self.bind()?.serve().await
    }

    /// Handle a single UDS connection
    async fn handle_connection(
        stream: UnixStream,
        adapter: TransportAdapter,
        error_handler: Arc<dyn TransportErrorHandler>,
        shutdown_state: ShutdownState,
    ) -> Result<()> {
        debug!("Configuring UDS connection");

        // Create framed stream with zero-copy codec (same as TCP)
        let mut framed = Framed::new(stream, TcpFrameCodec::new());
        let teardown_token = shutdown_state.teardown_token().clone();

        debug!("UDS connection ready for frames");

        loop {
            tokio::select! {
                frame_result = framed.next() => {
                    match frame_result {
                        Some(Ok((msg_type, header, payload))) => {
                            // During drain: reject new Message frames with ShuttingDown,
                            // but always pass through Response/Ack/Event frames.
                            if shutdown_state.is_draining() && msg_type == MessageType::Message {
                                debug!(
                                    "Rejecting Message frame during drain (sending ShuttingDown)"
                                );
                                // Echo original header back for correlation, empty payload
                                if let Err(e) = TcpFrameCodec::encode_frame(
                                    framed.get_mut(),
                                    MessageType::ShuttingDown,
                                    &header,
                                    &[],
                                )
                                .await
                                {
                                    warn!(
                                        "Failed to send ShuttingDown frame: {}",
                                        e
                                    );
                                }
                                continue;
                            }

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
                                    "Failed to route {:?} frame from UDS: {}",
                                    msg_type, e
                                );
                            }
                        }
                        Some(Err(e)) => {
                            error!("Frame decode error from UDS: {}", e);
                            break;
                        }
                        None => {
                            debug!("UDS connection closed gracefully");
                            break;
                        }
                    }
                }
                _ = teardown_token.cancelled() => {
                    debug!("UDS connection handler torn down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Route a decoded frame to the appropriate stream
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
            MessageType::ShuttingDown => {
                // ShuttingDown is an outbound-only frame type; receiving it here
                // means a remote peer rejected our request. Route to the response
                // stream so higher layers can handle the rejection via correlation.
                &adapter.response_stream
            }
        };

        match sender.send_async((header, payload)).await {
            Ok(_) => Ok(()),
            Err(e) => {
                error_handler.on_error(e.0.0, e.0.1, format!("Failed to route {:?}", msg_type));
                Err(anyhow::anyhow!("Failed to send to stream"))
            }
        }
    }
}

impl BoundUdsListener {
    /// Accept connections until the teardown token is cancelled.
    pub async fn serve(self) -> Result<()> {
        let teardown_token = self.shutdown_state.teardown_token().clone();

        loop {
            tokio::select! {
                accept_result = self.listener.accept() => {
                    match accept_result {
                        Ok((stream, _addr)) => {
                            debug!("Accepted UDS connection");

                            let adapter = self.adapter.clone();
                            let error_handler = self.error_handler.clone();
                            let shutdown_state = self.shutdown_state.clone();

                            tokio::spawn(async move {
                                if let Err(e) = UdsListener::handle_connection(
                                    stream,
                                    adapter,
                                    error_handler,
                                    shutdown_state,
                                )
                                .await
                                {
                                    warn!("Error handling UDS connection: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept UDS connection: {}", e);
                        }
                    }
                }
                _ = teardown_token.cancelled() => {
                    info!("UDS listener shutting down (teardown)");
                    break;
                }
            }
        }

        // Clean up socket file
        std::fs::remove_file(&self.socket_path).ok();

        Ok(())
    }
}

/// Builder for UdsListener
pub struct UdsListenerBuilder {
    socket_path: Option<PathBuf>,
    adapter: Option<TransportAdapter>,
    error_handler: Option<Arc<dyn TransportErrorHandler>>,
    shutdown_state: Option<ShutdownState>,
}

impl UdsListenerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            socket_path: None,
            adapter: None,
            error_handler: None,
            shutdown_state: None,
        }
    }

    /// Set the socket path
    pub fn socket_path(mut self, path: PathBuf) -> Self {
        self.socket_path = Some(path);
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

    /// Set the shutdown state for graceful drain coordination
    pub fn shutdown_state(mut self, state: ShutdownState) -> Self {
        self.shutdown_state = Some(state);
        self
    }

    /// Build the UdsListener
    pub fn build(self) -> Result<UdsListener> {
        let socket_path = self
            .socket_path
            .ok_or_else(|| anyhow::anyhow!("socket_path is required"))?;
        let adapter = self
            .adapter
            .ok_or_else(|| anyhow::anyhow!("adapter is required"))?;
        let error_handler = self
            .error_handler
            .ok_or_else(|| anyhow::anyhow!("error_handler is required"))?;
        let shutdown_state = self.shutdown_state.unwrap_or_default();

        Ok(UdsListener {
            socket_path,
            adapter,
            error_handler,
            shutdown_state,
        })
    }
}

impl Default for UdsListenerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::make_channels;

    struct TestErrorHandler;

    impl TransportErrorHandler for TestErrorHandler {
        fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
            eprintln!("Test error handler: {}", error);
        }
    }

    #[test]
    fn test_builder_requires_fields() {
        let result = UdsListener::builder().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_builder_with_all_fields() {
        let (adapter, _streams) = make_channels();
        let error_handler = Arc::new(TestErrorHandler);

        let result = UdsListener::builder()
            .socket_path(PathBuf::from("/tmp/test-uds-listener.sock"))
            .adapter(adapter)
            .error_handler(error_handler)
            .build();

        assert!(result.is_ok());
    }
}
