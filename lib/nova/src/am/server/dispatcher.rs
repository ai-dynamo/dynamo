// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message dispatcher for active message routing in the nova crate.
//!
//! The dispatcher receives decoded messages and routes them to registered handlers.
//! It supports inline and spawned dispatch modes with optional per-handler concurrency limits.

use bytes::Bytes;
use dashmap::DashMap;
use dynamo_identity::WorkerId;
use dynamo_nova_backend::NovaBackend;
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, trace, warn};

use crate::am::Nova;
use crate::am::common::events::{EventType, Outcome, encode_event_header};
use crate::am::common::messages::ResponseType;
use crate::am::common::responses::ResponseId;

/// Context passed to handlers during dispatch.
#[derive(Clone)]
pub(crate) struct HandlerContext {
    /// The response ID for correlation
    pub message_id: ResponseId,

    /// Message payload
    pub payload: Bytes,

    /// Response type (FireAndForget, AckNack, Unary)
    pub response_type: ResponseType,

    /// Optional user headers (for tracing, metadata, etc.)
    pub headers: Option<std::collections::HashMap<String, String>>,

    /// The active message system for handler access
    pub system: Arc<Nova>,
}

/// Base trait for active message handlers.
pub(crate) trait ActiveMessageHandler: Send + Sync {
    /// Handle a message asynchronously
    fn handle(
        &self,
        ctx: HandlerContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>>;

    /// Get the handler name
    fn name(&self) -> &str;
}

/// Trait for dispatching messages to handlers.
///
/// Implementations wrap handlers with dispatch logic (spawn, inline, etc.)
pub(crate) trait ActiveMessageDispatcher: Send + Sync {
    /// Get the handler name
    fn name(&self) -> &str;

    /// Dispatch a message to the handler (non-async, kicks off handler execution)
    fn dispatch(&self, ctx: HandlerContext);
}

/// Dispatcher implementation that spawns handlers on a task tracker.
pub(crate) struct SpawnedDispatcher<H: ActiveMessageHandler> {
    handler: Arc<H>,
    task_tracker: TaskTracker,
    semaphore: Option<Arc<Semaphore>>,
}

impl<H: ActiveMessageHandler> SpawnedDispatcher<H> {
    pub fn new(handler: H, task_tracker: TaskTracker) -> Self {
        Self {
            handler: Arc::new(handler),
            task_tracker,
            semaphore: None,
        }
    }

    /// This method is not used and is hidden from the public API.
    ///
    /// In the future, we may want to add a concurrency limit to the dispatcher.
    #[expect(dead_code)]
    #[doc(hidden)]
    pub fn with_concurrency_limit(handler: H, task_tracker: TaskTracker, limit: usize) -> Self {
        Self {
            handler: Arc::new(handler),
            task_tracker,
            semaphore: Some(Arc::new(Semaphore::new(limit))),
        }
    }
}

impl<H: ActiveMessageHandler + 'static> ActiveMessageDispatcher for SpawnedDispatcher<H> {
    fn name(&self) -> &str {
        self.handler.name()
    }

    fn dispatch(&self, ctx: HandlerContext) {
        let handler = self.handler.clone();
        let semaphore = self.semaphore.clone();
        let handler_name = handler.name().to_string();

        self.task_tracker.spawn(async move {
            // Acquire semaphore if concurrency limit is set
            let _permit = if let Some(sem) = &semaphore {
                Some(sem.acquire().await.expect("semaphore closed"))
            } else {
                None
            };

            trace!(target: "dynamo_nova::dispatcher", handler = %handler_name, "Handler task started");
            handler.handle(ctx).await;
            trace!(target: "dynamo_nova::dispatcher", handler = %handler_name, "Handler task completed");
        });
    }
}

/// Dispatcher implementation that executes handlers inline on the dispatcher task.
pub(crate) struct InlineDispatcher<H: ActiveMessageHandler> {
    handler: Arc<H>,
    semaphore: Option<Arc<Semaphore>>,
}

impl<H: ActiveMessageHandler> InlineDispatcher<H> {
    pub fn new(handler: H) -> Self {
        Self {
            handler: Arc::new(handler),
            semaphore: None,
        }
    }

    /// This method is not used and is hidden from the public API.
    ///
    /// In the future, we may want to add a concurrency limit to the dispatcher.
    #[expect(dead_code)]
    #[doc(hidden)]
    pub fn with_concurrency_limit(handler: H, limit: usize) -> Self {
        Self {
            handler: Arc::new(handler),
            semaphore: Some(Arc::new(Semaphore::new(limit))),
        }
    }
}

impl<H: ActiveMessageHandler + 'static> ActiveMessageDispatcher for InlineDispatcher<H> {
    fn name(&self) -> &str {
        self.handler.name()
    }

    fn dispatch(&self, ctx: HandlerContext) {
        let handler = self.handler.clone();
        let semaphore = self.semaphore.clone();

        // For inline dispatcher, we spawn a task but don't track it
        // This allows the dispatch call to return immediately
        tokio::spawn(async move {
            // Acquire semaphore if concurrency limit is set
            let _permit = if let Some(sem) = &semaphore {
                Some(sem.acquire().await.expect("semaphore closed"))
            } else {
                None
            };

            handler.handle(ctx).await;
        });
    }
}

/// Control messages for dispatcher management (registration only, not dispatch).
pub(crate) enum ControlMessage {
    /// Register a new handler
    Register {
        dispatcher: Arc<dyn ActiveMessageDispatcher>,
    },

    // TODO: add a register_internal_handler method to the dispatcher hub
    /// Unregister a handler
    #[expect(dead_code)]
    #[doc(hidden)]
    Unregister { name: String },

    // TODO: add a shutdown method to the dispatcher hub
    /// Shutdown the dispatcher
    #[expect(dead_code)]
    #[doc(hidden)]
    Shutdown,
}

/// Main message dispatcher hub that routes messages to handlers.
pub(crate) struct DispatcherHub {
    /// Handler registry (lock-free for fast dispatch)
    handlers: Arc<DashMap<String, Arc<dyn ActiveMessageDispatcher>>>,

    /// Backend for sending messages
    backend: Arc<NovaBackend>,

    /// Control channel for registration
    control_rx: flume::Receiver<ControlMessage>,

    /// Active message system reference (late-bound via OnceLock)
    system: OnceLock<Arc<Nova>>,
}

impl DispatcherHub {
    /// Create a new dispatcher hub
    pub fn new(backend: Arc<NovaBackend>, control_rx: flume::Receiver<ControlMessage>) -> Self {
        Self {
            handlers: Arc::new(DashMap::new()),
            backend,
            control_rx,
            system: OnceLock::new(),
        }
    }

    /// Initialize the system reference (must be called exactly once before dispatching)
    pub fn set_system(&self, system: Arc<Nova>) -> anyhow::Result<()> {
        self.system
            .set(system)
            .map_err(|_| anyhow::anyhow!("System already initialized"))
    }

    /// Get the system reference (panics if not initialized)
    pub(crate) fn system(&self) -> &Arc<Nova> {
        self.system
            .get()
            .expect("System must be initialized before dispatching messages")
    }

    // /// Get a shared reference to the handlers map (for cloning the hub)
    // pub fn handlers(&self) -> Arc<DashMap<String, Arc<dyn ActiveMessageDispatcher>>> {
    //     self.handlers.clone()
    // }

    /// Get a list of all registered handler names
    pub(crate) fn list_handlers(&self) -> Vec<String> {
        self.handlers
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Process control messages (registration, unregistration, shutdown)
    pub async fn process_control(&self) -> bool {
        match self.control_rx.recv_async().await {
            Ok(ControlMessage::Register { dispatcher }) => {
                let name = dispatcher.name().to_string();
                debug!(target: "dynamo_nova::dispatcher", handler = %name, "Registering handler");
                self.handlers.insert(name, dispatcher);
                true
            }
            Ok(ControlMessage::Unregister { name }) => {
                debug!(target: "dynamo_nova::dispatcher", handler = %name, "Unregistering handler");
                self.handlers.remove(&name);
                true
            }
            Ok(ControlMessage::Shutdown) => {
                debug!(target: "dynamo_nova::dispatcher", "Shutting down dispatcher hub");
                false
            }
            Err(_) => {
                warn!(target: "dynamo_nova::dispatcher", "Control channel closed");
                false
            }
        }
    }

    /// Dispatch a message to the appropriate handler
    pub fn dispatch_message(&self, handler_name: &str, ctx: HandlerContext) {
        match self.handlers.get(handler_name) {
            Some(dispatcher) => {
                dispatcher.dispatch(ctx);
            }
            None => {
                self.handle_unknown_handler(handler_name, ctx);
            }
        }
    }

    /// Handle messages for unknown handlers
    fn handle_unknown_handler(&self, handler_name: &str, ctx: HandlerContext) {
        error!(
            target: "dynamo_nova::dispatcher",
            handler = %handler_name,
            message_id = %ctx.message_id,
            "No handler registered for message"
        );

        // Send error response if caller expects one
        let backend = self.backend.clone();
        let message_id = ctx.message_id;
        let handler_name = handler_name.to_string();

        match ctx.response_type {
            ResponseType::AckNack | ResponseType::Unary => {
                let error_message = format!("Handler '{}' not found", handler_name);
                tokio::spawn(async move {
                    if let Err(e) =
                        Self::send_error_response_static(&backend, message_id, error_message).await
                    {
                        error!(
                            target: "dynamo_nova::dispatcher",
                            "Failed to send error response for unknown handler: {}", e
                        );
                    }
                });
            }
            ResponseType::FireAndForget => {
                // No response expected, just log the error
                warn!(
                    target: "dynamo_nova::dispatcher",
                    handler = %handler_name,
                    "Fire-and-forget message to unknown handler, no response sent"
                );
            }
        }
    }

    /// Send an error response back to the sender (static method)
    async fn send_error_response_static(
        backend: &NovaBackend,
        response_id: ResponseId,
        error_message: String,
    ) -> anyhow::Result<()> {
        use dynamo_nova_backend::MessageType;

        // Encode error response header using event encoding
        let header = encode_event_header(EventType::Ack(response_id, Outcome::Error));

        // Error message as payload
        let payload = Bytes::from(error_message.into_bytes());

        // Static error handler (created once, reused)
        struct DispatcherErrorHandler;
        impl dynamo_nova_backend::TransportErrorHandler for DispatcherErrorHandler {
            fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
                error!(target: "dynamo_nova::dispatcher", "Failed to send error response: {}", error);
            }
        }

        static ERROR_HANDLER: std::sync::OnceLock<
            Arc<dyn dynamo_nova_backend::TransportErrorHandler>,
        > = std::sync::OnceLock::new();
        let error_handler = ERROR_HANDLER
            .get_or_init(|| Arc::new(DispatcherErrorHandler))
            .clone();

        backend.send_message_to_worker(
            WorkerId::from_u64(response_id.worker_id()),
            header.to_vec(),
            payload.to_vec(),
            MessageType::Ack,
            error_handler,
        )?;

        Ok(())
    }
}
