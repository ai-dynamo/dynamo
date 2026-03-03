// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Message dispatcher for active message routing.

use bytes::Bytes;
use dashmap::DashMap;
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, trace, warn};
use velo_backend::VeloBackend;
use velo_common::WorkerId;

use crate::Messenger;
use crate::common::events::{EventType, Outcome, encode_event_header};
use crate::common::messages::ResponseType;
use crate::common::responses::ResponseId;

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

    /// The messenger system for handler access
    pub system: Arc<Messenger>,
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
            let _permit = if let Some(sem) = &semaphore {
                Some(sem.acquire().await.expect("semaphore closed"))
            } else {
                None
            };

            trace!(target: "velo_messenger::dispatcher", handler = %handler_name, "Handler task started");
            handler.handle(ctx).await;
            trace!(target: "velo_messenger::dispatcher", handler = %handler_name, "Handler task completed");
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

        tokio::spawn(async move {
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

    /// Unregister a handler
    #[expect(dead_code)]
    #[doc(hidden)]
    Unregister { name: String },

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
    backend: Arc<VeloBackend>,

    /// Control channel for registration
    control_rx: flume::Receiver<ControlMessage>,

    /// Messenger system reference (late-bound via OnceLock)
    system: OnceLock<Arc<Messenger>>,
}

impl DispatcherHub {
    /// Create a new dispatcher hub
    pub fn new(backend: Arc<VeloBackend>, control_rx: flume::Receiver<ControlMessage>) -> Self {
        Self {
            handlers: Arc::new(DashMap::new()),
            backend,
            control_rx,
            system: OnceLock::new(),
        }
    }

    /// Initialize the system reference (must be called exactly once before dispatching)
    pub fn set_system(&self, system: Arc<Messenger>) -> anyhow::Result<()> {
        self.system
            .set(system)
            .map_err(|_| anyhow::anyhow!("System already initialized"))
    }

    /// Get the system reference (panics if not initialized)
    pub(crate) fn system(&self) -> &Arc<Messenger> {
        self.system
            .get()
            .expect("System must be initialized before dispatching messages")
    }

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
                debug!(target: "velo_messenger::dispatcher", handler = %name, "Registering handler");
                self.handlers.insert(name, dispatcher);
                true
            }
            Ok(ControlMessage::Unregister { name }) => {
                debug!(target: "velo_messenger::dispatcher", handler = %name, "Unregistering handler");
                self.handlers.remove(&name);
                true
            }
            Ok(ControlMessage::Shutdown) => {
                debug!(target: "velo_messenger::dispatcher", "Shutting down dispatcher hub");
                false
            }
            Err(_) => {
                warn!(target: "velo_messenger::dispatcher", "Control channel closed");
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
            target: "velo_messenger::dispatcher",
            handler = %handler_name,
            message_id = %ctx.message_id,
            "No handler registered for message"
        );

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
                            target: "velo_messenger::dispatcher",
                            "Failed to send error response for unknown handler: {}", e
                        );
                    }
                });
            }
            ResponseType::FireAndForget => {
                warn!(
                    target: "velo_messenger::dispatcher",
                    handler = %handler_name,
                    "Fire-and-forget message to unknown handler, no response sent"
                );
            }
        }
    }

    /// Send an error response back to the sender (static method)
    async fn send_error_response_static(
        backend: &VeloBackend,
        response_id: ResponseId,
        error_message: String,
    ) -> anyhow::Result<()> {
        use velo_backend::MessageType;

        let header = encode_event_header(EventType::Ack(response_id, Outcome::Error));
        let payload = Bytes::from(error_message.into_bytes());

        struct DispatcherErrorHandler;
        impl velo_backend::TransportErrorHandler for DispatcherErrorHandler {
            fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
                error!(target: "velo_messenger::dispatcher", "Failed to send error response: {}", error);
            }
        }

        static ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
            std::sync::OnceLock::new();
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
