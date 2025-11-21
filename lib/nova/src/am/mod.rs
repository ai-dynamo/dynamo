// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Messages

mod client;
mod common;
mod handlers;
mod server;

use client::ActiveMessageClient;
use handlers::HandlerManager;
use server::ActiveMessageServer;

use anyhow::Result;
use std::sync::Arc;

// Re-export shared identity types
pub use dynamo_identity::{InstanceId, WorkerId};
pub use dynamo_nova_backend::{PeerInfo, WorkerAddress};
pub use handlers::NovaHandler;

use dynamo_nova_backend::{NovaBackend, Transport};

use crate::am::client::builders::MessageBuilder;
use crate::events::EventHandle;
use crate::am::handlers::events::EventMessenger;
use tracing::warn;

#[derive(Clone)]
pub struct Nova {
    instance_id: InstanceId,
    backend: Arc<NovaBackend>,
    client: Arc<ActiveMessageClient>,
    server: Arc<ActiveMessageServer>,
    handlers: HandlerManager,
    events: Arc<handlers::NovaEvents>,
    discovery: Arc<dynamo_discovery::peer::PeerDiscoveryManager>,
}

impl Nova {
    pub async fn new(transports: Vec<Arc<dyn Transport>>) -> anyhow::Result<Arc<Self>> {
        // 1. Setup infrastructure
        let (backend, data_streams) = NovaBackend::new(transports).await?;
        let backend = Arc::new(backend);
        let instance_id = backend.instance_id();
        let worker_id = instance_id.worker_id();
        let response_manager = common::responses::ResponseManager::new(worker_id.as_u64());
        let local_events = crate::events::LocalEventSystem::new(worker_id.as_u64());

        // Create peer discovery manager with local peer info
        let peer_info = backend.peer_info();
        let discovery = dynamo_discovery::peer::PeerDiscoveryManager::new(
            Some(peer_info),
            vec![], // No remote backends for now
        ).await?;
        let discovery = Arc::new(discovery);

        let event_manager =
            handlers::NovaEvents::new(worker_id, instance_id, local_events.clone(), backend.clone(), Arc::new(response_manager.clone()), discovery.clone());

        // 2. Create server (creates hub internally)
        let server = ActiveMessageServer::new(
            response_manager.clone(),
            local_events.clone(),
            data_streams,
            backend.clone(),
        )
        .await;
        let server = Arc::new(server);

        // 3. Create client with error handler
        struct DefaultErrorHandler;
        impl dynamo_nova_backend::TransportErrorHandler for DefaultErrorHandler {
            fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
                tracing::error!("Transport error: {}", error);
            }
        }

        let client = Arc::new(ActiveMessageClient::new(
            response_manager,
            backend.clone(),
            Arc::new(DefaultErrorHandler),
        ));

        // 4. Create handler manager
        let control_tx = server.control_tx();
        let handlers = HandlerManager::new(control_tx);

        // 5. Wrap everything in Arc<Nova>
        let system = Arc::new(Self {
            instance_id,
            backend: backend.clone(),
            client,
            server: server.clone(),
            handlers,
            events: event_manager.clone(),
            discovery: discovery.clone(),
        });

        // 5. Initialize hub's system reference (OnceLock)
        server.hub().set_system(system.clone())?;

        // 5b. Attach messenger to event system
        event_manager.set_messenger(system.clone());

        // 6. Register system handlers
        server::register_system_handlers(&system.handlers)?;
        register_event_handlers(&system.handlers, event_manager.clone())?;

        Ok(system)
    }

    /// Get the instance ID of this system
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Get the peer information for this Nova instance.
    ///
    /// Returns a `PeerInfo` containing the instance ID and worker address,
    /// which can be used to register this instance as a peer on other Nova instances.
    pub fn peer_info(&self) -> PeerInfo {
        self.backend.peer_info()
    }

    /// Get the backend (internal use for sending responses)
    pub(crate) fn backend(&self) -> &Arc<NovaBackend> {
        &self.backend
    }

    /// Fire-and-forget builder (no response expected).
    pub fn am_send(&self, handler: &str) -> anyhow::Result<client::builders::AmSendBuilder> {
        client::builders::AmSendBuilder::new(self.client.clone(), handler)
    }

    /// Active-message synchronous completion (await handler finish).
    pub fn am_sync(&self, handler: &str) -> anyhow::Result<client::builders::AmSyncBuilder> {
        client::builders::AmSyncBuilder::new(self.client.clone(), handler)
    }

    /// Unary builder returning raw bytes.
    pub fn unary(&self, handler: &str) -> anyhow::Result<client::builders::UnaryBuilder> {
        client::builders::UnaryBuilder::new(self.client.clone(), handler)
    }

    /// Typed unary builder returning deserialized response.
    pub fn typed_unary<R: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        handler: &str,
    ) -> anyhow::Result<client::builders::TypedUnaryBuilder<R>> {
        client::builders::TypedUnaryBuilder::new(self.client.clone(), handler)
    }

    // todo: trigger an event on a remote machine
    pub fn trigger_event(&self, event: EventHandle) -> Result<impl Future<Output = Result<()>>> {
        Ok(self
            .am_sync("_trigger_event")?
            .worker(event.owner_worker())
            .payload(event)?
            .send())
    }

    pub fn register_handler(&self, handler: NovaHandler) -> anyhow::Result<()> {
        self.handlers.register_handler(handler)
    }

    /// Access the remote-capable event system.
    pub fn events(&self) -> Arc<handlers::NovaEvents> {
        self.events.clone()
    }

    /// Connect to a peer by registering their peer information.
    ///
    /// This registers the peer with both the backend (for transport) and the client
    /// (for handler discovery). The first message sent to this peer will trigger
    /// an automatic handshake to exchange handler lists.
    pub fn register_peer(&self, peer_info: PeerInfo) -> anyhow::Result<()> {
        let instance_id = peer_info.instance_id();
        let worker_id = peer_info.worker_id();

        // Register with backend (registers with transports)
        self.backend.register_peer(peer_info)?;

        // Register in client peer registry
        self.client.register_peer(instance_id);

        // Pre-populate routing table for fast-path event lookups
        self.events
            .register_worker_mapping(worker_id.as_u64(), instance_id);

        Ok(())
    }

    /// Get the list of handlers available on a remote instance.
    ///
    /// This may trigger a handshake if handler information hasn't been queried yet.
    pub async fn available_handlers(&self, instance_id: InstanceId) -> anyhow::Result<Vec<String>> {
        self.client.get_peer_handlers(instance_id).await
    }

    /// Refresh the handler list for a remote instance.
    ///
    /// This forces a new query to the remote instance, updating the cached handler list.
    pub async fn refresh_handlers(&self, instance_id: InstanceId) -> anyhow::Result<()> {
        self.client.refresh_handler_list(instance_id).await
    }

    /// Wait for a specific handler to become available on a remote instance.
    ///
    /// This polls the remote instance periodically until the handler appears or a timeout occurs.
    pub async fn wait_for_handler(
        &self,
        instance_id: InstanceId,
        handler_name: &str,
    ) -> anyhow::Result<()> {
        const MAX_ATTEMPTS: u32 = 10;
        const DELAY: std::time::Duration = std::time::Duration::from_millis(100);

        for _ in 0..MAX_ATTEMPTS {
            self.refresh_handlers(instance_id).await?;

            let handlers = self.available_handlers(instance_id).await?;
            if handlers.contains(&handler_name.to_string()) {
                return Ok(());
            }

            tokio::time::sleep(DELAY).await;
        }

        anyhow::bail!(
            "Timeout waiting for handler '{}' on instance {}",
            handler_name,
            instance_id
        )
    }

    /// Get the list of handlers registered on this local instance.
    pub fn list_local_handlers(&self) -> Vec<String> {
        self.server.hub().list_handlers()
    }

    // /// Register a peer internally (used by system handlers)
    // pub(crate) fn register_peer_internal(&self, instance_id: InstanceId) {
    //     self.client.register_peer(instance_id);
    // }

    // /// Get a handler manager for registering message handlers.
    // ///
    // /// Use this to register your custom handlers for processing incoming active messages.
    // pub fn handler_manager(&self) -> HandlerManager {
    //     self.server.handler_manager()
    // }
}

fn register_event_handlers(
    manager: &HandlerManager,
    events: Arc<handlers::NovaEvents>,
) -> Result<()> {
    let subscribe = {
        let events = events.clone();
        NovaHandler::am_handler_async("_event_subscribe", move |ctx| {
            let events = events.clone();
            async move { events.handle_subscribe(ctx.payload) }
        })
        .spawn()
        .build()
    };

    let trigger = {
        let events = events.clone();
        NovaHandler::am_handler_async("_event_trigger", move |ctx| {
            let events = events.clone();
            async move { events.handle_trigger(ctx.payload) }
        })
        .spawn()
        .build()
    };

    let trigger_request = {
        let events = events.clone();
        NovaHandler::am_handler_async("_event_trigger_request", move |ctx| {
            let events = events.clone();
            async move { events.handle_trigger_request(ctx.payload) }
        })
        .spawn()
        .build()
    };

    manager.register_internal_handler(subscribe)?;
    manager.register_internal_handler(trigger)?;
    manager.register_internal_handler(trigger_request)?;
    Ok(())
}

impl EventMessenger for Nova {
    fn send_system(&self, target: InstanceId, handler: &str, payload: bytes::Bytes) -> Result<()> {
        let handler_name = handler.to_string();
        let fut = MessageBuilder::new_unchecked(self.client.clone(), handler)
            .instance(target)
            .raw_payload(payload)
            .fire();
        tokio::spawn(async move {
            if let Err(e) = fut.await {
                warn!("Failed to send system event {}: {}", handler_name, e);
            }
        });
        Ok(())
    }
}
