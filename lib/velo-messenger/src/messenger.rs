// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Messenger - the core active messaging system.

use anyhow::Result;
use std::num::NonZero;
use std::sync::Arc;

use velo_backend::{Transport, VeloBackend};
use velo_common::{InstanceId, PeerInfo};
use velo_events::DistributedEventFactory;

use crate::PeerDiscovery;
use crate::client::ActiveMessageClient;
use crate::client::builders::MessageBuilder;
use crate::events::VeloEvents;
use crate::handlers::{Handler, HandlerManager};
use crate::server::ActiveMessageServer;

/// The core active messaging system.
#[derive(Clone)]
pub struct Messenger {
    instance_id: InstanceId,
    backend: Arc<VeloBackend>,
    client: Arc<ActiveMessageClient>,
    server: Arc<ActiveMessageServer>,
    handlers: HandlerManager,
    discovery: Option<Arc<dyn PeerDiscovery>>,
    events: Arc<VeloEvents>,
    runtime: tokio::runtime::Handle,
    tracker: tokio_util::task::TaskTracker,
}

/// Builder for Messenger allowing incremental configuration.
pub struct MessengerBuilder {
    transports: Vec<Arc<dyn Transport>>,
    discovery: Option<Arc<dyn PeerDiscovery>>,
}

impl MessengerBuilder {
    /// Create a new empty MessengerBuilder.
    pub fn new() -> Self {
        Self {
            transports: Vec::new(),
            discovery: None,
        }
    }

    /// Add a transport to the Messenger system.
    pub fn add_transport(mut self, transport: Arc<dyn Transport>) -> Self {
        self.transports.push(transport);
        self
    }

    /// Set the peer discovery backend.
    pub fn discovery(mut self, discovery: Arc<dyn PeerDiscovery>) -> Self {
        self.discovery = Some(discovery);
        self
    }

    /// Build the Messenger system with the configured transports and discovery.
    pub async fn build(self) -> Result<Arc<Messenger>> {
        Messenger::new(self.transports, self.discovery).await
    }
}

impl Default for MessengerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Messenger {
    /// Create a builder for configuring Messenger.
    ///
    /// # Example
    /// ```ignore
    /// let messenger = Messenger::builder()
    ///     .add_transport(tcp_transport)
    ///     .discovery(my_discovery)
    ///     .build()
    ///     .await?;
    /// ```
    pub fn builder() -> MessengerBuilder {
        MessengerBuilder::new()
    }

    /// Create a new Messenger system.
    pub async fn new(
        transports: Vec<Arc<dyn Transport>>,
        discovery: Option<Arc<dyn PeerDiscovery>>,
    ) -> Result<Arc<Self>> {
        // 1. Setup infrastructure
        let (backend, data_streams) = VeloBackend::new(transports).await?;
        let backend = Arc::new(backend);
        let instance_id = backend.instance_id();
        let worker_id = instance_id.worker_id();
        let response_manager = crate::common::responses::ResponseManager::new(worker_id.as_u64());
        let runtime = tokio::runtime::Handle::current();
        let tracker = tokio_util::task::TaskTracker::new();

        // 2. Create distributed event system
        let system_id = NonZero::new(worker_id.as_u64())
            .expect("worker_id must be non-zero for distributed events");
        let factory = DistributedEventFactory::new(system_id);
        let local_base = factory.system().clone();
        let events = VeloEvents::new(
            instance_id,
            local_base,
            backend.clone(),
            response_manager.clone(),
        );

        // 3. Create server (no event frame handler — events use AM handlers)
        let server = ActiveMessageServer::new(
            response_manager.clone(),
            None,
            data_streams,
            backend.clone(),
            tracker.clone(),
        )
        .await;
        let server = Arc::new(server);

        // 4. Create client with error handler
        struct DefaultErrorHandler;
        impl velo_backend::TransportErrorHandler for DefaultErrorHandler {
            fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
                tracing::error!("Transport error: {}", error);
            }
        }

        let client = Arc::new(ActiveMessageClient::new(
            response_manager,
            backend.clone(),
            Arc::new(DefaultErrorHandler),
            discovery.clone(),
        ));

        // 5. Create handler manager
        let control_tx = server.control_tx();
        let handlers = HandlerManager::new(control_tx);

        // 6. Wrap everything in Arc<Messenger>
        let system = Arc::new(Self {
            instance_id,
            backend: backend.clone(),
            client,
            server: server.clone(),
            handlers,
            discovery,
            events: events.clone(),
            runtime,
            tracker,
        });

        // 7. Initialize hub's system reference (OnceLock)
        server.hub().set_system(system.clone())?;

        // 8. Wire events: set messenger reference and register event handlers
        events.set_messenger(system.clone());
        crate::events::handlers::register_event_handlers(&system.handlers, events)?;

        // 9. Register system handlers
        crate::server::register_system_handlers(&system.handlers)?;

        Ok(system)
    }

    /// Get the instance ID of this system
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// Get the peer information for this Messenger instance.
    pub fn peer_info(&self) -> PeerInfo {
        self.backend.peer_info()
    }

    /// Get the backend (internal use for sending responses)
    pub(crate) fn backend(&self) -> &Arc<VeloBackend> {
        &self.backend
    }

    /// Get the discovery backend, if configured.
    pub(crate) fn discovery(&self) -> Option<Arc<dyn PeerDiscovery>> {
        self.discovery.clone()
    }

    /// Get the distributed event system.
    pub fn events(&self) -> &Arc<VeloEvents> {
        &self.events
    }

    /// Convenience: create an EventManager wired with the distributed backend.
    pub fn event_manager(&self) -> velo_events::EventManager {
        self.events.event_manager()
    }

    /// Fire-and-forget builder (no response expected).
    pub fn am_send(&self, handler: &str) -> Result<crate::client::builders::AmSendBuilder> {
        crate::client::builders::AmSendBuilder::new(self.client.clone(), handler)
    }

    /// Active-message synchronous completion (await handler finish).
    pub fn am_sync(&self, handler: &str) -> Result<crate::client::builders::AmSyncBuilder> {
        crate::client::builders::AmSyncBuilder::new(self.client.clone(), handler)
    }

    /// Unary builder returning raw bytes.
    pub fn unary(&self, handler: &str) -> Result<crate::client::builders::UnaryBuilder> {
        crate::client::builders::UnaryBuilder::new(self.client.clone(), handler)
    }

    /// Typed unary builder returning deserialized response.
    pub fn typed_unary<R: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        handler: &str,
    ) -> Result<crate::client::builders::TypedUnaryBuilder<R>> {
        crate::client::builders::TypedUnaryBuilder::new(self.client.clone(), handler)
    }

    pub fn register_handler(&self, handler: Handler) -> Result<()> {
        self.handlers.register_handler(handler)
    }

    /// Connect to a peer by registering their peer information.
    pub fn register_peer(&self, peer_info: PeerInfo) -> Result<()> {
        let instance_id = peer_info.instance_id();

        // Register with backend (registers with transports)
        self.backend.register_peer(peer_info)?;

        // Register in client peer registry
        self.client.register_peer(instance_id);

        Ok(())
    }

    /// Discover a peer by instance_id and register it for communication.
    pub async fn discover_and_register_peer(&self, instance_id: InstanceId) -> Result<()> {
        tracing::debug!(
            target: "velo_messenger::discovery",
            %instance_id,
            "Discovering peer by instance_id"
        );

        let discovery = self.discovery.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No discovery backend configured. Cannot discover instance {}",
                instance_id
            )
        })?;

        let peer_info = discovery.discover_by_instance_id(instance_id).await?;

        tracing::info!(
            target: "velo_messenger::discovery",
            %instance_id,
            "Discovered peer, registering"
        );

        self.register_peer(peer_info)
    }

    /// Get the list of handlers available on a remote instance.
    pub async fn available_handlers(&self, instance_id: InstanceId) -> Result<Vec<String>> {
        self.client.get_peer_handlers(instance_id).await
    }

    /// Refresh the handler list for a remote instance.
    pub async fn refresh_handlers(&self, instance_id: InstanceId) -> Result<()> {
        self.client.refresh_handler_list(instance_id).await
    }

    /// Wait for a specific handler to become available on a remote instance.
    pub async fn wait_for_handler(
        &self,
        instance_id: InstanceId,
        handler_name: &str,
    ) -> Result<()> {
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

    pub fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    pub fn tracker(&self) -> &tokio_util::task::TaskTracker {
        &self.tracker
    }

    /// Internal: create an unchecked message builder (for system messages)
    pub(crate) fn message_builder_unchecked(&self, handler: &str) -> MessageBuilder {
        MessageBuilder::new_unchecked(self.client.clone(), handler)
    }
}
