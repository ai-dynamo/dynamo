// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Messages

mod client;
mod common;
mod server;

use client::ActiveMessageClient;
use server::ActiveMessageServer;

use std::sync::Arc;

// Re-export shared identity types
pub use dynamo_identity::{InstanceId, WorkerId};
pub use dynamo_nova_backend::{PeerInfo, WorkerAddress};

use dynamo_nova_backend::{NovaBackend, Transport};

#[derive(Clone)]
pub struct Nova {
    instance_id: InstanceId,
    backend: Arc<NovaBackend>,
    client: Arc<ActiveMessageClient>,
    server: Arc<ActiveMessageServer>,
}

impl Nova {
    pub async fn new(transports: Vec<Arc<dyn Transport>>) -> anyhow::Result<Arc<Self>> {
        // 1. Setup infrastructure
        let (backend, data_streams) = NovaBackend::new(transports).await?;
        let instance_id = backend.instance_id();
        let worker_id = instance_id.worker_id();
        let response_manager = common::responses::ResponseManager::new(worker_id.as_u64());
        let event_manager = crate::events::LocalEventSystem::new(worker_id.as_u64());

        let backend = Arc::new(backend);

        // 2. Create server (creates hub internally)
        let server = ActiveMessageServer::new(
            response_manager.clone(),
            event_manager.clone(),
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
            instance_id,
            response_manager,
            backend.clone(),
            Arc::new(DefaultErrorHandler),
        ));

        // 4. Wrap everything in Arc<Nova>
        let system = Arc::new(Self {
            instance_id,
            backend: backend.clone(),
            client,
            server: server.clone(),
        });

        // 5. Initialize hub's system reference (OnceLock)
        server.hub().set_system(system.clone())?;

        // 6. TODO: Register internal handlers here
        // system.register_internal_handlers()?;

        Ok(system)
    }

    /// Get the instance ID of this system
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
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

    pub async fn connect_to_peer(&self, _peer_info: PeerInfo) -> anyhow::Result<()> {
        unimplemented!()
    }

    pub async fn available_handlers(
        &self,
        _instance_id: InstanceId,
    ) -> anyhow::Result<Vec<String>> {
        unimplemented!()
    }

    pub async fn wait_for_handler(
        &self,
        _instance_id: InstanceId,
        _handler_name: &str,
    ) -> anyhow::Result<()> {
        unimplemented!()
    }
}
