// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Active Message Client

pub mod builders;
mod peer_registry;

use anyhow::Result;
use std::sync::Arc;

use crate::am::{
    InstanceId,
    common::{ActiveMessage, responses::ResponseManager},
};

use dynamo_nova_backend::{NovaBackend, TransportErrorHandler};
use peer_registry::PeerRegistry;

pub struct ActiveMessageClient {
    response_manager: ResponseManager,
    backend: Arc<NovaBackend>,
    error_handler: Arc<dyn TransportErrorHandler>,
    peer_registry: Arc<PeerRegistry>,
}

impl ActiveMessageClient {
    pub(crate) fn new(
        response_manager: ResponseManager,
        backend: Arc<NovaBackend>,
        error_handler: Arc<dyn TransportErrorHandler>,
    ) -> Self {
        Self {
            response_manager,
            backend,
            error_handler,
            peer_registry: Arc::new(PeerRegistry::new()),
        }
    }

    pub(crate) fn send_message(&self, target: InstanceId, message: ActiveMessage) -> Result<()> {
        let (header, payload, message_type) = message.encode()?;
        self.backend.send_message(
            target,
            header.to_vec(),
            payload.to_vec(),
            message_type,
            self.error_handler.clone(),
        )
    }

    /// Register a peer in the client peer registry (internal use)
    pub(crate) fn register_peer(&self, instance_id: InstanceId) {
        self.peer_registry.register_peer(instance_id);
    }

    /// Check if a peer is registered in the backend
    pub(crate) fn is_peer_registered(&self, instance_id: InstanceId) -> bool {
        self.backend.is_registered(instance_id)
    }

    /// Check if we have handler information for a peer
    pub(crate) fn has_handler_info(&self, instance_id: InstanceId) -> bool {
        self.peer_registry.has_handler_info(instance_id)
    }

    /// Check if we can send a message directly (fast path)
    pub(crate) fn can_send_directly(&self, target: InstanceId, handler: &str) -> bool {
        // 1. Peer must be registered
        if !self.is_peer_registered(target) {
            return false;
        }

        // 2. System handlers (starting with _) always allowed
        if handler.starts_with('_') {
            return true;
        }

        // 3. Must have handler info and handler must exist
        self.peer_registry.handler_exists(target, handler)
    }

    /// Perform handshake with a peer to exchange handler information
    async fn handshake_with_peer(&self, target: InstanceId) -> Result<()> {
        use crate::am::server::system_handlers::{HandlersResponse, HelloRequest};

        tracing::debug!(
            target: "dynamo_nova::client",
            target_instance = %target,
            "Initiating handshake with peer"
        );

        // Send _hello with our peer info
        let request = HelloRequest {
            peer_info: self.backend.peer_info(),
        };

        // Serialize request
        let payload = serde_json::to_vec(&request)
            .map_err(|e| anyhow::anyhow!("Failed to serialize _hello request: {}", e))?;

        // Register response and send message
        let mut outcome = self.register_outcome()?;
        let response_id = outcome.response_id();

        let message = crate::am::common::ActiveMessage {
            metadata: crate::am::common::messages::MessageMetadata::new_unary(
                response_id,
                "_hello".to_string(),
                None, // System message, no headers
            ),
            payload: bytes::Bytes::from(payload),
        };

        self.send_message(target, message)?;

        // Wait for response
        let result = outcome.recv().await;
        let response_bytes = match result {
            Ok(Some(bytes)) => bytes,
            Ok(None) => {
                anyhow::bail!("Expected response from _hello, got empty acknowledgment");
            }
            Err(err) => {
                anyhow::bail!("Handshake failed: {}", err);
            }
        };

        // Deserialize response
        let response: HandlersResponse = serde_json::from_slice(&response_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize _hello response: {}", e))?;

        // Update peer registry with handler list
        self.peer_registry
            .update_handlers(target, response.handlers.clone());

        tracing::debug!(
            target: "dynamo_nova::client",
            target_instance = %target,
            handler_count = response.handlers.len(),
            "Handshake completed successfully"
        );

        Ok(())
    }

    /// Ensure a peer is ready for communication (performs handshake if needed)
    pub(crate) async fn ensure_peer_ready(&self, target: InstanceId, handler: &str) -> Result<()> {
        // 1. Check if peer is registered (null discovery - error if not)
        if !self.is_peer_registered(target) {
            anyhow::bail!(
                "Peer {} not registered. Call nova.connect_to_peer() first.",
                target
            );
        }

        // 2. System handlers skip further checks
        if handler.starts_with('_') {
            return Ok(());
        }

        // 3. Ensure we have handler list (perform handshake if needed)
        if !self.has_handler_info(target) {
            self.handshake_with_peer(target).await?;
        }

        // 4. Verify handler exists
        if !self.peer_registry.handler_exists(target, handler) {
            anyhow::bail!(
                "Handler '{}' not found on instance {}. Available handlers: {:?}",
                handler,
                target,
                self.peer_registry.get_handlers(target).unwrap_or_default()
            );
        }

        Ok(())
    }

    /// Get the list of handlers for a peer (may trigger handshake)
    pub(crate) async fn get_peer_handlers(&self, instance_id: InstanceId) -> Result<Vec<String>> {
        if !self.has_handler_info(instance_id) {
            self.handshake_with_peer(instance_id).await?;
        }

        self.peer_registry
            .get_handlers(instance_id)
            .ok_or_else(|| anyhow::anyhow!("Failed to get handlers for instance {}", instance_id))
    }

    /// Refresh the handler list for a peer
    pub(crate) async fn refresh_handler_list(&self, instance_id: InstanceId) -> Result<()> {
        self.handshake_with_peer(instance_id).await
    }

    fn register_outcome(
        &self,
    ) -> Result<
        crate::am::common::responses::ResponseAwaiter,
        crate::am::common::responses::ResponseRegistrationError,
    > {
        self.response_manager.register_outcome()
    }
}
