// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS Multiplexed Server
//!
//! Provides a multiplexed NATS server that handles multiple endpoints on a single
//! NATS service group. This replaces the per-endpoint PushEndpoint pattern with
//! a unified multiplexed approach consistent with HTTP and TCP servers.

use super::*;
use crate::SystemHealth;
use crate::config::HealthStatus;
use crate::pipeline::network::ingress::push_endpoint::PushEndpoint;
use anyhow::Result;
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Multiplexed NATS server that handles multiple endpoints
///
/// Unlike the previous per-endpoint approach, this server manages multiple
/// endpoints, getting the service group dynamically from the component registry
/// for each endpoint registration.
pub struct NatsMultiplexedServer {
    nats_client: async_nats::Client,
    component_registry: crate::component::Registry,
    handlers: Arc<DashMap<String, EndpointTask>>,
    cancellation_token: CancellationToken,
}

struct EndpointTask {
    cancel_token: CancellationToken,
    join_handle: tokio::task::JoinHandle<()>,
    _endpoint_name: String,
}

impl NatsMultiplexedServer {
    /// Create a new multiplexed NATS server
    ///
    /// # Arguments
    ///
    /// * `nats_client` - NATS client for connection management
    /// * `component_registry` - Component registry to get service groups from
    /// * `cancellation_token` - Token for graceful shutdown
    pub fn new(
        nats_client: async_nats::Client,
        component_registry: crate::component::Registry,
        cancellation_token: CancellationToken,
    ) -> Arc<Self> {
        Arc::new(Self {
            nats_client,
            component_registry,
            handlers: Arc::new(DashMap::new()),
            cancellation_token,
        })
    }
}

#[async_trait]
impl super::unified_server::RequestPlaneServer for NatsMultiplexedServer {
    async fn register_endpoint(
        &self,
        endpoint_name: String,
        service_handler: Arc<dyn PushWorkHandler>,
        instance_id: u64,
        namespace: String,
        component_name: String,
        system_health: Arc<Mutex<SystemHealth>>,
    ) -> Result<()> {
        tracing::info!(
            endpoint_name = %endpoint_name,
            namespace = %namespace,
            component = %component_name,
            instance_id = instance_id,
            "NatsMultiplexedServer::register_endpoint called"
        );

        // Get the service group from the component registry
        // Service name format matches Component::service_name(): "{namespace}_{component}" slugified
        use crate::transports::nats::Slug;
        let service_name_raw = format!("{}_{}", namespace, component_name);
        let service_name = Slug::slugify(&service_name_raw).to_string();

        tracing::debug!(
            service_name_raw = %service_name_raw,
            service_name = %service_name,
            "Looking up service group in registry"
        );

        let registry = self.component_registry.inner.lock().await;
        let service_group = registry
            .services
            .get(&service_name)
            .map(|service| service.group(&service_name))
            .ok_or_else(|| anyhow::anyhow!("Service '{}' not found in registry", service_name))?;
        drop(registry);

        tracing::info!("Successfully retrieved service group");

        // Construct the full NATS subject with instance ID
        // Format: {endpoint_name}-{instance_id_hex}
        // This matches Endpoint::name_with_id() and subject_to() format
        let endpoint_with_id = format!("{}-{:x}", endpoint_name, instance_id);

        // Create NATS service endpoint with the full subject
        let service_endpoint = service_group
            .endpoint(&endpoint_with_id)
            .await
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create NATS endpoint '{}': {}",
                    endpoint_with_id,
                    e
                )
            })?;

        tracing::info!(
            endpoint_name = %endpoint_name,
            endpoint_with_id = %endpoint_with_id,
            namespace = %namespace,
            component = %component_name,
            instance_id = instance_id,
            "Registering NATS endpoint"
        );

        // Create cancellation token for this specific endpoint
        let endpoint_cancel = CancellationToken::new();
        let endpoint_cancel_clone = endpoint_cancel.clone();

        // Build the push endpoint
        let push_endpoint = PushEndpoint::builder()
            .service_handler(service_handler)
            .cancellation_token(endpoint_cancel_clone)
            .graceful_shutdown(true)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build NATS push endpoint: {}", e))?;

        tracing::info!(
            endpoint_name = %endpoint_name,
            endpoint_with_id = %endpoint_with_id,
            "Starting NATS push endpoint listener (blocking)"
        );

        // Spawn task to handle this endpoint using PushEndpoint
        // Note: PushEndpoint::start() is a blocking loop that runs until cancelled
        let endpoint_name_clone = endpoint_name.clone();
        let join_handle = tokio::spawn(async move {
            if let Err(e) = push_endpoint
                .start(
                    service_endpoint,
                    namespace,
                    component_name,
                    endpoint_name_clone.clone(),
                    instance_id,
                    system_health,
                )
                .await
            {
                tracing::error!(
                    endpoint_name = %endpoint_name_clone,
                    error = %e,
                    "NATS endpoint task failed"
                );
            } else {
                tracing::info!(
                    endpoint_name = %endpoint_name_clone,
                    "NATS push endpoint listener completed"
                );
            }
        });

        // Give the endpoint a moment to start listening
        // This prevents a race condition where discovery registers the endpoint
        // before NATS is actually ready to receive requests
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Store task info for later cleanup
        self.handlers.insert(
            endpoint_name.clone(),
            EndpointTask {
                cancel_token: endpoint_cancel,
                join_handle,
                _endpoint_name: endpoint_name,
            },
        );

        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint_name: &str) -> Result<()> {
        if let Some((_, task)) = self.handlers.remove(endpoint_name) {
            tracing::info!(
                endpoint_name = %endpoint_name,
                "Unregistering NATS endpoint"
            );
            // Cancel the token to trigger graceful shutdown
            task.cancel_token.cancel();

            // Wait for the endpoint task to complete (which includes waiting for inflight requests)
            tracing::debug!(
                endpoint_name = %endpoint_name,
                "Waiting for NATS endpoint task to complete"
            );
            if let Err(e) = task.join_handle.await {
                tracing::warn!(
                    endpoint_name = %endpoint_name,
                    error = %e,
                    "NATS endpoint task panicked during shutdown"
                );
            }
            tracing::info!(
                endpoint_name = %endpoint_name,
                "NATS endpoint unregistration complete"
            );
        }
        Ok(())
    }

    fn address(&self) -> String {
        // Return NATS server URL from connection info
        // NATS client doesn't expose server info directly, return generic address
        "nats://connected".to_string()
    }

    fn transport_name(&self) -> &'static str {
        "nats"
    }

    fn is_healthy(&self) -> bool {
        // Check if NATS client is connected
        // NATS client doesn't expose connection state directly, assume healthy
        true
    }
}
