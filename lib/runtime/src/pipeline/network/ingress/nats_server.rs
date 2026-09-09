// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS Multiplexed Server
//!
//! Provides a multiplexed NATS server that handles multiple endpoints on a single
//! NATS service group. This replaces the per-endpoint PushEndpoint pattern with
//! a unified multiplexed approach consistent with TCP server.

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
    registration: Arc<()>,
    cancel_token: CancellationToken,
    join_handle: Option<tokio::task::JoinHandle<()>>,
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

    fn remove_reservation(&self, endpoint_with_id: &str, registration: &Arc<()>) {
        if let dashmap::mapref::entry::Entry::Occupied(entry) =
            self.handlers.entry(endpoint_with_id.to_string())
        {
            if Arc::ptr_eq(&entry.get().registration, registration) {
                entry.remove();
            }
        }
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

        let endpoint_with_id = format!("{}-{:x}", endpoint_name, instance_id);
        let registration = Arc::new(());
        let endpoint_cancel = CancellationToken::new();

        match self.handlers.entry(endpoint_with_id.clone()) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(EndpointTask {
                    registration: Arc::clone(&registration),
                    cancel_token: endpoint_cancel.clone(),
                    join_handle: None,
                });
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                anyhow::bail!("Endpoint '{endpoint_name}' is already registered for this instance");
            }
        }

        use crate::transports::nats::Slug;
        let service_name_raw = format!("{}_{}", namespace, component_name);
        let service_name = Slug::slugify(&service_name_raw).to_string();

        tracing::debug!(
            service_name_raw = %service_name_raw,
            service_name = %service_name,
            "Looking up service group in registry"
        );

        let setup = async {
            let registry = self.component_registry.inner.lock().await;
            let service_group = registry
                .services
                .get(&service_name)
                .map(|service| service.group(&service_name))
                .ok_or_else(|| {
                    anyhow::anyhow!("Service '{}' not found in registry", service_name)
                })?;
            drop(registry);

            let service_endpoint =
                service_group
                    .endpoint(&endpoint_with_id)
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to create NATS endpoint '{}': {}",
                            endpoint_with_id,
                            e
                        )
                    })?;

            let push_endpoint = PushEndpoint::builder()
                .service_handler(service_handler)
                .cancellation_token(endpoint_cancel.clone())
                .graceful_shutdown(true)
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build NATS push endpoint: {}", e))?;

            Ok::<_, anyhow::Error>((service_endpoint, push_endpoint))
        }
        .await;

        let (service_endpoint, push_endpoint) = match setup {
            Ok(result) => result,
            Err(error) => {
                self.remove_reservation(&endpoint_with_id, &registration);
                return Err(error);
            }
        };

        tracing::info!("Successfully retrieved service group");

        tracing::info!(
            endpoint_name = %endpoint_name,
            endpoint_with_id = %endpoint_with_id,
            namespace = %namespace,
            component = %component_name,
            instance_id = instance_id,
            "Registering NATS endpoint"
        );

        tracing::info!(
            endpoint_name = %endpoint_name,
            endpoint_with_id = %endpoint_with_id,
            "Starting NATS push endpoint listener (blocking)"
        );

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

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        match self.handlers.entry(endpoint_with_id.clone()) {
            dashmap::mapref::entry::Entry::Occupied(mut entry)
                if Arc::ptr_eq(&entry.get().registration, &registration) =>
            {
                entry.insert(EndpointTask {
                    registration,
                    cancel_token: endpoint_cancel,
                    join_handle: Some(join_handle),
                });
            }
            _ => {
                endpoint_cancel.cancel();
                let _ = join_handle.await;
                anyhow::bail!("Endpoint '{endpoint_name}' was unregistered while starting");
            }
        }

        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint_name: &str, instance_id: u64) -> Result<()> {
        let endpoint_with_id = format!("{endpoint_name}-{instance_id:x}");
        if let Some((_, task)) = self.handlers.remove(&endpoint_with_id) {
            tracing::info!(
                endpoint_name = %endpoint_name,
                "Unregistering NATS endpoint"
            );
            task.cancel_token.cancel();

            tracing::debug!(
                endpoint_name = %endpoint_name,
                "Waiting for NATS endpoint task to complete"
            );
            if let Some(join_handle) = task.join_handle {
                if let Err(e) = join_handle.await {
                    tracing::warn!(
                        endpoint_name = %endpoint_name,
                        error = %e,
                        "NATS endpoint task panicked during shutdown"
                    );
                }
            }
            tracing::info!(
                endpoint_name = %endpoint_name,
                "NATS endpoint unregistration complete"
            );
        }
        Ok(())
    }

    fn address(&self) -> String {
        "nats://connected".to_string()
    }

    fn transport_name(&self) -> &'static str {
        "nats"
    }

    fn is_healthy(&self) -> bool {
        true
    }
}
