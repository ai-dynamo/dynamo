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
use crate::protocols::EndpointId;
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
    handlers: Arc<DashMap<(EndpointId, u64), EndpointTask>>,
    cancellation_token: CancellationToken,
}

#[derive(Clone)]
struct EndpointTask {
    registration: Arc<()>,
    cancel_token: CancellationToken,
    finished: CancellationToken,
}

/// Retains the reservation until setup and listener shutdown have both finished.
struct EndpointTaskGuard {
    handlers: Arc<DashMap<(EndpointId, u64), EndpointTask>>,
    endpoint_key: (EndpointId, u64),
    task: EndpointTask,
}

impl Drop for EndpointTaskGuard {
    fn drop(&mut self) {
        self.handlers.remove_if(&self.endpoint_key, |_, task| {
            Arc::ptr_eq(&task.registration, &self.task.registration)
        });
        self.task.finished.cancel();
    }
}

/// NATS subject within a namespace/component service group for one endpoint instance.
fn instance_subject(endpoint_name: &str, instance_id: u64) -> String {
    format!("{endpoint_name}-{instance_id:x}")
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

        let endpoint_with_id = instance_subject(&endpoint_name, instance_id);
        let endpoint_key = (
            EndpointId {
                namespace: namespace.clone(),
                component: component_name.clone(),
                name: endpoint_name.clone(),
            },
            instance_id,
        );
        let task = EndpointTask {
            registration: Arc::new(()),
            cancel_token: self.cancellation_token.child_token(),
            finished: CancellationToken::new(),
        };
        // Dropping the registration future must not leave an orphaned setup task.
        let cancel_on_drop = task.cancel_token.clone().drop_guard();
        let push_endpoint = PushEndpoint::builder()
            .service_handler(service_handler)
            .cancellation_token(task.cancel_token.clone())
            .graceful_shutdown(true)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build NATS push endpoint: {e}"))?;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let component_registry = self.component_registry.clone();
        let endpoint_name_for_task = endpoint_name.clone();
        match self.handlers.entry(endpoint_key.clone()) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(task.clone());
                let task_guard = EndpointTaskGuard {
                    handlers: self.handlers.clone(),
                    endpoint_key,
                    task: task.clone(),
                };
                // Setup runs in the owned task so caller cancellation cannot drop
                // service_group.endpoint halfway through creating a subscription.
                tokio::spawn(async move {
                    let _task_guard = task_guard;
                    let endpoint_cancel = &_task_guard.task.cancel_token;
                    use crate::transports::nats::Slug;
                    let service_name =
                        Slug::slugify(&format!("{namespace}_{component_name}")).to_string();
                    let setup = async {
                        let registry = component_registry.inner.lock().await;
                        let service_group = registry
                            .services
                            .get(&service_name)
                            .map(|service| service.group(&service_name))
                            .ok_or_else(|| {
                                anyhow::anyhow!("Service '{service_name}' not found in registry")
                            })?;
                        drop(registry);
                        if endpoint_cancel.is_cancelled() {
                            anyhow::bail!("NATS endpoint setup cancelled");
                        }
                        service_group
                            .endpoint(&endpoint_with_id)
                            .await
                            .map_err(|e| {
                                anyhow::anyhow!(
                                    "Failed to create NATS endpoint '{endpoint_with_id}': {e}"
                                )
                            })
                    }
                    .await;
                    let mut service_endpoint = match setup {
                        Ok(endpoint) => endpoint,
                        Err(error) => {
                            let _ = started_tx.send(Err(error));
                            return;
                        }
                    };
                    if endpoint_cancel.is_cancelled() || started_tx.send(Ok(())).is_err() {
                        if let Err(error) = service_endpoint.stop().await {
                            tracing::warn!(%error, "Failed to stop cancelled NATS endpoint setup");
                        }
                        return;
                    }
                    if let Err(error) = push_endpoint
                        .start(
                            service_endpoint,
                            namespace,
                            component_name,
                            endpoint_name_for_task.clone(),
                            instance_id,
                            system_health,
                        )
                        .await
                    {
                        tracing::error!(
                            endpoint_name = %endpoint_name_for_task,
                            %error,
                            "NATS endpoint task failed"
                        );
                    }
                });
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                anyhow::bail!("Endpoint '{endpoint_name}' is already registered for this instance");
            }
        }

        started_rx.await.map_err(|error| {
            anyhow::anyhow!("NATS endpoint setup task ended before ready: {error}")
        })??;
        if task.cancel_token.is_cancelled() {
            anyhow::bail!("Endpoint '{endpoint_name}' was unregistered while starting");
        }
        cancel_on_drop.disarm();
        Ok(())
    }

    async fn unregister_endpoint(&self, endpoint: &EndpointId, instance_id: u64) -> Result<()> {
        let task = self
            .handlers
            .get(&(endpoint.clone(), instance_id))
            .map(|entry| entry.value().clone());
        if let Some(task) = task {
            task.cancel_token.cancel();
            // Keep the reservation until setup and the listener have stopped, so
            // a replacement cannot overlap an outgoing subscription. Concurrent
            // unregister callers all observe the same completion signal.
            task.finished.cancelled().await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_subject_is_the_client_subject_and_unique_per_instance() {
        assert_eq!(instance_subject("generate", 0xa), "generate-a");
        assert_ne!(
            instance_subject("generate", 0xa),
            instance_subject("generate", 0xb),
            "two instances of one endpoint name must not share a handler-map key"
        );
    }
}
