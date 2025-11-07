// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Dissolve;
use tokio_util::sync::CancellationToken;

use crate::storage::key_value_store;

use super::*;

pub use async_nats::service::endpoint::Stats as EndpointStats;

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct EndpointConfig {
    #[builder(private)]
    endpoint: Endpoint,

    /// Endpoint handler
    #[educe(Debug(ignore))]
    handler: Arc<dyn PushWorkHandler>,

    /// Stats handler
    #[educe(Debug(ignore))]
    #[builder(default, private)]
    _stats_handler: Option<EndpointStatsHandler>,

    /// Additional labels for metrics
    #[builder(default, setter(into))]
    metrics_labels: Option<Vec<(String, String)>>,

    /// Whether to wait for inflight requests to complete during shutdown
    #[builder(default = "true")]
    graceful_shutdown: bool,

    /// Health check payload for this endpoint
    /// This payload will be sent to the endpoint during health checks
    /// to verify it's responding properly
    #[educe(Debug(ignore))]
    #[builder(default, setter(into, strip_option))]
    health_check_payload: Option<serde_json::Value>,
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    pub fn stats_handler<F>(self, handler: F) -> Self
    where
        F: FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static,
    {
        self._stats_handler(Some(Box::new(handler)))
    }

    pub async fn start(self) -> Result<()> {
        let (
            endpoint,
            handler,
            stats_handler,
            metrics_labels,
            graceful_shutdown,
            health_check_payload,
        ) = self.build_internal()?.dissolve();
        let connection_id = endpoint.drt().connection_id();

        tracing::debug!(
            "Starting endpoint: {}",
            endpoint.etcd_path_with_lease_id(connection_id)
        );

        let service_name = endpoint.component.service_name();

        // acquire the registry lock
        let registry = endpoint.drt().component_registry.inner.lock().await;

        let metrics_labels: Option<Vec<(&str, &str)>> = metrics_labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
        // Add metrics to the handler. The endpoint provides additional information to the handler.
        handler.add_metrics(&endpoint, metrics_labels.as_deref())?;

        // get the group
        let group = registry
            .services
            .get(&service_name)
            .map(|service| service.group(endpoint.component.service_name()))
            .ok_or(error!("Service not found"))?;

        // get the stats handler map
        let handler_map = registry
            .stats_handlers
            .get(&service_name)
            .cloned()
            .expect("no stats handler registry; this is unexpected");

        drop(registry);

        // insert the stats handler
        if let Some(stats_handler) = stats_handler {
            handler_map
                .lock()
                .insert(endpoint.subject_to(connection_id), stats_handler);
        }

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&endpoint.name_with_id(connection_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        // This creates a child token of the runtime's endpoint_shutdown_token. That token is
        // cancelled first as part of graceful shutdown. See Runtime::shutdown.
        let endpoint_shutdown_token = endpoint.drt().child_token();

        // Extract all values needed from endpoint before any spawns
        let namespace_name = endpoint.component.namespace.name.clone();
        let component_name = endpoint.component.name.clone();
        let endpoint_name = endpoint.name.clone();
        let system_health = endpoint.drt().system_health.clone();
        let subject = endpoint.subject_to(connection_id);

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &health_check_payload {
            let instance = Instance {
                component: component_name.clone(),
                endpoint: endpoint_name.clone(),
                namespace: namespace_name.clone(),
                instance_id: connection_id,
                transport: TransportType::NatsTcp(subject.clone()),
            };
            tracing::debug!(endpoint_name = %endpoint_name, "Registering endpoint health check target");
            let guard = system_health.lock();
            guard.register_health_check_target(
                &endpoint_name,
                instance,
                health_check_payload.clone(),
            );
            if let Some(notifier) = guard.get_endpoint_health_check_notifier(&endpoint_name) {
                handler.set_endpoint_health_check_notifier(notifier)?;
            }
        }

        // Register with graceful shutdown tracker if needed
        if graceful_shutdown {
            tracing::debug!(
                "Registering endpoint '{}' with graceful shutdown tracker",
                endpoint.name
            );
            let tracker = endpoint.drt().graceful_shutdown_tracker();
            tracker.register_endpoint();
        } else {
            tracing::debug!("Endpoint '{}' has graceful_shutdown=false", endpoint.name);
        }

        let push_endpoint = PushEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(endpoint_shutdown_token.clone())
            .graceful_shutdown(graceful_shutdown)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        let tracker_clone = if graceful_shutdown {
            Some(endpoint.drt().graceful_shutdown_tracker())
        } else {
            None
        };

        // Create clones for the async closure
        let namespace_name_for_task = namespace_name.clone();
        let component_name_for_task = component_name.clone();
        let endpoint_name_for_task = endpoint_name.clone();

        let task = tokio::spawn(async move {
            let result = push_endpoint
                .start(
                    service_endpoint,
                    namespace_name_for_task,
                    component_name_for_task,
                    endpoint_name_for_task,
                    connection_id,
                    system_health,
                )
                .await;

            // Unregister from graceful shutdown tracker
            if let Some(tracker) = tracker_clone {
                tracing::debug!("Unregistering endpoint from graceful shutdown tracker");
                tracker.unregister_endpoint();
            }

            result
        });

        let info = Instance {
            component: component_name.clone(),
            endpoint: endpoint_name.clone(),
            namespace: namespace_name.clone(),
            instance_id: connection_id,
            transport: TransportType::NatsTcp(subject),
        };

        let info = serde_json::to_vec_pretty(&info)?;

        let store = endpoint.drt().store();
        let instances_bucket = store
            .get_or_create_bucket(super::INSTANCE_ROOT_PATH, None)
            .await?;
        let key = key_value_store::Key::from_raw(endpoint.unique_path(connection_id));
        if let Err(err) = instances_bucket.insert(&key, info.into(), 0).await {
            tracing::error!(
                component_name,
                endpoint_name,
                error = %err,
                "Unable to register service for discovery"
            );
            endpoint_shutdown_token.cancel();
            return Err(error!(
                "Unable to register service for discovery. Check discovery service status"
            ));
        }
        task.await??;

        Ok(())
    }
}
