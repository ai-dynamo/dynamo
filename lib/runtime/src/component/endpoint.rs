// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
pub use async_nats::service::endpoint::Stats as EndpointStats;
use derive_builder::Builder;
use derive_getters::Dissolve;
use educe::Educe;
use tokio_util::sync::CancellationToken;

use crate::{
    component::{Endpoint, Instance, TransportType, service::EndpointStatsHandler},
    distributed::RequestPlaneMode,
    pipeline::network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint},
    storage::key_value_store,
    traits::DistributedRuntimeProvider,
};

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
            mut endpoint,
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

        let metrics_labels: Option<Vec<(&str, &str)>> = metrics_labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
        // Add metrics to the handler. The endpoint provides additional information to the handler.
        handler.add_metrics(&endpoint, metrics_labels.as_deref())?;

        // Determine request plane mode
        let request_plane_mode = endpoint.drt().request_plane();
        if request_plane_mode.is_nats() {
            // We only need the service if we want NATS metrics.
            // TODO: This is called for every endpoint of a component. Ideally we only call it once
            // on the component.
            endpoint.component.add_stats_service().await?;
        }
        tracing::info!(
            "Endpoint starting with request plane mode: {:?}",
            request_plane_mode
        );

        // Insert the stats handler. depends on NATS.
        if let Some(stats_handler) = stats_handler {
            let registry = endpoint.drt().component_registry().inner.lock().await;
            let handler_map = registry
                .stats_handlers
                .get(&service_name)
                .cloned()
                .expect("no stats handler registry; this is unexpected");
            handler_map
                .lock()
                .insert(endpoint.subject_to(connection_id), stats_handler);
        }

        // This creates a child token of the runtime's endpoint_shutdown_token. That token is
        // cancelled first as part of graceful shutdown. See Runtime::shutdown.
        let endpoint_shutdown_token = endpoint.drt().child_token();

        // Extract all values needed from endpoint before any spawns
        let namespace_name = endpoint.component.namespace.name.clone();
        let component_name = endpoint.component.name.clone();
        let endpoint_name = endpoint.name.clone();
        let system_health = endpoint.drt().system_health();
        let subject = endpoint.subject_to(connection_id);

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &health_check_payload {
            // Build transport based on request plane mode
            let transport = build_transport_type(
                request_plane_mode,
                &endpoint_name,
                &subject,
                TransportContext::HealthCheck,
            );

            let instance = Instance {
                component: component_name.clone(),
                endpoint: endpoint_name.clone(),
                namespace: namespace_name.clone(),
                instance_id: connection_id,
                transport,
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

        // Launch endpoint based on request plane mode
        let tracker_clone = if graceful_shutdown {
            Some(endpoint.drt().graceful_shutdown_tracker())
        } else {
            None
        };

        // Create clones for the async closure
        let namespace_name_for_task = namespace_name.clone();
        let component_name_for_task = component_name.clone();
        let endpoint_name_for_task = endpoint_name.clone();

        // Get the unified request plane server (works for all transport types)
        let server = endpoint.drt().request_plane_server().await?;

        tracing::info!(
            endpoint = %endpoint_name_for_task,
            transport = server.transport_name(),
            "Registering endpoint with request plane server"
        );

        // Register endpoint with the server (unified interface)
        server
            .register_endpoint(
                endpoint_name_for_task.clone(),
                handler,
                connection_id,
                namespace_name_for_task.clone(),
                component_name_for_task.clone(),
                system_health.clone(),
            )
            .await?;

        // Create cleanup task that unregisters on cancellation
        let endpoint_name_for_cleanup = endpoint_name_for_task.clone();
        let server_for_cleanup = server.clone();
        let cancel_token_for_cleanup = endpoint_shutdown_token.clone();

        let task: tokio::task::JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
            cancel_token_for_cleanup.cancelled().await;

            tracing::debug!(
                endpoint = %endpoint_name_for_cleanup,
                "Unregistering endpoint from request plane server"
            );

            // Unregister from server
            if let Err(e) = server_for_cleanup
                .unregister_endpoint(&endpoint_name_for_cleanup)
                .await
            {
                tracing::warn!(
                    endpoint = %endpoint_name_for_cleanup,
                    error = %e,
                    "Failed to unregister endpoint"
                );
            }

            // Unregister from graceful shutdown tracker
            if let Some(tracker) = tracker_clone {
                tracing::debug!("Unregister endpoint from graceful shutdown tracker");
                tracker.unregister_endpoint();
            }

            anyhow::Ok(())
        });

        // Register this endpoint instance in the discovery plane
        // The discovery interface abstracts storage backend (etcd, k8s, etc) and provides
        // consistent registration/discovery across the system.
        let discovery = endpoint.drt().discovery();

        // Build transport for discovery service based on request plane mode
        let transport = build_transport_type(
            request_plane_mode,
            &endpoint_name,
            &subject,
            TransportContext::Discovery,
        );

        let discovery_spec = crate::discovery::DiscoverySpec::Endpoint {
            namespace: namespace_name.clone(),
            component: component_name.clone(),
            endpoint: endpoint_name.clone(),
            transport,
        };

        if let Err(e) = discovery.register(discovery_spec).await {
            tracing::error!(
                component_name,
                endpoint_name,
                error = %e,
                "Unable to register service for discovery"
            );
            endpoint_shutdown_token.cancel();
            anyhow::bail!(
                "Unable to register service for discovery. Check discovery service status"
            );
        }

        task.await??;

        Ok(())
    }
}

/// Context for building transport type - determines port and formatting differences
enum TransportContext {
    /// For health check targets
    HealthCheck,
    /// For discovery service registration
    Discovery,
}

/// Build transport type based on request plane mode and context
///
/// This unified function handles both health check and discovery transport building,
/// with context-specific differences:
/// - HTTP: Both use the same port (default 8888, configurable via DYN_HTTP_RPC_PORT)
/// - TCP: Health check omits endpoint suffix, discovery includes it for routing
/// - NATS: Identical for both contexts
fn build_transport_type(
    mode: RequestPlaneMode,
    endpoint_name: &str,
    subject: &str,
    context: TransportContext,
) -> TransportType {
    match mode {
        RequestPlaneMode::Http => {
            let http_host = crate::utils::get_http_rpc_host_from_env();
            // Both health check and discovery use the same port (8888) where the HTTP server binds
            let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(8888);
            let rpc_root =
                std::env::var("DYN_HTTP_RPC_ROOT_PATH").unwrap_or_else(|_| "/v1/rpc".to_string());

            let http_endpoint = format!(
                "http://{}:{}{}/{}",
                http_host, http_port, rpc_root, endpoint_name
            );

            TransportType::Http(http_endpoint)
        }
        RequestPlaneMode::Tcp => {
            let tcp_host = crate::utils::get_tcp_rpc_host_from_env();
            let tcp_port = std::env::var("DYN_TCP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(9999);

            let tcp_endpoint = match context {
                TransportContext::HealthCheck => {
                    // Health check uses simple host:port format
                    format!("{}:{}", tcp_host, tcp_port)
                }
                TransportContext::Discovery => {
                    // Discovery includes endpoint name for routing
                    format!("{}:{}/{}", tcp_host, tcp_port, endpoint_name)
                }
            };

            TransportType::Tcp(tcp_endpoint)
        }
        RequestPlaneMode::Nats => TransportType::Nats(subject.to_string()),
    }
}
