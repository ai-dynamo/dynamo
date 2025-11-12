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
    config::RequestPlaneMode,
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

        let metrics_labels: Option<Vec<(&str, &str)>> = metrics_labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
        // Add metrics to the handler. The endpoint provides additional information to the handler.
        handler.add_metrics(&endpoint, metrics_labels.as_deref())?;

        let registry = endpoint.drt().component_registry().inner.lock().await;

        // get the group
        let group = registry
            .services
            .get(&service_name)
            .map(|service| service.group(endpoint.component.service_name()))
            .ok_or(anyhow::anyhow!("Service not found"))?;

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

        // Determine request plane mode
        let request_plane_mode = RequestPlaneMode::get();
        tracing::info!(
            "Endpoint starting with request plane mode: {:?}",
            request_plane_mode
        );

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

        let task = match request_plane_mode {
            RequestPlaneMode::Http => {
                // HTTP mode - use SharedHttpServer
                let http_server = endpoint.drt().http_server().await?;

                // Register this endpoint with the shared server
                http_server
                    .register_endpoint(
                        endpoint_name_for_task.clone(),
                        handler,
                        connection_id,
                        namespace_name_for_task.clone(),
                        component_name_for_task.clone(),
                        endpoint_name_for_task.clone(),
                        system_health.clone(),
                    )
                    .await?;

                // Create a task that waits for cancellation and then unregisters
                let endpoint_name_for_cleanup = endpoint_name_for_task.clone();
                let http_server_for_cleanup = http_server.clone();
                let cancel_token_for_cleanup = endpoint_shutdown_token.clone();

                tokio::spawn(async move {
                    cancel_token_for_cleanup.cancelled().await;

                    tracing::debug!("Unregistering endpoint from shared HTTP server");
                    http_server_for_cleanup
                        .unregister_endpoint(&endpoint_name_for_cleanup, &endpoint_name_for_cleanup)
                        .await;

                    // Unregister from graceful shutdown tracker
                    if let Some(tracker) = tracker_clone {
                        tracing::debug!("Unregister endpoint from graceful shutdown tracker");
                        tracker.unregister_endpoint();
                    }

                    Ok(())
                })
            }
            RequestPlaneMode::Tcp => {
                // TCP mode - use SharedTcpServer
                tracing::info!("Starting endpoint in TCP mode, initializing SharedTcpServer");
                let tcp_server = endpoint.drt().shared_tcp_server().await?;
                tracing::info!("SharedTcpServer obtained, registering endpoint");

                // Register this endpoint with the shared TCP server
                tcp_server
                    .register_endpoint(
                        endpoint_name_for_task.clone(),
                        handler,
                        connection_id,
                        namespace_name_for_task.clone(),
                        component_name_for_task.clone(),
                        endpoint_name_for_task.clone(),
                        system_health.clone(),
                    )
                    .await?;

                // Create a task that waits for cancellation and then unregisters
                let endpoint_name_for_cleanup = endpoint_name_for_task.clone();
                let tcp_server_for_cleanup = tcp_server.clone();
                let cancel_token_for_cleanup = endpoint_shutdown_token.clone();

                tokio::spawn(async move {
                    cancel_token_for_cleanup.cancelled().await;

                    tracing::debug!("Unregistering endpoint from shared TCP server");
                    tcp_server_for_cleanup
                        .unregister_endpoint(&endpoint_name_for_cleanup, &endpoint_name_for_cleanup)
                        .await;

                    // Unregister from graceful shutdown tracker
                    if let Some(tracker) = tracker_clone {
                        tracing::debug!("Unregister endpoint from graceful shutdown tracker");
                        tracker.unregister_endpoint();
                    }

                    Ok(())
                })
            }
            RequestPlaneMode::Nats => {
                let service_endpoint = group
                    .endpoint(&endpoint.name_with_id(connection_id))
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

                let push_endpoint = PushEndpoint::builder()
                    .service_handler(handler)
                    .cancellation_token(endpoint_shutdown_token.clone())
                    .graceful_shutdown(graceful_shutdown)
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

                tokio::spawn(async move {
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
                })
            }
        };

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
/// - HTTP: Health check uses port 8081, discovery uses 8080
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
            let default_port = match context {
                TransportContext::HealthCheck => 8081,
                TransportContext::Discovery => 8080,
            };
            let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(default_port);
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
                .unwrap_or(9090);

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
        RequestPlaneMode::Nats => TransportType::NatsTcp(subject.to_string()),
    }
}
