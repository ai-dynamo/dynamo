// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use derive_builder::Builder;
use derive_getters::Dissolve;
use educe::Educe;
use tokio_util::sync::CancellationToken;

use crate::{
    component::{Endpoint, Instance, TransportType},
    distributed::RequestPlaneMode,
    pipeline::network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint},
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::nats,
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

    /// Register an async engine in the local endpoint registry for direct in-process calls
    pub fn register_local_engine(
        self,
        engine: crate::local_endpoint_registry::LocalAsyncEngine,
    ) -> Result<Self> {
        if let Some(endpoint) = &self.endpoint {
            let registry = endpoint.drt().local_endpoint_registry();
            registry.register(endpoint.name.clone(), engine);
            tracing::debug!(
                "Registered engine for endpoint '{}' in local registry",
                endpoint.name
            );
        }
        Ok(self)
    }

    pub async fn start(self) -> Result<()> {
        let (endpoint, handler, metrics_labels, graceful_shutdown, health_check_payload) =
            self.build_internal()?.dissolve();
        let connection_id = endpoint.drt().connection_id();
        let endpoint_id = endpoint.id();

        tracing::debug!("Starting endpoint: {endpoint_id}");

        let metrics_labels: Option<Vec<(&str, &str)>> = metrics_labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());
        // Add metrics to the handler. The endpoint provides additional information to the handler.
        handler.add_metrics(&endpoint, metrics_labels.as_deref())?;

        // This creates a child token of the runtime's endpoint_shutdown_token. That token is
        // cancelled first as part of graceful shutdown. See Runtime::shutdown.
        let endpoint_shutdown_token = endpoint.drt().child_token();

        let system_health = endpoint.drt().system_health();

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
        let namespace_name_for_task = endpoint_id.namespace.clone();
        let component_name_for_task = endpoint_id.component.clone();
        let endpoint_name_for_task = endpoint_id.name.clone();

        // Get the unified request plane server
        let server = endpoint.drt().request_plane_server().await?;

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &health_check_payload {
            // Build transport based on request plane mode
            let transport = build_transport_type(&endpoint, &endpoint_id, connection_id).await?;

            let instance = Instance {
                component: endpoint_id.component.clone(),
                endpoint: endpoint_id.name.clone(),
                namespace: endpoint_id.namespace.clone(),
                instance_id: connection_id,
                transport,
            };
            tracing::debug!(endpoint_name = %endpoint.name, "Registering endpoint health check target");
            let guard = system_health.lock();
            guard.register_health_check_target(
                &endpoint.name,
                instance,
                health_check_payload.clone(),
            );
            if let Some(notifier) = guard.get_endpoint_health_check_notifier(&endpoint.name) {
                handler.set_endpoint_health_check_notifier(notifier)?;
            }
        }

        tracing::debug!(
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
        let transport = build_transport_type(&endpoint, &endpoint_id, connection_id).await?;

        let discovery_spec = crate::discovery::DiscoverySpec::Endpoint {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
            transport,
        };

        if let Err(e) = discovery.register(discovery_spec).await {
            tracing::error!(
                %endpoint_id,
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

/// Build transport type based on request plane mode
///
/// This function handles both health check and discovery transport building.
/// All transport modes use consistent addressing:
/// - HTTP: Uses full URL path including endpoint name (e.g., http://host:port/v1/rpc/endpoint_name)
/// - TCP: Includes instance_id and endpoint name for routing (e.g., host:port/instance_id_hex/endpoint_name)
/// - NATS: Uses subject-based addressing (unique per endpoint)
///
/// # Errors
/// Returns an error if TCP mode is used but the TCP server hasn't been started yet.
fn build_transport_type_inner(
    mode: RequestPlaneMode,
    endpoint_id: &EndpointId,
    connection_id: u64,
) -> Result<TransportType> {
    match mode {
        RequestPlaneMode::Http => {
            let http_host = crate::utils::get_http_rpc_host_from_env();
            let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(8888);
            let rpc_root =
                std::env::var("DYN_HTTP_RPC_ROOT_PATH").unwrap_or_else(|_| "/v1/rpc".to_string());

            let http_endpoint = format!(
                "http://{http_host}:{http_port}{rpc_root}/{}",
                endpoint_id.name
            );

            Ok(TransportType::Http(http_endpoint))
        }
        RequestPlaneMode::Tcp => {
            let tcp_host = crate::utils::get_tcp_rpc_host_from_env();
            // If a fixed port is explicitly configured, use it directly (no init ordering dependency).
            // Otherwise, use the actual bound port (set by TCP server after binding when port 0 is used).
            let tcp_port = std::env::var("DYN_TCP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(crate::pipeline::network::manager::get_actual_tcp_rpc_port()?);

            // Include instance_id and endpoint name for proper TCP routing.
            // Format: host:port/instance_id_hex/endpoint_name
            // This ensures each worker has a unique routing key when multiple workers
            // share the same TCP server (e.g., --num-workers > 1).
            let tcp_endpoint = format!(
                "{}:{}/{:x}/{}",
                tcp_host, tcp_port, connection_id, endpoint_id.name
            );

            Ok(TransportType::Tcp(tcp_endpoint))
        }
        RequestPlaneMode::Nats => Ok(TransportType::Nats(nats::instance_subject(
            endpoint_id,
            connection_id,
        ))),
    }
}

/// Build transport type, ensuring TCP server is initialized when needed.
///
/// In TCP mode with an OS-assigned port (`DYN_TCP_RPC_PORT` unset or invalid), the server must bind
/// before we can construct a correct transport address. This helper ensures that initialization
/// occurs, then delegates to the internal builder.
pub async fn build_transport_type(
    endpoint: &Endpoint,
    endpoint_id: &EndpointId,
    connection_id: u64,
) -> Result<TransportType> {
    let mode = endpoint.drt().request_plane();

    if mode == RequestPlaneMode::Tcp {
        // Only force server init when we *don't* have a valid explicit port.
        let has_fixed_port = std::env::var("DYN_TCP_RPC_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .is_some();

        if !has_fixed_port {
            // Ensure request plane server is initialized before building transport.
            let _ = endpoint.drt().request_plane_server().await?;
        }
    }

    build_transport_type_inner(mode, endpoint_id, connection_id)
}

impl Endpoint {
    /// Unregister this endpoint instance from discovery.
    ///
    /// This removes the endpoint from the instances bucket, preventing the router
    /// from sending requests to this worker. Use this when a worker is sleeping
    /// and should not receive any requests.
    pub async fn unregister_endpoint_instance(&self) -> anyhow::Result<()> {
        let drt = self.drt();
        let instance_id = drt.connection_id();
        let endpoint_id = self.id();

        // Get the transport type for the endpoint
        let transport = build_transport_type(self, &endpoint_id, instance_id).await?;

        let instance = crate::discovery::DiscoveryInstance::Endpoint(Instance {
            namespace: endpoint_id.namespace,
            component: endpoint_id.component,
            endpoint: endpoint_id.name,
            instance_id,
            transport,
        });

        let discovery = drt.discovery();
        if let Err(e) = discovery.unregister(instance).await {
            let endpoint_id = self.id();
            tracing::error!(
                %endpoint_id,
                error = %e,
                "Unable to unregister endpoint instance from discovery"
            );
            anyhow::bail!(
                "Unable to unregister endpoint instance from discovery. Check discovery service status"
            );
        }

        tracing::info!(
            instance_id = instance_id,
            "Successfully unregistered endpoint instance from discovery - worker removed from routing pool"
        );

        Ok(())
    }

    /// Re-register this endpoint instance to discovery.
    ///
    /// This adds the endpoint back to the instances bucket, allowing the router
    /// to send requests to this worker again. Use this when a worker wakes up
    /// and should start receiving requests.
    pub async fn register_endpoint_instance(&self) -> anyhow::Result<()> {
        let drt = self.drt();
        let instance_id = drt.connection_id();
        let endpoint_id = self.id();

        // Get the transport type for the endpoint
        let transport = build_transport_type(self, &endpoint_id, instance_id).await?;

        let spec = crate::discovery::DiscoverySpec::Endpoint {
            namespace: endpoint_id.namespace,
            component: endpoint_id.component,
            endpoint: endpoint_id.name,
            transport,
        };

        let discovery = drt.discovery();
        if let Err(e) = discovery.register(spec).await {
            let endpoint_id = self.id();
            tracing::error!(
                %endpoint_id,
                error = %e,
                "Unable to re-register endpoint instance to discovery"
            );
            anyhow::bail!(
                "Unable to re-register endpoint instance to discovery. Check discovery service status"
            );
        }

        tracing::info!(
            instance_id = instance_id,
            "Successfully re-registered endpoint instance to discovery - worker added back to routing pool"
        );

        Ok(())
    }
}
