// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use derive_builder::Builder;
use derive_getters::Dissolve;
use educe::Educe;
use tokio_util::sync::CancellationToken;

use crate::{
    component::{DeviceType, Endpoint, Instance, TransportType},
    distributed::RequestPlaneMode,
    pipeline::network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint},
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::nats,
};

fn endpoint_device_type() -> Option<DeviceType> {
    // Common CUDA masks that explicitly disable GPU visibility.
    if std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .map(|v| {
            let l = v.trim().to_ascii_lowercase();
            l.is_empty() || l == "-1" || l == "none" || l == "void"
        })
        .unwrap_or(false)
    {
        return Some(DeviceType::Cpu);
    }

    // Container runtimes often use NVIDIA_VISIBLE_DEVICES to gate GPU visibility.
    if std::env::var("NVIDIA_VISIBLE_DEVICES")
        .ok()
        .map(|v| {
            let l = v.trim().to_ascii_lowercase();
            l == "none" || l == "void"
        })
        .unwrap_or(false)
    {
        return Some(DeviceType::Cpu);
    }

    // Default: no explicit CPU override means this endpoint is CUDA-capable.
    Some(DeviceType::Cuda)
}

/// A registered endpoint whose exact callable instance is ready for use.
///
/// Dropping this handle does not stop the endpoint. Call [`shutdown`](Self::shutdown)
/// for scoped endpoint lifetimes, or [`wait`](Self::wait) for the traditional
/// runtime-owned lifetime.
pub struct StartedEndpoint {
    instance: Instance,
    shutdown_token: CancellationToken,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl StartedEndpoint {
    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub async fn shutdown(self) -> Result<()> {
        self.shutdown_token.cancel();
        self.task.await??;
        Ok(())
    }

    pub async fn wait(self) -> Result<()> {
        self.task.await??;
        Ok(())
    }
}

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
        self.start_with_registration().await?.wait().await
    }

    /// Start an endpoint and return once its exact discovery instance is callable.
    pub async fn start_with_registration(self) -> Result<StartedEndpoint> {
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

        // Create clones for the async closure
        let namespace_name_for_task = endpoint_id.namespace.clone();
        let component_name_for_task = endpoint_id.component.clone();
        let endpoint_name_for_task = endpoint_id.name.clone();

        // Get the unified request plane server
        let server = endpoint.drt().request_plane_server().await?;
        let transport = build_transport_type(&endpoint, &endpoint_id, connection_id).await?;

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &health_check_payload {
            if system_health.lock().health_check_enabled()
                && endpoint
                    .drt()
                    .local_endpoint_registry()
                    .get(&endpoint.name)
                    .is_none()
            {
                anyhow::bail!(
                    "Endpoint '{}' has a health_check_payload and canary is enabled, \
                     but no local engine is registered. Call .register_local_engine() \
                     before .start() so the canary health check can function.",
                    endpoint.name
                );
            }

            let instance = Instance {
                component: endpoint_id.component.clone(),
                endpoint: endpoint_id.name.clone(),
                namespace: endpoint_id.namespace.clone(),
                instance_id: connection_id,
                transport: transport.clone(),
                device_type: endpoint_device_type(),
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

        let tracker_clone = if graceful_shutdown {
            tracing::debug!(
                "Registering endpoint '{}' with graceful shutdown tracker",
                endpoint.name
            );
            let tracker = endpoint.drt().graceful_shutdown_tracker();
            tracker.register_endpoint();
            Some(tracker)
        } else {
            tracing::debug!("Endpoint '{}' has graceful_shutdown=false", endpoint.name);
            None
        };

        // Register this endpoint instance in the discovery plane
        // The discovery interface abstracts storage backend (etcd, k8s, etc) and provides
        // consistent registration/discovery across the system.
        let discovery = endpoint.drt().discovery();

        let discovery_spec = crate::discovery::DiscoverySpec::Endpoint {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
            transport,
            device_type: endpoint_device_type(),
        };

        let discovery_instance = match discovery.register(discovery_spec).await {
            Ok(instance) => instance,
            Err(e) => {
                tracing::error!(
                    %endpoint_id,
                    error = %e,
                    "Unable to register service for discovery"
                );
                let _ = server.unregister_endpoint(&endpoint_name_for_task).await;
                if let Some(tracker) = tracker_clone {
                    tracker.unregister_endpoint();
                }
                anyhow::bail!(
                    "Unable to register service for discovery. Check discovery service status"
                );
            }
        };
        let instance = match &discovery_instance {
            crate::discovery::DiscoveryInstance::Endpoint(instance) => instance.clone(),
            _ => unreachable!("endpoint discovery spec returned a non-endpoint instance"),
        };

        // Create cleanup task that unregisters on cancellation.
        let endpoint_name_for_cleanup = endpoint_name_for_task;
        let server_for_cleanup = server;
        let cancel_token_for_cleanup = endpoint_shutdown_token.clone();
        let discovery_for_cleanup = discovery;

        let task: tokio::task::JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
            cancel_token_for_cleanup.cancelled().await;

            if let Err(error) = discovery_for_cleanup.unregister(discovery_instance).await {
                tracing::warn!(%error, "Failed to unregister endpoint from discovery");
            }

            tracing::debug!(
                endpoint = %endpoint_name_for_cleanup,
                "Unregistering endpoint from request plane server"
            );

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

            if let Some(tracker) = tracker_clone {
                tracing::debug!("Unregister endpoint from graceful shutdown tracker");
                tracker.unregister_endpoint();
            }

            anyhow::Ok(())
        });

        Ok(StartedEndpoint {
            instance,
            shutdown_token: endpoint_shutdown_token,
            task,
        })
    }
}

/// Build transport type based on request plane mode
///
/// This function handles both health check and discovery transport building.
/// All transport modes use consistent addressing:
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
        RequestPlaneMode::Tcp => {
            let tcp_host = crate::utils::tcp_rpc_host_from_env();
            // If a fixed port is explicitly configured, use it directly (no init ordering dependency).
            // Otherwise, use the actual bound port (set by TCP server after binding when port 0 is used).
            let tcp_port = std::env::var("DYN_TCP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .filter(|&p| p != 0)
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

    // For TCP with OS-assigned ports, we must ensure the server is initialized
    // (bound to a port) before we can construct a correct transport address.
    let has_fixed_port = match mode {
        RequestPlaneMode::Tcp => std::env::var("DYN_TCP_RPC_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .filter(|&p| p != 0)
            .is_some(),
        RequestPlaneMode::Nats => true, // NATS doesn't need port init
    };

    if !has_fixed_port {
        // Ensure request plane server is initialized before building transport.
        let _ = endpoint.drt().request_plane_server().await?;
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
            device_type: endpoint_device_type(),
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
            device_type: endpoint_device_type(),
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
