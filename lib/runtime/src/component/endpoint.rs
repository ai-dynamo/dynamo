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
    pipeline::network::{
        PushWorkHandler, RequestPlanePayloadCodec, ingress::push_endpoint::PushEndpoint,
    },
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

    /// Engine published for direct calls while this endpoint is running.
    #[educe(Debug(ignore))]
    #[builder(default, setter(custom))]
    local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
}

struct EndpointScopedState {
    endpoint_name: String,
    registry: crate::local_endpoint_registry::LocalEndpointRegistry,
    system_health: Arc<parking_lot::Mutex<crate::system_health::SystemHealth>>,
    local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
    health_check_registration: Option<crate::system_health::HealthCheckRegistration>,
}

impl EndpointScopedState {
    fn acquire(
        endpoint_name: String,
        registry: crate::local_endpoint_registry::LocalEndpointRegistry,
        system_health: Arc<parking_lot::Mutex<crate::system_health::SystemHealth>>,
        local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
        health_check_target: Option<(Instance, serde_json::Value)>,
    ) -> (Self, Option<Arc<tokio::sync::Notify>>) {
        if let Some(engine) = &local_engine {
            // Publish the engine before exposing its health-check target.
            registry.register(endpoint_name.clone(), engine.clone());
            tracing::debug!("Registered engine for endpoint '{endpoint_name}' in local registry");
        }

        let mut notifier = None;
        let health_check_registration = health_check_target.map(|(instance, payload)| {
            tracing::debug!(endpoint_name = %endpoint_name, "Registering endpoint health check target");
            let guard = system_health.lock();
            let registration = guard.register_health_check_target(&endpoint_name, instance, payload);
            notifier = guard.get_endpoint_health_check_notifier(&endpoint_name);
            registration
        });

        (
            Self {
                endpoint_name,
                registry,
                system_health,
                local_engine,
                health_check_registration,
            },
            notifier,
        )
    }

    fn release(self) {
        if let Some(registration) = self.health_check_registration {
            self.system_health
                .lock()
                .release_health_check_target(registration);
        }
        if let Some(engine) = &self.local_engine {
            self.registry
                .remove_registration(&self.endpoint_name, engine);
        }
    }
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    /// Register an async engine for direct calls while the endpoint is running.
    pub fn register_local_engine(
        mut self,
        engine: crate::local_endpoint_registry::LocalAsyncEngine,
    ) -> Result<Self> {
        self.local_engine = Some(Some(engine));
        Ok(self)
    }

    pub async fn start(self) -> Result<()> {
        self.start_with_registration().await?.wait().await
    }

    /// Start an endpoint and return once its exact discovery instance is callable.
    pub async fn start_with_registration(self) -> Result<StartedEndpoint> {
        let (
            endpoint,
            handler,
            metrics_labels,
            graceful_shutdown,
            health_check_payload,
            local_engine,
        ) = self.build_internal()?.dissolve();
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

        let health_check_target = match &health_check_payload {
            Some(health_check_payload) => {
                if system_health.lock().health_check_enabled() && local_engine.is_none() {
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
                    request_plane_codec: Some(RequestPlanePayloadCodec::configured()),
                };
                Some((instance, health_check_payload.clone()))
            }
            None => None,
        };

        let (scoped_state, notifier) = EndpointScopedState::acquire(
            endpoint.name.clone(),
            endpoint.drt().local_endpoint_registry().clone(),
            system_health.clone(),
            local_engine,
            health_check_target,
        );

        if let Some(notifier) = notifier
            && let Err(error) = handler.set_endpoint_health_check_notifier(notifier)
        {
            scoped_state.release();
            return Err(error);
        }

        tracing::debug!(
            endpoint = %endpoint_name_for_task,
            transport = server.transport_name(),
            "Registering endpoint with request plane server"
        );

        // Register endpoint with the server (unified interface)
        if let Err(error) = server
            .register_endpoint(
                endpoint_name_for_task.clone(),
                handler,
                connection_id,
                namespace_name_for_task.clone(),
                component_name_for_task.clone(),
                system_health.clone(),
            )
            .await
        {
            scoped_state.release();
            return Err(error);
        }

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
            request_plane_codec: Some(RequestPlanePayloadCodec::configured()),
        };

        let discovery_instance = match discovery.register(discovery_spec).await {
            Ok(instance) => instance,
            Err(e) => {
                tracing::error!(
                    %endpoint_id,
                    error = %e,
                    "Unable to register service for discovery"
                );
                let _ = server
                    .unregister_endpoint(&endpoint_name_for_task, connection_id)
                    .await;
                if let Some(tracker) = tracker_clone {
                    tracker.unregister_endpoint();
                }
                scoped_state.release();
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
                .unregister_endpoint(&endpoint_name_for_cleanup, connection_id)
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

            scoped_state.release();

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
            request_plane_codec: Some(RequestPlanePayloadCodec::configured()),
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
            request_plane_codec: Some(RequestPlanePayloadCodec::configured()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HealthStatus;
    use crate::local_endpoint_registry::{
        LocalAsyncEngine, LocalEndpointRegistry, test_support::stub_engine,
    };
    use crate::system_health::SystemHealth;

    const ENDPOINT: &str = "generate";

    fn system_health() -> Arc<parking_lot::Mutex<SystemHealth>> {
        Arc::new(parking_lot::Mutex::new(SystemHealth::new(
            HealthStatus::NotReady,
            Vec::new(),
            true,
            "/health".to_string(),
            "/live".to_string(),
        )))
    }

    fn instance(instance_id: u64) -> Instance {
        Instance {
            component: "backend".to_string(),
            endpoint: ENDPOINT.to_string(),
            namespace: "dynamo".to_string(),
            instance_id,
            transport: TransportType::Tcp("127.0.0.1:0".to_string()),
            device_type: None,
            request_plane_codec: None,
        }
    }

    fn acquire(
        registry: &LocalEndpointRegistry,
        health: &Arc<parking_lot::Mutex<SystemHealth>>,
        engine: &LocalAsyncEngine,
        instance_id: u64,
        payload: serde_json::Value,
    ) -> (EndpointScopedState, Option<Arc<tokio::sync::Notify>>) {
        EndpointScopedState::acquire(
            ENDPOINT.to_string(),
            registry.clone(),
            health.clone(),
            Some(engine.clone()),
            Some((instance(instance_id), payload)),
        )
    }

    #[test]
    fn acquire_publishes_the_engine_target_notifier_and_notready_status() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        let engine = stub_engine();

        let (_scope, notifier) = acquire(
            &registry,
            &health,
            &engine,
            7,
            serde_json::json!({"probe": "payload"}),
        );

        let registered = registry.get(ENDPOINT).expect(
            "the canary dispatches through the local registry, so the engine must be there",
        );
        assert!(Arc::ptr_eq(&registered, &engine));

        let guard = health.lock();
        let target = guard
            .get_health_check_target(ENDPOINT)
            .expect("the canary needs a target to probe");
        assert_eq!(target.instance.instance_id, 7);
        assert_eq!(target.payload, serde_json::json!({"probe": "payload"}));
        assert_eq!(
            guard.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::NotReady),
            "an endpoint the canary has not verified yet must not count as ready"
        );
        let published = guard
            .get_endpoint_health_check_notifier(ENDPOINT)
            .expect("the handler signals the canary through this notifier");
        let handed_back = notifier.expect("acquire hands the notifier to the handler");
        assert!(
            Arc::ptr_eq(&published, &handed_back),
            "the handler must signal the same notifier the canary waits on"
        );
        assert!(
            !guard.get_health_status().0,
            "an unverified endpoint holds the worker unhealthy"
        );
    }

    #[test]
    fn release_leaves_no_endpoint_scoped_state_behind() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        health.lock().set_health_status(HealthStatus::Ready);
        let engine = stub_engine();

        let (scope, _notifier) = acquire(&registry, &health, &engine, 7, serde_json::json!({}));
        assert!(!health.lock().get_health_status().0);

        scope.release();

        assert!(
            registry.get(ENDPOINT).is_none(),
            "a stopped endpoint must not stay locally dispatchable"
        );
        let guard = health.lock();
        assert!(guard.get_health_check_target(ENDPOINT).is_none());
        assert!(guard.get_endpoint_health_check_notifier(ENDPOINT).is_none());
        assert!(guard.get_endpoint_health_status(ENDPOINT).is_none());
        assert!(
            guard.get_health_status().0,
            "an abandoned target would hold the worker unhealthy with no endpoint to blame"
        );
    }

    #[test]
    fn release_withdraws_the_engine_even_with_no_health_check_target() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        let engine = stub_engine();

        let (scope, notifier) = EndpointScopedState::acquire(
            ENDPOINT.to_string(),
            registry.clone(),
            health.clone(),
            Some(engine.clone()),
            None,
        );
        assert!(
            notifier.is_none(),
            "no target means nothing to notify about"
        );
        assert!(registry.get(ENDPOINT).is_some());
        assert!(health.lock().get_health_check_target(ENDPOINT).is_none());

        scope.release();

        assert!(registry.get(ENDPOINT).is_none());
    }

    #[test]
    fn a_restart_under_the_same_name_installs_its_own_engine_and_target() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        let first_engine = stub_engine();
        let second_engine = stub_engine();

        let (first, _) = acquire(
            &registry,
            &health,
            &first_engine,
            1,
            serde_json::json!({"generation": "first"}),
        );
        first.release();
        let (_second, _) = acquire(
            &registry,
            &health,
            &second_engine,
            2,
            serde_json::json!({"generation": "second"}),
        );

        let registered = registry.get(ENDPOINT).expect("the restart is dispatchable");
        assert!(
            Arc::ptr_eq(&registered, &second_engine),
            "requests must reach the engine that is serving now"
        );
        let guard = health.lock();
        let target = guard
            .get_health_check_target(ENDPOINT)
            .expect("the restart registered a target");
        assert_eq!(
            target.instance.instance_id, 2,
            "the canary must report on the instance that exists"
        );
        assert_eq!(target.payload, serde_json::json!({"generation": "second"}));
        assert!(
            guard.get_endpoint_health_check_notifier(ENDPOINT).is_some(),
            "the restart's handler needs a notifier to signal"
        );
    }

    #[test]
    fn releasing_an_overlapped_scope_leaves_the_newer_one_serving() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        let outgoing_engine = stub_engine();
        let live_engine = stub_engine();

        let (outgoing, _) = acquire(
            &registry,
            &health,
            &outgoing_engine,
            1,
            serde_json::json!({"generation": "outgoing"}),
        );
        let (_live, _) = acquire(
            &registry,
            &health,
            &live_engine,
            2,
            serde_json::json!({"generation": "live"}),
        );

        outgoing.release();

        let registered = registry
            .get(ENDPOINT)
            .expect("the live endpoint must stay dispatchable");
        assert!(
            Arc::ptr_eq(&registered, &live_engine),
            "the outgoing scope must not evict the engine that replaced its own"
        );
        let guard = health.lock();
        let target = guard
            .get_health_check_target(ENDPOINT)
            .expect("the live endpoint must keep its canary target");
        assert_eq!(target.instance.instance_id, 2);
        assert_eq!(target.payload, serde_json::json!({"generation": "live"}));
        assert!(
            guard.get_endpoint_health_check_notifier(ENDPOINT).is_some(),
            "the live endpoint's handler still signals through this notifier"
        );
        assert_eq!(
            guard.get_endpoint_health_status(ENDPOINT),
            Some(HealthStatus::NotReady),
            "the live endpoint is still tracked, awaiting its own canary verdict"
        );
    }

    #[test]
    fn releasing_the_newer_scope_hands_the_name_back_to_the_older_one() {
        let registry = LocalEndpointRegistry::new();
        let health = system_health();
        let displaced_engine = stub_engine();
        let newer_engine = stub_engine();

        let (_displaced, _) = acquire(
            &registry,
            &health,
            &displaced_engine,
            1,
            serde_json::json!({"generation": "displaced"}),
        );
        let (newer, _) = acquire(
            &registry,
            &health,
            &newer_engine,
            2,
            serde_json::json!({"generation": "newer"}),
        );

        newer.release();

        let registered = registry
            .get(ENDPOINT)
            .expect("the displaced endpoint is still running and must be reachable again");
        assert!(
            Arc::ptr_eq(&registered, &displaced_engine),
            "requests must reach the endpoint that is serving now"
        );
        let guard = health.lock();
        let target = guard
            .get_health_check_target(ENDPOINT)
            .expect("the displaced endpoint's target is re-exposed");
        assert_eq!(
            target.instance.instance_id, 1,
            "the canary must probe the instance the registry now dispatches to"
        );
        assert_eq!(
            target.payload,
            serde_json::json!({"generation": "displaced"})
        );
    }
}

#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::local_endpoint_registry::{LocalAsyncEngine, test_support::stub_engine};
    use crate::pipeline::PipelineError;
    use crate::pipeline::network::PushWorkHandler;
    use crate::system_health::SystemHealth;
    use async_trait::async_trait;
    use bytes::Bytes;

    const ENDPOINT: &str = "generate";

    struct TestHandler {
        refuse_notifier: bool,
    }

    #[async_trait]
    impl PushWorkHandler for TestHandler {
        async fn handle_payload(
            &self,
            _payload: Bytes,
            _request_id: Option<String>,
        ) -> Result<(), PipelineError> {
            Ok(())
        }

        fn add_metrics(
            &self,
            _endpoint: &Endpoint,
            _metrics_labels: Option<&[(&str, &str)]>,
        ) -> Result<()> {
            Ok(())
        }

        fn set_endpoint_health_check_notifier(
            &self,
            _notifier: Arc<tokio::sync::Notify>,
        ) -> Result<()> {
            if self.refuse_notifier {
                anyhow::bail!("handler rejected the health check notifier");
            }
            Ok(())
        }
    }

    fn handler(refuse_notifier: bool) -> Arc<dyn PushWorkHandler> {
        Arc::new(TestHandler { refuse_notifier })
    }

    fn assert_no_endpoint_state(
        registry: &crate::local_endpoint_registry::LocalEndpointRegistry,
        system_health: &Arc<parking_lot::Mutex<SystemHealth>>,
        context: &str,
    ) {
        assert!(
            registry.get(ENDPOINT).is_none(),
            "{context}: the engine must not stay locally dispatchable"
        );
        let guard = system_health.lock();
        assert!(
            guard.get_health_check_target(ENDPOINT).is_none(),
            "{context}: an abandoned canary target holds the whole worker unhealthy"
        );
        assert!(
            guard.get_endpoint_health_check_notifier(ENDPOINT).is_none(),
            "{context}: nothing is left to signal this notifier"
        );
        assert!(
            guard.get_endpoint_health_status(ENDPOINT).is_none(),
            "{context}: a stale health entry keeps counting towards worker health"
        );
    }

    async fn start(
        drt: &crate::DistributedRuntime,
        namespace: &str,
        engine: &LocalAsyncEngine,
        payload: serde_json::Value,
        refuse_notifier: bool,
    ) -> Result<StartedEndpoint> {
        drt.namespace(namespace)?
            .component("backend")?
            .endpoint(ENDPOINT)
            .endpoint_builder()
            .handler(handler(refuse_notifier))
            .health_check_payload(payload)
            .register_local_engine(engine.clone())?
            .start_with_registration()
            .await
    }

    #[tokio::test]
    async fn a_failed_start_leaves_no_endpoint_scoped_state_behind() {
        let drt = create_test_drt_async().await;
        let engine = stub_engine();

        let outcome = start(
            &drt,
            "rollback_ns",
            &engine,
            serde_json::json!({"probe": "payload"}),
            true,
        )
        .await;
        assert!(
            outcome.is_err(),
            "the handler refused the notifier, so the start cannot succeed"
        );

        assert_no_endpoint_state(
            drt.local_endpoint_registry(),
            &drt.system_health(),
            "after a failed start",
        );
    }

    #[tokio::test]
    async fn a_restart_after_shutdown_is_the_one_the_canary_reports_on() {
        let drt = create_test_drt_async().await;
        let first_engine = stub_engine();
        let second_engine = stub_engine();

        let started = start(
            &drt,
            "restart_ns",
            &first_engine,
            serde_json::json!({"generation": "first"}),
            false,
        )
        .await
        .expect("the first start succeeds");
        assert!(Arc::ptr_eq(
            &drt.local_endpoint_registry()
                .get(ENDPOINT)
                .expect("the first endpoint is dispatchable"),
            &first_engine
        ));

        started.shutdown().await.expect("shutdown runs cleanly");
        assert_no_endpoint_state(
            drt.local_endpoint_registry(),
            &drt.system_health(),
            "after shutdown",
        );

        let _restarted = start(
            &drt,
            "restart_ns",
            &second_engine,
            serde_json::json!({"generation": "second"}),
            false,
        )
        .await
        .expect("the endpoint can be started again under the same name");

        let registered = drt
            .local_endpoint_registry()
            .get(ENDPOINT)
            .expect("the restart is dispatchable");
        assert!(
            Arc::ptr_eq(&registered, &second_engine),
            "requests must reach the engine that is serving now"
        );
        let guard = drt.system_health();
        let guard = guard.lock();
        let target = guard
            .get_health_check_target(ENDPOINT)
            .expect("the restart registered a canary target");
        assert_eq!(
            target.payload,
            serde_json::json!({"generation": "second"}),
            "the canary must probe the incarnation that is serving, not the one that stopped"
        );
        assert!(
            guard.get_endpoint_health_status(ENDPOINT).is_some(),
            "the restart counts towards worker health again"
        );
        assert!(guard.get_endpoint_health_check_notifier(ENDPOINT).is_some());
    }
}
