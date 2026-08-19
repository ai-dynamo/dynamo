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

    /// Engine to publish in the local endpoint registry for direct in-process calls
    ///
    /// Held here rather than registered eagerly: the registry is process-wide state whose
    /// lifetime must match the endpoint's, so the start path owns both installing and
    /// releasing it.
    #[educe(Debug(ignore))]
    #[builder(default, setter(custom))]
    local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
}

/// Process-wide state a start installs on behalf of one endpoint: its engine in the local
/// registry and its health check target in [`SystemHealth`](crate::system_health::SystemHealth).
///
/// Neither is owned by the endpoint's task, so neither goes away when a start fails or an
/// endpoint stops. Both are keyed by endpoint name alone, so a leftover from one start is
/// indistinguishable from a live registration: the canary keeps dispatching into an engine
/// for an endpoint that has no request-plane or discovery presence, and the abandoned
/// target holds the whole worker unhealthy. This handle ties both to a single start.
struct EndpointScopedState {
    endpoint_name: String,
    registry: crate::local_endpoint_registry::LocalEndpointRegistry,
    system_health: Arc<parking_lot::Mutex<crate::system_health::SystemHealth>>,
    local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
    health_check_registration: Option<crate::system_health::HealthCheckRegistration>,
}

impl EndpointScopedState {
    /// Install this start's endpoint-scoped state.
    ///
    /// Returns the handle that releases it, together with the canary notifier the handler
    /// must be given when a health check target was registered.
    fn acquire(
        endpoint_name: String,
        registry: crate::local_endpoint_registry::LocalEndpointRegistry,
        system_health: Arc<parking_lot::Mutex<crate::system_health::SystemHealth>>,
        local_engine: Option<crate::local_endpoint_registry::LocalAsyncEngine>,
        health_check_target: Option<(Instance, serde_json::Value)>,
    ) -> (Self, Option<Arc<tokio::sync::Notify>>) {
        if let Some(engine) = &local_engine {
            // Before the health check target, so the canary never sees a target whose
            // engine has not landed yet.
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

    /// Release everything [`acquire`](Self::acquire) installed.
    ///
    /// Every exit from the start path runs this: a start that never reached a serving
    /// endpoint has to leave the process as it found it, and an endpoint that has stopped
    /// must stop being dispatchable and stop counting towards worker health.
    fn release(self) {
        if let Some(registration) = self.health_check_registration {
            // Conditional on registration identity: the target map is keyed by endpoint
            // name alone, so this removes only the registration this start made, and
            // whichever registration is still outstanding under that name takes the
            // subject. An unrelated endpoint sharing the name keeps its canary.
            self.system_health
                .lock()
                .release_health_check_target(registration);
        }
        if let Some(engine) = &self.local_engine {
            // By engine identity, for the same reason the target above is released by
            // registration: the registry is keyed by endpoint name alone, so this withdraws
            // only this start's engine and leaves whichever registration is still
            // outstanding under that name serving.
            self.registry
                .remove_registration(&self.endpoint_name, engine);
        }
    }
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    /// Register an async engine in the local endpoint registry for direct in-process calls
    ///
    /// The engine is published when the endpoint starts and withdrawn when it stops or
    /// fails to start.
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

        // Build the health check target in SystemHealth if provided
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

        // Everything from here on is rollback territory: each exit below releases.
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
                let _ = server.unregister_endpoint(&endpoint_name_for_task).await;
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

            // Last: the endpoint is off the request plane and out of discovery, so it must
            // also stop being locally dispatchable and stop counting towards worker health.
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
    //! What a start installs process-wide, and what stops being installed once it releases.
    //!
    //! [`EndpointScopedState`] is deliberately built out of a bare
    //! [`LocalEndpointRegistry`](crate::local_endpoint_registry::LocalEndpointRegistry) and a
    //! bare [`SystemHealth`](crate::system_health::SystemHealth) rather than reaching through a
    //! `DistributedRuntime`, so these assertions need no NATS, no discovery backend and no GPU.
    //! They read the same public getters the canary and the `/health` handler read.

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

    /// Acquire one endpoint's scoped state, as `start_with_registration` does once it has
    /// built the instance and knows the payload.
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

    /// The three things a start makes visible to the rest of the process: a dispatchable
    /// engine, a canary target for it, and a notifier its handler signals through — with the
    /// endpoint held `NotReady` until the canary says otherwise.
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

    /// The bug this work item exists for: a start that fails, or an endpoint that stops,
    /// used to leave all of the above behind. The engine stayed callable for an endpoint
    /// with no request-plane or discovery presence, and the abandoned target — which nothing
    /// would ever verify — held the whole worker unhealthy.
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

    /// An endpoint with no health check payload registers no canary target, but its engine
    /// is still endpoint-scoped state that has to go when the endpoint stops.
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

    /// Restarting an endpoint under the same name must serve from the new engine and let the
    /// canary probe the new instance with the new payload. Before this change the second
    /// registration was refused outright, so the restart silently inherited the dead
    /// incarnation's instance and payload.
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

    /// Both identity guards, together. Two endpoints can hold one name at once — a restart
    /// that overlaps the outgoing incarnation's cleanup, or two components whose endpoints
    /// share a name — and the registry and the target map are both keyed by that name alone.
    /// Releasing the older scope must therefore evict neither the engine nor the canary
    /// target the newer one installed; the whole receipt design exists for this case.
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

    /// The same overlap, released in the other order. When the scope that took the name goes
    /// first, the one it displaced is still serving, so both structures have to hand the name
    /// back to it — engine and canary target together. A registry that overwrote on
    /// registration could not: it had nothing left to re-expose, so the canary kept a target
    /// for an endpoint it could no longer dispatch to.
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

// ===============================
// Integration Tests (require DRT)
// ===============================
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    //! The same guarantees, driven through a real `start_with_registration`.
    //!
    //! These need a live NATS server, so they are gated behind the `integration` feature
    //! like every other DRT-backed test in this crate.

    use super::*;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use crate::local_endpoint_registry::{LocalAsyncEngine, test_support::stub_engine};
    use crate::pipeline::PipelineError;
    use crate::pipeline::network::PushWorkHandler;
    use crate::system_health::SystemHealth;
    use async_trait::async_trait;
    use bytes::Bytes;

    const ENDPOINT: &str = "generate";

    /// A handler that carries no pipeline. These tests assert on the process-wide state a
    /// start installs, never on request handling, so the handler only has to exist — and a
    /// bare one keeps a second start under the same endpoint name from tripping over
    /// metrics that were registered by the first.
    struct TestHandler {
        /// Fails the notifier handshake, which is the first fallible step after the start
        /// has installed its endpoint-scoped state.
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

    /// Assert that the endpoint name holds no engine, target, notifier or health entry.
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

    /// A start that fails partway leaves the process as it found it. Before this change the
    /// engine and the canary target it had already installed outlived the failed start: the
    /// engine stayed callable for an endpoint that never reached discovery, and the target —
    /// which nothing would ever verify — held the whole worker unhealthy.
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
        let Err(error) = outcome else {
            panic!("the handler refused the notifier, so the start cannot succeed");
        };
        assert!(error.to_string().contains("notifier"));

        assert_no_endpoint_state(
            drt.local_endpoint_registry(),
            &drt.system_health(),
            "after a failed start",
        );
    }

    /// Stop an endpoint and start it again under the same name. The restart must serve from
    /// its own engine and be probed with its own payload — and the shutdown in between must
    /// leave nothing that could answer for the endpoint while it is down.
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
        // The canary is off in this configuration, so registering with the request plane
        // marks the endpoint ready; what matters here is that the restart is tracked again
        // under its own registration, after the shutdown removed the previous entry.
        assert!(
            guard.get_endpoint_health_status(ENDPOINT).is_some(),
            "the restart counts towards worker health again"
        );
        assert!(guard.get_endpoint_health_check_notifier(ENDPOINT).is_some());
    }
}
