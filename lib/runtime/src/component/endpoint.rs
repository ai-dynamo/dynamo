// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Dissolve;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::transports::etcd;

/// Extract the full namespace hierarchy from a Namespace
fn get_namespace_hierarchy(namespace: &super::Namespace) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current: Option<&super::Namespace> = Some(namespace);

    // Walk up the parent chain to collect all namespace segments
    while let Some(ns) = current {
        segments.push(ns.name.clone());
        current = ns.parent.as_deref();
    }

    // Reverse to get root-to-leaf order
    segments.reverse();
    segments
}

pub use async_nats::service::endpoint::Stats as EndpointStats;

/// An endpoint instance that has been created but not yet started
pub struct EndpointInstance {
    // Core fields
    endpoint: Endpoint,
    handler: Arc<dyn PushWorkHandler>,

    // Optional fields (may be None in static mode)
    lease: Option<Lease>,
    etcd_client: Option<etcd::Client>,

    // Always available
    stats_handler: Option<EndpointStatsHandler>,
    metrics_labels: Option<Vec<(String, String)>>,
    graceful_shutdown: bool,
    local_engine_info: Option<(String, Arc<dyn crate::engine::AnyAsyncEngine>)>, // Prepared but not registered

    // Pre-computed values for start()
    lease_id: i64,
    service_name: String,
    health_check_payload: Option<serde_json::Value>,
}

impl EndpointInstance {
    /// Start the endpoint on the network
    pub async fn start(mut self) -> Result<()> {
        tracing::debug!(
            "Starting endpoint: {}",
            self.endpoint.etcd_path_with_lease_id(self.lease_id)
        );

        // Register local engine if configured
        let local_engine_key = if let Some((key, engine)) = self.local_engine_info.take() {
            tracing::debug!("Registering local engine for endpoint: {}", key);
            self.endpoint
                .drt()
                .register_local_engine(key.clone(), engine)
                .await?;
            Some(key)
        } else {
            None
        };

        // acquire the registry lock to get the service group
        let registry = self.endpoint.drt().component_registry.inner.lock().await;

        // get the group
        let group = registry
            .services
            .get(&self.service_name)
            .map(|service| service.group(self.endpoint.component.service_name()))
            .ok_or(error!("Service not found"))?;

        // get the stats handler map
        let handler_map = registry
            .stats_handlers
            .get(&self.service_name)
            .cloned()
            .expect("no stats handler registry; this is unexpected");

        drop(registry);

        // insert the stats handler
        if let Some(stats_handler) = self.stats_handler {
            handler_map
                .lock()
                .unwrap()
                .insert(self.endpoint.subject_to(self.lease_id), stats_handler);
        }

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&self.endpoint.name_with_id(self.lease_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        // Create a token that responds to both runtime shutdown and lease expiration
        let runtime_shutdown_token = self.endpoint.drt().child_token();

        // Extract all values needed from endpoint before any spawns
        let namespace_name = self.endpoint.component.namespace.name.clone();
        let component_name = self.endpoint.component.name.clone();
        let endpoint_name = self.endpoint.name.clone();
        let system_health = self.endpoint.drt().system_health.clone();
        let subject = self.endpoint.subject_to(self.lease_id);
        let etcd_path = self.endpoint.etcd_path_with_lease_id(self.lease_id);

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &self.health_check_payload {
            let instance = Instance {
                component: component_name.clone(),
                endpoint: endpoint_name.clone(),
                namespace: namespace_name.clone(),
                instance_id: self.lease_id,
                transport: TransportType::NatsTcp(subject.clone()),
            };
            tracing::debug!(subject = %subject, "Registering endpoint health check target");
            let guard = system_health.lock().unwrap();
            guard.register_health_check_target(&subject, instance, health_check_payload.clone());
            if let Some(notifier) = guard.get_endpoint_health_check_notifier(&subject) {
                self.handler.set_endpoint_health_check_notifier(notifier)?;
            }
        }

        let cancel_token = if let Some(lease) = self.lease.as_ref() {
            // Create a new token that will be cancelled when EITHER the lease expires OR runtime shutdown occurs
            let combined_token = CancellationToken::new();
            let combined_for_select = combined_token.clone();
            let lease_token = lease.child_token();
            // Use secondary runtime for this lightweight monitoring task
            self.endpoint.drt().runtime().secondary().spawn(async move {
                tokio::select! {
                    _ = lease_token.cancelled() => {
                        tracing::trace!("Lease cancelled, triggering endpoint shutdown");
                    }
                    _ = runtime_shutdown_token.cancelled() => {
                        tracing::trace!("Runtime shutdown triggered, cancelling endpoint");
                    }
                }
                combined_for_select.cancel();
            });
            combined_token
        } else {
            // No lease, just use runtime shutdown token
            runtime_shutdown_token
        };

        // Register with graceful shutdown tracker if needed
        if self.graceful_shutdown {
            tracing::debug!(
                "Registering endpoint '{}' with graceful shutdown tracker",
                self.endpoint.name
            );
            let tracker = self.endpoint.drt().graceful_shutdown_tracker();
            tracker.register_endpoint();
        } else {
            tracing::debug!(
                "Endpoint '{}' has graceful_shutdown=false",
                self.endpoint.name
            );
        }

        let push_endpoint = PushEndpoint::builder()
            .service_handler(self.handler)
            .cancellation_token(cancel_token.clone())
            .graceful_shutdown(self.graceful_shutdown)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        // launch in primary runtime
        let tracker_clone = if self.graceful_shutdown {
            Some(self.endpoint.drt().graceful_shutdown_tracker())
        } else {
            None
        };

        // Create clones for the async closure
        let namespace_name_for_task = namespace_name.clone();
        let component_name_for_task = component_name.clone();
        let endpoint_name_for_task = endpoint_name.clone();
        let drt_for_cleanup = self.endpoint.drt().clone();
        let local_engine_key_for_task = local_engine_key.clone();
        let lease_id_for_task = self.lease_id;

        let task = tokio::spawn(async move {
            let result = push_endpoint
                .start(
                    service_endpoint,
                    namespace_name_for_task,
                    component_name_for_task,
                    endpoint_name_for_task,
                    lease_id_for_task,
                    system_health,
                )
                .await;

            // Unregister from graceful shutdown tracker
            if let Some(tracker) = tracker_clone {
                tracing::debug!("Unregistering endpoint from graceful shutdown tracker");
                tracker.unregister_endpoint();
            }

            // Unregister from local engine registry if it was registered
            if let Some(key) = local_engine_key_for_task {
                tracing::debug!("Unregistering local engine for endpoint: {}", key);
                drt_for_cleanup.unregister_local_engine(&key).await;
            }

            result
        });

        // make the components service endpoint discovery in etcd

        // client.register_service()
        let info = Instance {
            component: component_name,
            endpoint: endpoint_name,
            namespace: namespace_name,
            instance_id: self.lease_id,
            transport: TransportType::NatsTcp(subject),
        };

        let info = serde_json::to_vec_pretty(&info)?;

        // Register in etcd (only if not static)
        if let Some(etcd_client) = &self.etcd_client
            && let Err(e) = etcd_client
                .kv_create(&etcd_path, info, Some(self.lease_id))
                .await
        {
            tracing::error!("Failed to register discoverable service: {:?}", e);
            cancel_token.cancel();
            return Err(error!("Failed to register discoverable service"));
        }

        task.await??;
        Ok(())
    }

    /// Start the endpoint in the background as a critical task
    ///
    /// Returns a handle that can be used to monitor or cancel the endpoint.
    /// If the endpoint fails critically, it will trigger parent token cancellation
    /// (usually leading to worker shutdown).
    pub fn start_background_handle(self) -> Result<crate::utils::tasks::critical::CriticalTaskExecutionHandle> {
        let endpoint_name = format!(
            "{}/{}/{}:{}",
            self.endpoint.component.namespace.name,
            self.endpoint.component.name,
            self.endpoint.name,
            self.lease_id
        );

        let parent_token = self.endpoint.drt().primary_token();
        let endpoint_name_for_closure = endpoint_name.clone();

        crate::utils::tasks::critical::CriticalTaskExecutionHandle::new(
            |cancel_token| async move {
                // Monitor cancellation while starting
                tokio::select! {
                    result = self.start() => result,
                    _ = cancel_token.cancelled() => {
                        tracing::info!("Endpoint {} cancelled during startup", endpoint_name_for_closure);
                        Ok(())
                    }
                }
            },
            parent_token,
            &format!("endpoint-{}", endpoint_name),
        )
    }

    /// Start the endpoint in the background and register it with the distributed runtime
    ///
    /// This method will:
    /// 1. Create a critical task to run the endpoint
    /// 2. Register the task handle for tracking and cleanup
    /// 3. If the endpoint fails critically, it will trigger worker shutdown
    pub async fn start_background(self) -> Result<()> {
        // Generate key for tracking (same format as local engine registry)
        let key = if let Some((ref engine_key, _)) = self.local_engine_info {
            engine_key.clone()
        } else {
            // Generate key even if no local engine
            format!(
                "{}/{}/{}:{}",
                self.endpoint.component.namespace.name,
                self.endpoint.component.name,
                self.endpoint.name,
                self.lease_id
            )
        };

        let drt = self.endpoint.drt().clone();
        let handle = self.start_background_handle()?;

        // Register the handle for tracking
        drt.register_background_endpoint(key, handle).await;

        Ok(())
    }
}

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct EndpointConfig {
    #[builder(private)]
    endpoint: Endpoint,

    // todo: move lease to component/service
    /// Lease
    #[educe(Debug(ignore))]
    #[builder(default)]
    lease: Option<Lease>,

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

    /// Start the endpoint directly (backwards compatible)
    /// This is equivalent to calling create().await?.start().await
    pub async fn start(self) -> Result<()> {
        let instance = self.create().await?;
        instance.start().await
    }

    /// Create the endpoint instance (setup phase)
    /// This validates configuration, registers local engines, and prepares for starting
    pub async fn create(self) -> Result<EndpointInstance> {
        let (
            endpoint,
            lease,
            handler,
            stats_handler,
            metrics_labels,
            graceful_shutdown,
            health_check_payload,
        ) = self.build_internal()?.dissolve();
        let lease = lease.or(endpoint.drt().primary_lease());
        let lease_id = lease.as_ref().map(|l| l.id()).unwrap_or(0);

        let service_name = endpoint.component.service_name();
        let etcd_client = endpoint.component.drt.etcd_client.clone();

        tracing::debug!(
            "Creating endpoint: {}",
            endpoint.etcd_path_with_lease_id(lease_id)
        );

        // acquire the registry lock
        let registry = endpoint.drt().component_registry.inner.lock().await;

        let metrics_labels: Option<Vec<(&str, &str)>> = metrics_labels
            .as_ref()
            .map(|v| v.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect());

        // Add metrics to the handler. The endpoint provides additional information to the handler.
        handler.add_metrics(&endpoint, metrics_labels.as_deref())?;

        // Check if local registry is enabled and handler has a type-erased engine
        let enable_local_registry = registry
            .service_enable_local_registry
            .get(&endpoint.component.service_name())
            .copied()
            .unwrap_or(true);

        let local_engine_info = if enable_local_registry
            && let Some(any_engine) = handler.as_any_engine()
        {
            use crate::v2::entity::{ComponentDescriptor, EndpointDescriptor, NamespaceDescriptor};

            // Extract the full namespace hierarchy
            let namespace_segments = get_namespace_hierarchy(&endpoint.component.namespace);

            // Create the descriptor for this endpoint, handling internal namespaces
            let namespace_desc = if namespace_segments.first().is_some_and(|s| s.starts_with('_')) {
                NamespaceDescriptor::new_internal(
                    &namespace_segments
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )
            } else {
                NamespaceDescriptor::new(
                    &namespace_segments
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )
            }
            .map_err(|e| anyhow::anyhow!("Invalid namespace: {}", e))?;

            let component_desc = namespace_desc
                .component(&endpoint.component.name)
                .map_err(|e| anyhow::anyhow!("Invalid component: {}", e))?;
            let endpoint_desc = component_desc
                .endpoint(&endpoint.name)
                .map_err(|e| anyhow::anyhow!("Invalid endpoint: {}", e))?;

            // Prepare registration info but don't register yet
            let key = endpoint_desc.to_string();
            tracing::debug!("Prepared local engine registration for endpoint: {}", key);
            Some((key, any_engine))
        } else {
            None
        };

        // Drop registry lock before returning
        drop(registry);

        // Convert metrics_labels back to owned version for storage
        let metrics_labels = metrics_labels.map(|v| {
            v.into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        });

        Ok(EndpointInstance {
            endpoint,
            handler,
            lease,
            etcd_client,
            stats_handler,
            metrics_labels,
            graceful_shutdown,
            local_engine_info,
            lease_id,
            service_name,
            health_check_payload,
        })
    }
}
