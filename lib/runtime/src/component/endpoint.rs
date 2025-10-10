// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Dissolve;
use tokio_util::sync::CancellationToken;

use super::*;

pub use async_nats::service::endpoint::Stats as EndpointStats;

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

    pub async fn start(self) -> Result<()> {
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

        tracing::debug!(
            "Starting endpoint: {}",
            endpoint.etcd_path_with_lease_id(lease_id)
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
                .insert(endpoint.subject_to(lease_id), stats_handler);
        }

        // Determine request plane mode
        use crate::config::RequestPlaneMode;
        let request_plane_mode = RequestPlaneMode::from_env();

        // creates an endpoint for the service (NATS only)
        let service_endpoint = if request_plane_mode.is_nats() {
            Some(
                group
                    .endpoint(&endpoint.name_with_id(lease_id))
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?,
            )
        } else {
            None
        };

        // Create a token that responds to both runtime shutdown and lease expiration
        let runtime_shutdown_token = endpoint.drt().child_token();

        // Extract all values needed from endpoint before any spawns
        let namespace_name = endpoint.component.namespace.name.clone();
        let component_name = endpoint.component.name.clone();
        let endpoint_name = endpoint.name.clone();
        let system_health = endpoint.drt().system_health.clone();
        let subject = endpoint.subject_to(lease_id);
        let etcd_path = endpoint.etcd_path_with_lease_id(lease_id);
        let etcd_client = endpoint.component.drt.etcd_client.clone();

        // Register health check target in SystemHealth if provided
        if let Some(health_check_payload) = &health_check_payload {
            // Build transport based on request plane mode
            let transport = if request_plane_mode.is_http() {
                // For HTTP mode, construct the HTTP endpoint URL
                let http_host =
                    std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
                let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                    .ok()
                    .and_then(|p| p.parse::<u16>().ok())
                    .unwrap_or(8081);
                let rpc_root = std::env::var("DYN_HTTP_RPC_ROOT_PATH")
                    .unwrap_or_else(|_| "/v1/dynamo".to_string());

                let http_endpoint =
                    format!("http://{}:{}{}/{}", http_host, http_port, rpc_root, subject);
                let tcp_endpoint = format!("{}:{}", http_host, http_port);

                TransportType::HttpTcp {
                    http_endpoint,
                    tcp_endpoint,
                }
            } else {
                TransportType::NatsTcp(subject.clone())
            };

            let instance = Instance {
                component: component_name.clone(),
                endpoint: endpoint_name.clone(),
                namespace: namespace_name.clone(),
                instance_id: lease_id,
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

        let cancel_token = if let Some(lease) = lease.as_ref() {
            // Create a new token that will be cancelled when EITHER the lease expires OR runtime shutdown occurs
            let combined_token = CancellationToken::new();
            let combined_for_select = combined_token.clone();
            let lease_token = lease.child_token();
            // Use secondary runtime for this lightweight monitoring task
            endpoint.drt().runtime().secondary().spawn(async move {
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

        let task = if request_plane_mode.is_http() {
            // HTTP mode - use HttpEndpoint
            use crate::pipeline::network::ingress::http_endpoint::HttpEndpoint;

            let http_host =
                std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
            let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(8081);
            let bind_addr: std::net::SocketAddr = format!("{}:{}", http_host, http_port)
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid HTTP bind address: {}", e))?;

            let http_endpoint = HttpEndpoint::builder()
                .service_handler(handler)
                .cancellation_token(cancel_token.clone())
                .graceful_shutdown(graceful_shutdown)
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HTTP endpoint: {e}"))?;

            tokio::spawn(async move {
                let result = http_endpoint
                    .start(
                        bind_addr,
                        namespace_name_for_task,
                        component_name_for_task,
                        endpoint_name_for_task,
                        lease_id,
                        system_health,
                    )
                    .await;

                // Unregister from graceful shutdown tracker
                if let Some(tracker) = tracker_clone {
                    tracing::debug!("Unregister endpoint from graceful shutdown tracker");
                    tracker.unregister_endpoint();
                }

                result
            })
        } else {
            // NATS mode - use PushEndpoint
            let service_endpoint = service_endpoint
                .ok_or_else(|| anyhow::anyhow!("NATS service endpoint not created"))?;

            let push_endpoint = PushEndpoint::builder()
                .service_handler(handler)
                .cancellation_token(cancel_token.clone())
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
                        lease_id,
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
        };

        // make the components service endpoint discovery in etcd

        // Build transport for etcd registration based on request plane mode
        let transport = if request_plane_mode.is_http() {
            let http_host =
                std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
            let http_port = std::env::var("DYN_HTTP_RPC_PORT")
                .ok()
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(8081);
            let rpc_root = std::env::var("DYN_HTTP_RPC_ROOT_PATH")
                .unwrap_or_else(|_| "/v1/dynamo".to_string());

            let http_endpoint =
                format!("http://{}:{}{}/{}", http_host, http_port, rpc_root, subject);
            let tcp_endpoint = format!("{}:{}", http_host, http_port);

            TransportType::HttpTcp {
                http_endpoint,
                tcp_endpoint,
            }
        } else {
            TransportType::NatsTcp(subject.clone())
        };

        let info = Instance {
            component: component_name.clone(),
            endpoint: endpoint_name.clone(),
            namespace: namespace_name.clone(),
            instance_id: lease_id,
            transport,
        };

        let info = serde_json::to_vec_pretty(&info)?;

        if let Some(etcd_client) = &etcd_client
            && let Err(e) = etcd_client
                .kv_create(&etcd_path, info, Some(lease_id))
                .await
        {
            tracing::error!(
                component_name,
                endpoint_name,
                error = %e,
                "Unable to register service for discovery"
            );
            cancel_token.cancel();
            return Err(error!(
                "Unable to register service for discovery. Check discovery service status"
            ));
        }
        task.await??;

        Ok(())
    }
}
