// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use derive_getters::Dissolve;

use super::*;

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

pub use async_nats::service::endpoint::Stats as EndpointStats;

#[derive(Clone)]
struct EndpointInfo {
    namespace: String,
    component: String,
    endpoint: String,
    instance_id: i64,
    lease: Option<Lease>,
    drt: Arc<DistributedRuntime>,
}

/// Aggregated information for HTTP management service covering multiple endpoints
#[derive(Clone)]
struct HttpManagementInfo {
    drt: Arc<DistributedRuntime>,
    port: u16,
    component: Component,
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

    /// HTTP management port (default: auto-assign)
    #[builder(default = Some(0))]
    http_management_port: Option<u16>,
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        tracing::info!("Building endpoint builder for {}", endpoint.path());
        Self::default().endpoint(endpoint)
    }

    pub fn stats_handler<F>(self, handler: F) -> Self
    where
        F: FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static,
    {
        self._stats_handler(Some(Box::new(handler)))
    }

    pub async fn start(self) -> Result<()> {
        let (endpoint, lease, handler, stats_handler, http_management_port) =
            self.build_internal()?.dissolve();
        let lease = lease.or(endpoint.drt().primary_lease());
        let lease_id = lease.as_ref().map(|l| l.id()).unwrap_or(0);

        tracing::debug!(
            "Starting endpoint: {}",
            endpoint.etcd_path_with_lease_id(lease_id)
        );

        let service_name = endpoint.component.service_name();

        // acquire the registry lock
        let registry = endpoint.drt().component_registry.inner.lock().await;

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
                .unwrap()
                .insert(endpoint.subject_to(lease_id), stats_handler);
        }

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&endpoint.name_with_id(lease_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        let cancel_token = lease
            .as_ref()
            .map(|l| l.child_token())
            .unwrap_or_else(|| endpoint.drt().child_token());

        let push_endpoint = PushEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(cancel_token.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        // Register service in etcd
        let info = Instance {
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
            namespace: endpoint.component.namespace.name.clone(),
            instance_id: lease_id,
            transport: TransportType::NatsTcp(endpoint.subject_to(lease_id)),
        };

        let info = serde_json::to_vec_pretty(&info)?;

        if let Some(etcd_client) = &endpoint.component.drt.etcd_client {
            if let Err(e) = etcd_client
                .kv_create(
                    endpoint.etcd_path_with_lease_id(lease_id),
                    info,
                    Some(lease_id),
                )
                .await
            {
                tracing::error!("Failed to register discoverable service: {:?}", e);
                cancel_token.cancel();
                return Err(error!("Failed to register discoverable service"));
            }
        }

        match endpoint.drt().try_start_http_management_service().await? {
            Some(()) => {
                // no HTTP started, start it
                let http_management_info = HttpManagementInfo {
                    drt: endpoint.component.drt.clone(),
                    port: 0,
                    component: endpoint.component.clone(),
                };

                // Determine port to use
                let port = http_management_port.unwrap_or(0); // Use 0 for auto-assignment

                let http_task = tokio::spawn({
                    let http_management_info = http_management_info.clone();
                    let cancel_token = cancel_token.clone();
                    let drt = endpoint.drt().clone();
                    async move {
                        match start_aggregated_http_service(
                            http_management_info,
                            port,
                            cancel_token.child_token(),
                        )
                        .await
                        {
                            Ok(_) => {
                                tracing::info!("HTTP management service ended");
                                // Clear the service info when it ends
                                drt.clear_http_management_service().await;
                            }
                            Err(e) => {
                                tracing::error!("HTTP management service failed: {}", e);
                                // Clear the service info on error
                                drt.clear_http_management_service().await;
                            }
                        }
                    }
                });

                // Complete the registration with the actual task handle
                endpoint
                    .drt()
                    .complete_http_management_service_registration(http_task)
                    .await;

                tracing::info!(
                    "HTTP management service started for endpoint {}",
                    endpoint.path()
                );
            }
            None => {
                // Another endpoint already started the HTTP service
                tracing::info!("HTTP management service already running for this DRT");
            }
        };

        // Start endpoint service
        let endpoint_task = tokio::spawn({
            let endpoint = push_endpoint;
            let service_endpoint = service_endpoint;
            async move {
                tracing::info!("Starting endpoint service");
                match endpoint.start(service_endpoint).await {
                    Ok(_) => tracing::info!("Endpoint service ended"),
                    Err(e) => tracing::error!("Endpoint service failed: {}", e),
                }
            }
        });

        tracing::info!(
            "Endpoint {} started with HTTP management service",
            endpoint.path()
        );

        // Wait for endpoint service to complete or shutdown signal
        tokio::select! {
            _ = endpoint_task => {
            }
            _ = cancel_token.cancelled() => {
            }
        }

        // // Start only endpoint service (original behavior)
        // let task = tokio::spawn(push_endpoint.start(service_endpoint));
        // task.await??;

        Ok(())
    }
}

async fn start_aggregated_http_service(
    http_management_info: HttpManagementInfo,
    requested_port: u16,
    cancel_token: crate::CancellationToken,
) -> Result<u16> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/info", get(info_handler))
        .with_state(http_management_info.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], requested_port));
    let listener = TcpListener::bind(addr).await?;
    let actual_addr = listener.local_addr()?;
    let actual_port = actual_addr.port();

    // Register HTTP management port to etcd for discovery
    if let Some(etcd_client) = &http_management_info.drt.etcd_client() {
        let lease_id = http_management_info
            .drt
            .primary_lease()
            .as_ref()
            .map(|l| l.id())
            .unwrap_or(0);
        let component_path = http_management_info.component.etcd_root();
        let http_management_path = format!("{}/http_management", component_path);
        let http_info = json!({
            "port": actual_port,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "lease_id": lease_id
        });
        tracing::info!("http_info: {:?}", http_info);
        tracing::info!("http_management_path: {:?}", http_management_path);

        if let Ok(http_info_bytes) = serde_json::to_vec_pretty(&http_info) {
            if let Err(e) = etcd_client
                .kv_put(http_management_path, http_info_bytes, Some(lease_id))
                .await
            {
                tracing::error!("Failed to update HTTP management port: {:?}", e);
                cancel_token.cancel();
                return Err(error!("Failed to register discoverable http"));
            }
        }
    }

    tracing::info!("HTTP management service listening on {}", actual_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(cancel_token.cancelled_owned())
        .await
        .map_err(|e| anyhow::anyhow!("HTTP management service error: {}", e))?;

    Ok(actual_port)
}

async fn health_handler(State(info): State<HttpManagementInfo>) -> Result<Json<Value>, StatusCode> {
    let mut health_checks = json!({});
    let mut overall_healthy = true;

    // Check lease status
    if let Some(ref lease) = info.drt.primary_lease() {
        let lease_valid = !lease.primary_token().is_cancelled();
        health_checks["lease"] = json!({
            "status": if lease_valid { "healthy" } else { "unhealthy" },
            "lease_id": lease.id(),
            "cancelled": lease.primary_token().is_cancelled(),
            "details": if lease_valid {
                "Lease is active and valid"
            } else {
                "Lease has been cancelled or expired"
            }
        });

        if !lease_valid {
            overall_healthy = false;
        }
    } else {
        health_checks["lease"] = json!({
            "status": "n/a",
            "details": "No lease configured for this distributed runtime"
        });
    }

    // Check NATS connection status
    let _nats_client = info.drt.nats_client();
    // For now, just check that we have a NATS client
    // TODO: Add more checks about network connectivity
    health_checks["nats"] = json!({
        "status": "healthy",
        "details": "NATS client is available"
    });

    // Check etcd connection status (if available)
    let etcd_healthy = match info.drt.etcd_client() {
        Some(etcd_client) => {
            // Try a simple operation to check connectivity
            let mut client = etcd_client.etcd_client().clone();
            match client.get("health_check_key", None).await {
                Ok(_) => {
                    health_checks["etcd"] = json!({
                        "status": "healthy",
                        "details": "etcd connection is active"
                    });
                    true
                }
                Err(e) => {
                    health_checks["etcd"] = json!({
                        "status": "unhealthy",
                        "error": format!("{}", e),
                        "details": "etcd connection failed"
                    });
                    false
                }
            }
        }
        None => {
            health_checks["etcd"] = json!({
                "status": "n/a",
                "details": "No etcd client configured"
            });
            true // Not having etcd isn't necessarily unhealthy
        }
    };

    // TODO: Add user-customizable health checks

    // Update overall health status based on critical checks
    if !etcd_healthy
        || (info.drt.primary_lease().is_some()
            && info
                .drt
                .primary_lease()
                .as_ref()
                .unwrap()
                .primary_token()
                .is_cancelled())
    {
        overall_healthy = false;
    }

    let health_status = json!({
        "status": if overall_healthy { "healthy" } else { "unhealthy" },
        "service_type": "distributed_runtime_management",
        "instance_id": info.drt.primary_lease().as_ref().map(|l| l.id()).unwrap_or(0),
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "checks": health_checks
    });

    Ok(Json(health_status))
}

async fn metrics_handler(
    State(info): State<HttpManagementInfo>,
) -> Result<Json<Value>, StatusCode> {
    // TODO: Implement actual metrics collection
    Ok(Json(json!({
        "service_type": "distributed_runtime_management",
        "instance_id": info.drt.primary_lease().as_ref().map(|l| l.id()).unwrap_or(0),
        "metrics": {
            "requests_total": 0,
            "requests_active": 0,
            "uptime_seconds": 0
        }
    })))
}

async fn info_handler(State(info): State<HttpManagementInfo>) -> Result<Json<Value>, StatusCode> {
    Ok(Json(json!({
        "service_type": "distributed_runtime_management",
        "instance_id": info.drt.primary_lease().as_ref().map(|l| l.id()).unwrap_or(0),
        "is_static": info.drt.is_static,
    })))
}
