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

pub use crate::component::Component;
use crate::{
    component::{self, ComponentBuilder, Endpoint, InstanceSource, Namespace},
    discovery::DiscoveryClient,
    metrics::MetricsRegistry,
    service::ServiceClient,
    transports::{etcd, nats, tcp},
    ErrorContext,
};

use super::{error, Arc, DistributedRuntime, OnceCell, Result, Runtime, SystemHealth, Weak, OK};
use std::sync::OnceLock;

use derive_getters::Dissolve;
use figment::error;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

impl MetricsRegistry for DistributedRuntime {
    fn basename(&self) -> String {
        "".to_string() // drt has no basename. Basename only begins with the Namespace.
    }

    fn parent_hierarchy(&self) -> Vec<String> {
        vec![] // drt is the root, so no parent hierarchy
    }
}

impl DistributedRuntime {
    pub async fn new(runtime: Runtime, config: DistributedConfig) -> Result<Self> {
        let secondary = runtime.secondary();
        let (etcd_config, nats_config, is_static) = config.dissolve();

        let runtime_clone = runtime.clone();

        let etcd_client = if is_static {
            None
        } else {
            Some(
                secondary
                    .spawn(async move {
                        let client = etcd::Client::new(etcd_config.clone(), runtime_clone)
                            .await
                            .context(format!(
                                "Failed to connect to etcd server with config {:?}",
                                etcd_config
                            ))?;
                        OK(client)
                    })
                    .await??,
            )
        };

        let nats_client = secondary
            .spawn(async move {
                let client = nats_config.clone().connect().await.context(format!(
                    "Failed to connect to NATS server with config {:?}",
                    nats_config
                ))?;
                anyhow::Ok(client)
            })
            .await??;

        // Start system metrics server for health and metrics if enabled in configuration
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        // IMPORTANT: We must extract cancel_token from runtime BEFORE moving runtime into the struct below.
        // This is because after moving, runtime is no longer accessible in this scope (ownership rules).
        let cancel_token = if config.system_server_enabled() {
            Some(runtime.clone().child_token())
        } else {
            None
        };
        let starting_health_status = config.starting_health_status.clone();
        let use_endpoint_health_status = config.use_endpoint_health_status.clone();
        let health_endpoint_path = config.system_health_path.clone();
        let live_endpoint_path = config.system_live_path.clone();
        let system_health = Arc::new(std::sync::Mutex::new(SystemHealth::new(
            starting_health_status,
            use_endpoint_health_status,
            health_endpoint_path,
            live_endpoint_path,
        )));

        let distributed_runtime = Self {
            runtime,
            etcd_client,
            nats_client,
            tcp_server: Arc::new(OnceCell::new()),
            metrics_server: Arc::new(OnceLock::new()),
            component_registry: component::Registry::new(),
            is_static,
            instance_sources: Arc::new(Mutex::new(HashMap::new())),
            prometheus_registries_by_prefix: Arc::new(std::sync::Mutex::new(HashMap::<
                String,
                prometheus::Registry,
            >::new())),
            system_health,
        };

        // Start metrics server if enabled
        if let Some(cancel_token) = cancel_token {
            let host = config.system_host.clone();
            let port = config.system_port;

            // Start metrics server (it spawns its own task internally)
            match crate::metrics_server::spawn_metrics_server(
                &host,
                port,
                cancel_token,
                Arc::new(distributed_runtime.clone()),
            )
            .await
            {
                Ok((addr, handle)) => {
                    tracing::info!("Metrics server started successfully on {}", addr);

                    // Store metrics server information
                    let metrics_server_info =
                        crate::metrics_server::MetricsServerInfo::new(addr, Some(handle));

                    // Initialize the metrics_server field
                    distributed_runtime
                        .metrics_server
                        .set(Arc::new(metrics_server_info))
                        .expect("Metrics server info should only be set once");
                }
                Err(e) => {
                    tracing::error!("Metrics server startup failed: {}", e);
                }
            }
        } else {
            tracing::debug!("Health and metrics server is disabled via DYN_SYSTEM_ENABLED");
        }

        Ok(distributed_runtime)
    }

    pub async fn from_settings(runtime: Runtime) -> Result<Self> {
        let config = DistributedConfig::from_settings(false);
        Self::new(runtime, config).await
    }

    // Call this if you are using static workers that do not need etcd-based discovery.
    pub async fn from_settings_without_discovery(runtime: Runtime) -> Result<Self> {
        let config = DistributedConfig::from_settings(true);
        Self::new(runtime, config).await
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn primary_token(&self) -> CancellationToken {
        self.runtime.primary_token()
    }

    /// The etcd lease all our components will be attached to.
    /// Not available for static workers.
    pub fn primary_lease(&self) -> Option<etcd::Lease> {
        self.etcd_client.as_ref().map(|c| c.primary_lease())
    }

    pub fn shutdown(&self) {
        self.runtime.shutdown();
    }

    /// Create a [`Namespace`]
    pub fn namespace(&self, name: impl Into<String>) -> Result<Namespace> {
        Namespace::new(self.clone(), name.into(), self.is_static)
    }

    // /// Create a [`Component`]
    // pub fn component(
    //     &self,
    //     name: impl Into<String>,
    //     namespace: impl Into<String>,
    // ) -> Result<Component> {
    //     Ok(ComponentBuilder::from_runtime(self.clone())
    //         .name(name.into())
    //         .namespace(namespace.into())
    //         .build()?)
    // }

    pub(crate) fn discovery_client(&self, namespace: impl Into<String>) -> DiscoveryClient {
        DiscoveryClient::new(
            namespace.into(),
            self.etcd_client
                .clone()
                .expect("Attempt to get discovery_client on static DistributedRuntime"),
        )
    }

    pub(crate) fn service_client(&self) -> ServiceClient {
        ServiceClient::new(self.nats_client.clone())
    }

    pub async fn tcp_server(&self) -> Result<Arc<tcp::server::TcpStreamServer>> {
        Ok(self
            .tcp_server
            .get_or_try_init(async move {
                let options = tcp::server::ServerOptions::default();
                let server = tcp::server::TcpStreamServer::new(options).await?;
                OK(server)
            })
            .await?
            .clone())
    }

    pub fn nats_client(&self) -> nats::Client {
        self.nats_client.clone()
    }

    /// Get metrics server information if available
    pub fn metrics_server_info(&self) -> Option<Arc<crate::metrics_server::MetricsServerInfo>> {
        self.metrics_server.get().cloned()
    }

    // todo(ryan): deprecate this as we move to Discovery traits and Component Identifiers
    pub fn etcd_client(&self) -> Option<etcd::Client> {
        self.etcd_client.clone()
    }

    pub fn child_token(&self) -> CancellationToken {
        self.runtime.child_token()
    }

    pub fn instance_sources(&self) -> Arc<Mutex<HashMap<Endpoint, Weak<InstanceSource>>>> {
        self.instance_sources.clone()
    }
}

#[derive(Dissolve)]
pub struct DistributedConfig {
    pub etcd_config: etcd::ClientOptions,
    pub nats_config: nats::ClientOptions,
    pub is_static: bool,
}

impl DistributedConfig {
    pub fn from_settings(is_static: bool) -> DistributedConfig {
        DistributedConfig {
            etcd_config: etcd::ClientOptions::default(),
            nats_config: nats::ClientOptions::default(),
            is_static,
        }
    }

    pub fn for_cli() -> DistributedConfig {
        let mut config = DistributedConfig {
            etcd_config: etcd::ClientOptions::default(),
            nats_config: nats::ClientOptions::default(),
            is_static: false,
        };

        config.etcd_config.attach_lease = false;

        config
    }
}
