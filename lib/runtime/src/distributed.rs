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
    service::ServiceClient,
    transports::{etcd, nats, tcp},
    ErrorContext,
};

use super::{error, Arc, DistributedRuntime, OnceCell, Result, Runtime, Weak, OK};

use derive_getters::Dissolve;
use figment::error;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

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

        let distributed_runtime = Self {
            runtime,
            etcd_client,
            nats_client,
            tcp_server: Arc::new(OnceCell::new()),
            component_registry: component::Registry::new(),
            is_static,
            instance_sources: Arc::new(Mutex::new(HashMap::new())),
            start_time: std::time::Instant::now(),
        };

        // Start HTTP server for health and metrics (if enabled)
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        if config.system_server_enabled() {
            let drt_arc = Arc::new(distributed_runtime.clone());
            let runtime_clone = distributed_runtime.runtime.clone();
            // spawn_http_server spawns its own background task:
            match crate::http_server::spawn_http_server(
                &config.system_host,
                config.system_port,
                runtime_clone.child_token(),
                drt_arc,
            )
            .await
            {
                Ok((addr, _handle)) => {
                    tracing::info!("HTTP server started successfully on {}", addr);
                }
                Err(e) => {
                    tracing::error!("HTTP server startup failed: {}", e);
                }
            }
        } else {
            tracing::debug!("Health and metrics HTTP server is disabled via DYN_SYSTEM_ENABLED");
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

    /// Get the uptime of this DistributedRuntime in seconds
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
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
