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
    component::{self, ComponentBuilder, Namespace},
    discovery::DiscoveryClient,
    service::ServiceClient,
    transports::{etcd, nats, tcp},
    ErrorContext,
};

use super::{error, Arc, DistributedRuntime, OnceCell, Result, Runtime, OK};

use derive_getters::Dissolve;
use figment::error;
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

        Ok(Self {
            runtime,
            etcd_client,
            nats_client,
            tcp_server: Arc::new(OnceCell::new()),
            component_registry: component::Registry::new(),
            is_static,
        })
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

    pub fn etcd_client(&self) -> Option<etcd::Client> {
        self.etcd_client.clone()
    }

    pub fn child_token(&self) -> CancellationToken {
        self.runtime.child_token()
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
