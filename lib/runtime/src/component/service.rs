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
use std::collections::HashMap;
use std::sync::Mutex;

use super::*;

pub use super::endpoint::EndpointStats;
pub type StatsHandler =
    Box<dyn FnMut(String, EndpointStats) -> serde_json::Value + Send + Sync + 'static>;
pub type EndpointStatsHandler =
    Box<dyn FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static>;

pub const PROJECT_NAME: &str = "Dynamo";

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct ServiceConfig {
    #[builder(private)]
    component: Component,

    /// Description
    #[builder(default)]
    description: Option<String>,
}

impl ServiceConfigBuilder {
    /// Create the [`Component`]'s service and store it in the registry.
    pub async fn create(self) -> Result<Component> {
        let (component, description) = self.build_internal()?.dissolve();

        let version = "0.0.1".to_string();

        let service_name = component.service_name();
        log::debug!("component: {component}; creating, service_name: {service_name}");

        let description = description.unwrap_or(format!(
            "{PROJECT_NAME} component {} in namespace {}",
            component.name, component.namespace
        ));

        let stats_handler_registry: Arc<Mutex<HashMap<String, EndpointStatsHandler>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let stats_handler_registry_clone = stats_handler_registry.clone();

        let mut guard = component.drt.component_registry.inner.lock().await;

        if guard.services.contains_key(&service_name) {
            return Err(anyhow::anyhow!("Service already exists"));
        }

        // create service on the secondary runtime
        let builder = component.drt.nats_client.client().service_builder();

        tracing::debug!("Starting service: {}", service_name);
        let service_builder = builder
            .description(description)
            .stats_handler(move |name, stats| {
                log::trace!("stats_handler: {name}, {stats:?}");
                let mut guard = stats_handler_registry.lock().unwrap();
                match guard.get_mut(&name) {
                    Some(handler) => handler(stats),
                    None => serde_json::Value::Null,
                }
            });
        tracing::debug!("Got builder");
        let service = service_builder
            .start(service_name.clone(), version)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start service: {e}"))?;

        // new copy of service_name as the previous one is moved into the task above
        let service_name = component.service_name();

        // insert the service into the registry
        guard.services.insert(service_name.clone(), service);

        // insert the stats handler into the registry
        guard
            .stats_handlers
            .insert(service_name, stats_handler_registry_clone);

        // drop the guard to unlock the mutex
        drop(guard);

        Ok(component)
    }
}

impl ServiceConfigBuilder {
    pub(crate) fn from_component(component: Component) -> Self {
        Self::default().component(component)
    }
}
