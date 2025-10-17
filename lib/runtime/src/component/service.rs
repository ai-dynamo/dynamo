// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_nats::service::Service as NatsService;
use async_nats::service::ServiceExt as _;
use derive_builder::Builder;
use derive_getters::Dissolve;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::component::Component;

pub use super::endpoint::EndpointStats;

use educe::Educe;

type StatsHandlerRegistry = Arc<Mutex<HashMap<String, EndpointStatsHandler>>>;
pub type StatsHandler =
    Box<dyn FnMut(String, EndpointStats) -> serde_json::Value + Send + Sync + 'static>;
pub type EndpointStatsHandler =
    Box<dyn FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static>;

pub const PROJECT_NAME: &str = "Dynamo";
const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

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
    pub async fn create(self) -> anyhow::Result<Component> {
        let (component, description) = self.build_internal()?.dissolve();

        let service_name = component.service_name();

        // Pre-check to save cost of creating the service, but don't hold the lock
        if component
            .drt
            .component_registry
            .inner
            .lock()
            .await
            .services
            .contains_key(&service_name)
        {
            anyhow::bail!("Service {service_name} already exists");
        }

        let Some(nats_client) = component.drt.nats_client() else {
            anyhow::bail!("Cannot create NATS service without NATS.");
        };
        let (nats_service, stats_reg) =
            build_nats_service(nats_client, &component, description).await?;

        let mut guard = component.drt.component_registry.inner.lock().await;
        if !guard.services.contains_key(&service_name) {
            // Normal case
            guard.services.insert(service_name.clone(), nats_service);
            guard.stats_handlers.insert(service_name, stats_reg);
            drop(guard);
        } else {
            drop(guard);
            let _ = nats_service.stop().await;
            return Err(anyhow::anyhow!(
                "Service create race for {service_name}, now already exists"
            ));
        }

        // Register metrics callback. CRITICAL: Never fail service creation for metrics issues.
        if let Err(err) = component.start_scraping_nats_service_component_metrics() {
            tracing::debug!(
                "Metrics registration failed for '{}': {}",
                component.service_name(),
                err
            );
        }
        Ok(component)
    }
}

async fn build_nats_service(
    nats_client: &crate::transports::nats::Client,
    component: &Component,
    description: Option<String>,
) -> anyhow::Result<(NatsService, StatsHandlerRegistry)> {
    let service_name = component.service_name();
    tracing::trace!("component: {component}; creating, service_name: {service_name}");

    let description = description.unwrap_or(format!(
        "{PROJECT_NAME} component {} in namespace {}",
        component.name, component.namespace
    ));

    let stats_handler_registry: StatsHandlerRegistry = Arc::new(Mutex::new(HashMap::new()));
    let stats_handler_registry_clone = stats_handler_registry.clone();

    let nats_service_builder = nats_client.client().service_builder();

    let nats_service_builder =
        nats_service_builder
            .description(description)
            .stats_handler(move |name, stats| {
                tracing::trace!("stats_handler: {name}, {stats:?}");
                let mut guard = stats_handler_registry.lock().unwrap();
                match guard.get_mut(&name) {
                    Some(handler) => handler(stats),
                    None => serde_json::Value::Null,
                }
            });
    let nats_service = nats_service_builder
        .start(service_name, SERVICE_VERSION.to_string())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start NATS service: {e}"))?;

    Ok((nats_service, stats_handler_registry_clone))
}

impl ServiceConfigBuilder {
    pub(crate) fn from_component(component: Component) -> Self {
        Self::default().component(component)
    }
}
