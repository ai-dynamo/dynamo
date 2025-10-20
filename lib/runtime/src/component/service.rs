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

type StatsHandlerRegistry = Arc<Mutex<HashMap<String, EndpointStatsHandler>>>;
pub type StatsHandler =
    Box<dyn FnMut(String, EndpointStats) -> serde_json::Value + Send + Sync + 'static>;
pub type EndpointStatsHandler =
    Box<dyn FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static>;

pub const PROJECT_NAME: &str = "Dynamo";
const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn build_nats_service(
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
