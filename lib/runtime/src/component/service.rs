// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::Component;
use async_nats::service::{Service as NatsService, ServiceExt};

pub const PROJECT_NAME: &str = "Dynamo";
const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimal NATS service builder to support legacy NATS request plane.
/// This will be removed once all components migrate to TCP request plane.
pub async fn build_nats_service(
    nats_client: &crate::transports::nats::Client,
    component: &Component,
    description: Option<String>,
) -> anyhow::Result<NatsService> {
    let service_name = component.service_name();
    tracing::trace!("component: {component}; creating NATS service, service_name: {service_name}");

    let description = description.unwrap_or(format!(
        "{PROJECT_NAME} component {} in namespace {}",
        component.name, component.namespace
    ));

    let nats_service = nats_client
        .client()
        .service_builder()
        .description(description)
        .start(service_name, SERVICE_VERSION.to_string())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start NATS service: {e}"))?;

    Ok(nats_service)
}
