// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Instance management functions for the distributed runtime.
//!
//! This module provides functionality to list and manage instances across
//! the entire distributed system, complementing the component-specific
//! instance listing in `component.rs`.

use std::sync::Arc;

use crate::component::Instance;
use crate::discovery::{Discovery, DiscoveryQuery};

pub async fn list_all_instances(
    discovery_client: Arc<dyn Discovery>,
) -> anyhow::Result<Vec<Instance>> {
    let discovery_instances = discovery_client.list(DiscoveryQuery::AllEndpoints).await?;

    let mut instances: Vec<Instance> = discovery_instances
        .into_iter()
        .filter_map(|di| match di {
            crate::discovery::DiscoveryInstance::Endpoint(instance) => Some(instance),
            _ => None, // Ignore all other variants (ModelCard, etc.)
        })
        .collect();

    instances.sort();

    Ok(instances)
}
