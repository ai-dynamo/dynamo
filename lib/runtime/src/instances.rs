// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Instance management functions for the distributed runtime.
//!
//! This module provides functionality to list and manage instances across
//! the entire distributed system, complementing the component-specific
//! instance listing in `component.rs`.

use std::sync::Arc;

use crate::component::{INSTANCE_ROOT_PATH, Instance};
use crate::storage::key_value_store::KeyValueStore;
use crate::transports::etcd::Client as EtcdClient;

pub async fn list_all_instances(client: Arc<dyn KeyValueStore>) -> anyhow::Result<Vec<Instance>> {
    let Some(bucket) = client.get_bucket(INSTANCE_ROOT_PATH).await? else {
        return Ok(vec![]);
    };

    let entries = bucket.entries().await?;
    let mut instances = Vec::with_capacity(entries.len());
    for (name, bytes) in entries.into_iter() {
        match serde_json::from_slice::<Instance>(&bytes) {
            Ok(instance) => instances.push(instance),
            Err(err) => {
                tracing::warn!(%err, key = name, "Failed to parse instance from storage");
            }
        }
    }
    instances.sort();

    Ok(instances)
}
