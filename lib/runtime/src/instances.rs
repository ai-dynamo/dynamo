// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Instance management functions for the distributed runtime.
//!
//! This module provides functionality to list and manage instances across
//! the entire distributed system, complementing the component-specific
//! instance listing in `component.rs`.

use crate::component::{INSTANCE_ROOT_PATH, Instance};
use crate::transports::etcd::Client as EtcdClient;

pub async fn list_all_instances(etcd_client: &EtcdClient) -> anyhow::Result<Vec<Instance>> {
    let mut instances = Vec::new();

    for kv in etcd_client
        .kv_get_prefix(format!("{}/", INSTANCE_ROOT_PATH))
        .await?
    {
        match serde_json::from_slice::<Instance>(kv.value()) {
            Ok(instance) => instances.push(instance),
            Err(err) => {
                tracing::warn!(
                    "Failed to parse instance from etcd: {}. Key: {}, Value: {}",
                    err,
                    kv.key_str().unwrap_or("invalid_key"),
                    kv.value_str().unwrap_or("invalid_value")
                );
            }
        }
    }

    Ok(instances)
}
