// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context as _;

use crate::discovery::{ModelEntry, MODEL_ROOT_PATH};
use dynamo_runtime::component::{self, Instance};
use dynamo_runtime::slug::Slug;
use dynamo_runtime::transports::etcd;

#[derive(Debug, Clone)]
pub struct ModelNetworkName(String);

impl ModelNetworkName {
    /// Key to store this model entry in networked key-value store (etcd).
    ///
    /// It looks like this:
    /// ns.cp.ep-694d967ca5efd804
    fn from_parts(namespace: &str, component: &str, endpoint: &str, lease_id: i64) -> Self {
        let model_root = MODEL_ROOT_PATH;
        let slug = Slug::slugify(&format!("{namespace}.{component}.{endpoint}-{lease_id:x}"));
        ModelNetworkName(format!("{model_root}/{slug}"))
    }

    // We can't do From<&component::Endpoint> here because we also need the lease_id
    pub fn from_local(endpoint: &component::Endpoint, lease_id: i64) -> Self {
        Self::from_parts(
            &endpoint.component().namespace().to_string(),
            &endpoint.component().name(),
            endpoint.name(),
            lease_id,
        )
    }

    pub fn from_entry(entry: &ModelEntry, lease_id: i64) -> Self {
        Self::from_parts(
            &entry.endpoint.namespace,
            &entry.endpoint.component,
            &entry.endpoint.name,
            lease_id,
        )
    }

    /// Fetch the ModelEntry from etcd.
    pub async fn load_entry(&self, etcd_client: &etcd::Client) -> anyhow::Result<ModelEntry> {
        let mut model_entries = etcd_client.kv_get(self.to_string(), None).await?;
        if model_entries.is_empty() {
            anyhow::bail!("No ModelEntry in etcd for key {self}");
        }
        let model_entry = model_entries.remove(0);
        serde_json::from_slice(model_entry.value()).with_context(|| {
            format!(
                "Error deserializing JSON. Key={self}. JSON={}",
                model_entry.value_str().unwrap_or("INVALID UTF-8")
            )
        })
    }
}

impl From<&Instance> for ModelNetworkName {
    fn from(cei: &Instance) -> Self {
        Self::from_parts(
            &cei.namespace,
            &cei.component,
            &cei.endpoint,
            cei.instance_id,
        )
    }
}

impl std::fmt::Display for ModelNetworkName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
