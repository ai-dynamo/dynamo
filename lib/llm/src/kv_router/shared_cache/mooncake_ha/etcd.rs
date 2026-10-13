// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! etcd-backed Mooncake master leader discovery.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::{MooncakeLeaderResolver, MooncakeLeaderUnavailable};

struct EtcdMooncakeLeaderResolver {
    client: Mutex<etcd_client::Client>,
    master_view_key: String,
}

#[async_trait]
impl MooncakeLeaderResolver for EtcdMooncakeLeaderResolver {
    async fn current_leader(&self) -> anyhow::Result<String> {
        let mut client = self.client.lock().await;
        let response = client
            .get(self.master_view_key.as_str(), None)
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "failed to read Mooncake master view {} from etcd: {error}",
                    self.master_view_key
                )
            })?;
        let kv = response
            .kvs()
            .first()
            .ok_or_else(|| MooncakeLeaderUnavailable {
                message: format!(
                    "Mooncake master view {} is absent in etcd",
                    self.master_view_key
                ),
            })?;
        let leader_address = std::str::from_utf8(kv.value())?.trim();
        if leader_address.is_empty() {
            return Err(MooncakeLeaderUnavailable {
                message: format!(
                    "Mooncake master view {} contains an empty leader address",
                    self.master_view_key
                ),
            }
            .into());
        }
        Ok(leader_address.to_string())
    }
}

pub(super) async fn build_leader_resolver(
    endpoints: Vec<String>,
    cluster_id: &str,
) -> anyhow::Result<Arc<dyn MooncakeLeaderResolver>> {
    let client = etcd_client::Client::connect(endpoints, None).await?;
    Ok(Arc::new(EtcdMooncakeLeaderResolver {
        client: Mutex::new(client),
        master_view_key: mooncake_master_view_key(cluster_id),
    }))
}

pub(super) fn mooncake_master_view_key(cluster_id: &str) -> String {
    let normalized = cluster_id.strip_suffix('/').unwrap_or(cluster_id);
    format!("mooncake-store/{normalized}/master_view")
}
