// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-advertised Mooncake HA metadata and rolling-upgrade reconciliation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub(in crate::kv_router::shared_cache) struct MooncakeHaConfig {
    #[serde(default)]
    pub(in crate::kv_router::shared_cache) master_server_address: Option<String>,
    #[serde(default)]
    pub(in crate::kv_router::shared_cache) cluster_id: Option<String>,
    #[serde(default)]
    pub(in crate::kv_router::shared_cache) redis_db_index: Option<u16>,
}

impl MooncakeHaConfig {
    pub(in crate::kv_router::shared_cache) fn is_compatible_with(&self, other: &Self) -> bool {
        optional_values_compatible(
            self.master_server_address.as_ref(),
            other.master_server_address.as_ref(),
        ) && optional_values_compatible(self.cluster_id.as_ref(), other.cluster_id.as_ref())
            && optional_values_compatible(
                self.redis_db_index.as_ref(),
                other.redis_db_index.as_ref(),
            )
    }

    pub(in crate::kv_router::shared_cache) fn enriched_with(&self, previous: &Self) -> Self {
        Self {
            master_server_address: self
                .master_server_address
                .clone()
                .or_else(|| previous.master_server_address.clone()),
            cluster_id: self
                .cluster_id
                .clone()
                .or_else(|| previous.cluster_id.clone()),
            redis_db_index: self.redis_db_index.or(previous.redis_db_index),
        }
    }

    pub(in crate::kv_router::shared_cache) fn merge<'a>(
        configs: impl IntoIterator<Item = &'a Self>,
    ) -> anyhow::Result<Self> {
        let mut merged = Self::default();
        for config in configs {
            merge_optional_string(
                &mut merged.master_server_address,
                config.master_server_address.as_deref(),
                "master locators",
            )?;
            merge_optional_string(
                &mut merged.cluster_id,
                config.cluster_id.as_deref(),
                "cluster IDs",
            )?;
            merge_optional_value(
                &mut merged.redis_db_index,
                config.redis_db_index,
                "Redis DB indices",
            )?;
        }
        Ok(merged)
    }
}

fn optional_values_compatible<T: PartialEq>(lhs: Option<&T>, rhs: Option<&T>) -> bool {
    lhs.is_none() || rhs.is_none() || lhs == rhs
}

fn merge_optional_string(
    merged: &mut Option<String>,
    candidate: Option<&str>,
    field: &'static str,
) -> anyhow::Result<()> {
    let candidate = candidate.filter(|value| !value.is_empty());
    merge_optional_value(merged, candidate.map(str::to_string), field)
}

fn merge_optional_value<T: PartialEq>(
    merged: &mut Option<T>,
    candidate: Option<T>,
    field: &'static str,
) -> anyhow::Result<()> {
    let Some(candidate) = candidate else {
        return Ok(());
    };
    if let Some(current) = merged.as_ref() {
        anyhow::ensure!(
            current == &candidate,
            "Mooncake HA {field} differ across workers"
        );
    } else {
        *merged = Some(candidate);
    }
    Ok(())
}
