// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelRuntimeConfig {
    /// The total number of KV blocks available
    pub total_kv_blocks: Option<u64>,

    /// The maximum number of sequences that can be batched together
    pub max_num_seqs: Option<u64>,

    /// GPU memory utilization percentage configured
    pub gpu_memory_utilization: Option<u64>,

    /// Mapping of engine-specific runtime configs
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub runtime_data: HashMap<String, serde_json::Value>,
}

impl ModelRuntimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_total_kv_blocks(&mut self, total_kv_blocks: u64) {
        self.total_kv_blocks = Some(total_kv_blocks);
    }

    pub fn with_max_num_seqs(&mut self, max_num_seqs: u64) {
        self.max_num_seqs = Some(max_num_seqs);
    }

    pub fn with_gpu_memory_utilization(&mut self, gpu_memory_utilization: u64) {
        self.gpu_memory_utilization = Some(gpu_memory_utilization);
    }

    pub fn set_engine_specific<T: Serialize>(&mut self, key: &str, value: T) -> anyhow::Result<()> {
        self.runtime_data
            .insert(key.to_string(), serde_json::to_value(value)?);
        Ok(())
    }

    pub fn get_engine_specific<T: DeserializeOwned>(&self, key: &str) -> anyhow::Result<Option<T>> {
        if let Some(value) = self.runtime_data.get(key) {
            Ok(Some(serde_json::from_value(value.clone())?))
        } else {
            Ok(None)
        }
    }
}
