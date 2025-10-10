// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::protocols::tensor;

#[derive(Debug, Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRuntimeConfig {
    pub total_kv_blocks: Option<u64>,

    pub max_num_seqs: Option<u64>,

    pub max_num_batched_tokens: Option<u64>,

    pub tool_call_parser: Option<String>,

    pub reasoning_parser: Option<String>,

    /// Mapping of engine-specific runtime configs
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub runtime_data: HashMap<String, serde_json::Value>,

    // Provide tensor model config in the case where the model type is Tensor.
    // Currently use JSON object for convinence, the programmatic way is to
    // define the model config struct as part of the tensor protocol and
    // import it here.
    // [gluo TODO] switch to ModelConfig if desired and workout a way to
    // prepare it in a convinent way, the protobuf library used by tonic
    // doesn't provide JSON parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_model_config: Option<tensor::TensorModelConfig>,
}

impl ModelRuntimeConfig {
    pub fn new() -> Self {
        Self::default()
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
