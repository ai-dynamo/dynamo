// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_backend_common::{EngineConfig, LlmRegistration};

/// Model identity and registration metadata for the TensorRT-LLM sidecar.
#[derive(Clone, Debug)]
pub(crate) struct ConfiguredModel {
    /// HF repo name or local path used for tokenization and templates.
    pub source: String,
    /// Maximum sequence length (input + output), from the `--context-length`
    /// argument or a server `GetModelInfo` report, if known.
    pub context_length: Option<u32>,
}

impl ConfiguredModel {
    pub(crate) fn engine_config(&self) -> EngineConfig {
        let mut runtime_data = HashMap::new();
        runtime_data.insert(
            "grpc_service".to_string(),
            serde_json::Value::String("trtllm.TrtllmService".to_string()),
        );

        EngineConfig {
            model: self.source.clone(),
            served_model_name: None,
            runtime_data,
            llm: Some(LlmRegistration {
                context_length: self.context_length,
                ..Default::default()
            }),
        }
    }
}
