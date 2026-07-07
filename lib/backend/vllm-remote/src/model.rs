// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{EngineConfig, LlmRegistration};

#[derive(Clone, Debug)]
pub(crate) struct ConfiguredModel {
    pub source: String,
    pub served_name: String,
}

impl ConfiguredModel {
    pub(crate) fn engine_config(&self) -> EngineConfig {
        EngineConfig {
            model: self.source.clone(),
            served_model_name: Some(self.served_name.clone()),
            runtime_data: Default::default(),
            llm: Some(LlmRegistration::default()),
        }
    }
}
