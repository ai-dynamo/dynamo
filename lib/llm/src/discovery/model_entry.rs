// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{protocols, slug::Slug};
use serde::{Deserialize, Serialize};

use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    model_type::{ModelInput, ModelType},
};

/// [ModelEntry] contains the information to discover models
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelEntry {
    /// Public name of the model
    /// Used to identify the model in the HTTP service from the value used in an OpenAI ChatRequest.
    pub name: String,

    /// How to address this on the network
    #[serde(rename = "endpoint")]
    pub endpoint_id: protocols::EndpointId,

    /// Specifies whether the model is a chat, completions, etc model.
    pub model_type: ModelType,

    /// Runtime configuration specific to this model instance
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_config: Option<ModelRuntimeConfig>,

    /// Specifies the model input type.
    /// `Tokens` for engines that expect pre-processed input.
    /// `Text` for engines that take care of pre-processing themselves.
    pub model_input: ModelInput,
}

impl ModelEntry {
    /// Slugified display name for use in network storage, or URL-safe environments
    pub fn slug(&self) -> Slug {
        Slug::from_string(&self.name)
    }

    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_input, ModelInput::Tokens)
    }
}
