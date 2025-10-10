// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::fmt;
use strum::Display;

bitflags! {
    /// Represents the set of model capabilities (endpoints) a model can support.
    ///
    /// This type is implemented using `bitflags` instead of a plain `enum`
    /// so that multiple capabilities can be combined in a single value:
    ///
    /// - `ModelType::Chat`
    /// - `ModelType::Completions`
    /// - `ModelType::Embedding`
    /// - `ModelType::TensorBased`
    ///
    /// For example, a model that supports both chat and completions can be
    /// expressed as:
    ///
    /// ```rust
    /// use dynamo_llm::model_type::ModelType;
    /// let mt = ModelType::Chat | ModelType::Completions;
    /// assert!(mt.supports_chat());
    /// assert!(mt.supports_completions());
    /// ```
    ///
    /// Using bitflags avoids deep branching on a single enum variant,
    /// simplifies checks like `supports_chat()`, and enables efficient,
    /// type-safe combinations of multiple endpoint types within a single byte.
    #[derive(Copy, Debug, Default, Clone, Serialize, Deserialize, Eq, PartialEq)]
    pub struct ModelType: u8 {
        const Chat = 1 << 0;
        const Completions = 1 << 1;
        const Embedding = 1 << 2;
        const TensorBased = 1 << 3;
    }
}

impl ModelType {
    pub fn as_str(&self) -> String {
        self.as_vec().join(",")
    }

    pub fn supports_chat(&self) -> bool {
        self.contains(ModelType::Chat)
    }
    pub fn supports_completions(&self) -> bool {
        self.contains(ModelType::Completions)
    }
    pub fn supports_embedding(&self) -> bool {
        self.contains(ModelType::Embedding)
    }
    pub fn supports_tensor(&self) -> bool {
        self.contains(ModelType::TensorBased)
    }

    pub fn as_vec(&self) -> Vec<&'static str> {
        let mut result = Vec::new();
        if self.supports_chat() {
            result.push("chat");
        }
        if self.supports_completions() {
            result.push("completions");
        }
        if self.supports_embedding() {
            result.push("embedding");
        }
        if self.supports_tensor() {
            result.push("tensor");
        }
        result
    }

    /// Decompose the bitflag into it's component units:
    /// Chat | Completion -> [Chat, Completion]
    pub fn units(&self) -> Vec<ModelType> {
        let mut result = Vec::new();
        if self.supports_chat() {
            result.push(ModelType::Chat);
        }
        if self.supports_completions() {
            result.push(ModelType::Completions);
        }
        if self.supports_embedding() {
            result.push(ModelType::Embedding);
        }
        if self.supports_tensor() {
            result.push(ModelType::TensorBased);
        }
        result
    }

    /// Returns all endpoint types supported by this model type.
    /// This properly handles combinations like Chat | Completions.
    pub fn as_endpoint_types(&self) -> Vec<crate::endpoint_type::EndpointType> {
        let mut endpoint_types = Vec::new();
        if self.contains(Self::Chat) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Chat);
        }
        if self.contains(Self::Completions) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Completion);
        }
        if self.contains(Self::Embedding) {
            endpoint_types.push(crate::endpoint_type::EndpointType::Embedding);
        }
        // [gluo NOTE] ModelType::Tensor doesn't map to any endpoint type,
        // current use of endpoint type is LLM specific and so does the HTTP
        // server that uses it.
        endpoint_types
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Copy, Debug, Default, Clone, Display, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelInput {
    /// Raw text input
    #[default]
    Text,
    /// Pre-processed input
    Tokens,
    /// Tensor input
    Tensor,
}

impl ModelInput {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Tokens => "tokens",
            Self::Tensor => "tensor",
        }
    }
}
