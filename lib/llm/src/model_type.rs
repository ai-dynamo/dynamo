// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::{Deserialize, Serialize};
use strum::Display;

#[derive(Copy, Debug, Clone, Display, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelType {
    // Chat Completions API
    Chat,
    /// Older completions API
    Completion,
    /// Embeddings API
    Embedding,
    // Pre-processed requests
    Backend,
}

impl ModelType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
            Self::Embedding => "embedding",
            Self::Backend => "backend",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![Self::Chat, Self::Completion, Self::Embedding, Self::Backend]
    }

    pub fn as_endpoint_type(&self) -> crate::endpoint_type::EndpointType {
        match self {
            Self::Chat => crate::endpoint_type::EndpointType::Chat,
            Self::Completion => crate::endpoint_type::EndpointType::Completion,
            Self::Embedding => crate::endpoint_type::EndpointType::Embedding,
            Self::Backend => panic!("Backend model type does not map to an endpoint type"),
        }
    }
}
