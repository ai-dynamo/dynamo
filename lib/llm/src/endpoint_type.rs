// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use strum::Display;

#[derive(Copy, Debug, Clone, Display, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum EndpointType {
    // Chat Completions API
    Chat,
    /// Older completions API
    Completion,
    /// Embeddings API
    Embedding,
    /// Responses API
    Responses,
}

impl EndpointType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
            Self::Embedding => "embedding",
            Self::Responses => "responses",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::Chat,
            Self::Completion,
            Self::Embedding,
            Self::Responses,
        ]
    }
}
