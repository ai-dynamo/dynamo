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
    /// Images API (Diffusion/DALL-E)
    Images,
    /// Audios API (speech/audio generation)
    Audios,
    /// Videos API (video generation)
    Videos,
    /// Realtime API (bidirectional streaming over WebSocket)
    Realtime,
    /// Responses API
    Responses,
    /// Anthropic Messages API
    AnthropicMessages,
    /// Generate API (token-in/token-out)
    Generate,
    /// OpenAI Batch API
    Batch,
}

impl EndpointType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
            Self::Embedding => "embedding",
            Self::Images => "images",
            Self::Audios => "audios",
            Self::Videos => "videos",
            Self::Realtime => "realtime",
            Self::Responses => "responses",
            Self::AnthropicMessages => "anthropic_messages",
            Self::Generate => "generate",
            Self::Batch => "batch",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            Self::Chat,
            Self::Completion,
            Self::Embedding,
            Self::Images,
            Self::Audios,
            Self::Videos,
            Self::Realtime,
            Self::Responses,
            Self::AnthropicMessages,
            Self::Generate,
            Self::Batch,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_as_str() {
        assert_eq!(EndpointType::Realtime.as_str(), "realtime");
    }

    #[test]
    fn realtime_in_all() {
        assert!(EndpointType::all().contains(&EndpointType::Realtime));
    }

    #[test]
    fn generate_as_str() {
        assert_eq!(EndpointType::Generate.as_str(), "generate");
    }

    #[test]
    fn generate_in_all() {
        assert!(EndpointType::all().contains(&EndpointType::Generate));
    }

    #[test]
    fn batch_as_str() {
        assert_eq!(EndpointType::Batch.as_str(), "batch");
    }

    #[test]
    fn batch_in_all() {
        assert!(EndpointType::all().contains(&EndpointType::Batch));
    }
}
