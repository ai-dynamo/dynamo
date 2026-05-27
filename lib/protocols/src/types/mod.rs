// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Types used in inference API requests and responses.
//
// Base OpenAI types are re-exported from upstream async-openai.
// Inference-serving extensions and Anthropic types are locally defined.

// --- Locally defined modules ---
pub mod anthropic;
mod chat;
mod completion;
mod embeddings;
pub mod realtime;
pub mod responses;

// --- Local type re-exports ---
pub use chat::*;
pub use completion::*;

// Embeddings: `Embedding` and `CreateEmbeddingResponse` are owned
// locally (in the `embeddings` submodule) so a per-input embedding
// can be either a float vector or a base64 string per the OpenAI
// `encoding_format` spec. Every other embedding type is still
// re-exported from `async_openai::types::embeddings::*`.
pub use embeddings::*;

// Images: re-exported wholesale from async-openai (no Dynamo overrides).
pub use async_openai::types::images::*;

// --- Convenience impls for locally-defined types ---
mod impls;

use crate::error::OpenAIError;
use derive_builder::UninitializedFieldError;

impl From<UninitializedFieldError> for OpenAIError {
    fn from(value: UninitializedFieldError) -> Self {
        OpenAIError::InvalidArgument(value.to_string())
    }
}
